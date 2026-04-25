"""Reidemeister-augmented variant of `KnotDataset`.

PyG idiom for one-to-many stochastic augmentation that needs precomputation:
expand the dataset N -> N*K at process() time and persist to disk. PyG's
`processed_dir` then provides content-addressed caching as long as the augment
config is encoded into the dataset `root` (loader's responsibility).

NOTE: augmentation is intended to be applied to the *training* split only.
Splitting must therefore happen by knot identity (`knot_name`) so a knot's
augmentations cannot leak across train/val/test.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import snappy
import torch
from torch_geometric.data import Data

from src.data.dataset import KnotDataset, build_pyg_data_from_pd, read_csv
from src.data.knot import KnotDiagramTopology

SIGN_FLIP_TASKS = {"Signature", "Rasmussen <i>s</i>", "Ozsvath-Szabo <i>tau</i>"}
MIRROR_SAFE_TASKS = {
    "Alternating",
    "Crossing Number",
    "Unknotting Number",
    "Determinant",
    "Genus-3D",
    "Genus-4D",
    "Genus-4D (Top.)",
    "Arf Invariant",
}


def _fix_pd_notation(pd_notation: str) -> str:
    return pd_notation.replace(";", ",")


def _mirror_target(y: torch.Tensor, task_col: str) -> torch.Tensor:
    if task_col in MIRROR_SAFE_TASKS:
        return y
    if task_col in SIGN_FLIP_TASKS:
        return (-y).to(y.dtype)
    raise ValueError(f"Unknown mirror behavior for task_col={task_col!r}")


@dataclass(frozen=True)
class AugmentConfig:
    n_diagrams: int = 10
    tries: int = 100
    method: str = "backtrack"
    include_mirror: bool = True
    seed: int = 0


class KnotAugmentedDataset(KnotDataset):
    """Pre-expands every CSV row to N+K Data objects via SnapPy's
    `Link.many_diagrams`, optionally including the mirror image (with
    task-aware target sign flip)."""

    def __init__(
        self,
        root,
        csv_path,
        target_column,
        *,
        n_diagrams: int = 10,
        tries: int = 100,
        method: str = "backtrack",
        include_mirror: bool = True,
        aug_seed: int = 0,
        **kwargs,
    ) -> None:
        if method not in {"backtrack", "exterior"}:
            raise ValueError("method must be 'backtrack' or 'exterior'")
        self.aug_cfg = AugmentConfig(
            n_diagrams=int(n_diagrams),
            tries=int(tries),
            method=str(method),
            include_mirror=bool(include_mirror),
            seed=int(aug_seed),
        )
        super().__init__(root, csv_path, target_column, **kwargs)

    def process(self) -> None:
        rows = read_csv(self.csv_path, target=self.target_column, limit=self.limit)
        cfg = self.aug_cfg
        rng = np.random.default_rng(int(cfg.seed))
        task_col = self.target_column

        data_list: list[Data] = []
        seen: set[tuple[str, int]] = set()
        skipped_rows = 0

        for row in rows:
            base = build_pyg_data_from_pd(
                row, label_shift=self.label_shift, strict=self.strict
            )
            if base is None:
                skipped_rows += 1
                continue
            data_list.append(base)

            pd_notation = row.get("__pd_notation")
            if pd_notation is None:
                continue

            try:
                link = snappy.Link(ast.literal_eval(_fix_pd_notation(str(pd_notation))))
                diagrams = link.many_diagrams(
                    target=cfg.n_diagrams, tries=cfg.tries, method=cfg.method
                )
                rng.shuffle(diagrams)
            except Exception:
                continue

            for diagram in diagrams:
                variants = [(diagram, lambda y: y)]
                if cfg.include_mirror:
                    variants.append(
                        (diagram.mirror(), lambda y, c=task_col: _mirror_target(y, c))
                    )
                for variant_diagram, y_fn in variants:
                    try:
                        new_pd = str(variant_diagram.PD_code())
                        y_new = y_fn(base.y) if base.y is not None else None
                        key = (new_pd, int(y_new) if y_new is not None else -1)
                        if key in seen:
                            continue
                        seen.add(key)
                        topo = KnotDiagramTopology.from_pd(new_pd)
                        aug = topo.topology_to_pyg_data(
                            y=y_new,
                            knot_name=row.get("__knot_name"),
                            pd_notation=new_pd,
                        )
                        data_list.append(aug)
                    except Exception:
                        continue

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        if skipped_rows:
            print(f"[KnotAugmentedDataset] skipped {skipped_rows}/{len(rows)} base rows")
        print(
            f"[KnotAugmentedDataset] total samples after augmentation: {len(data_list)}"
        )
        self.save(data_list, self.processed_paths[0])
