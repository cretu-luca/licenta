"""Reidemeister-augmented variant of `KnotDataset`.

PyG idiom for one-to-many stochastic augmentation that needs precomputation:
expand the dataset N -> N*K at process() time and persist to disk.

Cache invalidation: PyG fingerprints `pre_transform.repr()` into
`processed_dir` and warns (does not invalidate) on changes there. Augment
kwargs (`n_diagrams`, `tries`, `method`, `include_mirror`, `aug_seed`) live
on the dataset instance itself, not on `pre_transform`, so PyG cannot see
them. `KnotAugmentedDatasetLoader` is responsible for hashing the augment
kwargs into the on-disk `root` path; do not instantiate this dataset
directly without doing the equivalent fingerprinting.

Reproducibility caveat: spherogram's `Link.many_diagrams` is sensitive to
Python's per-process hash randomization (set/dict iteration order at the C
level). Seeding `random` and `numpy.random` is necessary but insufficient.
For bit-exact reproducibility across runs, set ``PYTHONHASHSEED=<fixed>``
*in the environment before Python starts*; Python itself cannot pin it after
import. If unset, this dataset emits a warning and the loader fingerprints
``PYTHONHASHSEED`` into the cache path so two hash seeds cannot silently
share a cached `data.pt`.

NOTE: augmentation is intended to be applied to the *training* split only.
Splitting must therefore happen by knot identity (`knot_name`) so a knot's
augmentations cannot leak across train/val/test.
"""
from __future__ import annotations

import ast
import os
import random as py_random
import warnings
from dataclasses import dataclass

import numpy as np
import snappy
import torch
from torch_geometric.data import Data

from src.data.dataset import KnotDataset, build_pyg_data_from_pd, read_csv
from src.data.knot import KnotDiagramTopology

# Modulus for derived per-row seeds. Below 2**32 so it fits numpy's seed range
# and torch's; comfortably above any plausible CSV row count.
_SEED_MOD = 2**31 - 1
_SEED_MULT = 1_000_003


def _hashseed_warning_once() -> None:
    """Warn once per process if PYTHONHASHSEED is unset or 'random'.

    spherogram's `Link.many_diagrams` consults set/dict iteration order, which
    is hash-randomized per process by default. Without a pinned hash seed,
    `aug_seed` alone does not guarantee reproducibility across runs.
    """
    if getattr(_hashseed_warning_once, "_done", False):
        return
    _hashseed_warning_once._done = True  # type: ignore[attr-defined]
    hs = os.environ.get("PYTHONHASHSEED", "random")
    if hs == "random" or hs == "":
        warnings.warn(
            "PYTHONHASHSEED is unset; spherogram's many_diagrams() is not "
            "fully reproducible across processes. For bit-exact reruns set "
            "PYTHONHASHSEED=<int> in the environment BEFORE invoking Python.",
            RuntimeWarning,
            stacklevel=3,
        )

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


def _mirror_target(
    y: torch.Tensor, task_col: str, label_shift: float = 0
) -> torch.Tensor:
    """Map a base knot's stored target to the target of its mirror diagram.

    `y` is in *shifted* representation: ``y = y_orig - label_shift`` (see
    `dataset._coerce_label`). For SIGN_FLIP_TASKS the conceptual operation is
    ``y_orig -> -y_orig`` in the unshifted space, which in the shifted space
    becomes ``y_stored -> -y_stored - 2*label_shift``. Forgetting the
    ``-2*label_shift`` correction silently produces out-of-range targets on
    every SIGN_FLIP task whose ``min_val != 0`` (signature, rasmussen_s,
    ozsvath_szabo_tau), tripping `cross_entropy` with a negative class index.
    """
    if task_col in MIRROR_SAFE_TASKS:
        return y
    if task_col in SIGN_FLIP_TASKS:
        if y.dtype.is_floating_point:
            return (-y - 2 * float(label_shift)).to(y.dtype)
        new_val = int(-int(y.item()) - 2 * int(label_shift))
        return torch.tensor(new_val, dtype=y.dtype)
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
        _hashseed_warning_once()
        super().__init__(root, csv_path, target_column, **kwargs)

    def process(self) -> None:
        rows = read_csv(self.csv_path, target=self.target_column, limit=self.limit)
        cfg = self.aug_cfg
        task_col = self.target_column

        data_list: list[Data] = []
        # Dedup is by PD code only. Y is NOT part of the key:
        #   - PD code is the canonical identity of a knot diagram, so two
        #     equal PDs are necessarily the same Data.
        #   - Including y would silently truncate continuous targets via
        #     int(y), collapsing distinct rows.
        seen_pds: set[str] = set()
        skipped_rows = 0
        n_aug = 0

        for row_idx, row in enumerate(rows):
            base = build_pyg_data_from_pd(
                row,
                label_shift=self.label_shift,
                strict=self.strict,
                target_dtype=self.target_dtype,
            )
            if base is None:
                skipped_rows += 1
                continue
            base.is_augmented = torch.tensor([0], dtype=torch.long)
            data_list.append(base)
            base_pd = getattr(base, "pd_notation", None)
            if base_pd is not None:
                seen_pds.add(str(base_pd))

            pd_notation = row.get("__pd_notation")
            if pd_notation is None:
                continue

            # Per-row deterministic seeding. Both Python's `random` (used
            # internally by spherogram's `Link.many_diagrams` backtrack) and
            # numpy's RNG (used for the shuffle) must be seeded; otherwise
            # the augmented set is non-reproducible across runs even with
            # `aug_seed` fixed. Per-row (not global) so a single failing row
            # cannot shift downstream rows' diagram sets.
            row_seed = (int(cfg.seed) * _SEED_MULT + row_idx) % _SEED_MOD
            py_random.seed(row_seed)
            rng = np.random.default_rng(row_seed)

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
                        (
                            diagram.mirror(),
                            lambda y, c=task_col, s=self.label_shift: _mirror_target(
                                y, c, s
                            ),
                        )
                    )
                for variant_diagram, y_fn in variants:
                    try:
                        new_pd = str(variant_diagram.PD_code())
                        if new_pd in seen_pds:
                            continue
                        seen_pds.add(new_pd)
                        y_new = y_fn(base.y) if base.y is not None else None
                        topo = KnotDiagramTopology.from_pd(new_pd)
                        aug = topo.topology_to_pyg_data(
                            y=y_new,
                            knot_name=row.get("__knot_name"),
                            pd_notation=new_pd,
                        )
                        aug.is_augmented = torch.tensor([1], dtype=torch.long)
                        data_list.append(aug)
                        n_aug += 1
                    except Exception:
                        continue

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        if skipped_rows:
            print(f"[KnotAugmentedDataset] skipped {skipped_rows}/{len(rows)} base rows")
        if not data_list:
            raise RuntimeError(
                f"KnotAugmentedDataset.process() produced 0 samples for "
                f"target_column={self.target_column!r}, csv={self.csv_path}, "
                f"limit={self.limit}, strict={self.strict}. "
                f"Skipped {skipped_rows}/{len(rows)} base rows. "
                f"Refusing to save an empty data.pt."
            )
        print(
            f"[KnotAugmentedDataset] {len(data_list)} samples "
            f"({len(data_list) - n_aug} base + {n_aug} augmented)"
        )
        self.save(data_list, self.processed_paths[0])
