"""Thin PyG `InMemoryDataset` shell over the KnotInfo CSV.

Reads CSV -> builds a `Data(x, edge_index, faces, y, num_crossings, knot_name,
pd_notation)` per row via SnapPy. Heavy CW connectivity construction lives in
`src.transforms.graph2cell_face_lifting.Graph2CellFaceLifting`, which is wired
in as `pre_transform` so the cached `data.pt` already carries `x_0`,
`incidence_*`, etc.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset


_PD_COLUMN_CANDIDATES = ("PD Notation", "PD_Notation", "pd_notation", "PD")
_NAME_COLUMN_CANDIDATES = ("Name", "name", "knot_name")


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(s).strip().lower())
    return s.strip("_") or "target"


def _coerce_label(raw: Any, label_shift: int) -> torch.Tensor | None:
    """Y/N -> {1,0}; numeric -> int after shift. Returns None on failure."""
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "d.n.e.", "dne", "unknown", "?"}:
        return None
    if s in {"Y", "y", "Yes", "yes", "True", "true"}:
        return torch.tensor(int(1) - int(label_shift), dtype=torch.long)
    if s in {"N", "n", "No", "no", "False", "false"}:
        return torch.tensor(int(0) - int(label_shift), dtype=torch.long)
    try:
        f = float(s)
    except ValueError:
        return None
    if not f.is_integer():
        return torch.tensor(f, dtype=torch.float)
    return torch.tensor(int(f) - int(label_shift), dtype=torch.long)


def _pick(d: dict, candidates: Iterable[str]) -> Any:
    for k in candidates:
        if k in d:
            return d[k]
    return None


def read_csv(
    csv_path: str | Path,
    *,
    target: str,
    limit: int | None = None,
) -> list[dict]:
    """Read the KnotInfo CSV; return one dict per row containing the columns
    needed downstream (PD notation, target value, knot name).
    """
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(int(limit))
    rows: list[dict] = []
    for _, r in df.iterrows():
        d = {k: r[k] for k in df.columns}
        d["__pd_notation"] = _pick(d, _PD_COLUMN_CANDIDATES)
        d["__knot_name"] = _pick(d, _NAME_COLUMN_CANDIDATES)
        d["__target_raw"] = d.get(target)
        rows.append(d)
    return rows


def build_pyg_data_from_pd(
    row: dict,
    *,
    label_shift: int = 0,
    strict: bool = False,
) -> Data | None:
    """Parse one CSV row into a `Data`. Returns None on coerce/parse failure
    (drops the row) unless `strict=True`."""
    from src.data.knot import KnotDiagramTopology

    pd_notation = row.get("__pd_notation")
    if pd_notation is None or str(pd_notation).strip() == "":
        if strict:
            raise ValueError(f"missing PD notation for row {row.get('__knot_name')!r}")
        return None

    y = _coerce_label(row.get("__target_raw"), label_shift=label_shift)
    if y is None:
        if strict:
            raise ValueError(
                f"un-coercible target {row.get('__target_raw')!r} for "
                f"{row.get('__knot_name')!r}"
            )
        return None

    try:
        topo = KnotDiagramTopology.from_pd(str(pd_notation))
    except (ValueError, SyntaxError, KeyError, RuntimeError) as e:
        if strict:
            raise
        return None

    return topo.topology_to_pyg_data(
        y=y,
        knot_name=row.get("__knot_name"),
        pd_notation=str(pd_notation),
    )


class KnotDataset(InMemoryDataset):
    """KnotInfo CSV -> PyG dataset.

    Cache layout: ``<root>/<target_slug>/{raw,processed}/``. Switching targets
    does not invalidate other targets' caches. The configured ``pre_transform``
    runs once at process() time and its repr is fingerprinted in
    ``processed_dir`` automatically by PyG (warning, not invalidation), so for
    full safety also pass a transform-keyed ``root`` from the loader if needed.
    """

    def __init__(
        self,
        root: str | Path,
        csv_path: str | Path,
        target_column: str,
        label_shift: int = 0,
        limit: int | None = None,
        strict: bool = False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload: bool = False,
    ) -> None:
        self.csv_path = str(csv_path)
        self.target_column = str(target_column)
        self.label_shift = int(label_shift)
        self.limit = None if limit is None else int(limit)
        self.strict = bool(strict)
        self._target_slug = _slug(target_column)
        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            force_reload=force_reload,
        )
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return str(Path(self.root) / self._target_slug / "raw")

    @property
    def processed_dir(self) -> str:
        return str(Path(self.root) / self._target_slug / "processed")

    @property
    def raw_file_names(self) -> list[str]:
        return [Path(self.csv_path).name]

    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt"]

    def download(self) -> None:
        # The CSV is produced out-of-band (knotinfo.org xls -> csv). If the
        # raw_dir copy is missing, fall back to the absolute csv_path. PyG's
        # _download() only triggers when raw_paths are missing.
        src = Path(self.csv_path)
        if not src.exists():
            raise FileNotFoundError(f"KnotInfo CSV not found at {src}")
        dst = Path(self.raw_dir) / src.name
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())

    def process(self) -> None:
        rows = read_csv(self.csv_path, target=self.target_column, limit=self.limit)
        data_list: list[Data] = []
        skipped = 0
        for row in rows:
            d = build_pyg_data_from_pd(
                row, label_shift=self.label_shift, strict=self.strict
            )
            if d is None:
                skipped += 1
                continue
            data_list.append(d)
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        if skipped:
            print(f"[KnotDataset] skipped {skipped}/{len(rows)} rows")
        self.save(data_list, self.processed_paths[0])
