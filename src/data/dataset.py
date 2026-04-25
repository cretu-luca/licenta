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


_VALID_TARGET_DTYPES = {"long", "float"}


def _coerce_label(
    raw: Any, label_shift: float, target_dtype: str = "long"
) -> torch.Tensor | None:
    """Coerce a CSV cell to the requested tensor dtype.

    `target_dtype` is fixed at config time per task semantics:
      - "long"  : classification target (binary or multiclass). Y/N strings
                  map to 1/0; numeric strings must be integer-valued (else
                  the row is dropped).
      - "float" : regression target. Y/N strings raise; numeric strings are
                  cast directly with `label_shift` subtracted.

    Returns None on un-coercible / sentinel values; the caller then drops
    the row.
    """
    if target_dtype not in _VALID_TARGET_DTYPES:
        raise ValueError(
            f"target_dtype must be one of {_VALID_TARGET_DTYPES}, got {target_dtype!r}"
        )
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "d.n.e.", "dne", "unknown", "?"}:
        return None
    if s in {"Y", "y", "Yes", "yes", "True", "true"}:
        if target_dtype == "float":
            raise ValueError("Y/N target encountered with target_dtype='float'")
        return torch.tensor(int(1) - int(label_shift), dtype=torch.long)
    if s in {"N", "n", "No", "no", "False", "false"}:
        if target_dtype == "float":
            raise ValueError("Y/N target encountered with target_dtype='float'")
        return torch.tensor(int(0) - int(label_shift), dtype=torch.long)
    try:
        f = float(s)
    except ValueError:
        return None
    if target_dtype == "float":
        return torch.tensor(f - float(label_shift), dtype=torch.float)
    if not f.is_integer():
        return None
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

    Fails loud on missing target/PD columns. Silent failure here was the
    most damaging bug class in the previous pipeline: a misspelled
    `target_column` in a sweep override would produce an empty dataset, an
    Optuna trial completing with garbage, and no visible trace.
    """
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        cols_preview = sorted(df.columns)
        if len(cols_preview) > 30:
            cols_preview = cols_preview[:30] + ["..."]
        raise KeyError(
            f"target column {target!r} not found in {csv_path}. "
            f"Available columns: {cols_preview}"
        )
    if not any(c in df.columns for c in _PD_COLUMN_CANDIDATES):
        raise KeyError(
            f"no PD-notation column found in {csv_path}; expected one of "
            f"{list(_PD_COLUMN_CANDIDATES)}"
        )
    if limit is not None:
        df = df.head(int(limit))
    if len(df) == 0:
        raise ValueError(
            f"CSV {csv_path} (limit={limit}) yielded zero rows; cannot build dataset"
        )
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
    label_shift: float = 0,
    strict: bool = False,
    target_dtype: str = "long",
) -> Data | None:
    """Parse one CSV row into a `Data`. Returns None on coerce/parse failure
    (drops the row) unless `strict=True`."""
    from src.data.knot import KnotDiagramTopology

    pd_notation = row.get("__pd_notation")
    if pd_notation is None or str(pd_notation).strip() == "":
        if strict:
            raise ValueError(f"missing PD notation for row {row.get('__knot_name')!r}")
        return None

    y = _coerce_label(
        row.get("__target_raw"),
        label_shift=label_shift,
        target_dtype=target_dtype,
    )
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

    data = topo.topology_to_pyg_data(
        y=y,
        knot_name=row.get("__knot_name"),
        pd_notation=str(pd_notation),
    )
    data.is_augmented = torch.tensor([0], dtype=torch.long)
    return data


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
        label_shift: float = 0,
        limit: int | None = None,
        strict: bool = False,
        target_dtype: str = "long",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload: bool = False,
    ) -> None:
        if target_dtype not in _VALID_TARGET_DTYPES:
            raise ValueError(
                f"target_dtype must be one of {_VALID_TARGET_DTYPES}, got {target_dtype!r}"
            )
        self.csv_path = str(csv_path)
        self.target_column = str(target_column)
        self.label_shift = float(label_shift)
        self.limit = None if limit is None else int(limit)
        self.strict = bool(strict)
        self.target_dtype = str(target_dtype)
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
                row,
                label_shift=self.label_shift,
                strict=self.strict,
                target_dtype=self.target_dtype,
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
        if not data_list:
            raise RuntimeError(
                f"KnotDataset.process() produced 0 samples for "
                f"target_column={self.target_column!r}, csv={self.csv_path}, "
                f"limit={self.limit}, strict={self.strict}. "
                f"Skipped {skipped}/{len(rows)} rows due to coerce/parse "
                f"failures, then pre_filter/pre_transform reduced to 0. "
                f"Refusing to save an empty data.pt; this would silently "
                f"produce empty splits and meaningless training metrics."
            )
        self.save(data_list, self.processed_paths[0])
