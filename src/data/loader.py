"""TopoBench-compatible loader wrappers for `KnotDataset` /
`KnotAugmentedDataset`."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from topobench.data.loaders.base import AbstractLoader

from src.data.augmented_dataset import KnotAugmentedDataset
from src.data.dataset import KnotDataset


def _stable_fingerprint(payload: dict, n: int = 10) -> str:
    """Deterministic short hash of a dict; values that aren't JSON-native are
    coerced via `default=str`. Sort keys so insertion order doesn't matter."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:n]


def _to_python(value):
    """Convert OmegaConf containers to plain dicts/lists for hashing."""
    if isinstance(value, (DictConfig, list)) or OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _instantiate_optional(cfg):
    """Instantiate a Hydra config if it still looks like one; otherwise pass
    through. Handles both code paths:
      - top-level loader instantiated via `hydra.utils.instantiate(cfg.dataset.loader)`
        (Hydra's recursive instantiation has already turned `pre_transform` into
        an object), and
      - bare dataset usage where `parameters` is still a DictConfig dict.
    """
    if cfg is None:
        return None
    if isinstance(cfg, DictConfig) and "_target_" in cfg:
        return hydra.utils.instantiate(cfg)
    return cfg


class KnotDatasetLoader(AbstractLoader):
    """Returns the configured `KnotDataset` plus its on-disk directory.

    Hydra params (DictConfig):
      data_dir         path under which the dataset cache lives
      data_name        sub-folder name (default: "knots")
      csv_path         absolute path to the KnotInfo CSV
      target_column    KnotInfo column to predict
      label_shift      subtracted from numeric targets to land in [0, num_classes)
      num_classes      forwarded for downstream consumers (unused here)
      limit            optional row cap for debugging
      strict           if true, raise on first un-coercible row
      force_reload     bypass PyG's processed-cache check
      pre_transform    optional Hydra-instantiable transform
      transform        optional Hydra-instantiable transform
    """

    def __init__(self, parameters: DictConfig, **_: dict) -> None:
        super().__init__(parameters)

    def load_dataset(self, **kwargs) -> KnotDataset:
        p = self.parameters
        root = Path(p["data_dir"]) / p.get("data_name", "knots")
        pre_transform = _instantiate_optional(p.get("pre_transform"))
        transform = _instantiate_optional(p.get("transform"))
        dataset = KnotDataset(
            root=str(root),
            csv_path=str(p["csv_path"]),
            target_column=str(p["target_column"]),
            label_shift=float(p.get("label_shift", 0)),
            limit=p.get("limit"),
            strict=bool(p.get("strict", False)),
            target_dtype=str(p.get("target_dtype", "long")),
            transform=transform,
            pre_transform=pre_transform,
            force_reload=bool(p.get("force_reload", False)),
            **kwargs,
        )
        self.data_dir = Path(dataset.processed_dir).parent
        return dataset

    def get_data_dir(self) -> Path:
        return Path(self.parameters["data_dir"]) / self.parameters.get(
            "data_name", "knots"
        )


class KnotAugmentedDatasetLoader(KnotDatasetLoader):
    """Same shape as `KnotDatasetLoader`, but instantiates
    `KnotAugmentedDataset` and forwards augment kwargs.

    Augment kwargs (`n_diagrams`, `tries`, `method`, `include_mirror`,
    `aug_seed`) live on the dataset, not on `pre_transform`, so PyG's
    pre_transform-repr fingerprint does not see them. To prevent silent reuse
    of a stale `processed/data.pt` when augment kwargs change, the loader
    deterministically hashes them into the on-disk `root` path:

        <data_dir>/<data_name>/aug-<sha1[:10]>/<target_slug>/processed/data.pt

    Changing any augment kwarg lands the dataset in a new directory; the old
    cache stays available for the previous config (no overwrite, no race).
    """

    _AUG_KEYS = ("n_diagrams", "tries", "method", "include_mirror", "aug_seed")

    def _augment_fingerprint(self, p: DictConfig) -> str:
        aug_payload = {k: _to_python(p[k]) for k in self._AUG_KEYS if k in p}
        # PYTHONHASHSEED is part of the determinism contract for spherogram's
        # many_diagrams (see KnotAugmentedDataset docstring). Fold it into the
        # cache path so two hash-seed environments cannot silently share a
        # `processed/data.pt` that was built under different iteration order.
        aug_payload["__pythonhashseed"] = os.environ.get("PYTHONHASHSEED", "random")
        return _stable_fingerprint(aug_payload)

    def load_dataset(self, **kwargs) -> KnotAugmentedDataset:
        p = self.parameters
        fp = self._augment_fingerprint(p)
        root = (
            Path(p["data_dir"])
            / p.get("data_name", "knots_augmented")
            / f"aug-{fp}"
        )
        pre_transform = _instantiate_optional(p.get("pre_transform"))
        transform = _instantiate_optional(p.get("transform"))
        aug_kwargs = {k: p[k] for k in self._AUG_KEYS if k in p}
        dataset = KnotAugmentedDataset(
            root=str(root),
            csv_path=str(p["csv_path"]),
            target_column=str(p["target_column"]),
            label_shift=float(p.get("label_shift", 0)),
            limit=p.get("limit"),
            strict=bool(p.get("strict", False)),
            target_dtype=str(p.get("target_dtype", "long")),
            transform=transform,
            pre_transform=pre_transform,
            force_reload=bool(p.get("force_reload", False)),
            **aug_kwargs,
            **kwargs,
        )
        self.data_dir = Path(dataset.processed_dir).parent
        return dataset

    def get_data_dir(self) -> Path:
        fp = self._augment_fingerprint(self.parameters)
        return (
            Path(self.parameters["data_dir"])
            / self.parameters.get("data_name", "knots_augmented")
            / f"aug-{fp}"
        )
