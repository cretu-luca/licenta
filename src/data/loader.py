"""TopoBench-compatible loader wrappers for `KnotDataset` /
`KnotAugmentedDataset`."""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig
from topobench.data.loaders.base import AbstractLoader

from src.data.augmented_dataset import KnotAugmentedDataset
from src.data.dataset import KnotDataset


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
            label_shift=int(p.get("label_shift", 0)),
            limit=p.get("limit"),
            strict=bool(p.get("strict", False)),
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
    `KnotAugmentedDataset` and forwards augment kwargs."""

    _AUG_KEYS = ("n_diagrams", "tries", "method", "include_mirror", "aug_seed")

    def load_dataset(self, **kwargs) -> KnotAugmentedDataset:
        p = self.parameters
        root = Path(p["data_dir"]) / p.get("data_name", "knots_augmented")
        pre_transform = _instantiate_optional(p.get("pre_transform"))
        transform = _instantiate_optional(p.get("transform"))
        aug_kwargs = {k: p[k] for k in self._AUG_KEYS if k in p}
        dataset = KnotAugmentedDataset(
            root=str(root),
            csv_path=str(p["csv_path"]),
            target_column=str(p["target_column"]),
            label_shift=int(p.get("label_shift", 0)),
            limit=p.get("limit"),
            strict=bool(p.get("strict", False)),
            transform=transform,
            pre_transform=pre_transform,
            force_reload=bool(p.get("force_reload", False)),
            **aug_kwargs,
            **kwargs,
        )
        self.data_dir = Path(dataset.processed_dir).parent
        return dataset
