"""TopoBench + Lightning + Hydra training entrypoint.

One Hydra job = one hyperparameter configuration = K fits across K splits.
The objective returned to Optuna is the mean of `cfg.optimized_metric` across
folds. Splits are group-aware (by `knot_name`) via sklearn's
`GroupShuffleSplit`; Lightning has no native CV in 2026 (KFoldLoop removed
in 2.0; issue #20544 still open) and Hydra-Optuna's sweeper conflates multirun
jobs with trials, so a manual loop returning the mean is the canonical
pattern.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
import lightning as L
import numpy as np
from omegaconf import DictConfig, OmegaConf
from topobench.dataloader import TBDataloader
from topobench.dataloader.dataload_dataset import DataloadDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.splitting import five_splits_by_knot_name  # noqa: E402
from src.utils.instantiators import (  # noqa: E402
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
)


def _knot_names_of(dataset) -> list[str]:
    names: list[str] = []
    for i in range(len(dataset)):
        d = dataset[i]
        n = getattr(d, "knot_name", None)
        names.append(str(n) if n is not None else f"_idx_{i}")
    return names


def _build_datamodule(
    dataset, train_idx, val_idx, test_idx, dataloader_params: DictConfig
) -> TBDataloader:
    train_set = DataloadDataset([dataset[int(i)] for i in train_idx])
    val_set = DataloadDataset([dataset[int(i)] for i in val_idx])
    test_set = DataloadDataset([dataset[int(i)] for i in test_idx])
    return TBDataloader(
        dataset_train=train_set,
        dataset_val=val_set,
        dataset_test=test_set,
        **OmegaConf.to_container(dataloader_params, resolve=True),
    )


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="run_tb.yaml"
)
def main(cfg: DictConfig) -> float:
    base_seed = int(cfg.get("seed") or 0)

    loader = hydra.utils.instantiate(cfg.dataset.loader)
    dataset, _ = loader.load()
    knot_names = _knot_names_of(dataset)

    splits = five_splits_by_knot_name(
        knot_names,
        proportions=tuple(cfg.split.proportions),
        seeds=tuple(cfg.split.seeds),
    )

    scores: list[float] = []
    for fold, (tr, va, te) in enumerate(splits):
        L.seed_everything(base_seed * 1000 + fold, workers=True)

        datamodule = _build_datamodule(
            dataset, tr, va, te, cfg.dataset.dataloader_params
        )

        loss = hydra.utils.instantiate(cfg.loss)
        evaluator = hydra.utils.instantiate(cfg.evaluator)
        optimizer = hydra.utils.instantiate(cfg.optimizer)
        model = hydra.utils.instantiate(
            cfg.model, loss=loss, evaluator=evaluator, optimizer=optimizer
        )
        callbacks = instantiate_callbacks(cfg.get("callbacks"))
        loggers = instantiate_loggers(cfg.get("logger"))

        trainer = hydra.utils.instantiate(
            cfg.trainer, callbacks=callbacks, logger=loggers
        )

        fit_metrics: dict = {}
        if cfg.get("train", True):
            trainer.fit(model=model, datamodule=datamodule)
            fit_metrics = dict(trainer.callback_metrics)
        if cfg.get("test", True):
            trainer.test(model=model, datamodule=datamodule)

        # Optimized metric (Optuna objective) is read from fit-time callback
        # metrics, since trainer.test() rewrites callback_metrics to test-only.
        # Fall back to test-time metrics if train was skipped.
        source = fit_metrics or dict(trainer.callback_metrics)
        scores.append(get_metric_value(source, cfg.optimized_metric))

    mean = float(np.mean(scores))
    std = float(np.std(scores))
    print(f"[train_tb] folds={len(scores)} mean={mean:.4f} std={std:.4f}")
    return mean


if __name__ == "__main__":
    main()
