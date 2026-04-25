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
from collections import Counter
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import torch
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


def _index_dataset(dataset) -> tuple[list[int], list[str], dict[str, list[int]]]:
    """Single pass over the dataset that returns:
      - `base_indices`     : positions of non-augmented samples (one per knot
                             per CSV row), used as the splitting universe.
      - `base_knot_names`  : knot_name of each entry in `base_indices`,
                             aligned 1:1.
      - `aug_by_knot`      : knot_name -> list of dataset positions of that
                             knot's augmented copies.

    Datasets that do not carry `is_augmented` (e.g. plain `KnotDataset`) are
    treated as all-base, in which case `aug_by_knot` is empty and the train
    expansion below is a no-op.
    """
    base_indices: list[int] = []
    base_knot_names: list[str] = []
    aug_by_knot: dict[str, list[int]] = {}
    for i in range(len(dataset)):
        d = dataset[i]
        is_aug_attr = getattr(d, "is_augmented", None)
        is_aug = bool(int(is_aug_attr.item())) if is_aug_attr is not None else False
        name_attr = getattr(d, "knot_name", None)
        name = str(name_attr) if name_attr is not None else f"_idx_{i}"
        if is_aug:
            aug_by_knot.setdefault(name, []).append(i)
        else:
            base_indices.append(i)
            base_knot_names.append(name)
    return base_indices, base_knot_names, aug_by_knot


_INTEGER_DTYPES = (torch.long, torch.int, torch.int32, torch.int64, torch.int16, torch.int8)


def _class_distribution(dataset, indices, max_show: int = 12) -> dict | None:
    """Return {class_id: count} for classification targets, or None for
    regression / mixed dtypes (where 'class balance' is undefined)."""
    counts: Counter = Counter()
    is_int = True
    for i in indices:
        d = dataset[int(i)]
        if d.y is None:
            continue
        if d.y.dtype not in _INTEGER_DTYPES:
            is_int = False
            break
        counts[int(d.y.item())] += 1
    if not is_int:
        return None
    if not counts:
        return {}
    items = sorted(counts.items())
    if len(items) > max_show:
        items = items[:max_show] + [("...", sum(c for _, c in items[max_show:]))]
    return dict(items)


def _check_split_classes(fold, va_dist, te_dist, min_classes: int) -> None:
    """Raise on degenerate val/test splits. Group-aware splitting can produce
    a fold whose val (or test) set contains only one class, particularly
    on imbalanced binary tasks like Alternating/Arf. Validation metrics on
    such a fold are meaningless; best-checkpoint selection optimizes noise
    and Optuna sees garbage.
    """
    if min_classes <= 1:
        return
    for name, dist in (("val", va_dist), ("test", te_dist)):
        if dist is None:
            continue  # regression
        if len(dist) < min_classes:
            raise RuntimeError(
                f"fold={fold}: degenerate {name} split — only "
                f"{len(dist)} class(es) present ({dist}); "
                f"need >= {min_classes}. Group-aware splitting collided "
                f"with class imbalance. Either: switch task to a less "
                f"imbalanced target, raise val_proportion, or set "
                f"`min_classes_per_split=1` to ignore (sweep results for "
                f"this fold will be noise)."
            )


def _expand_train_with_augments(
    base_train_idx, base_indices, aug_by_knot, base_knot_names
) -> list[int]:
    """Map base-relative train indices to absolute dataset indices, then
    extend with all augmented copies of those train knots. Val/test stay
    base-only by virtue of never being expanded."""
    train_abs = [int(base_indices[int(i)]) for i in base_train_idx]
    if not aug_by_knot:
        return train_abs
    train_knots = {str(base_knot_names[int(i)]) for i in base_train_idx}
    extra: list[int] = []
    for knot in train_knots:
        extra.extend(aug_by_knot.get(knot, []))
    return train_abs + extra


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

    # Splitting universe = base rows only. Augmented copies follow their
    # knot into whichever split it lands, but only after the split is
    # decided, and only into the train side. This keeps val/test free of
    # augmented copies so reported metrics measure generalization to unseen
    # knots, not robustness to augmentation.
    base_indices, base_knot_names, aug_by_knot = _index_dataset(dataset)
    print(
        f"[train_tb] dataset: {len(dataset)} samples "
        f"({len(base_indices)} base, "
        f"{sum(len(v) for v in aug_by_knot.values())} augmented)",
        flush=True,
    )

    base_splits = five_splits_by_knot_name(
        base_knot_names,
        proportions=tuple(cfg.split.proportions),
        seeds=tuple(cfg.split.seeds),
    )

    min_classes_per_split = int(cfg.get("min_classes_per_split", 2))

    scores: list[float] = []
    for fold, (tr_b, va_b, te_b) in enumerate(base_splits):
        L.seed_everything(base_seed * 1000 + fold, workers=True)

        tr = _expand_train_with_augments(
            tr_b, base_indices, aug_by_knot, base_knot_names
        )
        va = [int(base_indices[int(i)]) for i in va_b]
        te = [int(base_indices[int(i)]) for i in te_b]
        n_aug_in_train = len(tr) - len(tr_b)

        tr_dist = _class_distribution(dataset, tr)
        va_dist = _class_distribution(dataset, va)
        te_dist = _class_distribution(dataset, te)
        print(
            f"[train_tb] fold={fold} "
            f"train={len(tr)} (base={len(tr_b)}, +aug={n_aug_in_train}) "
            f"val={len(va)} test={len(te)}",
            flush=True,
        )
        print(
            f"[train_tb] fold={fold} class dist: "
            f"train={tr_dist} val={va_dist} test={te_dist}",
            flush=True,
        )
        _check_split_classes(fold, va_dist, te_dist, min_classes_per_split)

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
            # Test must run on the best-val checkpoint, not the last-epoch
            # weights. If no checkpoint exists (validation never ran during
            # fit, e.g. check_val_every_n_epoch > max_epochs) we refuse
            # rather than silently testing on whatever weights happen to be
            # in memory.
            if fit_metrics:
                ckpt_cb = getattr(trainer, "checkpoint_callback", None)
                ckpt_path = getattr(ckpt_cb, "best_model_path", "") if ckpt_cb else ""
                if not ckpt_path:
                    raise RuntimeError(
                        f"fold={fold}: no best checkpoint to test on. "
                        f"Validation never produced a checkpoint during fit. "
                        f"Likely cause: trainer.check_val_every_n_epoch="
                        f"{cfg.trainer.get('check_val_every_n_epoch', 1)} > "
                        f"max_epochs={cfg.trainer.max_epochs}, or the "
                        f"ModelCheckpoint callback is disabled. Refusing "
                        f"to test on last-epoch weights silently."
                    )
                trainer.test(
                    model=model, datamodule=datamodule, ckpt_path=ckpt_path
                )
            else:
                # Train was skipped entirely; testing on the in-memory model
                # is the only option (and the no-test-leak guard below will
                # catch it if optimized_metric is requested).
                trainer.test(model=model, datamodule=datamodule)

        # Optuna objective MUST come from validation metrics. trainer.test()
        # rewrites trainer.callback_metrics to test-only, so we read from the
        # snapshot taken right after fit. Falling back to test metrics here
        # would let Optuna optimize on the test set; refuse instead.
        if not fit_metrics:
            raise RuntimeError(
                "fit-time callback metrics are empty (was train=false?). "
                "Refusing to read the optimized metric from test-time "
                "metrics, which would leak the test set into Optuna."
            )
        score = get_metric_value(fit_metrics, cfg.optimized_metric)
        scores.append(score)
        print(
            f"[train_tb] fold={fold}/{len(base_splits) - 1} "
            f"{cfg.optimized_metric}={score:.4f}",
            flush=True,
        )

    mean = float(np.mean(scores))
    std = float(np.std(scores))
    print(f"[train_tb] folds={len(scores)} mean={mean:.4f} std={std:.4f}")
    return mean


if __name__ == "__main__":
    main()
