"""Hydra -> Lightning instantiators (ashleve/lightning-hydra-template idiom)."""
from __future__ import annotations

from typing import Any

import hydra
from lightning.pytorch import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig


def instantiate_callbacks(callbacks_cfg: DictConfig | None) -> list[Callback]:
    """Instantiate every entry of `callbacks_cfg` that has a `_target_`."""
    callbacks: list[Callback] = []
    if callbacks_cfg is None:
        return callbacks
    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("callbacks_cfg must be a DictConfig")
    for _, cb_cfg in callbacks_cfg.items():
        if isinstance(cb_cfg, DictConfig) and "_target_" in cb_cfg:
            callbacks.append(hydra.utils.instantiate(cb_cfg))
    return callbacks


def instantiate_loggers(logger_cfg: DictConfig | None) -> list[Logger]:
    """Instantiate every entry of `logger_cfg` that has a `_target_`."""
    loggers: list[Logger] = []
    if logger_cfg is None:
        return loggers
    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("logger_cfg must be a DictConfig")
    for _, lg_cfg in logger_cfg.items():
        if isinstance(lg_cfg, DictConfig) and "_target_" in lg_cfg:
            loggers.append(hydra.utils.instantiate(lg_cfg))
    return loggers


def get_metric_value(metric_dict: dict[str, Any], metric_name: str) -> float:
    """Read a Lightning callback metric, raising a clear error if it's missing."""
    if metric_name not in metric_dict:
        raise KeyError(
            f"metric {metric_name!r} not in trainer.callback_metrics; "
            f"available: {sorted(metric_dict)}"
        )
    v = metric_dict[metric_name]
    return float(v.item()) if hasattr(v, "item") else float(v)
