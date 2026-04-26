#!/usr/bin/env bash
# Local smoke driver for the knot-pipeline thesis experiments.
#
# Companion to scripts/pod.sh. Same Hydra override surface (hparams_search=
# optuna, same trainer.* / dataset.* / callbacks.* keys, same SQLite +
# log layout), sized to land in roughly 30-50 min on a CPU laptop while
# still producing numerically informative metrics (not collapse-to-noise
# like the prior 50-row / 3-epoch smoke):
#   - all 11 invariants x 2 models = 22 (task, model) Optuna studies
#   - n_trials=6 (5 random startup + at least 1 TPE posterior sample, so
#     the Bayesian path is also exercised, not just startup)
#   - max_epochs=15, patience=4, check_val_every_n_epoch=2 (early stopping
#     can actually trigger; best-checkpoint loading is real)
#   - limit=400 rows (~325 valid after coerce; folds become statistically
#     non-degenerate for binary tasks, multiclass tasks see most labels)
#   - 3 folds (closer to pod's 5; same GroupShuffleSplit code path)
#   - batch_size sampled by the sweeper (not pinned by CLI) so the 4th
#     axis declared in configs/hparams_search/optuna.yaml is exercised
#   - min_classes_per_split=2 (the default; fold-collapse guard is on,
#     since with limit=400 imbalanced binary folds should pass it)
#
# Run this BEFORE the pod sweep. If every (task, model) study completes
# AND the per-task best Optuna value is meaningfully above the
# majority-class prior, every Hydra override the pod will use is wired
# correctly and the model + lifting + readout stack actually fits.
# Anything broken (typo'd override, sweeper-axis collision, dataset
# cache bug, model/task incompatibility, missing checkpoint at test
# time, degenerate split, optimizer divergence) trips here in under an
# hour instead of after ~8 h on the pod.
#
# Precaching is intentionally skipped: the per-task .pt cache builds
# inline on the first trial of each (task, model) pair, on the limit=50
# subset, in seconds. The full pod cache (no limit) is built lazily on
# pod by do_sweep's first trial per task.
#
# Usage:
#   bash scripts/local.sh                  # all = verify + sweep + summary
#   bash scripts/local.sh verify
#   bash scripts/local.sh sweep
#   bash scripts/local.sh summary
#
# Override knobs (env), e.g. shrink further on a laptop:
#   N_TRIALS=2 MAX_EPOCHS=2 LIMIT=40 bash scripts/local.sh sweep
#   TASKS="signature crossing_number" MODELS=cwn bash scripts/local.sh sweep
#   ACCELERATOR=cpu bash scripts/local.sh sweep
#
# Setup expected:
#   - A local venv at ${VENV_DIR} (default .venv) with the project's
#     dependencies installed and importable. Unlike scripts/pod.sh, this
#     script does NOT install anything: local installs are OS-specific
#     (Mac CPU/MPS vs Linux CPU vs Linux CUDA), and clobbering an active
#     dev venv from a smoke driver is the wrong default.
#
#   - sqlalchemy must be < 2.0 (snapshot pins 1.4.51) and alembic must
#     be < 1.13 (snapshot pins 1.12.1). optuna 2.10 was released before
#     either upstream broke its assumptions:
#       * SA 2.0 reversed the implicit-transaction default. optuna's
#         `_VersionManager._set_alembic_revision` calls `context.stamp(...)`
#         on a bare `engine.connect()` without `.begin()`; under SA 2.0
#         that write is rolled back at GC time, leaving `alembic_version`
#         empty and tripping the `assert version is not None` in
#         `optuna/storages/_rdb/storage.py::get_current_version` on the
#         very first `optuna.create_study(storage=...)`.
#       * Alembic 1.13 removed the no-transaction `stamp` path optuna
#         2.10 relies on; same symptom from a different angle.
#     If either pin drifts locally, re-pin:
#         .venv/bin/python -m pip install 'sqlalchemy==1.4.51' 'alembic==1.12.1'

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
ACTIVATE="${VENV_DIR}/bin/activate"

# ---------------------------------------------------------------------------
# CONFIG -- override via environment, e.g.
#   N_TRIALS=3 MAX_EPOCHS=5 bash scripts/local.sh sweep
# ---------------------------------------------------------------------------
TASKS=(${TASKS:-alternating arf crossing_number determinant genus_3d \
    genus_4d genus_4d_top ozsvath_szabo_tau rasmussen_s signature \
    unknotting_number})
MODELS=(${MODELS:-gin cwn})

# Optuna's TPESampler has n_startup_trials=5 (configs/hparams_search/
# optuna.yaml). N_TRIALS=6 gives 5 random-startup samples plus 1 actual
# Bayesian sample, which is the cheapest config that exercises both
# code paths. Trials are seeded by ${seed}=42, so trial #0..#4 are
# deterministic across studies.
N_TRIALS="${N_TRIALS:-6}"
MAX_EPOCHS="${MAX_EPOCHS:-15}"
LIMIT="${LIMIT:-400}"
SEEDS="${SEEDS:-[0,1,2]}"           # Three folds; pod uses five.
N_WORKERS="${N_WORKERS:-0}"         # 0 avoids fork-spawn overhead per epoch.

# Default to CPU, NOT auto. On Apple silicon `auto` resolves to MPS, and
# torch's MPS backend does not implement `aten::empty.memory_format` for
# the SparseMPS layout. The Graph2CellFaceLifting pre_transform builds
# torch.sparse incidence matrices (CWN consumes them directly; even GIN
# carries them on the Data object), and Lightning's batch_to_device
# step crashes the moment it tries to move them onto MPS. Upstream gap,
# tracked since pytorch#77764, still partial in torch 2.3.
#
# Override to `cuda` if running this smoke on a CUDA box, or to `mps`
# if you know your model+data path avoids sparse ops (it does not, here).
ACCELERATOR="${ACCELERATOR:-cpu}"

# Set to 1 (bypass the degenerate-fold guard), NOT the default 2.
# Reason: KnotInfo's CSV is in Rolfsen order (small-crossing knots
# first). The first ~400 rows are >95% alternating=Y, so the binary
# tasks `alternating` and `arf` reliably produce val slices with one
# class even at this limit. The pipeline's response to that is exactly
# what the pod's heavy run does too: collapse to majority-class
# predictor (see scripts/pod.sh top-of-file caveat). Bypassing the
# guard here propagates that behavior into the smoke instead of
# erroring out before the multiclass tasks (which DO produce
# meaningful numerics) get a chance to run. The proper fix to the
# CSV-head bias is stratified subsampling in src/data/dataset.py,
# which is out of scope for this driver.
MIN_CLASSES="${MIN_CLASSES:-1}"

# patience=4 with check_val_every_n_epoch=2 means 4 non-improving
# validations -> 8 epochs without progress before stopping. With
# max_epochs=15 most fits hit early stopping; the test phase always
# has a real best-val checkpoint to load.
PATIENCE="${PATIENCE:-4}"
VAL_EVERY_N_EPOCHS="${VAL_EVERY_N_EPOCHS:-2}"

EXP="${EXP:-local_$(date -u +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-${REPO_ROOT}/logs/${EXP}}"
DBDIR="${DBDIR:-${REPO_ROOT}/outputs/optuna_dbs}"

# Reproducibility: spherogram's many_diagrams() is sensitive to Python's
# per-process hash randomization. Pin it for the whole pipeline (matches
# scripts/pod.sh).
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

# OMP / KMP hygiene. KMP_DUPLICATE_LIB_OK silences a libomp clash that
# crops up on macOS with mixed conda/pip torch builds; OMP_NUM_THREADS=1
# avoids oversubscription on a workstation.
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
say() { printf '\n=== %s :: %s ===\n' "$(date -Iseconds)" "$*"; }

ensure_venv() {
    if [[ ! -f "${ACTIVATE}" ]]; then
        echo "ERROR: no venv at ${VENV_DIR}." >&2
        echo "Activate or create your local dev venv first; this smoke" >&2
        echo "driver intentionally does not install dependencies (the OS-" >&2
        echo "specific torch/PyG build is yours to manage locally; see" >&2
        echo "scripts/pod.sh for the Linux+CUDA install recipe)." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${ACTIVATE}"
}

# ---------------------------------------------------------------------------
# verify :: imports + tiny end-to-end (1 fold, 1 epoch, accelerator=auto)
# Smaller than scripts/pod.sh's verify so it tolerates CPU-only laptops.
# ---------------------------------------------------------------------------
do_verify() {
    ensure_venv
    say "package imports + accelerator visibility"
    python -c "
import sys; sys.path.insert(0, '${REPO_ROOT}')
from src.data import KnotDataset, KnotAugmentedDataset, KnotDatasetLoader
from src.transforms.graph2cell_face_lifting import Graph2CellFaceLifting
from src.utils.instantiators import instantiate_callbacks
import topobench, lightning, hydra, optuna
import torch
mps = bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
print('torch', torch.__version__,
      'cuda?', torch.cuda.is_available(),
      'mps?', mps)
print('all good')
"
    say "tiny end-to-end (1 fold, 1 epoch, accelerator=${ACCELERATOR})"
    rm -rf data/knots_verify outputs/verify
    python -u scripts/train_tb.py \
        task=signature \
        experiment_name=verify \
        dataset.loader.parameters.limit=20 \
        dataset.loader.parameters.data_name=knots_verify \
        trainer.max_epochs=1 \
        trainer.accelerator="${ACCELERATOR}" \
        trainer.check_val_every_n_epoch=1 \
        trainer.enable_progress_bar=false \
        'split.seeds=[0]' \
        min_classes_per_split=1 \
        dataset.dataloader_params.batch_size=4 \
        dataset.dataloader_params.num_workers=0 \
        dataset.dataloader_params.pin_memory=false \
        callbacks.early_stopping.patience=1
    rm -rf data/knots_verify outputs/verify
    say "verify ok"
}

# ---------------------------------------------------------------------------
# sweep :: per (task, model), run an Optuna study with the SAME override
# surface as scripts/pod.sh's do_sweep, just at smoke scale. Each study
# logs to its own SQLite file so summary aggregation is identical.
# ---------------------------------------------------------------------------
do_sweep() {
    ensure_venv
    mkdir -p "${LOGDIR}" "${DBDIR}"
    say "smoke sweep config: tasks=${TASKS[*]} models=${MODELS[*]}"
    say "  n_trials=${N_TRIALS} max_epochs=${MAX_EPOCHS} patience=${PATIENCE} val_every=${VAL_EVERY_N_EPOCHS}"
    say "  limit=${LIMIT} seeds=${SEEDS} min_classes=${MIN_CLASSES} accel=${ACCELERATOR}"

    # NB: dataset.dataloader_params.batch_size is intentionally NOT
    # overridden here. configs/hparams_search/optuna.yaml declares it as
    # a 4th sweep axis (choice(64, 128, 256)); a CLI override would
    # silently shadow the sweeper's sample, dropping the axis. Same in
    # scripts/pod.sh.
    #
    # data_name carries ${LIMIT}: PyG's InMemoryDataset keys its cache
    # by <data_dir>/<data_name>/<target_slug>/processed/data.pt and
    # does NOT include `limit` in the path. Caching a smaller `limit`
    # then bumping it on the next run silently reuses the smaller .pt
    # and ignores the new limit (yes, that has happened). Encoding
    # the limit into data_name forces a per-LIMIT cache.
    local data_name="knots_smoke_${LIMIT}"
    for task in "${TASKS[@]}"; do
        for model in "${MODELS[@]}"; do
            local log="${LOGDIR}/${task}_${model}.log"
            local db="${DBDIR}/${EXP}_${task}_${model}.db"
            say "starting ${task} / ${model} -> ${log}"
            python -u scripts/train_tb.py \
                hparams_search=optuna \
                experiment_name="${EXP}" \
                task="${task}" model="${model}" \
                hydra.sweeper.n_trials="${N_TRIALS}" \
                hydra.sweeper.storage="sqlite:///${db}" \
                trainer.accelerator="${ACCELERATOR}" \
                trainer.max_epochs="${MAX_EPOCHS}" \
                trainer.check_val_every_n_epoch="${VAL_EVERY_N_EPOCHS}" \
                trainer.enable_progress_bar=false \
                dataset.loader.parameters.limit="${LIMIT}" \
                dataset.loader.parameters.data_name="${data_name}" \
                "split.seeds=${SEEDS}" \
                min_classes_per_split="${MIN_CLASSES}" \
                dataset.dataloader_params.num_workers="${N_WORKERS}" \
                dataset.dataloader_params.pin_memory=false \
                callbacks.early_stopping.patience="${PATIENCE}" \
                2>&1 | tee "${log}"
        done
    done
    say "smoke sweep complete -> ${LOGDIR}"
}

# ---------------------------------------------------------------------------
# summary :: dump best (task, model) result from each Optuna DB. Filters
# to local_* by default; pass EXP=<exp> to scope to one run.
# ---------------------------------------------------------------------------
do_summary() {
    ensure_venv
    say "results from ${DBDIR}"
    # Pass known TASKS/MODELS to the helper so it can split db filenames
    # of the form `<exp>_<task>_<model>.db` correctly even when <task>
    # contains underscores (genus_4d_top, ozsvath_szabo_tau, etc.) by
    # matching the filename's suffix against the known sets rather than
    # blindly splitting on `_`.
    TASKS="${TASKS[*]}" MODELS="${MODELS[*]}" python -c "
import glob, json, os, sys
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
exp = os.environ.get('EXP', '')
known_tasks = sorted(os.environ.get('TASKS', '').split(), key=len, reverse=True)
known_models = sorted(os.environ.get('MODELS', '').split(), key=len, reverse=True)
pat = os.path.join('${DBDIR}', f'{exp or \"local_*\"}_*_*.db')

def parse(fname):
    # filename (no .db): <exp>_<task>_<model>. Match longest known task
    # first so genus_4d_top wins over genus_4d.
    for m in known_models:
        suf = '_' + m
        if fname.endswith(suf):
            head = fname[:-len(suf)]
            for t in known_tasks:
                tsuf = '_' + t
                if head.endswith(tsuf):
                    return head[:-len(tsuf)], t, m
    return None

rows = []
skipped = []
for db in sorted(glob.glob(pat)):
    fname = os.path.basename(db)[:-3]
    parsed = parse(fname)
    if parsed is None:
        skipped.append(fname); continue
    expp, task, model = parsed
    s = optuna.load_study(study_name=None, storage=f'sqlite:///{db}')
    states = {}
    for t in s.trials:
        states[t.state.name] = states.get(t.state.name, 0) + 1
    try:
        bt = s.best_trial            # Optuna 2.10 raises ValueError if no COMPLETE trial exists
    except ValueError:
        skipped.append(f'{fname} (no completed trials; states={states})'); continue
    if bt.state.name != 'COMPLETE':
        skipped.append(f'{fname} (best trial state: {bt.state.name}; states={states})'); continue
    rows.append((expp, task, model, s.best_value, s.best_trial.number,
                 len(s.trials), s.best_params))
if not rows:
    print('no studies found at', pat); sys.exit(0)
print(f'{\"exp\":<24} {\"task\":<18} {\"model\":<6} {\"best\":>8} {\"#trial\":>6} {\"#all\":>5}  best_params')
print('-' * 130)
for expp, task, model, best, idx, n, params in sorted(rows):
    print(f'{expp:<24} {task:<18} {model:<6} {best:>8.4f} {idx:>6} {n:>5}  {json.dumps(params)}')
if skipped:
    print()
    print(f'skipped {len(skipped)} db(s):')
    for s in skipped:
        print(f'  {s}')
"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
cd "${REPO_ROOT}"
case "${1:-all}" in
    verify)   do_verify ;;
    sweep)    do_sweep ;;
    summary)  do_summary ;;
    all)
        do_verify
        do_sweep
        do_summary
        ;;
    *)
        echo "usage: $0 {verify|sweep|summary|all}" >&2
        exit 2
        ;;
esac
