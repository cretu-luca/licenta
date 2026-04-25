#!/usr/bin/env bash
# Pod setup + sweep driver for the knot-pipeline thesis experiments.
#
# Assumed pod state on entry:
#   - Linux + Python 3.11 available as `python3.11` (or system `python`).
#   - CUDA 12 toolchain on the host (drivers + runtime). `nvidia-smi` works.
#   - This repository uploaded to ${REPO_ROOT}.
#   - TopoBench checked out to ${TOPOBENCH_ROOT} (the requirements snapshot
#     pins `topobench @ file://...`; the path must exist).
#   - `data/knotinfo.csv` is present in the repo.
#
# Usage (run sections one at a time the first time, then re-run as needed):
#   bash scripts/pod.sh install
#   bash scripts/pod.sh verify
#   bash scripts/pod.sh precache
#   bash scripts/pod.sh sweep              # runs sequentially, ~5h budget
#   bash scripts/pod.sh summary
#
# Or run end-to-end:
#   bash scripts/pod.sh all
#
# Live monitoring from a second shell on the pod:
#   tail -f logs/pod_*/*.log
#   tail -f logs/pod_*/*.log | grep --line-buffered '\[train_tb\]'
#   optuna-dashboard sqlite:///outputs/optuna_dbs/pod_*_<task>_<model>.db --port 8080

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
if [[ "${REPO_ROOT}" != /workspace/* ]]; then
    echo "ERROR: REPO_ROOT=${REPO_ROOT} is not under /workspace." >&2
    echo "Move the repo to /workspace, or override with REPO_ROOT=/workspace/... if intentional." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# CONFIG -- override via environment, e.g.
#   TASKS="signature crossing_number" N_TRIALS=20 bash scripts/pod.sh sweep
# ---------------------------------------------------------------------------
TOPOBENCH_ROOT="${TOPOBENCH_ROOT:-${REPO_ROOT}/../TopoBench}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"

# Tasks with non-degenerate class priors (multi-class, ~uniform-ish).
# `alternating`, `arf`, `unknotting_number` are skipped: they are
# class-imbalanced binary tasks that collapse to majority-class predictor
# under accuracy-monitored early stopping. Re-add only after switching
# `optimized_metric` to macro-F1 and adding class-weighted loss.
TASKS=(${TASKS:-crossing_number signature genus_3d genus_4d})
MODELS=(${MODELS:-gin cwn})

N_TRIALS="${N_TRIALS:-15}"          # Optuna trials per (task, model)
N_WORKERS="${N_WORKERS:-8}"         # DataLoader workers
BATCH_SIZE="${BATCH_SIZE:-64}"      # Default; sweep search space overrides
GPUS="${GPUS:-1}"                   # Number of GPUs on this pod
EXP="${EXP:-pod_$(date -u +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-${REPO_ROOT}/logs/${EXP}}"
DBDIR="${DBDIR:-${REPO_ROOT}/outputs/optuna_dbs}"

# Reproducibility: spherogram's many_diagrams() is sensitive to Python's
# per-process hash randomization. Pin it for the entire pipeline.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

# CUDA / OMP hygiene.
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

ACTIVATE="${VENV_DIR}/bin/activate"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
say() { printf '\n=== %s :: %s ===\n' "$(date -Iseconds)" "$*"; }

ensure_venv() {
    if [[ ! -f "${ACTIVATE}" ]]; then
        say "creating venv at ${VENV_DIR}"
        "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    fi
    # shellcheck disable=SC1090
    source "${ACTIVATE}"
    python -m pip install --upgrade pip wheel setuptools >/dev/null
}

# ---------------------------------------------------------------------------
# install :: system + python deps
# ---------------------------------------------------------------------------
do_install() {
    say "system deps (best-effort apt; skip if not root)"
    if command -v apt-get >/dev/null && [[ "$(id -u)" == "0" ]]; then
        apt-get update -y
        # SnapPy uses tk for some optional features; the python `snappy`
        # package itself does not strictly require it for headless use,
        # but `plink` (a transitive dep) does. libgl/libglib are needed
        # by snappy_manifolds C extensions on minimal containers.
        apt-get install -y --no-install-recommends \
            python3.11 python3.11-venv python3.11-dev \
            build-essential git curl ca-certificates \
            libgl1 libglib2.0-0 libxext6 libxrender1
    else
        echo "(non-root or no apt; assuming system deps already present)"
    fi

    ensure_venv

    say "torch + CUDA wheels (cu121 to match snapshot's torch==2.3.0)"
    pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.3.0 torchvision==0.18.0

    # PyG companion wheels must match torch+CUDA exactly. The snapshot pins
    # 2.3.0+cu121 wheels; install from PyG's wheel index.
    pip install --no-cache-dir \
        torch-scatter==2.1.2 torch-sparse==0.6.18 \
        torch-cluster==1.6.3 torch-spline-conv==1.2.2 \
        -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

    pip install torch-geometric==2.7.0

    say "snappy + spherogram + topology stack"
    # SnapPy publishes manylinux wheels; pip install works on standard pods.
    pip install snappy==3.3.2 spherogram==2.4.1
    pip install \
        "toponetx @ git+https://github.com/pyt-team/TopoNetX.git@c378925c52169fd46bf81868dc6bf7c18f81a0bc" \
        "topomodelx @ git+https://github.com/pyt-team/TopoModelX.git@8fc047c85886642cf53a784fc712cbae89b015fd"

    say "TopoBench (local checkout at ${TOPOBENCH_ROOT})"
    if [[ ! -d "${TOPOBENCH_ROOT}" ]]; then
        echo "ERROR: TopoBench not found at ${TOPOBENCH_ROOT}." >&2
        echo "Either:" >&2
        echo "  - clone https://github.com/pyt-team/TopoBenchmark to that path, OR" >&2
        echo "  - set TOPOBENCH_ROOT=/path/to/TopoBench" >&2
        exit 1
    fi
    pip install -e "${TOPOBENCH_ROOT}"

    say "remaining python deps from snapshot"
    # Skip lines we already installed above (or pin elsewhere) to avoid
    # version conflicts.
    grep -v -E '^(torch|topobench|toponetx|topomodelx|snappy|spherogram|pyg-nightly)([ =@]|$)' \
        "${REPO_ROOT}/requirements-snapshot.txt" \
        > "${REPO_ROOT}/requirements-pod.txt"
    pip install -r "${REPO_ROOT}/requirements-pod.txt"

    say "thesis-only extras"
    pip install optuna-dashboard

    say "install complete"
}

# ---------------------------------------------------------------------------
# verify :: imports + GPU + tiny CWN forward pass
# ---------------------------------------------------------------------------
do_verify() {
    ensure_venv
    say "torch CUDA visibility"
    python -c "
import torch, sys
print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(),
      'devices:', torch.cuda.device_count())
assert torch.cuda.is_available(), 'CUDA not visible to torch'
print('device 0:', torch.cuda.get_device_name(0))
"
    say "package imports"
    python -c "
import sys; sys.path.insert(0, '$REPO_ROOT')
from src.data import KnotDataset, KnotAugmentedDataset, KnotDatasetLoader
from src.transforms.graph2cell_face_lifting import Graph2CellFaceLifting
from src.utils.instantiators import instantiate_callbacks
import topobench, lightning, hydra, optuna
print('all good')
"
    say "tiny end-to-end (1 fold, 1 epoch, GPU)"
    rm -rf data/knots_verify outputs/verify
    python -u scripts/train_tb.py \
        task=signature \
        experiment_name=verify \
        dataset.loader.parameters.limit=20 \
        dataset.loader.parameters.data_name=knots_verify \
        trainer.max_epochs=1 \
        trainer.accelerator=gpu trainer.devices=1 \
        trainer.check_val_every_n_epoch=1 \
        trainer.enable_progress_bar=false \
        'split.seeds=[0]' \
        dataset.dataloader_params.batch_size=4 \
        callbacks.early_stopping.patience=1
    rm -rf data/knots_verify outputs/verify
    say "verify ok"
}

# ---------------------------------------------------------------------------
# precache :: build the PyG data.pt for each task once, sequentially
# Avoids repeated SnapPy parsing across 25 trials per (task, model).
# Skip per-task if the cache already exists.
# ---------------------------------------------------------------------------
do_precache() {
    ensure_venv
    for task in "${TASKS[@]}"; do
        say "precache ${task}"
        python -u scripts/train_tb.py \
            task="$task" \
            train=false test=false \
            trainer.max_epochs=1 \
            trainer.limit_train_batches=1 \
            trainer.limit_val_batches=1 \
            trainer.limit_test_batches=1 \
            'split.seeds=[0]' \
            trainer.accelerator=cpu \
            trainer.enable_progress_bar=false 2>&1 \
            | grep -E 'Processing|Done|train_tb|skipped' || true
    done
}

# ---------------------------------------------------------------------------
# sweep :: per (task, model), run an Optuna study, log to file + SQLite DB
# Sequential by default (1 GPU). For multi-GPU, set GPUS>1 and uncomment
# the parallel branch below.
# ---------------------------------------------------------------------------
do_sweep() {
    ensure_venv
    mkdir -p "${LOGDIR}" "${DBDIR}"
    say "sweep config: tasks=${TASKS[*]} models=${MODELS[*]} n_trials=${N_TRIALS} gpus=${GPUS}"

    if [[ "${GPUS}" -le 1 ]]; then
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
                    trainer.accelerator=gpu trainer.devices=1 \
                    trainer.enable_progress_bar=false \
                    dataset.dataloader_params.batch_size="${BATCH_SIZE}" \
                    dataset.dataloader_params.num_workers="${N_WORKERS}" \
                    dataset.dataloader_params.pin_memory=true \
                    2>&1 | tee "${log}"
            done
        done
    else
        # Parallel across GPUs: round-robin (task, model) pairs onto GPUs.
        local i=0
        for task in "${TASKS[@]}"; do
            for model in "${MODELS[@]}"; do
                local gpu=$(( i % GPUS ))
                local log="${LOGDIR}/${task}_${model}.log"
                local db="${DBDIR}/${EXP}_${task}_${model}.db"
                say "starting ${task} / ${model} on GPU ${gpu} -> ${log}"
                CUDA_VISIBLE_DEVICES="${gpu}" python -u scripts/train_tb.py \
                    hparams_search=optuna \
                    experiment_name="${EXP}" \
                    task="${task}" model="${model}" \
                    hydra.sweeper.n_trials="${N_TRIALS}" \
                    hydra.sweeper.storage="sqlite:///${db}" \
                    trainer.accelerator=gpu trainer.devices=1 \
                    trainer.enable_progress_bar=false \
                    dataset.dataloader_params.batch_size="${BATCH_SIZE}" \
                    dataset.dataloader_params.num_workers="${N_WORKERS}" \
                    dataset.dataloader_params.pin_memory=true \
                    > "${log}" 2>&1 &
                i=$(( i + 1 ))
                # Drain a wave once we've launched one job per GPU.
                if (( i % GPUS == 0 )); then wait; fi
            done
        done
        wait
    fi
    say "sweep complete -> ${LOGDIR}"
}

# ---------------------------------------------------------------------------
# summary :: dump best (task, model) result from each Optuna DB
# ---------------------------------------------------------------------------
do_summary() {
    ensure_venv
    say "results from ${DBDIR}"
    python -c "
import glob, json, os, sys
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
exp = os.environ.get('EXP', '')
pat = os.path.join('${DBDIR}', f'{exp or \"pod_*\"}_*_*.db')
rows = []
for db in sorted(glob.glob(pat)):
    fname = os.path.basename(db)[:-3]
    parts = fname.split('_')
    # exp prefix is everything up to second-from-last underscore
    model = parts[-1]; task = parts[-2]; expp = '_'.join(parts[:-2])
    s = optuna.load_study(study_name=None, storage=f'sqlite:///{db}')
    rows.append((expp, task, model, s.best_value, s.best_trial.number,
                 len(s.trials), s.best_params))
if not rows:
    print('no studies found at', pat); sys.exit(0)
print(f'{\"exp\":<22} {\"task\":<18} {\"model\":<6} {\"best\":>8} {\"#trial\":>6} {\"#all\":>5}  best_params')
print('-' * 110)
for expp, task, model, best, idx, n, params in rows:
    print(f'{expp:<22} {task:<18} {model:<6} {best:>8.4f} {idx:>6} {n:>5}  {json.dumps(params)}')
"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
cd "${REPO_ROOT}"
case "${1:-all}" in
    install)  do_install ;;
    verify)   do_verify ;;
    precache) do_precache ;;
    sweep)    do_sweep ;;
    summary)  do_summary ;;
    all)
        do_install
        do_verify
        do_precache
        do_sweep
        do_summary
        ;;
    *)
        echo "usage: $0 {install|verify|precache|sweep|summary|all}" >&2
        exit 2
        ;;
esac
