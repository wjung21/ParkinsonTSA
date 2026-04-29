#!/usr/bin/env bash
set -euo pipefail

# ─── Usage ────────────────────────────────────────────────────────────────────
# bash setup.sh [--download_raw]
#
#   --download_raw   Also download the raw OpenNeuro dataset into data/.
#                    Omit this flag when only the preprocessed data in
#                    data_preprocessed/ is needed (the default).

# ─── Configuration ────────────────────────────────────────────────────────────
ENV_NAME="parkinson-tsa"
DATASET_URL="https://github.com/OpenNeuroDatasets/ds007526.git"
DATA_DIR="data"
DOWNLOAD_RAW=false

# ─── Helpers ──────────────────────────────────────────────────────────────────
info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }

# ─── Parse arguments ──────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --download_raw)
            DOWNLOAD_RAW=true
            ;;
        *)
            error "Unknown argument: $arg  (usage: bash setup.sh [--download_raw])"
            ;;
    esac
done

# ─── 1. Verify conda ──────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    error "conda not found. Please install Anaconda or Miniconda and try again."
fi
info "Found conda: $(conda --version)"

# ─── 2. Raw dataset (optional — only when --download_raw is passed) ───────────
if [ "$DOWNLOAD_RAW" = true ]; then

    # ── 2a. Install / verify git-annex ────────────────────────────────────────
    # git-annex is not available on conda-forge for osx-arm64 (Apple Silicon).
    # On macOS we install it via Homebrew; on Linux it is available via conda-forge.
    if ! command -v git-annex &>/dev/null; then
        case "$(uname -s)" in
            Darwin)
                if ! command -v brew &>/dev/null; then
                    error "Homebrew not found. Install it from https://brew.sh and try again."
                fi
                info "Installing git-annex via Homebrew..."
                brew install git-annex
                ;;
            Linux)
                info "Installing git-annex via conda-forge..."
                conda install -y -c conda-forge git-annex
                ;;
            *)
                error "Unsupported OS: $(uname -s). Please install git-annex manually."
                ;;
        esac
    fi
    info "Found git-annex: $(git-annex version | head -n1)"

    # ── 2b. Install / verify datalad ──────────────────────────────────────────
    # DataLad is required to fetch git-annex annexed files from OpenNeuro repos.
    # Installed via conda-forge on all platforms (no platform gap for datalad).
    if ! command -v datalad &>/dev/null; then
        info "datalad not found — installing into base environment via conda-forge..."
        conda install -y -c conda-forge datalad
        # conda install doesn't update the current shell's PATH; add it explicitly
        CONDA_BASE=$(conda info --base)
        export PATH="$CONDA_BASE/bin:$PATH"
    fi
    info "Found datalad: $(datalad --version 2>&1 | head -n1)"

    # ── 2c. Download dataset ───────────────────────────────────────────────────
    mkdir -p "$DATA_DIR"
    DATASET_DEST="$DATA_DIR/ds007526"

    if [ -d "$DATASET_DEST/.datalad" ]; then
        info "Dataset already downloaded at $DATASET_DEST — skipping."
    else
        info "Cloning dataset from $DATASET_URL into $DATASET_DEST ..."
        datalad install --source "$DATASET_URL" "$DATASET_DEST"
        info "Downloading dataset files (this may take a while) ..."
        datalad get --dataset "$DATASET_DEST" "$DATASET_DEST"
        info "Dataset download complete."
    fi

else
    info "Skipping raw data download (pass --download_raw to enable)."
fi

# ─── 4. Create conda environment ──────────────────────────────────────────────
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    warn "Conda environment '$ENV_NAME' already exists — skipping creation."
    warn "To recreate it, run:  conda env remove -n $ENV_NAME"
else
    info "Creating conda environment '$ENV_NAME' ..."
    conda create -y -n "$ENV_NAME" -c conda-forge \
        python=3.11 \
        numpy \
        pandas \
        scipy \
        matplotlib \
        seaborn \
        scikit-learn \
        statsmodels \
        nibabel \
        nilearn \
        mne \
        jupyter \
        ipykernel \
        tqdm \
        xarray \
        onnxruntime

    # mne-icalabel is not on conda-forge; install via pip after the env is created
    conda run -n "$ENV_NAME" pip install mne-icalabel

    # Register the environment as a Jupyter kernel so notebooks can use it
    conda run -n "$ENV_NAME" python -m ipykernel install \
        --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

    info "Environment '$ENV_NAME' created and registered as a Jupyter kernel."
fi

# ─── 5. Done ──────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Setup complete!"
echo ""
echo " Activate your environment with:"
echo "   conda activate $ENV_NAME"
if [ "$DOWNLOAD_RAW" = true ]; then
echo ""
echo " Raw dataset location:  $DATA_DIR/ds007526"
fi
echo "========================================================"
