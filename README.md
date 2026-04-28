# ParkinsonTSA
GitHub repository for the Fall 2026 STAT 248 final project — time series analysis of Parkinson's disease neuroimaging data.

## Dataset

This project uses the OpenNeuro dataset [ds007526](https://openneuro.org/datasets/ds007526). The dataset is **not stored in this repository**; `setup.sh` downloads it automatically into the `data/` folder.

## Prerequisites

| Requirement | Notes |
|---|---|
| [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) | Required for environment creation |
| [Homebrew](https://brew.sh) | macOS only — required to install `git-annex` |

## Setup

Clone the repository and run the setup script from the project root:

```bash
git clone https://github.com/wjung21/ParkinsonTSA.git
cd ParkinsonTSA
bash setup.sh
```

The script will:
1. Verify that `conda` is available
2. Install `git-annex` — via Homebrew on macOS, or conda-forge on Linux
3. Install `datalad` via conda-forge (used to download annexed dataset files)
4. Download the dataset into `data/ds007526/`
5. Create a conda environment named `parkinson-tsa` (Python 3.11) with the packages listed below
6. Register the environment as a Jupyter kernel

The script is **idempotent** — re-running it safely skips any step already completed.

## Activating the Environment

```bash
conda activate parkinson-tsa
```

## Python Packages

| Category | Packages |
|---|---|
| Core numerics | `numpy`, `pandas`, `scipy` |
| Statistics / ML | `statsmodels`, `scikit-learn` |
| Neuroimaging | `nibabel`, `nilearn`, `mne` |
| Visualization | `matplotlib`, `seaborn` |
| Utilities | `tqdm` |
| Notebooks | `jupyter`, `ipykernel` |

## Project Structure

```
ParkinsonTSA/
├── data/               # Downloaded dataset (git-ignored)
│   └── ds007526/
├── setup.sh            # Environment & dataset setup script
├── LICENSE
└── README.md
```
