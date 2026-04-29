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
| Core numerics | `numpy`, `pandas`, `scipy`, `xarray` |
| Statistics / ML | `statsmodels`, `scikit-learn` |
| Neuroimaging | `nibabel`, `nilearn`, `mne` |
| Visualization | `matplotlib`, `seaborn` |
| Utilities | `tqdm` |
| Notebooks | `jupyter`, `ipykernel` |

## Analysis Pipeline

### 1. Participant Selection — Propensity Score Matching (`utils/psm.py`)

Not all participants have both a resting-state and a walking recording. Before any analysis, subjects missing either recording are excluded, and the remaining Healthy Controls (HC) and Parkinson's Disease (PD) participants are balanced using **Propensity Score Matching (PSM)**.

PSM fits a logistic regression to estimate each participant's probability of belonging to the HC group given their covariates (age, sex, MoCA score). Each HC participant is then matched to the most similar PD participant by propensity score distance.

```python
import pandas as pd
from utils.psm import PropensityScoreMatching

df = pd.read_csv("data/ds007526/participants.tsv", sep="\t", na_values="n/a")

psm = PropensityScoreMatching(
    target_col="group",
    features=["age", "sex", "moca"],
    data_dir="data/ds007526",
    required_tasks=["rest", "walk"],   # exclude subjects missing either recording
    caliper=0.2,                        # max allowed propensity score distance
    random_state=42,
)
matched = psm.fit_match(df)
psm.summary()                           # prints SMD before/after matching
```

The returned dataframe contains the matched HC–PD pairs along with all original participant metadata (`participant_id`, `subject_id`, demographic variables, etc.) and two added columns:

| Column | Description |
|---|---|
| `_pscore` | Estimated propensity score |
| `_match_id` | Integer linking each HC to its matched PD partner |

### 2. EEG Loading & Preprocessing (`utils/eeg_loader.py`)

EEG recordings are stored in EEGLAB `.set` format under `data/ds007526/{participant_id}/eeg/`. Each participant has up to two task recordings:

| Task | Filename pattern |
|---|---|
| Resting-state | `{participant_id}_task-rest_eeg.set` |
| Walking | `{participant_id}_task-walk_eeg.set` |

The function `load_preprocess` loads and preprocesses both recordings for a single participant and returns them as [`xarray.DataArray`](https://docs.xarray.dev) objects (shape `channel × time`).

```python
from utils.eeg_loader import load_preprocess

data = load_preprocess("sub-002", data_dir="data/ds007526")
rest = data["rest"]   # xr.DataArray  (65 channels × ~60 000 samples)
walk = data["walk"]
```

#### Preprocessing steps

The following steps are applied to each recording in order:

**1. Electrode positions**
Electrode 3-D positions (CapTrak coordinate system, units: metres) are loaded from the BIDS sidecar `{participant_id}_space-CapTrak_electrodes.tsv` and attached to the raw object as a digitisation montage. This is required for subsequent spherical spline interpolation.

**2. Bad channel marking**
Channels flagged as `"bad"` in the BIDS sidecar `{participant_id}_task-{task}_channels.tsv` are recorded in `raw.info["bads"]`.

**3. Bandpass filtering**
A zero-phase FIR bandpass filter is applied (default: **1–45 Hz**). The high-pass at 1 Hz removes slow drifts and DC offset; the low-pass at 45 Hz suppresses line-noise harmonics and high-frequency muscle artefacts while retaining all classical EEG frequency bands (delta 1–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, beta 13–30 Hz, gamma 30–45 Hz).

**4. Average reference**
The signal is re-referenced to the **common average** of all electrodes, which is the standard reference for source-independent EEG analysis and minimises the spatial bias introduced by any single reference electrode.

**5. Bad channel interpolation**
Channels marked bad in step 2 are reconstructed from their neighbours using **spherical spline interpolation** (MNE default). The interpolated channel names are recorded in `da.attrs["interpolated_bads"]`.

**6. Export to `xarray.DataArray`**
The preprocessed signal is exported to an `xr.DataArray` with:
- **dims** `("channel", "time")` — rows are electrodes, columns are time points
- **`coords["channel"]`** — electrode labels (e.g. `"Fz"`, `"Cz"`)
- **`coords["time"]`** — time in seconds from recording onset
- **`attrs`** — `sfreq`, `l_freq`, `h_freq`, `reference`, `interpolated_bads`, `participant_id`, `task`

`xarray` is chosen over a plain `numpy.ndarray` because channel names and the time axis are retained as named coordinates, enabling label-based selection (`da.sel(channel="Fz")`, `da.sel(time=slice(10, 60))`), while still supporting all NumPy/SciPy operations. The underlying array is always accessible via `da.values`.

#### Preprocessing parameters

| Parameter | Default | Description |
|---|---|---|
| `l_freq` | `1.0` Hz | High-pass cut-off |
| `h_freq` | `45.0` Hz | Low-pass cut-off |
| `reference` | `"average"` | EEG reference (`"average"` or a channel name) |
| `interpolate_bads` | `True` | Interpolate bad channels after referencing |

## Project Structure

```
ParkinsonTSA/
├── data/               # Downloaded dataset (git-ignored)
│   └── ds007526/
├── utils/
│   ├── psm.py          # Propensity Score Matching
│   └── eeg_loader.py   # EEG loading & preprocessing
├── setup.sh            # Environment & dataset setup script
├── LICENSE
└── README.md
```
