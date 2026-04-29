#!/usr/bin/env python
"""
create_preprocessed_data.py
============================
Runs the full preprocessing pipeline for all PSM-matched participants and
saves the results to ``data_preprocessed/``.

Pipeline per subject
---------------------
1. Propensity Score Matching   (utils/psm.py)
2. Load + preprocess EEG       (utils/eeg_loader.py  → load_preprocess)
3. ICA artifact reduction      (utils/artifact.py    → apply_ica)
4. Save to HDF5                (data_preprocessed/{participant_id}_{task}.h5)

Output format
-------------
Each file is a self-contained HDF5 archive:

    data_preprocessed/
        sub-002_rest.h5
        sub-002_walk.h5
        sub-047_rest.h5
        ...

HDF5 internal layout::

    /data        float64  (n_channels, n_times)   — EEG signal in µV
    /channels    str      (n_channels,)            — electrode labels
    /time        float64  (n_times,)               — time axis in seconds
    /attrs       HDF5 group attributes             — all metadata

List-valued attributes (notch_freqs, ica_excluded_*, …) are stored as
JSON strings and automatically deserialised by load_h5().

Usage
-----
Run from the project root::

    python utils/create_preprocessed_data.py

Already-processed files are skipped automatically; delete or re-run to
overwrite.

Loading saved files
-------------------
Use the bundled helper::

    from utils.create_preprocessed_data import load_h5
    da = load_h5("data_preprocessed/sub-002_rest.h5")
    # da is an xr.DataArray with all metadata in .attrs
"""

from __future__ import annotations

import json
import sys
import traceback
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# ── project root on sys.path so utils.* imports work when run as a script ──
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.artifact import apply_ica
from utils.eeg_loader import load_preprocess
from utils.psm import PropensityScoreMatching

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = _ROOT / "data" / "ds007526"
OUTPUT_DIR = _ROOT / "data_preprocessed"

PARTICIPANTS_TSV = DATA_DIR / "participants.tsv"

# PSM settings
PSM_TARGET_COL     = "group"
PSM_FEATURES       = ["age", "sex", "moca"]
PSM_REQUIRED_TASKS = ["rest", "walk"]
PSM_CALIPER        = 0.2
PSM_RANDOM_STATE   = 42

# Preprocessing settings
PREPROCESS_L_FREQ     = 1.0
PREPROCESS_H_FREQ     = 60.0
PREPROCESS_NOTCH      = [50.0, 100.0]
PREPROCESS_REFERENCE  = "average"

# ICA settings
ICA_FIT_ON          = "rest"
ICA_EXCLUDE_LABELS  = ["eye blink", "muscle artifact", "heart beat"]
ICA_LABEL_THRESHOLD = 0.8
ICA_RANDOM_STATE    = 42


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_preprocessed(da: xr.DataArray, path: Path) -> None:
    """Save a preprocessed ``xr.DataArray`` to an HDF5 file.

    Parameters
    ----------
    da : xr.DataArray
        Array with dims ``(channel, time)`` as returned by
        :func:`~utils.artifact.apply_ica`.
    path : Path
        Destination ``.h5`` file path.  Parent directory must exist.
    """
    path = Path(path)
    with h5py.File(path, "w") as f:
        # ── Signal data (gzip-compressed, level 4) ────────────────────
        f.create_dataset(
            "data",
            data=da.values.astype(np.float64),
            compression="gzip",
            compression_opts=4,
        )

        # ── Coordinates ───────────────────────────────────────────────
        ch_names = da.coords["channel"].values.tolist()
        f.create_dataset(
            "channels",
            data=np.array(ch_names, dtype=h5py.string_dtype()),
        )
        f.create_dataset("time", data=da.coords["time"].values.astype(np.float64))

        # ── Metadata attributes ───────────────────────────────────────
        # Lists are serialised to JSON strings so HDF5 can store them;
        # load_h5() deserialises them back to Python lists.
        for key, val in da.attrs.items():
            if isinstance(val, (list, dict)):
                f.attrs[key] = json.dumps(val)
            else:
                f.attrs[key] = val


def load_h5(path: str | Path) -> xr.DataArray:
    """Load a preprocessed EEG recording from an HDF5 file.

    Parameters
    ----------
    path : str | Path
        Path to a ``.h5`` file created by :func:`save_preprocessed`.

    Returns
    -------
    xr.DataArray
        Array with dims ``(channel, time)`` and all metadata in ``.attrs``.
        List-valued attributes (e.g. ``notch_freqs``, ``ica_excluded_labels``)
        are returned as Python lists.

    Examples
    --------
    >>> da = load_h5("data_preprocessed/sub-002_rest.h5")
    >>> da.sel(channel="Fz")          # time series for electrode Fz
    >>> da.attrs["ica_excluded_labels"]
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        data     = f["data"][:]
        channels = f["channels"][:].astype(str).tolist()
        time     = f["time"][:]

        attrs: dict = {}
        for key, val in f.attrs.items():
            # Attempt JSON deserialisation for list/dict attrs
            if isinstance(val, str):
                try:
                    attrs[key] = json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    attrs[key] = val
            else:
                attrs[key] = val

    return xr.DataArray(
        data,
        dims=("channel", "time"),
        coords={"channel": channels, "time": time},
        attrs=attrs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── 1. Propensity Score Matching ──────────────────────────────────
    print("=" * 60)
    print("  Step 1 — Propensity Score Matching")
    print("=" * 60)

    df = pd.read_csv(PARTICIPANTS_TSV, sep="\t", na_values="n/a")

    psm = PropensityScoreMatching(
        target_col=PSM_TARGET_COL,
        features=PSM_FEATURES,
        data_dir=str(DATA_DIR),
        required_tasks=PSM_REQUIRED_TASKS,
        caliper=PSM_CALIPER,
        random_state=PSM_RANDOM_STATE,
    )
    matched = psm.fit_match(df)
    psm.summary()

    participant_ids: list[str] = matched["participant_id"].unique().tolist()
    print(f"\n  {len(participant_ids)} subjects to process → {OUTPUT_DIR}\n")

    # ── 2. Per-subject preprocessing loop ────────────────────────────
    print("=" * 60)
    print("  Step 2 — Preprocessing & artifact reduction")
    print("=" * 60)

    succeeded: list[str] = []
    skipped:   list[str] = []
    failed:    dict[str, str] = {}

    for pid in tqdm(participant_ids, desc="Subjects", unit="subj"):

        # Check whether both output files already exist
        out_rest = OUTPUT_DIR / f"{pid}_rest.h5"
        out_walk = OUTPUT_DIR / f"{pid}_walk.h5"
        if out_rest.exists() and out_walk.exists():
            skipped.append(pid)
            tqdm.write(f"  [SKIP]  {pid} — already processed")
            continue

        try:
            # ── Preprocess ────────────────────────────────────────────
            data = load_preprocess(
                pid,
                data_dir=str(DATA_DIR),
                l_freq=PREPROCESS_L_FREQ,
                h_freq=PREPROCESS_H_FREQ,
                notch_freqs=PREPROCESS_NOTCH,
                reference=PREPROCESS_REFERENCE,
            )

            # ── ICA artifact reduction ────────────────────────────────
            clean = apply_ica(
                data,
                participant_id=pid,
                data_dir=str(DATA_DIR),
                fit_on=ICA_FIT_ON,
                exclude_labels=ICA_EXCLUDE_LABELS,
                label_threshold=ICA_LABEL_THRESHOLD,
                random_state=ICA_RANDOM_STATE,
            )

            # ── Save ──────────────────────────────────────────────────
            save_preprocessed(clean["rest"], out_rest)
            save_preprocessed(clean["walk"], out_walk)

            n_excl = len(clean["rest"].attrs["ica_excluded_indices"])
            tqdm.write(f"  [OK]    {pid} — {n_excl} ICA component(s) removed")
            succeeded.append(pid)

        except Exception:
            msg = traceback.format_exc().strip().splitlines()[-1]
            tqdm.write(f"  [FAIL]  {pid} — {msg}")
            failed[pid] = msg

    # ── 3. Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Processed : {len(succeeded)}")
    print(f"  Skipped   : {len(skipped)}  (output files already existed)")
    print(f"  Failed    : {len(failed)}")
    if failed:
        print("\n  Failed subjects:")
        for pid, msg in failed.items():
            print(f"    {pid}: {msg}")
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
