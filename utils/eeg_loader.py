"""
eeg_loader.py — EEG file loader for BIDS-formatted .set recordings
====================================================================
Loads and preprocesses EEGLAB .set files from a BIDS EEG dataset.

Two levels of API are provided:

``load_preprocess(participant_id, data_dir)``
    Per-subject entry point.  Loads both resting-state and walking
    recordings for one participant, runs the full preprocessing pipeline,
    and returns a ``dict[str, xr.DataArray]`` with keys ``"rest"`` and
    ``"walk"``.  Each DataArray has labeled dimensions ``(channel, time)``
    so channel names and a seconds-based time axis are always accessible.

``load_eeg`` / ``iter_eeg``
    Lower-level helpers that work over a PSM-matched dataframe and return
    raw :class:`mne.io.Raw` objects (no preprocessing applied).

Preprocessing pipeline (``load_preprocess``)
--------------------------------------------
1. Load EEGLAB ``.set`` file via MNE.
2. Set electrode positions from the BIDS ``_electrodes.tsv`` sidecar.
3. Mark bad channels from ``_channels.tsv``.
4. Bandpass filter (default 1–45 Hz).
5. Re-reference to average.
6. Interpolate bad channels (spherical spline).
7. Export to ``xarray.DataArray`` (dims: channel × time).

Why xarray over numpy.ndarray
------------------------------
A plain ndarray discards all metadata the moment it leaves MNE.  An
``xr.DataArray`` keeps:

* **Named dimensions** – ``da.sel(channel="Fz")`` works without knowing
  the channel index.
* **Coordinates** – ``da.coords["time"]`` gives absolute time in seconds;
  ``da.coords["channel"]`` gives electrode labels.
* **Attributes** – sampling rate, applied filters, subject ID, task, and
  reference are stored in ``da.attrs`` and travel with the array.
* **NumPy compatibility** – ``da.values`` returns the underlying ndarray
  instantly; all scipy/numpy operations work directly on the DataArray.

Typical usage
-------------
>>> from utils.eeg_loader import load_preprocess
>>>
>>> data = load_preprocess("sub-002", data_dir="data/ds007526")
>>> rest = data["rest"]   # xr.DataArray  (channel, time)
>>> walk = data["walk"]
>>>
>>> rest.shape                        # (65, 60904)
>>> rest.coords["channel"].values     # array(['F10', 'AF4', ...])
>>> rest.coords["time"].values        # array([0.   , 0.004, ...])  seconds
>>> rest.attrs["sfreq"]               # 250.0
>>>
>>> # Select a single channel as a 1-D array
>>> fz = rest.sel(channel="Fz")
>>>
>>> # Convert to plain numpy whenever needed
>>> arr = rest.values                 # shape (n_channels, n_times)
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Literal

import mne
import numpy as np
import pandas as pd
import xarray as xr


# ── Public API ────────────────────────────────────────────────────────────────

def load_eeg(
    matched_df: pd.DataFrame,
    task: Literal["rest", "walk"],
    data_dir: str | Path,
    *,
    preload: bool = True,
    verbose: bool = False,
) -> dict[str, mne.io.BaseRaw]:
    """Load EEG recordings for all subjects in *matched_df*.

    Parameters
    ----------
    matched_df : pd.DataFrame
        PSM output dataframe; must contain a ``participant_id`` column.
    task : {"rest", "walk"}
        Which task recording to load.
    data_dir : str | Path
        Root of the BIDS dataset (e.g. ``"data/ds007526"``).
    preload : bool, default True
        If ``True``, load the signal data into memory immediately.
        Set to ``False`` for lazy loading when RAM is limited.
    verbose : bool, default False
        Passed to :func:`mne.io.read_raw_eeglab`. Set to ``True``
        to see MNE's channel/info messages.

    Returns
    -------
    dict[str, mne.io.BaseRaw]
        Mapping of ``participant_id`` → :class:`mne.io.Raw`.
        Bad channels from the ``_channels.tsv`` sidecar are marked
        in ``raw.info["bads"]``.
    """
    return {
        pid: raw
        for pid, _, raw in iter_eeg(
            matched_df, task=task, data_dir=data_dir,
            preload=preload, verbose=verbose,
        )
    }


def iter_eeg(
    matched_df: pd.DataFrame,
    task: Literal["rest", "walk"],
    data_dir: str | Path,
    *,
    preload: bool = True,
    verbose: bool = False,
) -> Generator[tuple[str, str, mne.io.BaseRaw], None, None]:
    """Yield ``(participant_id, group, raw)`` for each subject in *matched_df*.

    A generator alternative to :func:`load_eeg` that reads one file at a
    time, avoiding loading all recordings into memory simultaneously.

    Parameters
    ----------
    matched_df : pd.DataFrame
        PSM output dataframe; must contain ``participant_id`` and ``group``
        columns.
    task : {"rest", "walk"}
        Which task recording to load.
    data_dir : str | Path
        Root of the BIDS dataset.
    preload : bool, default True
        Whether to preload signal data into memory.
    verbose : bool, default False
        MNE verbosity flag.

    Yields
    ------
    participant_id : str
    group : str
        Group label (e.g. ``"HC"`` or ``"PD"``).
    raw : mne.io.BaseRaw
        Loaded recording with bad channels marked.
    """
    if "participant_id" not in matched_df.columns:
        raise ValueError("matched_df must contain a 'participant_id' column.")
    if "group" not in matched_df.columns:
        raise ValueError("matched_df must contain a 'group' column.")

    data_dir = Path(data_dir)
    seen = set()

    for _, row in matched_df.iterrows():
        pid = row["participant_id"]
        group = row["group"]

        if pid in seen:          # guard against duplicate rows
            continue
        seen.add(pid)

        eeg_dir = data_dir / pid / "eeg"
        set_file = eeg_dir / f"{pid}_task-{task}_eeg.set"

        if not set_file.exists():
            raise FileNotFoundError(
                f"EEG file not found for {pid} (task={task}): {set_file}\n"
                "Run PropensityScoreMatching with required_tasks=[..., "
                f"'{task}'] to exclude subjects missing this recording."
            )

        raw = mne.io.read_raw_eeglab(str(set_file), preload=preload, verbose=verbose)

        # ── Mark bad channels from BIDS sidecar ───────────────────────
        channels_file = eeg_dir / f"{pid}_task-{task}_channels.tsv"
        if channels_file.exists():
            ch_df = pd.read_csv(channels_file, sep="\t")
            if "status" in ch_df.columns and "name" in ch_df.columns:
                bads = ch_df.loc[ch_df["status"] == "bad", "name"].tolist()
                raw.info["bads"] = bads

        yield pid, group, raw


# ── Per-subject load + preprocess ─────────────────────────────────────────────

def load_preprocess(
    participant_id: str,
    data_dir: str | Path,
    *,
    l_freq: float = 1.0,
    h_freq: float = 60.0,
    reference: str = "average",
    interpolate_bads: bool = True,
    verbose: bool = False,
) -> dict[str, xr.DataArray]:
    """Load and preprocess both EEG recordings for one participant.

    Runs the full preprocessing pipeline (filter → re-reference →
    interpolate bad channels) on the resting-state and walking recordings
    and returns them as labeled ``xr.DataArray`` objects.

    Parameters
    ----------
    participant_id : str
        BIDS participant ID, e.g. ``"sub-002"``.
    data_dir : str | Path
        Root of the BIDS dataset (e.g. ``"data/ds007526"``).
    l_freq : float, default 1.0
        High-pass cut-off frequency (Hz).
    h_freq : float, default 45.0
        Low-pass cut-off frequency (Hz).
    reference : str, default ``"average"``
        EEG reference to apply.  Passed directly to
        :meth:`mne.io.Raw.set_eeg_reference`.  Use ``"average"`` for
        common average reference, or a channel name (e.g. ``"Cz"``) for
        a single-electrode reference.
    interpolate_bads : bool, default True
        If ``True``, interpolate channels marked bad in the
        ``_channels.tsv`` sidecar using spherical spline interpolation.
        If ``False``, bad channels are marked but left in the data.
    verbose : bool, default False
        MNE verbosity flag.

    Returns
    -------
    dict[str, xr.DataArray]
        Keys are ``"rest"`` and ``"walk"``.  Each value is an
        ``xr.DataArray`` with:

        * **dims** ``("channel", "time")``
        * **coords["channel"]** – electrode labels (str)
        * **coords["time"]** – time in seconds (float)
        * **attrs** – ``sfreq``, ``participant_id``, ``task``,
          ``l_freq``, ``h_freq``, ``reference``,
          ``interpolated_bads`` (list of interpolated channel names)

    Raises
    ------
    FileNotFoundError
        If either the rest or walk ``.set`` file is missing for this
        participant.
    """
    data_dir = Path(data_dir)
    result: dict[str, xr.DataArray] = {}

    for task in ("rest", "walk"):
        raw = _load_raw(participant_id, task, data_dir, verbose=verbose)
        da  = _preprocess(
            raw,
            participant_id=participant_id,
            task=task,
            l_freq=l_freq,
            h_freq=h_freq,
            reference=reference,
            interpolate_bads=interpolate_bads,
            verbose=verbose,
        )
        result[task] = da

    return result


# ── Private helpers ───────────────────────────────────────────────────────────

def _load_raw(
    participant_id: str,
    task: str,
    data_dir: Path,
    *,
    verbose: bool,
) -> mne.io.BaseRaw:
    """Load a single .set file and mark bad channels from the sidecar."""
    eeg_dir  = data_dir / participant_id / "eeg"
    set_file = eeg_dir / f"{participant_id}_task-{task}_eeg.set"

    if not set_file.exists():
        raise FileNotFoundError(
            f"EEG file not found for {participant_id} (task={task}): {set_file}"
        )

    raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=verbose)

    # ── Electrode positions from BIDS sidecar ─────────────────────────
    elec_file = eeg_dir / f"{participant_id}_space-CapTrak_electrodes.tsv"
    if elec_file.exists():
        elec_df = pd.read_csv(elec_file, sep="\t")
        # Build a DigMontage from the stored x/y/z coordinates (metres)
        ch_pos = {
            row["name"]: np.array([row["x"], row["y"], row["z"]], dtype=float)
            for _, row in elec_df.iterrows()
            if row["name"] in raw.ch_names
        }
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
        raw.set_montage(montage, on_missing="warn", verbose=verbose)

    # ── Bad channels from BIDS sidecar ────────────────────────────────
    ch_file = eeg_dir / f"{participant_id}_task-{task}_channels.tsv"
    if ch_file.exists():
        ch_df = pd.read_csv(ch_file, sep="\t")
        if {"name", "status"}.issubset(ch_df.columns):
            raw.info["bads"] = ch_df.loc[
                ch_df["status"] == "bad", "name"
            ].tolist()

    return raw


def _preprocess(
    raw: mne.io.BaseRaw,
    *,
    participant_id: str,
    task: str,
    l_freq: float,
    h_freq: float,
    reference: str,
    interpolate_bads: bool,
    verbose: bool,
) -> xr.DataArray:
    """Apply the preprocessing pipeline and return an xr.DataArray."""

    # ── 1. Bandpass filter ────────────────────────────────────────────
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=verbose)

    # ── 2. Re-reference ───────────────────────────────────────────────
    raw.set_eeg_reference(reference, verbose=verbose)

    # ── 3. Interpolate bad channels ───────────────────────────────────
    interpolated: list[str] = []
    if interpolate_bads and raw.info["bads"]:
        interpolated = list(raw.info["bads"])
        raw.interpolate_bads(reset_bads=True, verbose=verbose)

    # ── 4. Export to xarray.DataArray ────────────────────────────────
    data, times = raw.get_data(return_times=True)   # (n_ch, n_times) float64

    da = xr.DataArray(
        data,
        dims=("channel", "time"),
        coords={
            "channel": raw.ch_names,
            "time":    times,          # seconds
        },
        attrs={
            "participant_id":     participant_id,
            "task":               task,
            "sfreq":              raw.info["sfreq"],
            "l_freq":             l_freq,
            "h_freq":             h_freq,
            "reference":          reference,
            "interpolated_bads":  interpolated,
        },
    )
    return da
