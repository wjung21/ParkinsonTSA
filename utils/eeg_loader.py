"""
eeg_loader.py — EEG file loader for BIDS-formatted .set recordings
====================================================================
Loads EEGLAB .set files for subjects listed in a PSM-matched dataframe.
Bad channels declared in the BIDS ``_channels.tsv`` sidecar are marked
automatically on each :class:`mne.io.Raw` object.

Typical usage
-------------
>>> import pandas as pd
>>> import mne
>>> from utils.psm import PropensityScoreMatching
>>> from utils.eeg_loader import load_eeg
>>>
>>> df = pd.read_csv("data/ds007526/participants.tsv", sep="\\t", na_values="n/a")
>>> psm = PropensityScoreMatching(
...     target_col="group", features=["age", "sex", "moca"],
...     data_dir="data/ds007526", required_tasks=["rest", "walk"],
... )
>>> matched = psm.fit_match(df)
>>>
>>> # Load all resting-state recordings
>>> recordings = load_eeg(matched, task="rest", data_dir="data/ds007526")
>>> raw = recordings["sub-002"]          # mne.io.Raw object
>>>
>>> # Iterate memory-efficiently (one file at a time)
>>> for pid, group, raw in iter_eeg(matched, task="walk", data_dir="data/ds007526"):
...     print(pid, group, raw.n_times)
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Literal

import mne
import pandas as pd


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
