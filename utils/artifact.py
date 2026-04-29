"""
artifact.py — ICA-based EEG artifact reduction
================================================
Removes ocular, muscle, and cardiac artifacts from preprocessed EEG
recordings using Independent Component Analysis (ICA) with automatic
component labelling via ICLabel.

Typical usage
-------------
>>> from utils.eeg_loader import load_preprocess
>>> from utils.artifact import apply_ica
>>>
>>> data  = load_preprocess("sub-002", data_dir="data/ds007526")
>>> clean = apply_ica(data, participant_id="sub-002", data_dir="data/ds007526")
>>>
>>> clean["rest"]   # xr.DataArray, same shape as data["rest"]
>>> clean["rest"].attrs["ica_excluded_labels"]   # e.g. ["eye blink", "muscle artifact"]

How it works
------------
1. The preprocessed ``xr.DataArray`` for the ``fit_on`` task (default: rest)
   is converted back to an MNE ``RawArray``.  Electrode positions are
   reloaded from the BIDS ``_electrodes.tsv`` sidecar so that ICLabel can
   use spatial features.
2. FastICA is run to decompose the signal into independent components (ICs).
3. ICLabel (a pre-trained CNN classifier) assigns each IC one of seven labels:
   ``brain``, ``muscle artifact``, ``eye blink``, ``heart beat``,
   ``line noise``, ``channel noise``, ``other``.
4. ICs whose predicted label is in ``exclude_labels`` *and* whose prediction
   confidence exceeds ``label_threshold`` are marked for removal.
5. The same unmixing matrix fitted on the ``fit_on`` task is applied to
   *all* tasks in ``data``.  Fitting on the resting-state recording (cleaner,
   more stationary signal) and applying to walking avoids overfitting ICA to
   movement-contaminated data.
6. Cleaned signals are exported back to ``xr.DataArray``, preserving all
   original ``attrs`` and appending ICA metadata.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import xarray as xr
from mne.preprocessing import ICA
from mne_icalabel import label_components


# Default artifact labels to remove
_DEFAULT_EXCLUDE = ["eye blink", "muscle artifact", "heart beat"]


def apply_ica(
    data: dict[str, xr.DataArray],
    participant_id: str,
    data_dir: str | Path,
    *,
    fit_on: str = "rest",
    n_components: int | None = None,
    exclude_labels: list[str] | None = None,
    label_threshold: float = 0.8,
    random_state: int = 42,
    verbose: bool = False,
) -> dict[str, xr.DataArray]:
    """Remove artifacts from preprocessed EEG recordings using ICA.

    ICA is fitted once on the ``fit_on`` recording and the resulting unmixing
    matrix is applied to all tasks in ``data``.  Artifactual independent
    components are identified automatically by ICLabel.

    Parameters
    ----------
    data : dict[str, xr.DataArray]
        Output of :func:`~utils.eeg_loader.load_preprocess`.
        Must contain at least the task specified by ``fit_on``.
    participant_id : str
        BIDS participant ID (e.g. ``"sub-002"``).  Used to reload electrode
        positions from ``_electrodes.tsv`` for ICLabel's spatial features.
    data_dir : str | Path
        Root of the BIDS dataset (e.g. ``"data/ds007526"``).
    fit_on : str, default ``"rest"``
        Task on whose data ICA is fitted.  The resting-state recording is
        recommended because it has a more stationary, lower-artefact signal
        that produces more stable ICA decompositions.  The fitted unmixing
        matrix is then applied to all tasks.
    n_components : int | None, default ``None``
        Number of ICA components to estimate.  ``None`` uses the data rank
        (n_channels − 1 after average reference, typically 64 for this
        dataset).  Reducing this (e.g. to 20) speeds up fitting and can
        improve stability when data quality is low.
    exclude_labels : list[str] | None, default ``["eye blink", "muscle artifact", "heart beat"]``
        ICLabel component labels to treat as artifacts.  Full set of labels:
        ``"brain"``, ``"muscle artifact"``, ``"eye blink"``, ``"heart beat"``,
        ``"line noise"``, ``"channel noise"``, ``"other"``.
    label_threshold : float, default 0.8
        Minimum ICLabel confidence (0–1) required to exclude a component.
        A component is only removed when its predicted label is in
        ``exclude_labels`` *and* the prediction probability is ≥ this value.
        Lowering the threshold removes more components (more aggressive);
        raising it retains borderline components (more conservative).
    random_state : int, default 42
        Random seed passed to FastICA for reproducibility.
    verbose : bool, default False
        MNE verbosity flag.

    Returns
    -------
    dict[str, xr.DataArray]
        Cleaned recordings in the same format as the input ``data``.
        Each DataArray preserves all original ``attrs`` from
        :func:`~utils.eeg_loader.load_preprocess` and additionally contains:

        * ``ica_fit_on`` – task used to fit ICA (e.g. ``"rest"``)
        * ``ica_n_components`` – number of ICA components estimated
        * ``ica_excluded_indices`` – list of removed component indices
        * ``ica_excluded_labels`` – ICLabel label for each removed component
        * ``ica_excluded_probs`` – ICLabel confidence for each removed component
        * ``ica_label_threshold`` – threshold used for exclusion

    Raises
    ------
    KeyError
        If ``fit_on`` is not a key in ``data``.
    """
    if fit_on not in data:
        raise KeyError(
            f"fit_on='{fit_on}' not found in data. "
            f"Available tasks: {list(data.keys())}"
        )

    if exclude_labels is None:
        exclude_labels = _DEFAULT_EXCLUDE

    data_dir = Path(data_dir)

    # ── 1. Convert fit_on DataArray → MNE Raw ─────────────────────────
    raw_fit = _dataarray_to_raw(
        data[fit_on], participant_id=participant_id, data_dir=data_dir,
        verbose=verbose,
    )

    # ── 2. Drop channels without electrode positions ───────────────────
    # Channels without scalp positions (e.g. VREF, the zero-valued average-
    # reference ghost channel added by EEGLAB) must be physically removed
    # from the Raw before ICA: ICLabel's feature extractor reads positions
    # directly from raw.info and raises an error for any channel with a
    # missing or zero loc vector.
    no_pos_chs = [
        ch["ch_name"] for ch in raw_fit.info["chs"]
        if not np.any(ch["loc"][:3] != 0)   # all-zero position
        or np.any(np.isnan(ch["loc"][:3]))   # NaN position (e.g. VREF)
    ]
    if no_pos_chs:
        raw_fit.drop_channels(no_pos_chs)

    # ── 3. Fit ICA ────────────────────────────────────────────────────
    # Rank is computed after dropping invalid channels; average reference
    # reduces rank by 1.
    rank = mne.compute_rank(raw_fit, rank="info", verbose=verbose)
    effective_rank = sum(rank.values())

    if n_components is None:
        n_components = effective_rank

    ica = ICA(
        n_components=n_components,
        method="infomax",
        fit_params={"extended": True},  # extended Infomax (handles both sub- and super-Gaussian sources)
        random_state=random_state,
        verbose=verbose,
    )
    ica.fit(raw_fit, verbose=verbose)

    # ── 4. Label components with ICLabel ──────────────────────────────
    ic_labels = label_components(raw_fit, ica, method="iclabel")
    labels = ic_labels["labels"]           # list[str], one per component
    probs  = ic_labels["y_pred_proba"]     # (n_components, n_classes)

    # ── 5. Select components to exclude ───────────────────────────────
    exclude_idx: list[int] = []
    exclude_lbl: list[str] = []
    exclude_prob: list[float] = []

    for i, label in enumerate(labels):
        confidence = float(probs[i].max())
        if label in exclude_labels and confidence >= label_threshold:
            exclude_idx.append(i)
            exclude_lbl.append(label)
            exclude_prob.append(round(confidence, 4))

    ica.exclude = exclude_idx

    # ── 5. Apply ICA to every task ────────────────────────────────────
    ica_attrs = {
        "ica_fit_on":           fit_on,
        "ica_n_components":     n_components,
        "ica_excluded_indices": exclude_idx,
        "ica_excluded_labels":  exclude_lbl,
        "ica_excluded_probs":   exclude_prob,
        "ica_label_threshold":  label_threshold,
    }

    clean: dict[str, xr.DataArray] = {}
    for task, da in data.items():
        raw_task = _dataarray_to_raw(
            da, participant_id=participant_id, data_dir=data_dir,
            verbose=verbose,
        )
        # Drop the same no-position channels before applying ICA
        if no_pos_chs:
            raw_task.drop_channels(
                [c for c in no_pos_chs if c in raw_task.ch_names]
            )
        ica.apply(raw_task, verbose=verbose)

        signal, times = raw_task.get_data(return_times=True)
        clean[task] = xr.DataArray(
            signal,
            dims=("channel", "time"),
            coords={"channel": raw_task.ch_names, "time": times},
            attrs={**da.attrs, **ica_attrs},
        )

    return clean


# ── Private helpers ───────────────────────────────────────────────────────────

def _dataarray_to_raw(
    da: xr.DataArray,
    *,
    participant_id: str,
    data_dir: Path,
    verbose: bool,
) -> mne.io.RawArray:
    """Convert an ``xr.DataArray`` back to an MNE ``RawArray``.

    Reloads electrode positions from the BIDS ``_electrodes.tsv`` sidecar
    and sets them as a digitisation montage.  This is required by ICLabel,
    which uses scalp topography as a spatial feature for classification.
    """
    sfreq = da.attrs["sfreq"]
    ch_names = da.coords["channel"].values.tolist()

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg",
                           verbose=verbose)
    raw = mne.io.RawArray(da.values, info, verbose=verbose)

    # Reload subject-specific electrode positions for ICLabel spatial features
    elec_file = data_dir / participant_id / "eeg" / \
                f"{participant_id}_space-CapTrak_electrodes.tsv"
    if elec_file.exists():
        elec_df = pd.read_csv(elec_file, sep="\t")
        ch_pos = {
            row["name"]: np.array([row["x"], row["y"], row["z"]], dtype=float)
            for _, row in elec_df.iterrows()
            if row["name"] in ch_names
            and pd.notna(row["x"])   # skip channels with missing positions (e.g. VREF)
        }
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos,
                                                coord_frame="head")
        raw.set_montage(montage, on_missing="warn", verbose=verbose)

    return raw
