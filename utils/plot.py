"""
plot.py — EEG visualisation utilities
======================================
Provides :func:`plot_eeg`, a single-function API for rendering multi-channel
EEG recordings stored as ``xr.DataArray`` in the classic stacked-trace layout.

Typical usage
-------------
>>> from utils.plot import plot_eeg
>>> from utils.create_preprocessed_data import load_h5
>>>
>>> da = load_h5("data_preprocessed/sub-002_rest.h5")
>>> fig, ax = plot_eeg(da, title="sub-002  |  rest")
>>> fig.savefig("example_eeg.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_eeg(
    da: xr.DataArray,
    *,
    t_start: float = 0.0,
    t_end: float = 10.0,
    channels: Sequence[str] | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    offset_uv: float | None = None,
    linewidth: float = 0.4,
    color: str = "steelblue",
    figsize: tuple[float, float] = (7, 10),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot EEG channel traces in the classic stacked layout.

    Each channel is shown as a separate horizontal trace, offset vertically so
    that channels do not overlap.  The top of the panel is the first channel in
    ``da`` (or the first entry in ``channels``); the bottom is the last.

    Parameters
    ----------
    da : xr.DataArray
        Array with dims ``(channel, time)`` as returned by
        :func:`~utils.create_preprocessed_data.load_h5`.
    t_start : float, default ``0.0``
        Start of the display window in seconds.
    t_end : float, default ``10.0``
        End of the display window in seconds.
    channels : sequence of str | None
        Subset of channel names to display.  ``None`` displays all channels in
        the order they appear in ``da``.
    title : str | None
        Axes title.  Auto-generated from ``da.attrs`` (``participant_id`` and
        ``task``) when ``None``.
    ax : plt.Axes | None
        Existing axes to draw into.  A new figure is created when ``None``.
    offset_uv : float | None
        Vertical spacing between channel baselines in µV.  Computed
        automatically from the data amplitude when ``None`` (6× the median
        per-channel RMS over the selected window).
    linewidth : float, default ``0.4``
        Line width for each channel trace.
    color : str, default ``"steelblue"``
        Line colour for all channel traces.
    figsize : tuple[float, float], default ``(7, 10)``
        Figure size in inches.  Only used when ``ax`` is ``None``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes

    Examples
    --------
    Plot a single recording:

    >>> fig, ax = plot_eeg(da_rest, title="Resting state")
    >>> fig.savefig("rest.png", dpi=150, bbox_inches="tight")

    Side-by-side rest vs walk comparison:

    >>> fig, axes = plt.subplots(1, 2, figsize=(14, 10), sharey=True)
    >>> plot_eeg(da_rest, title="Rest",  ax=axes[0])
    >>> plot_eeg(da_walk, title="Walk",  ax=axes[1])
    >>> fig.savefig("comparison.png", dpi=150, bbox_inches="tight")
    """
    # ── Select time window ────────────────────────────────────────────
    segment = da.sel(time=slice(t_start, t_end))

    if channels is not None:
        segment = segment.sel(channel=list(channels))

    ch_names: list[str] = segment.coords["channel"].values.tolist()
    times: np.ndarray   = segment.coords["time"].values
    data:  np.ndarray   = segment.values          # (n_ch, n_times)
    n_ch = len(ch_names)

    # ── Auto-compute vertical offset ──────────────────────────────────
    if offset_uv is None:
        rms_per_ch = np.sqrt(np.mean(data ** 2, axis=1))
        offset_uv = float(np.median(rms_per_ch)) * 6.0

    # ── Create figure / axes ─────────────────────────────────────────
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # ── Draw traces (top channel = highest y position) ────────────────
    for i, (ch, trace) in enumerate(zip(ch_names, data)):
        baseline = (n_ch - 1 - i) * offset_uv
        ax.plot(times, trace + baseline,
                linewidth=linewidth, color=color, rasterized=True)

    # ── Y-axis: channel labels ────────────────────────────────────────
    yticks = [(n_ch - 1 - i) * offset_uv for i in range(n_ch)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ch_names, fontsize=6)
    ax.tick_params(axis="y", length=0)

    # ── X-axis ────────────────────────────────────────────────────────
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel("Time (s)", fontsize=10)

    # ── Title ────────────────────────────────────────────────────────
    if title is None:
        pid  = da.attrs.get("participant_id", "")
        task = da.attrs.get("task", "")
        title = f"{pid}  |  {task}" if pid and task else (task or pid or "EEG")
    ax.set_title(title, fontsize=20, fontweight="bold", pad=6)

    # ── Styling ───────────────────────────────────────────────────────
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    if own_fig:
        fig.tight_layout()

    return fig, ax
