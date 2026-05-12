import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from utils.create_preprocessed_data import load_h5
from statsmodels.graphics.tsaplots import plot_ccf
from statsmodels.tsa.stattools import ccf as compute_ccf
from statsmodels.stats.multitest import multipletests
from scipy.signal import butter, sosfilt, hilbert
from scipy.stats import mannwhitneyu
from config import GROUP


def bandpass(signal, band, sfreq, order=5):
    # Create the filter coefficients
    sos = butter(order, band, btype='band', fs=sfreq, output='sos')
    # Apply the filter to the data
    y = sosfilt(sos, signal)
    return y

def plot_walk_band_ccf(
                  bandname, 
                  data_HC = None,
                  data_PD = None,
                  comparisons = [('Fz','Pz'), ('C1','C2'), ('C3','C4'), ('CP1', 'CP2')]):
    if data_HC is None:
        data_HC = load_h5(f'./data_preprocessed/sub-002_walk.h5') # example HC subject
    if data_PD is None:
        data_PD = load_h5(f'./data_preprocessed/sub-047_walk.h5') # matched PD subject

    fig, axs = plt.subplots(1, len(comparisons), figsize = (23, 5))
    for idx, comparison in enumerate(comparisons):
        #if 'Fz' in comparison:
        #   bandname, band = 'alpha', (8, 12)
        #else:
        band = GROUP[bandname]['band']
        idx_1, idx_2 = np.where(np.isin(data_HC.channel, comparison))[0]

        HC1 = bandpass(signal = data_HC.values[idx_1],
                        band = band,
                        sfreq = data_HC.sfreq
                        )

        HC2 = bandpass(signal = data_HC.values[idx_2],
                        band = band,
                        sfreq = data_HC.sfreq
                        )

        PD1 = bandpass(signal = data_PD.values[idx_1],
                        band = band,
                        sfreq = data_PD.sfreq
                        )

        PD2 = bandpass(signal = data_PD.values[idx_2],
                        band = band,
                        sfreq = data_PD.sfreq
                        )
        
        plot_ccf(HC1, HC2, lags=20, ax=axs[idx], label = 'HC', use_vlines = False)
        plot_ccf(PD1, PD2, lags=20, ax=axs[idx], label = 'PD', use_vlines = False)
        axs[idx].legend()
        axs[idx].set_title(f'{comparison[0]} vs {comparison[1]} {bandname} band cross-correlation')
        axs[idx].set_xlabel('lags (s)')
        axs[idx].set_ylabel('CCF')
    plt.show()
    
def test_group_ccf(
    bandname,
    task='walk',
    lags=20,
    comparisons=[('Fz', 'Pz'), ('C1', 'C2'), ('C3', 'C4'), ('CP1', 'CP2')],
    alpha=0.05,
    data_dir='./data_preprocessed',
    participants_csv='./results/matched_df.csv',
):
    """Compare CCF between HC and PD groups across all matched subjects.

    The CCF is a function of lag, so a pointwise Mann-Whitney U test is run at
    every lag, followed by Benjamini-Hochberg FDR correction across lags.  A
    lag-agnostic summary (peak absolute CCF per subject) is tested separately
    with a second Mann-Whitney U to give a single overall effect.

    Parameters
    ----------
    bandname : str
        One of 'theta', 'alpha', 'beta'.
    task : str
        Recording condition; 'walk' or 'rest'.
    lags : int
        Number of positive lags to compute (lags 0 … lags).
    comparisons : list of 2-tuples
        Channel pairs to analyse.
    alpha : float
        FDR threshold.
    data_dir : str
        Directory containing the preprocessed .h5 files.
    participants_tsv : str
        Path to the participants.tsv file with a 'group' column.

    Returns
    -------
    dict
        Keyed by channel-pair tuple.  Each value contains:
        - 'hc_ccfs'          : (n_HC, lags+1) array of per-subject CCF curves
        - 'pd_ccfs'          : (n_PD, lags+1) array of per-subject CCF curves
        - 'lag_axis_s'       : lag values in seconds
        - 'pvals_raw'        : pointwise p-values (lags+1,)
        - 'pvals_fdr'        : FDR-corrected p-values (lags+1,)
        - 'significant_lags' : lag indices where pvals_fdr < alpha
        - 'peak_ccf_pval'    : p-value for the peak-|CCF| summary test
        - 'peak_ccf_hc'      : per-subject peak |CCF| for HC
        - 'peak_ccf_pd'      : per-subject peak |CCF| for PD
    """
    # ── 1. Resolve group membership from participants.tsv ─────────────────────
    participants = pd.read_csv(participants_csv)

    hc_ids = participants.loc[participants['group'] == 'HC', 'participant_id'].tolist()
    pd_ids = participants.loc[participants['group'] == 'PD', 'participant_id'].tolist()

    band = GROUP[bandname]['band']

    # ── 2. Collect per-subject CCF curves ─────────────────────────────────────
    def _subject_ccfs(subject_ids):
        curves = []
        used = []
        for sid in subject_ids:
            try:
                da = load_h5(f'{data_dir}/{sid}_{task}.h5')
                ch_idx = np.where(np.isin(da.channel.values, list(comparison)))[0]
                if len(ch_idx) < 2:
                    continue
                sfreq = da.sfreq
                sig1 = bandpass(da.values[ch_idx[0]], band, sfreq)
                sig2 = bandpass(da.values[ch_idx[1]], band, sfreq)
                # compute_ccf returns correlation at lags 0 … nlags
                curve = compute_ccf(sig1, sig2, nlags=lags, adjusted=False)
                curves.append(curve)
                used.append(sid)
            except Exception:
                pass
        return np.array(curves), used, sfreq  # sfreq from last successful load

    results = {}
    fig, axs = plt.subplots(2, len(comparisons), figsize=(23, 8),
                            gridspec_kw={'height_ratios': [3, 1]})

    for idx, comparison in enumerate(comparisons):
        hc_ccfs, hc_used, sfreq = _subject_ccfs(hc_ids)
        pd_ccfs, pd_used, _     = _subject_ccfs(pd_ids)

        if hc_ccfs.size == 0 or pd_ccfs.size == 0:
            print(f'  Skipping {comparison}: insufficient data for one or both groups.')
            continue

        lag_axis = np.arange(lags + 1)

        # ── 3. Pointwise Mann-Whitney U + FDR correction ──────────────────────
        pvals_raw = np.array([
            mannwhitneyu(hc_ccfs[:, k], pd_ccfs[:, k], alternative='two-sided').pvalue
            for k in range(lags + 1)
        ])
        _, pvals_fdr, _, _ = multipletests(pvals_raw, alpha=alpha, method='fdr_bh')
        sig_lags = np.where(pvals_fdr < alpha)[0]

        # ── 4. Lag-agnostic summary: peak |CCF| per subject ───────────────────
        peak_hc = np.max(np.abs(hc_ccfs), axis=1)
        peak_pd = np.max(np.abs(pd_ccfs), axis=1)
        _, peak_pval = mannwhitneyu(peak_hc, peak_pd, alternative='two-sided')

        # ── 5. Top row: mean ± SEM CCF per group ──────────────────────────────
        hc_mean = hc_ccfs.mean(axis=0)
        hc_sem  = hc_ccfs.std(axis=0, ddof=1) / np.sqrt(len(hc_ccfs))
        pd_mean = pd_ccfs.mean(axis=0)
        pd_sem  = pd_ccfs.std(axis=0, ddof=1) / np.sqrt(len(pd_ccfs))

        ax_ccf = axs[0, idx]
        ax_ccf.plot(lag_axis, hc_mean, color='royalblue', label=f'HC (n={len(hc_used)})')
        ax_ccf.fill_between(lag_axis, hc_mean - hc_sem, hc_mean + hc_sem,
                            alpha=0.25, color='royalblue')
        ax_ccf.plot(lag_axis, pd_mean, color='tomato', label=f'PD (n={len(pd_used)})')
        ax_ccf.fill_between(lag_axis, pd_mean - pd_sem, pd_mean + pd_sem,
                            alpha=0.25, color='tomato')
        ax_ccf.axhline(0, color='k', linewidth=0.5, linestyle='--')
        ax_ccf.set_title(f'{comparison[0]} vs {comparison[1]}\n{bandname} (peak |CCF| p={peak_pval:.3f})')
        ax_ccf.set_ylabel('Mean CCF ± SEM')
        ax_ccf.legend(fontsize=8)
        ax_ccf.set_xticklabels([])

        # ── 6. Bottom row: -log10(FDR p-value) per lag ────────────────────────
        log_p = -np.log10(np.clip(pvals_fdr, 1e-10, 1.0))
        threshold = -np.log10(alpha)

        ax_p = axs[1, idx]
        ax_p.bar(lag_axis, log_p, width=0.8, color='steelblue', alpha=0.7)
        ax_p.axhline(threshold, color='firebrick', linewidth=1.2, linestyle='--',
                     label=f'FDR α={alpha}')
        ax_p.set_xlabel('Lag (samples)')
        ax_p.set_ylabel('−log₁₀(p)')
        ax_p.legend(fontsize=7)

        results[comparison] = {
            'hc_ccfs':          hc_ccfs,
            'pd_ccfs':          pd_ccfs,
            'lag_axis':         lag_axis,
            'pvals_raw':        pvals_raw,
            'pvals_fdr':        pvals_fdr,
            'significant_lags': sig_lags,
            'peak_ccf_pval':    peak_pval,
            'peak_ccf_hc':      peak_hc,
            'peak_ccf_pd':      peak_pd,
        }

        n_sig = len(sig_lags)
        print(f'{comparison[0]}-{comparison[1]} [{bandname}]: '
              f'{n_sig}/{lags+1} lags significant after FDR, '
              f'peak-|CCF| p={peak_pval:.3f}  '
              f'(HC n={len(hc_used)}, PD n={len(pd_used)})')

    plt.suptitle(f'{bandname} band CCF — HC vs PD ({task})', fontsize=13)
    plt.tight_layout()
    plt.show()

    return results


if __name__ == '__main__':
    data_HC = load_h5(f'./data_preprocessed/sub-002_walk.h5')
    data_PD = load_h5(f'./data_preprocessed/sub-047_walk.h5')
    for bandname in ['theta','alpha','beta']:
        plot_walk_band_ccf(bandname = bandname,
                        data_HC=data_HC,
                        data_PD=data_PD,)