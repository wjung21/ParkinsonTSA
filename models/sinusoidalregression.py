import numpy as np
import pandas as pd
from numpy.fft import rfft, rfftfreq
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import statsmodels.api as sm

warnings.filterwarnings("ignore")

FS          = 250          # Hz
N_SAMPLES   = 1250         # samples per series
N_SUBJECTS  = 19           # total subjects
N_SPLITS    = 5            # CV folds
T           = np.arange(N_SAMPLES) / FS   # time axis in seconds

def build_feature_matrix(t: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Build the sin/cos design matrix for a set of frequencies with intercept.
    
    Parameters:
        t     : (n_subjects * n_segments * n_samples,) time axis
        freqs : (n_components,) frequencies in Hz
    
    Returns:
        X : (n_subjects * n_segments * n_samples, 1 + 2 * n_components) feature matrix
    """
    cols = [np.ones(len(t))] # intercept term
    for f in freqs:
        cols.append(np.sin(2 * np.pi * f * t))
        cols.append(np.cos(2 * np.pi * f * t))
    return t, np.column_stack(cols)

def estimate_top_frequencies(t: np.ndarray, 
                             X: np.ndarray,
                             n_sinusoids: int = 3,
                             n_samples: int = 1250,
                             sfreq: int = 250) -> np.ndarray:
    """
    Estimate the top-n frequencies.
    
    Parameters:
        t = (n_subjects * n_segments * n_samples,) flattened time axis.
        X = (n_subjects * n_segments * n_samples,) flattened time series to analyse
        n_sinusoids = number of top frequencies to estimate
        sfreq = sampling frequency in Hz
    
    Returns:
        best_f = (n_sinusoids,) array of top frequencies in Hz
    """
    f_grid = np.linspace(0, sfreq / 2, n_samples // 2) # input is already band-passed filtered so we can limit to the band range instead of the full Nyquist range, but I just wanted to keep it simple for now
    n_freqs = len(f_grid)

    # Accumulate total (summed) RSS across all series for every frequency
    rss_per_freq = np.zeros(n_freqs)

    for idx, f_hat in enumerate(f_grid):
        # Design matrix: [1, cos(2πft), sin(2πft)]  shape: (n_samples, 3)
        X_cos = np.cos(2 * np.pi * f_hat * t)
        X_sin = np.sin(2 * np.pi * f_hat * t)
        design = np.column_stack([np.ones(len(t)), X_cos, X_sin])

        # Fit every series and add its RSS to the running total
        model = sm.OLS(X, design).fit()
        rss_per_freq[idx] = np.sum(model.resid ** 2)

    # ── Identify the best frequency ────────────────────────────────────────────
    best_f = f_grid[np.argsort(rss_per_freq)][:n_sinusoids]  # frequency with lowest RSS
    
    return best_f

# ══════════════════════════════════════════════════════════════════════════════
# 2.  MODEL CLASS
# ══════════════════════════════════════════════════════════════════════════════

class MultiSinusoidalRegressor:
    """
    Linearised multi-component sinusoidal regressor.
    1. Find best frequencies on pooled training data (fixed across series)
    2. Fit a Ridge regression model on the pooled data to get global weights

    Parameters:
        n_components : number of sinusoidal components
        fs           : sampling frequency in Hz
        ridge_alpha  : Ridge regularisation strength (for the global model)
        refit_test   : if True, re-fit weights on each test series individually
    """

    def __init__(
        self,
        n_components: int  = 3,
        sfreq: float = 250,
        ridge_alpha: float = 1e-4,
        refit_test:   bool  = True,
    ):
        self.n_components = n_components
        self.sfreq        = sfreq
        self.ridge_alpha  = ridge_alpha
        self.refit_test   = refit_test

        # Set after fit()
        self.freqs_        = None   # (n_components,) selected Hz
        self.global_model_ = None   # Ridge fitted on pooled training data

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, t: np.ndarray, Y: np.ndarray) -> "MultiSinusoidalRegressor":
        """
        Parameters
        ----------
        t : (n_samples,)         shared time axis
        Y : (n_series, n_samples) training time series
        """
        # Step 1 — frequency selection
        self.freqs_ = estimate_top_frequencies(Y, self.n_components, self.fs)

        # Step 2 — build pooled feature matrix
        X_base = build_feature_matrix(t, self.freqs_)   # (n_samples, 2n)
        X_all  = np.tile(X_base, (len(Y), 1))           # (n_series*n_samples, 2n)
        y_all  = Y.ravel()                               # (n_series*n_samples,)

        # Step 3 — fit global model
        self.global_model_ = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        self.global_model_.fit(X_all, y_all)

        return self

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, t: np.ndarray, Y_test: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        t      : (n_samples,)          time axis
        Y_test : (n_series, n_samples) test series (used only when refit_test=True)

        Returns
        -------
        Y_pred : (n_series, n_samples) predictions
        """
        X_base = build_feature_matrix(t, self.freqs_)
        n_test = len(Y_test)

        if not self.refit_test:
            # Global model — same prediction for every series
            y_hat = self.global_model_.predict(X_base)
            return np.tile(y_hat, (n_test, 1))

        # Series-specific refit  (frequencies fixed; weights free)
        Y_pred = np.zeros_like(Y_test)
        for i, y in enumerate(Y_test):
            local_model = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
            local_model.fit(X_base, y)
            Y_pred[i] = local_model.predict(X_base)

        return Y_pred

    # ── Convenience ───────────────────────────────────────────────────────────
    def get_components(self, t: np.ndarray, y_series: np.ndarray):
        """
        Decompose a single series into individual sinusoidal components.

        Returns
        -------
        components : dict  {frequency_hz: (amplitude, phase_deg, waveform)}
        """
        X_base      = build_feature_matrix(t, self.freqs_)
        local_model = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        local_model.fit(X_base, y_series)

        components = {}
        coef       = local_model.coef_

        for k, f in enumerate(self.freqs_):
            A_k       = coef[2 * k]       # sin coefficient
            B_k       = coef[2 * k + 1]   # cos coefficient
            amplitude = np.sqrt(A_k**2 + B_k**2)
            phase_deg = np.degrees(np.arctan2(B_k, A_k))
            waveform  = A_k * np.sin(2 * np.pi * f * t) + B_k * np.cos(2 * np.pi * f * t)
            components[round(f, 4)] = {
                "amplitude": amplitude,
                "phase_deg": phase_deg,
                "waveform":  waveform,
            }

        return components


# ══════════════════════════════════════════════════════════════════════════════
# 3.  5-FOLD CROSS-SERIES CROSS VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def cross_validate_sinusoidal(
    Y:            np.ndarray,
    t:            np.ndarray   = T,
    n_components: int          = 3,
    n_splits:     int          = N_SPLITS,
    ridge_alpha:  float        = 1e-4,
    refit_test:   bool         = True,
    random_state: int          = 42,
) -> dict:
    """
    Cross-series 5-fold CV for MultiSinusoidalRegressor.

    The split unit is the SUBJECT (series), not individual time steps.
    Temporal order within each series is always preserved.

    Parameters
    ----------
    Y            : (n_subjects, n_samples) — one row per subject
    t            : (n_samples,) time axis
    n_components : sinusoidal components to use
    n_splits     : number of CV folds
    ridge_alpha  : Ridge regularisation
    refit_test   : whether to re-fit weights per test series
    random_state : KFold seed

    Returns
    -------
    results dict with per-fold and aggregate metrics
    """
    subject_ids = np.arange(len(Y))
    kf          = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(subject_ids)):
        Y_train = Y[train_idx]   # (n_train, n_samples)
        Y_test  = Y[test_idx]    # (n_test,  n_samples)

        # ── Fit ───────────────────────────────────────────────────────────────
        model = MultiSinusoidalRegressor(
            n_components=n_components,
            fs=FS,
            ridge_alpha=ridge_alpha,
            refit_test=refit_test,
        )
        model.fit(t, Y_train)

        # ── Predict ───────────────────────────────────────────────────────────
        Y_pred = model.predict(t, Y_test)

        # ── Metrics per test subject ──────────────────────────────────────────
        subject_metrics = []
        for i, subj_idx in enumerate(test_idx):
            rmse = np.sqrt(mean_squared_error(Y_test[i], Y_pred[i]))
            mae  = mean_absolute_error(Y_test[i], Y_pred[i])

            ss_res = np.sum((Y_test[i] - Y_pred[i]) ** 2)
            ss_tot = np.sum((Y_test[i] - Y_test[i].mean()) ** 2)
            r2     = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            subject_metrics.append({
                "subject_id": subj_idx,
                "rmse":       rmse,
                "mae":        mae,
                "r2":         r2,
            })

        fold_rmse = np.mean([m["rmse"] for m in subject_metrics])
        fold_mae  = np.mean([m["mae"]  for m in subject_metrics])
        fold_r2   = np.mean([m["r2"]   for m in subject_metrics])

        fold_results.append({
            "fold":             fold + 1,
            "train_subjects":   train_idx.tolist(),
            "test_subjects":    test_idx.tolist(),
            "selected_freqs_hz": model.freqs_.tolist(),
            "fold_rmse":        fold_rmse,
            "fold_mae":         fold_mae,
            "fold_r2":          fold_r2,
            "subject_metrics":  subject_metrics,
            "model":            model,          # keep for inspection / plotting
            "Y_test":           Y_test,
            "Y_pred":           Y_pred,
        })

        print(
            f"  Fold {fold+1}/{n_splits} | "
            f"train={len(train_idx)} test={len(test_idx)} | "
            f"RMSE={fold_rmse:.4f}  MAE={fold_mae:.4f}  R²={fold_r2:.4f} | "
            f"freqs(Hz)={np.round(model.freqs_, 2).tolist()}"
        )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    all_rmse = [r["fold_rmse"] for r in fold_results]
    all_mae  = [r["fold_mae"]  for r in fold_results]
    all_r2   = [r["fold_r2"]   for r in fold_results]

    summary = {
        "n_components":   n_components,
        "refit_test":     refit_test,
        "fold_results":   fold_results,
        "mean_rmse":      np.mean(all_rmse),
        "std_rmse":       np.std(all_rmse),
        "mean_mae":       np.mean(all_mae),
        "std_mae":        np.std(all_mae),
        "mean_r2":        np.mean(all_r2),
        "std_r2":         np.std(all_r2),
    }

    print(
        f"\n  ── CV Summary (n_components={n_components}) ──\n"
        f"  RMSE : {summary['mean_rmse']:.4f} ± {summary['std_rmse']:.4f}\n"
        f"  MAE  : {summary['mean_mae']:.4f}  ± {summary['std_mae']:.4f}\n"
        f"  R²   : {summary['mean_r2']:.4f}  ± {summary['std_r2']:.4f}"
    )

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# 4.  COMPONENT SELECTION — compare different values of n
# ══════════════════════════════════════════════════════════════════════════════

def select_n_components(
    Y:                np.ndarray,
    t:                np.ndarray  = T,
    n_range:          range       = range(1, 11),
    n_splits:         int         = N_SPLITS,
    random_state:     int         = 42,
) -> pd.DataFrame:
    """
    Run CV for each value of n_components and return a comparison DataFrame.
    Useful for picking the optimal number of sinusoidal components.
    """
    rows = []
    for n in n_range:
        print(f"\n── n_components = {n} ──────────────────────────────")
        result = cross_validate_sinusoidal(
            Y, t,
            n_components=n,
            n_splits=n_splits,
            random_state=random_state,
        )
        rows.append({
            "n_components": n,
            "mean_rmse":    result["mean_rmse"],
            "std_rmse":     result["std_rmse"],
            "mean_r2":      result["mean_r2"],
            "std_r2":       result["std_r2"],
        })

    df = pd.DataFrame(rows).set_index("n_components")
    print("\n" + df.to_string())
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PLOTTING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def plot_fold_predictions(fold_result: dict, t: np.ndarray = T, max_subjects: int = 4):
    """Plot actual vs predicted for test subjects of one fold."""
    Y_test  = fold_result["Y_test"]
    Y_pred  = fold_result["Y_pred"]
    n_show  = min(max_subjects, len(Y_test))

    fig, axes = plt.subplots(n_show, 1, figsize=(12, 2.8 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i in range(n_show):
        subj_id = fold_result["test_subjects"][i]
        r2      = fold_result["subject_metrics"][i]["r2"]
        rmse    = fold_result["subject_metrics"][i]["rmse"]

        axes[i].plot(t, Y_test[i],  color="#2c7bb6", lw=1.2, label="Actual",    alpha=0.85)
        axes[i].plot(t, Y_pred[i],  color="#d7191c", lw=1.5, label="Predicted", linestyle="--")
        axes[i].set_ylabel("Amplitude")
        axes[i].set_title(f"Subject {subj_id}  |  R²={r2:.3f}  RMSE={rmse:.4f}")
        axes[i].legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fold_n = fold_result["fold"]
    freqs  = np.round(fold_result["selected_freqs_hz"], 2)
    fig.suptitle(f"Fold {fold_n} — Selected frequencies: {freqs} Hz", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"/mnt/user-data/outputs/fold_{fold_n}_predictions.png", dpi=120, bbox_inches="tight")
    plt.show()


def plot_component_decomposition(model: MultiSinusoidalRegressor, t: np.ndarray, y: np.ndarray, subject_id: int = 0):
    """Decompose a series into its sinusoidal components and plot each one."""
    components = model.get_components(t, y)
    n          = len(components)

    fig = plt.figure(figsize=(13, 2.5 * (n + 2)))
    gs  = gridspec.GridSpec(n + 2, 1, hspace=0.5)

    # Full signal
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, y, color="#555", lw=1.2, label="Original signal")
    ax0.set_title(f"Subject {subject_id} — Original signal")
    ax0.set_ylabel("Amplitude")

    # Reconstruction
    y_hat = sum(info["waveform"] for info in components.values())
    ax1   = fig.add_subplot(gs[1])
    ax1.plot(t, y,     color="#2c7bb6", lw=1.2, label="Actual",        alpha=0.8)
    ax1.plot(t, y_hat, color="#d7191c", lw=1.5, label="Reconstructed", linestyle="--")
    ax1.legend(fontsize=8)
    ax1.set_title("Reconstructed signal (sum of components)")
    ax1.set_ylabel("Amplitude")

    # Individual components
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    for k, (freq, info) in enumerate(components.items()):
        ax = fig.add_subplot(gs[k + 2])
        ax.plot(t, info["waveform"], color=colors[k], lw=1.5)
        ax.set_title(
            f"Component {k+1}: f={freq} Hz  |  "
            f"Amplitude={info['amplitude']:.4f}  Phase={info['phase_deg']:.1f}°"
        )
        ax.set_ylabel("Amplitude")

    fig.axes[-1].set_xlabel("Time (s)")
    plt.savefig(
        f"/mnt/user-data/outputs/components_subject_{subject_id}.png",
        dpi=120, bbox_inches="tight"
    )
    plt.show()


def plot_n_components_comparison(comparison_df: pd.DataFrame):
    """Plot RMSE and R² as a function of n_components."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    n = comparison_df.index

    axes[0].errorbar(n, comparison_df["mean_rmse"], yerr=comparison_df["std_rmse"],
                     marker="o", color="#2c7bb6", capsize=4, lw=1.8)
    axes[0].set_xlabel("n_components")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("CV RMSE vs Number of Components")
    axes[0].set_xticks(n)
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(n, comparison_df["mean_r2"], yerr=comparison_df["std_r2"],
                     marker="o", color="#d7191c", capsize=4, lw=1.8)
    axes[1].set_xlabel("n_components")
    axes[1].set_ylabel("R²")
    axes[1].set_title("CV R² vs Number of Components")
    axes[1].set_xticks(n)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/n_components_comparison.png", dpi=120, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  DEMO — synthetic data (replace Y with your real data)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # ── Generate synthetic multi-subject data ─────────────────────────────────
    # Each subject has the same dominant frequencies but different
    # amplitudes, phases, and noise levels — a realistic scenario.
    TRUE_FREQS = [2.0, 5.5, 10.0]           # Hz — ground truth
    Y = np.zeros((N_SUBJECTS, N_SAMPLES))

    for i in range(N_SUBJECTS):
        signal = np.zeros(N_SAMPLES)
        for f in TRUE_FREQS:
            A     = rng.uniform(0.5, 2.0)
            phi   = rng.uniform(0, 2 * np.pi)
            signal += A * np.sin(2 * np.pi * f * T + phi)
        noise     = rng.normal(0, 0.3, N_SAMPLES)
        Y[i]      = signal + noise

    # ── Single CV run with n=3 components ────────────────────────────────────
    print("=" * 65)
    print("5-Fold Cross-Series CV  |  n_components=3")
    print("=" * 65)
    cv_results = cross_validate_sinusoidal(Y, T, n_components=3)

    # ── Plot predictions for the first fold ──────────────────────────────────
    plot_fold_predictions(cv_results["fold_results"][0], T)

    # ── Decompose one test subject from fold 1 ───────────────────────────────
    fold1_model = cv_results["fold_results"][0]["model"]
    fold1_subj0 = cv_results["fold_results"][0]["test_subjects"][0]
    plot_component_decomposition(fold1_model, T, Y[fold1_subj0], subject_id=fold1_subj0)

    # ── Sweep n_components from 1 to 8 and compare ───────────────────────────
    print("\n" + "=" * 65)
    print("Sweeping n_components = 1 … 8")
    print("=" * 65)
    comp_df = select_n_components(Y, T, n_range=range(1, 9))
    plot_n_components_comparison(comp_df)