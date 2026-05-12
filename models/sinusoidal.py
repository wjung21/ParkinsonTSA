import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


def build_design_matrix(t: np.ndarray, f: float, n_harmonics: int = 1):
    cols = [np.ones(len(t))]
    for k in range(1, n_harmonics + 1):
        cols.append(np.cos(2 * np.pi * k * f * t))
        cols.append(np.sin(2 * np.pi * k * f * t))
    return np.column_stack(cols) # (n_samples, 1 + 2*K)


def train_sinusoidal_model(t_train, X_train, sfreq=250):
    """
    Fit a sinusoidal regression over a grid of frequencies to multiple time series.

    Parameters
    ----------
    t_train : np.ndarray, shape (n_samples,)
        Time vector for training.
    X_train : np.ndarray, shape (n_samples,) or (n_samples, n_series)
        One or more time series to fit.
    sfreq : float
        Sampling frequency in Hz (default 250).

    Returns
    -------
    best_freq : float
        Frequency (Hz) that minimises total RSS across all series.
    coefficients : np.ndarray, shape (n_series, 3)
        OLS coefficients [intercept, cos_coef, sin_coef] for each series
        at the best frequency.
    rss_per_freq : np.ndarray, shape (n_freqs,)
        Total RSS at each candidate frequency (summed across series).
    f_grid : np.ndarray, shape (n_freqs,)
        The frequency grid used.
    """
    # ── Normalise input to 2-D (n_samples, n_series) ──────────────────────────
    X_train = np.atleast_2d(X_train)
    if X_train.shape[0] == 1:          # single row → probably transposed
        X_train = X_train.T
    n_series, n_samples = X_train.shape

    # ── Frequency grid up to the Nyquist limit ─────────────────────────────────
    f_grid = np.linspace(0, sfreq / 2, n_samples // 2)
    n_freqs = len(f_grid)

    # Accumulate total (summed) RSS across all series for every frequency
    rss_per_freq = np.zeros(n_freqs)

    for idx, f_hat in enumerate(f_grid):
        # Design matrix: [1, cos(2πft), sin(2πft)]  shape: (n_samples, 3)
        X_cos = np.cos(2 * np.pi * f_hat * t_train)
        X_sin = np.sin(2 * np.pi * f_hat * t_train)
        design = np.column_stack([np.ones(n_samples), X_cos, X_sin])

        # Fit every series and add its RSS to the running total
        for s in range(n_series):
            model = sm.OLS(X_train[s], design).fit()
            rss_per_freq[idx] += np.sum(model.resid ** 2)

    # ── Identify the best frequency ────────────────────────────────────────────
    best_idx  = np.argmin(rss_per_freq)
    best_freq = f_grid[best_idx]

    # ── Re-fit at the best frequency to recover per-series coefficients ────────
    X_cos_best = np.cos(2 * np.pi * best_freq * t_train)
    X_sin_best = np.sin(2 * np.pi * best_freq * t_train)
    design_best = np.column_stack([np.ones(n_samples), X_cos_best, X_sin_best])

    coefficients = np.zeros((n_series, 3))   # [intercept, cos_coef, sin_coef]
    for s in range(n_series):
        model = sm.OLS(X_train[s], design_best).fit()
        coefficients[s] = model.params

    # ── Diagnostic plot ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: RSS landscape
    axes[0].plot(f_grid, rss_per_freq, linewidth=1.2)
    axes[0].axvline(best_freq, color="red", linestyle="--",
                    label=f"Best f = {best_freq:.2f} Hz")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Total RSS (summed across series)")
    axes[0].set_title("RSS vs. Candidate Frequency")
    axes[0].legend()

    # Right: fitted sinusoids overlaid on each series
    t_fine = np.linspace(t_train[0], t_train[-1], 500)
    colors = plt.cm.tab10(np.linspace(0, 1, n_series))

    for s in range(n_series):
        a0, a1, a2 = coefficients[s]
        y_fit = a0 + a1 * np.cos(2 * np.pi * best_freq * t_fine) \
                   + a2 * np.sin(2 * np.pi * best_freq * t_fine)
        axes[1].plot(t_train, X_train[:, s], alpha=0.4,
                     color=colors[s], label=f"Series {s+1} (data)")
        axes[1].plot(t_fine, y_fit, color=colors[s], linewidth=2,
                     linestyle="--", label=f"Series {s+1} (fit)")

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title(f"Fitted Sinusoids at f = {best_freq:.2f} Hz")
    axes[1].legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.show()

    print(f"Best frequency : {best_freq:.4f} Hz")
    print(f"Coefficients   : [intercept, cos_coef, sin_coef] per series")
    for s in range(n_series):
        print(f"  Series {s+1:>2d}: {coefficients[s]}")

    return best_freq, coefficients, rss_per_freq, f_grid

def train_sinusoidal_model_multi(
    t_train: np.ndarray,
    X_train: np.ndarray,
    sfreq: float = 250,
    n_harmonics: int = 1,
    ):
    """
    Fit a multi-harmonic sinusoidal regression to multiple time series by
    scanning a frequency grid and picking the fundamental frequency that
    minimises total RSS across all series.
    """

    n_params = 1 + 2 * n_harmonics # intercept + cos/sin per harmonic

    n_series, n_samples = X_train.shape
    f_grid = np.linspace(0, sfreq / 2, n_samples // 2) # frequency grid up to Nyquist frequency
    n_freqs = len(f_grid)

    rss_per_freq = np.zeros(n_freqs)

    for idx, f_hat in enumerate(f_grid):
        design = build_design_matrix(t_train, f_hat, n_harmonics)  # (n_samples, 1+2K)
        for s in range(n_series):
            model = sm.OLS(X_train[s], design).fit()
            rss_per_freq[idx] += np.sum(model.resid ** 2)

    best_idx  = np.argmin(rss_per_freq)
    best_freq = f_grid[best_idx]

    design_best  = build_design_matrix(t_train, best_freq, n_harmonics)
    coefficients = np.zeros((n_series, n_params))

    for s in range(n_series):
        model = sm.OLS(X_train[s], design_best).fit()
        coefficients[s] = model.params        # [intercept, cos_1, sin_1, …, cos_K, sin_K]

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, n_series))

    # Left — RSS landscape with harmonic tick marks
    axes[0].plot(f_grid, rss_per_freq, linewidth=1.2, color="steelblue")
    axes[0].axvline(best_freq, color="red", linestyle="--",
                    label=f"Best f₀ = {best_freq:.3f} Hz")
    for k in range(1, n_harmonics + 1):
        axes[0].axvline(k * best_freq, color="orange", linestyle=":",
                        linewidth=0.8,
                        label=f"Harmonic {k}" if k == n_harmonics else f"Harmonic {k}")
    axes[0].set_xlabel("Fundamental Frequency (Hz)")
    axes[0].set_ylabel("Total RSS (summed across series)")
    axes[0].set_title(f"RSS vs Frequency  (K = {n_harmonics} harmonic{'s' if n_harmonics > 1 else ''})")
    axes[0].legend(fontsize=8)

    # Right — fitted waveforms overlaid on raw data
    t_fine = np.linspace(t_train[0], t_train[-1], 1000)
    design_fine = build_design_matrix(t_fine, best_freq, n_harmonics)

    for s in range(n_series):
        y_fit = design_fine @ coefficients[s]          # (1000,)
        axes[1].plot(t_train, X_train[s], alpha=0.35, color=colors[s])
        axes[1].plot(t_fine,  y_fit,      color=colors[s], linewidth=2,
                     linestyle="--", label=f"Series {s + 1}")

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title(f"Multi-harmonic fit  f₀ = {best_freq:.3f} Hz,  K = {n_harmonics}")
    axes[1].legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.show()

    # ── Summary ────────────────────────────────────────────────────────────────
    col_names = ["intercept"] + [f"{fn}(2π·{k}·f·t)"
                                  for k in range(1, n_harmonics + 1)
                                  for fn in ("cos", "sin")]
    print(f"\nBest fundamental frequency : {best_freq:.4f} Hz")
    print(f"Harmonics included         : {n_harmonics}  →  "
          f"{[round(k * best_freq, 3) for k in range(1, n_harmonics + 1)]} Hz")
    print(f"Design-matrix columns      : {col_names}")
    print(f"\nCoefficients  (n_series={n_series}, n_params={n_params})")
    header = f"{'Series':>8} | " + " | ".join(f"{c:>14}" for c in col_names)
    print(header)
    print("-" * len(header))
    for s in range(n_series):
        row = f"{s + 1:>8} | " + " | ".join(f"{v:>14.4f}" for v in coefficients[s])
        print(row)

    return best_freq, coefficients, rss_per_freq, f_grid


# ── Demo ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sfreq       = 250
    t_train     = np.arange(0, 2, 1 / sfreq)   # 2-second window, 500 samples
    rng         = np.random.default_rng(0)
    true_freq   = 8.0                           # Hz
    n_harmonics = 3

    # Build three series: each shares the same harmonics but different weights
    def make_series(amps, phases, noise_std):
        """amps & phases: list of length n_harmonics"""
        x = np.zeros(len(t_train))
        for k, (a, phi) in enumerate(zip(amps, phases), start=1):
            x += a * np.sin(2 * np.pi * k * true_freq * t_train + phi)
        return x + noise_std * rng.standard_normal(len(t_train))

    series = np.vstack([
        make_series([1.0, 0.5, 0.25], [0,          np.pi/4, np.pi/3], 0.3),
        make_series([0.8, 0.3, 0.10], [np.pi/2,    np.pi/6, 0      ], 0.2),
        make_series([1.2, 0.6, 0.40], [np.pi/3,    np.pi/2, np.pi/5], 0.4),
    ])   # shape: (3, 500)

    best_freq, coefs, rss, f_grid = train_sinusoidal_model_multi(
        t_train, series, sfreq=sfreq, n_harmonics=n_harmonics
    )