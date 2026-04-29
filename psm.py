"""
psm.py — Propensity Score Matching (PSM)
=========================================
Matches a majority group to a minority group using 1:N nearest-neighbour
matching on estimated propensity scores from logistic regression.

Typical usage
-------------
>>> import pandas as pd
>>> from psm import PropensityScoreMatching
>>>
>>> df = pd.read_csv("data/ds007526/participants.tsv", sep="\t", na_values="n/a")
>>> psm = PropensityScoreMatching(
...     target_col="group",
...     features=["age", "sex", "moca"],
... )
>>> matched_df = psm.fit_match(df)
>>> psm.summary()
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler


class PropensityScoreMatching:
    """Nearest-neighbour propensity score matching.

    Parameters
    ----------
    target_col : str
        Name of the binary group column (e.g. ``"group"``).  The minority
        class is treated as the *reference* group; majority-class rows are
        down-sampled to match it.
    features : list[str]
        Column names used to estimate propensity scores (e.g.
        ``["age", "sex", "moca"]``).  Categorical (non-numeric) columns are
        label-encoded automatically.
    ratio : int, default 1
        Number of majority-group matches to draw per minority-group subject
        (1 → 1:1 matching, 2 → 1:2, …).
    caliper : float | None, default None
        Maximum allowed difference in propensity score between a matched
        pair.  Expressed as a fraction of the propensity score standard
        deviation (e.g. ``0.2`` is the commonly used rule-of-thumb).
        ``None`` disables caliper trimming.
    replacement : bool, default False
        If ``True``, majority-group subjects may be reused across matches.
        If ``False``, each majority-group subject is used at most once.
    missing : {"drop", "raise"}, default "drop"
        How to handle rows with missing values in *features*:
        ``"drop"`` silently removes them before matching;
        ``"raise"`` raises a ``ValueError`` instead.
    random_state : int, default 42
        Seed for reproducibility.
    """

    def __init__(
        self,
        target_col: str,
        features: list[str],
        *,
        ratio: int = 1,
        caliper: float | None = None,
        replacement: bool = False,
        missing: Literal["drop", "raise"] = "drop",
        random_state: int = 42,
    ) -> None:
        self.target_col = target_col
        self.features = list(features)
        self.ratio = ratio
        self.caliper = caliper
        self.replacement = replacement
        self.missing = missing
        self.random_state = random_state

        # Populated by fit()
        self._minority_label: object = None
        self._majority_label: object = None
        self._propensity_scores: np.ndarray | None = None
        self._working_df: pd.DataFrame | None = None  # cleaned copy used for matching
        self._matched_df: pd.DataFrame | None = None
        self._encoders: dict[str, LabelEncoder] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "PropensityScoreMatching":
        """Estimate propensity scores from *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing ``target_col`` and all ``features``.

        Returns
        -------
        self
        """
        required_cols = [self.target_col] + self.features
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in dataframe: {missing_cols}")

        work = df[required_cols].copy()

        # ── Handle missing values ──────────────────────────────────────
        n_before = len(work)
        has_na = work[self.features].isna().any(axis=1)
        if has_na.any():
            if self.missing == "raise":
                raise ValueError(
                    f"{has_na.sum()} row(s) have missing values in features. "
                    "Set missing='drop' to remove them automatically."
                )
            work = work[~has_na].copy()
            warnings.warn(
                f"Dropped {has_na.sum()} row(s) with missing feature values "
                f"({n_before} → {len(work)} rows).",
                UserWarning,
                stacklevel=2,
            )

        # ── Identify minority / majority labels ───────────────────────
        counts = work[self.target_col].value_counts()
        if len(counts) != 2:
            raise ValueError(
                f"target_col '{self.target_col}' must have exactly 2 unique "
                f"values; found: {counts.index.tolist()}"
            )
        self._minority_label = counts.index[-1]   # smaller group
        self._majority_label = counts.index[0]    # larger group

        # ── Encode features ───────────────────────────────────────────
        X = self._encode_features(work, fit=True)

        # ── Scale + fit logistic regression ───────────────────────────
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            solver="lbfgs",
        )
        # Target: 1 = minority label
        y = (work[self.target_col] == self._minority_label).astype(int)
        clf.fit(X_scaled, y)

        self._propensity_scores = clf.predict_proba(X_scaled)[:, 1]
        self._working_df = work.copy()
        self._working_df["_pscore"] = self._propensity_scores

        return self

    def match(self) -> pd.DataFrame:
        """Perform nearest-neighbour matching and return the matched subset.

        Must be called after :meth:`fit`.

        Returns
        -------
        pd.DataFrame
            Subset of the *original* rows (all original columns preserved)
            with an added ``_pscore`` column and a ``_match_id`` column that
            links each minority subject to its matched majority subject(s).
        """
        if self._working_df is None:
            raise RuntimeError("Call fit() before match().")

        minority_df = self._working_df[
            self._working_df[self.target_col] == self._minority_label
        ].copy()
        majority_df = self._working_df[
            self._working_df[self.target_col] == self._majority_label
        ].copy()

        minority_scores = minority_df["_pscore"].values.reshape(-1, 1)
        majority_scores = majority_df["_pscore"].values.reshape(-1, 1)

        n_neighbors = min(self.ratio, len(majority_df))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nn.fit(majority_scores)
        distances, indices = nn.kneighbors(minority_scores)

        # ── Caliper trimming ──────────────────────────────────────────
        if self.caliper is not None:
            ps_std = self._working_df["_pscore"].std()
            threshold = self.caliper * ps_std
        else:
            threshold = np.inf

        minority_rows = []
        majority_rows = []
        used_majority = set()
        match_id = 0

        majority_index = majority_df.index.tolist()

        for i, (dists_i, idxs_i) in enumerate(zip(distances, indices)):
            matched_minority = minority_df.iloc[[i]].copy()
            matched_minority["_match_id"] = match_id

            matched_any = False
            for dist, j in zip(dists_i, idxs_i):
                if dist > threshold:
                    continue
                if not self.replacement and j in used_majority:
                    continue

                matched_majority = majority_df.iloc[[j]].copy()
                matched_majority["_match_id"] = match_id
                majority_rows.append(matched_majority)

                if not self.replacement:
                    used_majority.add(j)
                matched_any = True

            if matched_any:
                minority_rows.append(matched_minority)
                match_id += 1

        if not minority_rows:
            raise RuntimeError(
                "No matches found. Consider relaxing the caliper or enabling replacement."
            )

        self._matched_df = pd.concat(minority_rows + majority_rows).sort_values(
            ["_match_id", self.target_col]
        )

        n_unmatched = len(minority_df) - len(minority_rows)
        if n_unmatched > 0:
            warnings.warn(
                f"{n_unmatched} minority subject(s) could not be matched "
                "(caliper too strict or insufficient majority subjects).",
                UserWarning,
                stacklevel=2,
            )

        return self._matched_df

    def fit_match(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method: calls :meth:`fit` then :meth:`match`.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.

        Returns
        -------
        pd.DataFrame
            Matched subset (see :meth:`match`).
        """
        return self.fit(df).match()

    def summary(self) -> None:
        """Print covariate balance statistics before and after matching.

        Reports standardized mean difference (SMD) for continuous features
        and proportion difference for binary/categorical features.  An
        SMD < 0.1 is the conventional threshold for acceptable balance.
        """
        if self._matched_df is None or self._working_df is None:
            raise RuntimeError("Call fit_match() (or fit() + match()) first.")

        min_lbl = self._minority_label
        maj_lbl = self._majority_label

        print("=" * 65)
        print(f"  Propensity Score Matching Summary")
        print(f"  Reference (minority) : {min_lbl}")
        print(f"  Matched  (majority)  : {maj_lbl}")
        print("=" * 65)

        # ── Sample sizes ──────────────────────────────────────────────
        before_min = (self._working_df[self.target_col] == min_lbl).sum()
        before_maj = (self._working_df[self.target_col] == maj_lbl).sum()
        after_min  = (self._matched_df[self.target_col] == min_lbl).sum()
        after_maj  = (self._matched_df[self.target_col] == maj_lbl).sum()

        print(f"\n  {'':25s} {'Before':>10s}   {'After':>10s}")
        print(f"  {'-'*50}")
        print(f"  {min_lbl:<25s} {before_min:>10d}   {after_min:>10d}")
        print(f"  {maj_lbl:<25s} {before_maj:>10d}   {after_maj:>10d}")

        # ── Covariate balance ─────────────────────────────────────────
        print(f"\n  Covariate Balance (SMD < 0.1 = well-balanced)")
        print(f"  {'Feature':<20s} {'SMD Before':>12s}  {'SMD After':>10s}  {'p Before':>10s}  {'p After':>10s}")
        print(f"  {'-'*68}")

        for feat in self.features:
            smd_b, p_b = self._balance_stat(self._working_df, feat)
            smd_a, p_a = self._balance_stat(self._matched_df, feat)
            flag = " ✓" if smd_a < 0.1 else " ✗"
            print(
                f"  {feat:<20s} {smd_b:>12.4f}  {smd_a:>10.4f}  {p_b:>10.4f}  {p_a:>10.4f}{flag}"
            )

        print("=" * 65)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_features(self, df: pd.DataFrame, *, fit: bool) -> np.ndarray:
        """Label-encode categorical columns; return numeric array."""
        cols = []
        for feat in self.features:
            col = df[feat]
            if pd.api.types.is_numeric_dtype(col):
                cols.append(col.values.astype(float))
            else:
                if fit:
                    le = LabelEncoder()
                    encoded = le.fit_transform(col.astype(str))
                    self._encoders[feat] = le
                else:
                    le = self._encoders[feat]
                    encoded = le.transform(col.astype(str))
                cols.append(encoded.astype(float))
        return np.column_stack(cols)

    def _balance_stat(
        self, df: pd.DataFrame, feature: str
    ) -> tuple[float, float]:
        """Return (SMD, p-value) for *feature* between the two groups."""
        grp_a = df[df[self.target_col] == self._minority_label][feature]
        grp_b = df[df[self.target_col] == self._majority_label][feature]

        # Drop NaN for stat tests
        grp_a = grp_a.dropna()
        grp_b = grp_b.dropna()

        if pd.api.types.is_numeric_dtype(df[feature]):
            mean_a, std_a = grp_a.mean(), grp_a.std()
            mean_b, std_b = grp_b.mean(), grp_b.std()
            pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
            smd = abs(mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0
            _, p = ttest_ind(grp_a, grp_b, equal_var=False)
        else:
            # Categorical: SMD via proportion of most-common level
            all_levels = pd.concat([grp_a, grp_b]).unique()
            smds = []
            for lvl in all_levels:
                p1 = (grp_a == lvl).mean()
                p2 = (grp_b == lvl).mean()
                denom = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
                smds.append(abs(p1 - p2) / denom if denom > 0 else 0.0)
            smd = float(np.mean(smds))
            values = pd.concat([grp_a, grp_b]).reset_index(drop=True)
            labels = pd.Series(
                ["minority"] * len(grp_a) + ["majority"] * len(grp_b)
            )
            contingency = pd.crosstab(values, labels)
            _, p, _, _ = chi2_contingency(contingency)

        return smd, float(p)
