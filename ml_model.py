"""
ml_model.py — Machine Learning Pipeline for the AI Bear Risk Dashboard.

Architecture overview
---------------------
This file sits on top of the deterministic scoring layer (Prompt 2) and adds
three independent probabilistic models — one per prediction horizon:

    bear_3m  : >=20% S&P 500 drawdown within next 3 months
    bear_6m  : >=20% S&P 500 drawdown within next 6 months
    bear_12m : >=20% S&P 500 drawdown within next 12 months

For each horizon, the pipeline is:
    1. Feature engineering (raw indicators -> 60-80 features)
    2. Walk-forward training: XGBoost + RandomForest + LogReg -> stacked meta-LR
    3. SMOTE oversampling strictly inside each training fold
    4. Isotonic calibration of ensemble output probabilities
    5. Evaluation: PR-AUC, Brier Score, decade breakdown, lead-time analysis
    6. SHAP explainability for real-time and historical use

Critical design constraints enforced
-------------------------------------
- NO shuffled CV: TimeSeriesSplit expanding window only
- NO SMOTE before split: applied only inside each training fold
- NO ROC-AUC or accuracy as primary metrics on imbalanced data
- All random seeds set and logged
- Model artifacts saved with joblib (timestamp + metrics in filename)
- Bootstrap 95% CI on all reported metrics
- Explicit threshold warning if PR-AUC < 0.40 or Brier > 0.15

Dependencies beyond Prompt 3 requirements.txt
----------------------------------------------
    hmmlearn>=0.3.0        -- 3-state HMM for regime meta-feature
    imbalanced-learn>=0.12.0  -- SMOTE oversampling
Add both to requirements.txt before running.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# Optional heavy imports
try:
    from hmmlearn import hmm as _hmm  # type: ignore
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "hmmlearn not installed -- HMM regime feature disabled. "
        "pip install hmmlearn>=0.3.0"
    )

try:
    from imblearn.over_sampling import SMOTE  # type: ignore
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "imbalanced-learn not installed -- SMOTE disabled, class_weight used only. "
        "pip install imbalanced-learn>=0.12.0"
    )

log = logging.getLogger(__name__)

# ── Config import with graceful fallback ──────────────────────────────────────
# config.py is required for CACHE_DIR (where latest_prediction.json is saved).
# If config is unavailable (e.g. running ml_model.py in isolation), we fall
# back to a local .cache directory next to this file.
try:
    import config as config                          # noqa: PLC0414
    _CACHE_DIR = config.CACHE_DIR
except Exception:
    _CACHE_DIR = Path(__file__).parent / ".cache"
    log.warning(
        "config.py not found or failed to import — "
        "using fallback cache directory: %s", _CACHE_DIR
    )
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path(__file__).parent / ".models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED: int = 42
np.random.seed(RANDOM_SEED)
log.info("Random seed: %d", RANDOM_SEED)

HORIZONS: list[str] = ["bear_3m", "bear_6m", "bear_12m"]
HORIZON_MONTHS: dict[str, int] = {"bear_3m": 3, "bear_6m": 6, "bear_12m": 12}

MIN_TRAIN_YEARS = 10
TEST_FOLD_MONTHS = 24

PR_AUC_MIN = 0.40
BRIER_SCORE_MAX = 0.15
N_BOOTSTRAP = 500

CORRELATION_THRESHOLD = 0.95


# =============================================================================
# COMPONENT 1: TARGET VARIABLE CONSTRUCTION
# =============================================================================

def build_targets(
    sp500_prices: pd.Series,
    usrec: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Construct three binary bear market target variables.

    Parameters
    ----------
    sp500_prices : Monthly S&P 500 closing prices (DatetimeIndex, month-end).
    usrec        : Optional FRED USREC series (1=NBER recession month).
                   Used as secondary validation label only.

    Returns
    -------
    DataFrame with columns: bear_3m, bear_6m, bear_12m, in_recession.

    Target definition -- bear_Xm at date t:
        1 if S&P 500 falls >=20% from price[t] at any point in the
          next X calendar months.
        0 otherwise.
        NaN for the last X months (no forward data yet).

    Drawdown measurement:
        peak       = price[t]   (current price is the reference)
        min_price  = min(price[t+1 .. t+H])
        drawdown   = (min_price - peak) / peak
        label      = 1 if drawdown <= -0.20 else 0

    This is a forward-looking binary: did buying TODAY lead to a >=20%
    loss at any point in the next X months?
    """
    prices = sp500_prices.sort_index().astype(float)
    # Ensure prices is a 1-D Series, not a DataFrame or 2-D structure
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    prices = prices.squeeze()   # converts single-column DataFrame → Series
    n = len(prices)
    targets: dict[str, np.ndarray] = {}

    for target_name, months in HORIZON_MONTHS.items():
        labels = np.full(n, np.nan)
        for i in range(n - months):
            current_price = float(prices.iloc[i])   # explicit float — avoids Series comparison
            if current_price <= 0 or np.isnan(current_price):
                continue
            forward_prices = prices.iloc[i + 1: i + months + 1]
            if forward_prices.empty:
                continue
            min_forward = float(forward_prices.min())
            drawdown = (min_forward - current_price) / current_price
            labels[i] = 1.0 if drawdown <= -0.20 else 0.0
        targets[target_name] = labels

    df = pd.DataFrame(targets, index=prices.index)

    # NBER recession: secondary validation only (not a training label)
    if usrec is not None:
        rec_monthly = usrec.resample("ME").last().reindex(prices.index, method="ffill")
        df["in_recession"] = rec_monthly.astype(float)
    else:
        df["in_recession"] = np.nan

    # Log label distribution
    for col in HORIZONS:
        valid = df[col].dropna()
        n_pos = int(valid.sum())
        n_tot = len(valid)
        pct = 100.0 * n_pos / n_tot if n_tot > 0 else 0.0
        log.info(
            "Target %s: %d positive / %d total (%.1f%% positive, base rate=%.3f)",
            col, n_pos, n_tot, pct, pct / 100.0,
        )

    return df


# =============================================================================
# COMPONENT 2: FEATURE ENGINEERING
# =============================================================================

# Interaction pairs from Prompt 1 framework interaction effects.
# Format: (col_a, col_b, interaction_type)
# "product" = standardized a * standardized b (amplification signal)
INTERACTION_PAIRS: list[tuple[str, str, str]] = [
    ("YieldCurve",   "HY_OAS",         "product"),  # GFC/2022 archetype
    ("CAPE",         "RealFFR",         "product"),  # valuation + rate shock
    ("SahmRule",     "LEI_6M_Change",   "product"),  # recession confirmed
    ("YieldCurve",   "RealFFR",         "product"),  # max credit tightening
    ("HY_OAS",       "IG_OAS",          "product"),  # systemic vs sector stress
    ("NFCI",         "YieldCurve",      "product"),  # institutional credit shock
    ("VIX",          "HY_OAS",          "product"),  # multi-asset stress
    ("MarketBreadth","VIX",             "product"),  # broad-based panic
    ("MarketBreadth","PriceMomentum",   "product"),  # confirmed downtrend
    ("ISM_PMI",      "HY_OAS",          "product"),  # recession not soft landing
    ("AAII_BullBear","CAPE",            "product"),  # speculative excess
    ("DebtGDP",      "RealFFR",         "product"),  # debt service crisis
]

# Method B threshold parameters for pre-scoring (from Prompt 2)
_METHOD_B_PARAMS: dict[str, dict] = {
    "RealFFR":          {"x_min": -5.0,  "G": 0.0,  "A": 2.0,   "x_max": 6.0,   "direction": "higher_worse"},
    "ISM_PMI":          {"x_min": 30.0,  "G": 53.0, "A": 47.0,  "x_max": 65.0,  "direction": "lower_worse"},
    "SahmRule":         {"x_min": 0.0,   "G": 0.2,  "A": 0.5,   "x_max": 2.0,   "direction": "higher_worse"},
    "ProfitMargin":     {"x_min": 4.0,   "G": 10.0, "A": 8.0,   "x_max": 14.0,  "direction": "lower_worse"},
    "InterestCoverage": {"x_min": 1.0,   "G": 5.0,  "A": 3.0,   "x_max": 10.0,  "direction": "lower_worse"},
    "LEI_6M_Change":    {"x_min": -15.0, "G": 1.5,  "A": -2.0,  "x_max": 8.0,   "direction": "lower_worse"},
}

# Rolling window sizes for percentile rank features (Method A, Prompt 2)
_PERCENTILE_WINDOWS: dict[str, int] = {
    "CAPE": 480, "PriceToSales": 360, "ERP": 360,
    "YieldCurve": 600, "HY_OAS": 336, "IG_OAS": 324,
    "NFCI": 600, "DebtGDP": 120, "VIX": 408,
}
# These indicators have "lower = more bearish" so percentile is inverted
_INVERT_PERCENTILE = {"ERP", "YieldCurve"}


def _score_method_b(x: float, params: dict) -> float:
    """
    Piecewise linear Method B scoring from Prompt 2. Returns [0, 100].
    NaN -> 50.0 (neutral sentinel).
    """
    if np.isnan(x):
        return 50.0
    x_min, x_max = params["x_min"], params["x_max"]
    G, A = params["G"], params["A"]
    direction = params["direction"]

    if direction == "higher_worse":
        if x <= G:
            s = 33.0 * (x - x_min) / (G - x_min) if G != x_min else 0.0
        elif x <= A:
            s = 33.0 + 34.0 * (x - G) / (A - G) if A != G else 33.0
        else:
            s = 66.0 + 34.0 * (x - A) / (x_max - A) if x_max != A else 100.0
    else:  # lower_worse
        if x <= A:
            s = 100.0 - 34.0 * (x - x_min) / (A - x_min) if A != x_min else 100.0
        elif x <= G:
            s = 66.0 - 33.0 * (x - A) / (G - A) if G != A else 66.0
        else:
            s = 33.0 - 33.0 * (x - G) / (x_max - G) if x_max != G else 0.0

    return float(np.clip(s, 0.0, 100.0))


def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank of series against its own history."""
    def _rank(x: np.ndarray) -> float:
        if len(x) < 2 or np.isnan(x[-1]):
            return np.nan
        return float(np.mean(x[:-1] <= x[-1])) * 100.0
    return series.rolling(
        window=window, min_periods=max(12, window // 10)
    ).apply(_rank, raw=True)


def _fit_hmm_regime(
    yield_curve: pd.Series,
    hy_oas: pd.Series,
    n_states: int = 3,
) -> pd.Series:
    """
    3-state Gaussian HMM on [yield_curve, hy_oas].

    States ordered by mean yield_curve ascending so labels are consistent:
        0 = steep curve / low spreads (benign)
        1 = flat / moderate (transition)
        2 = inverted / wide (stress)

    Returns Series of float state labels (0/1/2), NaN where data absent.
    Falls back to all-NaN if hmmlearn unavailable or fit fails.
    """
    if not _HMM_AVAILABLE:
        return pd.Series(np.nan, index=yield_curve.index, name="hmm_regime")

    combined = pd.concat([yield_curve, hy_oas], axis=1).dropna()
    if len(combined) < 60:
        log.warning("HMM: insufficient data (%d rows). Returning NaN.", len(combined))
        return pd.Series(np.nan, index=yield_curve.index, name="hmm_regime")

    X = combined.values.astype(float)
    # Standardise so both series contribute equally
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    model = _hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=RANDOM_SEED,
        verbose=False,
    )
    try:
        model.fit(X)
    except Exception as exc:
        log.error("HMM fit failed: %s. Returning NaN.", exc)
        return pd.Series(np.nan, index=yield_curve.index, name="hmm_regime")

    labels = model.predict(X)

    # Remap so lowest yield-curve-mean state = 0 (benign)
    state_means = [model.means_[s][0] for s in range(n_states)]
    order = np.argsort(state_means)
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in labels])

    result = pd.Series(np.nan, index=yield_curve.index, name="hmm_regime")
    result.loc[combined.index] = labels.astype(float)
    return result


def _remove_correlated_features(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Greedy removal of features with pairwise Pearson |r| > threshold.
    When a pair is correlated, the feature with more NaNs is removed.
    """
    corr = df.corr(method="pearson", min_periods=20).abs()
    to_drop: set[str] = set()
    cols = list(corr.columns)
    for i, col_i in enumerate(cols):
        if col_i in to_drop:
            continue
        for col_j in cols[i + 1:]:
            if col_j in to_drop:
                continue
            if corr.loc[col_i, col_j] > threshold:
                nan_i = df[col_i].isna().sum()
                nan_j = df[col_j].isna().sum()
                to_drop.add(col_j if nan_j >= nan_i else col_i)
    if to_drop:
        log.debug("Dropping %d correlated features.", len(to_drop))
    return df.drop(columns=list(to_drop))


def engineer_features(
    indicators: pd.DataFrame,
    composite_score: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Transform raw 20-indicator DataFrame into ML feature matrix.

    Parameters
    ----------
    indicators      : Monthly raw indicator DataFrame (from indicator_calculator).
    composite_score : Optional deterministic composite score (hybrid model feature).

    Returns
    -------
    Feature DataFrame after de-correlation (>0.95 pairs removed).
    Target feature count: 60-80 depending on available indicators.

    Feature groups
    --------------
    A -- Raw indicator levels (up to 20 features)
    B -- Rolling percentile rank, Method A indicators (up to 9)
    C -- Method B pre-scored levels (up to 6)
    D -- 3M and 6M momentum (rate of change) + direction sign (up to 60)
    E -- Interaction cross-products, 12 pairs (up to 12)
    F -- HMM regime label (1)
    G -- Deterministic composite score pass-through (0 or 1)
    """
    df = indicators.copy().astype(float)
    features: dict[str, pd.Series] = {}

    # A: Raw levels
    for col in df.columns:
        features[f"raw_{col}"] = df[col]

    # B: Percentile ranks
    for col, window in _PERCENTILE_WINDOWS.items():
        if col in df.columns:
            pct = _rolling_percentile_rank(df[col], window)
            if col in _INVERT_PERCENTILE:
                pct = 100.0 - pct
            features[f"pct_{col}"] = pct

    # C: Method B pre-scored levels
    for col, params in _METHOD_B_PARAMS.items():
        if col in df.columns:
            features[f"scored_{col}"] = df[col].apply(
                lambda x: _score_method_b(x, params)
            )

    # D: 3M and 6M momentum + direction
    for col in df.columns:
        s = df[col]
        chg3 = s.diff(3)
        chg6 = s.diff(6)
        features[f"mom3_{col}"] = chg3
        features[f"mom6_{col}"] = chg6
        features[f"dir3_{col}"] = np.sign(chg3)

    # E: Interaction cross-products (standardised)
    for col_a, col_b, itype in INTERACTION_PAIRS:
        if col_a in df.columns and col_b in df.columns:
            a = df[col_a].fillna(df[col_a].median())
            b = df[col_b].fillna(df[col_b].median())
            if itype == "product":
                a_s = (a - a.mean()) / (a.std() + 1e-8)
                b_s = (b - b.mean()) / (b.std() + 1e-8)
                features[f"ix_{col_a}_x_{col_b}"] = a_s * b_s

    # F: HMM regime
    if "YieldCurve" in df.columns and "HY_OAS" in df.columns:
        features["hmm_regime"] = _fit_hmm_regime(df["YieldCurve"], df["HY_OAS"])

    # G: Composite score
    if composite_score is not None:
        features["deterministic_composite"] = composite_score.reindex(df.index)

    feat_df = pd.DataFrame(features, index=df.index)
    initial_count = feat_df.shape[1]

    feat_df = _remove_correlated_features(feat_df, CORRELATION_THRESHOLD)

    log.info(
        "Feature engineering: %d features (%d removed as correlated).",
        feat_df.shape[1], initial_count - feat_df.shape[1],
    )
    return feat_df


# =============================================================================
# COMPONENT 3: MODEL TRAINING
# =============================================================================

@dataclass
class HorizonModel:
    """Container for a trained ensemble for one prediction horizon."""
    horizon: str
    feature_names: list[str]
    xgb_model: Optional[object] = None
    rf_model: Optional[object] = None
    lr_model: Optional[object] = None
    meta_calibrated: Optional[object] = None
    scaler: Optional[StandardScaler] = None
    train_end_date: Optional[pd.Timestamp] = None
    metrics: dict = field(default_factory=dict)


def _make_xgb(n_pos: int, n_total: int, seed: int) -> XGBClassifier:
    """XGBoost with scale_pos_weight for class imbalance."""
    spw = (n_total - n_pos) / max(n_pos, 1)
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        scale_pos_weight=spw,
        use_label_encoder=False, eval_metric="aucpr",
        random_state=seed, verbosity=0, tree_method="hist",
    )


def _make_rf(seed: int) -> RandomForestClassifier:
    """Random Forest with balanced class weights."""
    return RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=3,
        max_features="sqrt", class_weight="balanced",
        random_state=seed, n_jobs=-1,
    )


def _make_lr() -> LogisticRegression:
    """Logistic Regression with L2 and balanced class weights."""
    return LogisticRegression(
        penalty="l2", C=0.1, class_weight="balanced",
        max_iter=1000, random_state=RANDOM_SEED, solver="lbfgs",
    )


def _apply_smote_within_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE inside a single training fold.

    SAFETY: Only called with fold-internal data; never on the full dataset.
    Skipped if imbalanced-learn unavailable or positive class < 5 samples.
    k_neighbors capped at n_pos - 1 to avoid SMOTE errors on tiny classes.
    """
    n_pos = int(y_train.sum())
    if not _SMOTE_AVAILABLE or n_pos < 5:
        return X_train, y_train
    k = min(5, n_pos - 1)
    try:
        sm = SMOTE(k_neighbors=k, random_state=seed)
        return sm.fit_resample(X_train, y_train)
    except Exception as exc:
        log.warning("SMOTE failed in fold: %s. Using original data.", exc)
        return X_train, y_train


def _evaluate_fold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    dates: pd.Index,
    fold_idx: int,
    horizon: str,
) -> dict:
    """Compute fold-level evaluation metrics."""
    result: dict = {"fold": fold_idx, "horizon": horizon}
    if y_true.sum() == 0:
        result.update({
            "pr_auc": np.nan, "brier": np.nan,
            "precision_top20": np.nan, "recall_at_50": np.nan,
            "n_positive": 0, "note": "no_positives",
        })
        return result
    try:
        result["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        result["pr_auc"] = np.nan
    try:
        result["brier"] = float(brier_score_loss(y_true, y_prob))
    except Exception:
        result["brier"] = np.nan
    try:
        n_top = max(1, int(0.20 * len(y_prob)))
        top_idx = np.argsort(y_prob)[-n_top:]
        result["precision_top20"] = float(np.mean(y_true[top_idx]))
    except Exception:
        result["precision_top20"] = np.nan
    try:
        result["recall_at_50"] = float(
            recall_score(y_true, (y_prob >= 0.50).astype(int), zero_division=0)
        )
    except Exception:
        result["recall_at_50"] = np.nan
    result["n_positive"] = int(y_true.sum())
    result["date_start"] = str(dates[0].date()) if len(dates) > 0 else None
    result["date_end"]   = str(dates[-1].date()) if len(dates) > 0 else None
    return result


def train_horizon_model(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    horizon: str,
) -> tuple[HorizonModel, list[dict]]:
    """
    Train the full stacked ensemble for one prediction horizon.

    Walk-forward expanding window:
        - Minimum training window: 10 years (120 months)
        - Test fold: 24 months
        - Hyperparameter search via RandomizedSearchCV within each fold
        - SMOTE applied inside each training fold after splitting

    Base models: XGBoost, RandomForest, LogisticRegression
    Meta model: LogisticRegression stacking base model probabilities

    Final model trained on ALL available data for production use.

    Returns
    -------
    (HorizonModel, fold_results_list)
    """
    log.info("Training model: %s", horizon)

    y = targets[horizon].astype(float)
    X_full = features.copy()

    valid_mask = X_full.notna().all(axis=1) & y.notna()
    X_valid = X_full.loc[valid_mask].copy()
    y_valid = y.loc[valid_mask].copy()

    # Impute: ffill then median
    X_valid = X_valid.ffill().fillna(X_valid.median())

    n_samples = len(X_valid)
    min_train_n = MIN_TRAIN_YEARS * 12

    if n_samples < min_train_n + 24:
        log.error(
            "%s: insufficient data (%d samples; need >=%d). Skipping.",
            horizon, n_samples, min_train_n + 24,
        )
        return HorizonModel(horizon=horizon, feature_names=list(X_valid.columns)), []

    feature_names = list(X_valid.columns)
    X_arr = X_valid.values.astype(float)
    y_arr = y_valid.values.astype(float)

    n_folds = max(2, (n_samples - min_train_n) // TEST_FOLD_MONTHS)
    tscv = TimeSeriesSplit(
        n_splits=n_folds, gap=0,
        max_train_size=None,       # expanding window: no cap
        test_size=TEST_FOLD_MONTHS,
    )

    fold_results: list[dict] = []
    xgb_hyperparam_space = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8],
        "min_child_weight": [1, 3, 5],
    }

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_arr)):
        if len(train_idx) < min_train_n:
            log.debug("Fold %d: train too small (%d). Skipping.", fold_idx, len(train_idx))
            continue

        X_tr_raw, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr,     y_te = y_arr[train_idx], y_arr[test_idx]
        fold_seed = RANDOM_SEED + fold_idx

        # Scale: fit on train fold only
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_raw)
        X_te_s = scaler.transform(X_te)

        # SMOTE: ONLY inside training fold (anti-leakage critical path)
        X_tr_sm, y_tr_sm = _apply_smote_within_fold(X_tr_s, y_tr, fold_seed)
        n_pos = int(y_tr_sm.sum())
        n_tot = len(y_tr_sm)

        # XGBoost with HP tuning on train fold only
        inner_cv = TimeSeriesSplit(n_splits=3, test_size=12)
        try:
            xgb_search = RandomizedSearchCV(
                XGBClassifier(
                    use_label_encoder=False, eval_metric="aucpr",
                    random_state=fold_seed, verbosity=0, tree_method="hist",
                    scale_pos_weight=(n_tot - n_pos) / max(n_pos, 1),
                ),
                param_distributions=xgb_hyperparam_space,
                n_iter=10, scoring="average_precision",
                cv=inner_cv, random_state=fold_seed, n_jobs=-1, refit=True,
            )
            xgb_search.fit(X_tr_sm, y_tr_sm)
            xgb_fold = xgb_search.best_estimator_
        except Exception as exc:
            log.warning("XGB search failed fold %d: %s. Using default.", fold_idx, exc)
            xgb_fold = _make_xgb(n_pos, n_tot, fold_seed)
            xgb_fold.fit(X_tr_sm, y_tr_sm)

        # Random Forest
        rf_fold = _make_rf(fold_seed)
        rf_fold.fit(X_tr_sm, y_tr_sm)

        # Logistic Regression (receives already-scaled SMOTE data)
        lr_fold = _make_lr()
        lr_fold.fit(X_tr_sm, y_tr_sm)

        # Ensemble = mean of base model test probabilities
        xgb_p = xgb_fold.predict_proba(X_te_s)[:, 1]
        rf_p  = rf_fold.predict_proba(X_te_s)[:, 1]
        lr_p  = lr_fold.predict_proba(X_te_s)[:, 1]
        ensemble_p = (xgb_p + rf_p + lr_p) / 3.0

        fold_results.append(_evaluate_fold(
            y_te, ensemble_p,
            dates=y_valid.index[test_idx],
            fold_idx=fold_idx, horizon=horizon,
        ))
        log.info(
            "Fold %d | %s | PR-AUC=%.3f | Brier=%.3f | n_pos=%d",
            fold_idx, horizon,
            fold_results[-1].get("pr_auc", np.nan),
            fold_results[-1].get("brier", np.nan),
            int(y_te.sum()),
        )

    # ---- Final production model: ALL data ----
    log.info("Fitting final %s model on full dataset (%d samples).", horizon, len(X_arr))
    scaler_final = StandardScaler()
    X_all_s = scaler_final.fit_transform(X_arr)

    X_all_sm, y_all_sm = _apply_smote_within_fold(X_all_s, y_arr, RANDOM_SEED)
    n_pos_f = int(y_all_sm.sum())
    n_tot_f = len(y_all_sm)

    xgb_final = _make_xgb(n_pos_f, n_tot_f, RANDOM_SEED)
    xgb_final.fit(X_all_sm, y_all_sm)

    rf_final = _make_rf(RANDOM_SEED)
    rf_final.fit(X_all_sm, y_all_sm)

    lr_final = _make_lr()
    lr_final.fit(X_all_sm, y_all_sm)

    # Meta-learner: train on original (non-SMOTE) scaled data to avoid
    # distorted calibration targets from synthetic samples
    xgb_ap = xgb_final.predict_proba(X_all_s)[:, 1]
    rf_ap  = rf_final.predict_proba(X_all_s)[:, 1]
    lr_ap  = lr_final.predict_proba(X_all_s)[:, 1]
    meta_X = np.column_stack([xgb_ap, rf_ap, lr_ap])

    meta_lr = LogisticRegression(
        penalty="l2", C=1.0, max_iter=500, random_state=RANDOM_SEED
    )
    meta_lr.fit(meta_X, y_arr)

    model = HorizonModel(
        horizon=horizon,
        feature_names=feature_names,
        xgb_model=xgb_final,
        rf_model=rf_final,
        lr_model=lr_final,
        meta_calibrated=meta_lr,   # replaced by isotonic calibrator in Component 4
        scaler=scaler_final,
        train_end_date=y_valid.index[-1],
        metrics={},
    )
    return model, fold_results


# =============================================================================
# COMPONENT 4: CALIBRATION
# =============================================================================

def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error.
    ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc  = np.mean(y_true[mask])
        conf = np.mean(y_prob[mask])
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def calibrate_model(
    model: HorizonModel,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
) -> HorizonModel:
    """
    Apply isotonic regression calibration to the meta-learner output.

    Calibration is fitted on a held-out calibration set (last 20% of
    in-sample data) so it does NOT reuse the full training set.

    Populates model.metrics with:
        ece                 -- Expected Calibration Error (float)
        calibration_flag    -- "OK" or "UNCALIBRATED"
        calibration_curve   -- dict with fraction_of_positives, mean_predicted_value

    Flags "UNCALIBRATED" and logs a warning if ECE > 0.05.
    """
    if model.xgb_model is None:
        log.warning("Calibrate: %s not trained. Skipping.", model.horizon)
        return model

    # Generate base model probabilities on calibration set
    X_cal_s = model.scaler.transform(X_cal)
    xgb_p = model.xgb_model.predict_proba(X_cal_s)[:, 1]
    rf_p  = model.rf_model.predict_proba(X_cal_s)[:, 1]
    lr_p  = model.lr_model.predict_proba(X_cal_s)[:, 1]
    meta_X_cal = np.column_stack([xgb_p, rf_p, lr_p])

    # Wrap raw meta-LR in isotonic calibration
    calibrated = CalibratedClassifierCV(
        model.meta_calibrated, method="isotonic", cv="prefit"
    )
    try:
        calibrated.fit(meta_X_cal, y_cal)
        model.meta_calibrated = calibrated
        cal_prob = calibrated.predict_proba(meta_X_cal)[:, 1]
    except Exception as exc:
        log.warning("Isotonic calibration failed: %s. Using raw.", exc)
        cal_prob = model.meta_calibrated.predict_proba(meta_X_cal)[:, 1]

    ece = _compute_ece(y_cal, cal_prob)
    model.metrics["ece"] = float(ece)
    log.info("Calibration | %s | ECE=%.4f (threshold=0.05)", model.horizon, ece)

    if ece > 0.05:
        log.warning(
            "CALIBRATION WARNING: %s ECE=%.4f > 0.05. "
            "Probabilities may be unreliable.", model.horizon, ece
        )
        model.metrics["calibration_flag"] = "UNCALIBRATED"
    else:
        model.metrics["calibration_flag"] = "OK"

    # Reliability diagram data for Streamlit
    try:
        frac_pos, mean_pred = calibration_curve(y_cal, cal_prob, n_bins=10)
        model.metrics["calibration_curve"] = {
            "fraction_of_positives": frac_pos.tolist(),
            "mean_predicted_value": mean_pred.tolist(),
        }
    except Exception as exc:
        log.warning("Calibration curve failed: %s", exc)

    return model


# =============================================================================
# COMPONENT 5: EVALUATION & BACKTESTING
# =============================================================================

def _bootstrap_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_iter: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> tuple[float, float, float]:
    """
    Bootstrap 95% CI for a metric function.
    Returns (point_estimate, ci_lower_2.5, ci_upper_97.5).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot: list[float] = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        try:
            boot.append(float(metric_fn(yt, yp)))
        except Exception:
            pass
    if not boot:
        return np.nan, np.nan, np.nan
    point = float(metric_fn(y_true, y_prob))
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def run_backtest(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    sp500_prices: pd.Series,
    trained_models: dict[str, HorizonModel],
) -> pd.DataFrame:
    """
    Full backtesting and evaluation report.

    Reports per horizon and per decade (1990s, 2000s, 2010s, 2020s, ALL):
        PR-AUC with 95% bootstrap CI (primary metric)
        Brier Score with 95% bootstrap CI
        Precision at top 20% probability predictions
        Recall at 50% probability threshold
        Average lead time in months before bear market
        Number of structural bears flagged >=3M in advance

    Prints "MODEL PERFORMANCE BELOW THRESHOLD -- USE WITH EXTREME CAUTION"
    if PR-AUC < 0.40, Brier > 0.15, or fewer than 3 of 5 bears flagged.

    Returns
    -------
    Wide DataFrame suitable for display in Streamlit.
    """
    rows: list[dict] = []
    decades = {
        "1990s": (pd.Timestamp("1990-01-01"), pd.Timestamp("1999-12-31")),
        "2000s": (pd.Timestamp("2000-01-01"), pd.Timestamp("2009-12-31")),
        "2010s": (pd.Timestamp("2010-01-01"), pd.Timestamp("2019-12-31")),
        "2020s": (pd.Timestamp("2020-01-01"), pd.Timestamp("2029-12-31")),
        "ALL":   (pd.Timestamp("1990-01-01"), pd.Timestamp("2099-12-31")),
    }
    # Known structural bear market peaks for lead-time analysis
    bear_windows = {
        "2000 Dot-com": pd.Timestamp("2000-03-01"),
        "2007 GFC":     pd.Timestamp("2007-10-01"),
        "2022 Rate":    pd.Timestamp("2022-01-01"),
    }
    all_pass = True

    for horizon, model in trained_models.items():
        if model.xgb_model is None:
            log.warning("Backtest: %s has no trained model. Skipping.", horizon)
            continue

        y = targets[horizon].dropna()
        X_aligned = features.reindex(y.index).ffill().fillna(features.median())
        X_s = model.scaler.transform(X_aligned.values)
        xgb_p = model.xgb_model.predict_proba(X_s)[:, 1]
        rf_p  = model.rf_model.predict_proba(X_s)[:, 1]
        lr_p  = model.lr_model.predict_proba(X_s)[:, 1]
        meta_X = np.column_stack([xgb_p, rf_p, lr_p])
        prob_s = pd.Series(
            model.meta_calibrated.predict_proba(meta_X)[:, 1],
            index=y.index,
        )

        # Lead-time analysis per known bear market
        lead_times: list[float] = []
        flagged_count = 0
        for bear_name, peak_ts in bear_windows.items():
            lookback_start = peak_ts - pd.DateOffset(months=24)
            pre_peak = prob_s.loc[
                (prob_s.index >= lookback_start) & (prob_s.index < peak_ts)
            ]
            if pre_peak.empty:
                continue
            early = pre_peak[pre_peak >= 0.50]
            if not early.empty:
                months_lead = (peak_ts - early.index[0]).days / 30.4
                if months_lead >= 3:
                    flagged_count += 1
                    lead_times.append(months_lead)
                    log.info(
                        "Backtest %s | %s: %.1f months lead.",
                        horizon, bear_name, months_lead,
                    )

        avg_lead = float(np.mean(lead_times)) if lead_times else 0.0

        # Metrics per decade
        for decade_name, (d_start, d_end) in decades.items():
            mask = (y.index >= d_start) & (y.index <= d_end)
            y_d = y[mask].values.astype(float)
            p_d = prob_s.loc[y.index[mask]].values.astype(float)

            if len(y_d) == 0:
                continue

            n_pos = int(y_d[~np.isnan(y_d)].sum())
            if n_pos == 0:
                rows.append({
                    "horizon": horizon, "decade": decade_name,
                    "pr_auc": np.nan, "pr_auc_lo": np.nan, "pr_auc_hi": np.nan,
                    "brier": np.nan, "brier_lo": np.nan, "brier_hi": np.nan,
                    "precision_top20": np.nan, "recall_at_50": np.nan,
                    "n_positive": 0, "n_total": len(y_d),
                    "avg_lead_months": avg_lead if decade_name == "ALL" else np.nan,
                    "bears_flagged_3m": flagged_count if decade_name == "ALL" else np.nan,
                    "note": "no_positives",
                })
                continue

            pr, pr_lo, pr_hi = _bootstrap_metric(y_d, p_d, average_precision_score)
            br, br_lo, br_hi = _bootstrap_metric(y_d, p_d, brier_score_loss)
            n_top = max(1, int(0.20 * len(p_d)))
            prec20 = float(np.mean(y_d[np.argsort(p_d)[-n_top:]]))
            rec50  = float(recall_score(
                y_d, (p_d >= 0.50).astype(int), zero_division=0
            ))
            rows.append({
                "horizon": horizon, "decade": decade_name,
                "pr_auc": pr, "pr_auc_lo": pr_lo, "pr_auc_hi": pr_hi,
                "brier": br, "brier_lo": br_lo, "brier_hi": br_hi,
                "precision_top20": prec20, "recall_at_50": rec50,
                "n_positive": n_pos, "n_total": len(y_d),
                "avg_lead_months": avg_lead if decade_name == "ALL" else np.nan,
                "bears_flagged_3m": flagged_count if decade_name == "ALL" else np.nan,
                "note": "",
            })

        # Threshold check (ALL decade row)
        all_row = next(
            (r for r in rows if r["horizon"] == horizon and r["decade"] == "ALL"),
            None,
        )
        if all_row:
            pr = all_row.get("pr_auc", np.nan)
            br = all_row.get("brier", np.nan)
            fl = all_row.get("bears_flagged_3m", 0)
            passes = (
                not np.isnan(pr) and pr >= PR_AUC_MIN
                and not np.isnan(br) and br <= BRIER_SCORE_MAX
                and fl >= 3
            )
            if not passes:
                all_pass = False
                log.warning(
                    "\n================================================\n"
                    "MODEL PERFORMANCE BELOW THRESHOLD -- USE WITH EXTREME CAUTION\n"
                    "Horizon       : %s\n"
                    "PR-AUC        : %.3f (minimum %.2f)\n"
                    "Brier Score   : %.3f (maximum %.2f)\n"
                    "Bears flagged : %d of 3 (minimum 3)\n"
                    "================================================",
                    horizon,
                    pr if not np.isnan(pr) else -1.0, PR_AUC_MIN,
                    br if not np.isnan(br) else -1.0, BRIER_SCORE_MAX,
                    fl,
                )

    if all_pass and rows:
        log.info("All horizons passed minimum performance thresholds.")

    return pd.DataFrame(rows)


# =============================================================================
# COMPONENT 6: SHAP EXPLAINABILITY
# =============================================================================

def compute_shap_values(
    model: HorizonModel,
    X: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Compute SHAP values for all three base models.

    Uses:
        TreeExplainer  -- XGBoost and RandomForest
        LinearExplainer -- Logistic Regression (with background masker)

    Returns
    -------
    Dict with keys "xgb", "rf", "lr", "mean".
    Each value is a DataFrame: index=X.index, columns=feature_names.
    "mean" is a weighted ensemble (XGB 40%, RF 30%, LR 30%).
    Missing models are excluded and weights renormalized.
    """
    if model.xgb_model is None:
        log.warning("SHAP: %s model not trained.", model.horizon)
        return {}

    X_aligned = X.reindex(columns=model.feature_names, fill_value=0.0)
    X_aligned = X_aligned.ffill().fillna(0.0)
    X_scaled = model.scaler.transform(X_aligned.values)
    X_scaled_df = pd.DataFrame(
        X_scaled, index=X_aligned.index, columns=model.feature_names
    )

    shap_dict: dict[str, pd.DataFrame] = {}

    # XGBoost -- TreeExplainer
    try:
        expl_xgb = shap.TreeExplainer(model.xgb_model, feature_names=model.feature_names)
        sv = expl_xgb.shap_values(X_scaled_df.values)
        shap_dict["xgb"] = pd.DataFrame(sv, index=X_aligned.index, columns=model.feature_names)
    except Exception as exc:
        log.error("XGB SHAP failed: %s", exc)

    # RandomForest -- TreeExplainer (returns list [class0, class1])
    try:
        expl_rf = shap.TreeExplainer(model.rf_model, feature_names=model.feature_names)
        sv = expl_rf.shap_values(X_scaled_df.values)
        if isinstance(sv, list):
            sv = sv[1]   # take positive class
        shap_dict["rf"] = pd.DataFrame(sv, index=X_aligned.index, columns=model.feature_names)
    except Exception as exc:
        log.error("RF SHAP failed: %s", exc)

    # LogisticRegression -- LinearExplainer with background sample
    try:
        bg_n = min(100, len(X_scaled_df))
        rng = np.random.default_rng(RANDOM_SEED)
        bg_idx = rng.integers(0, len(X_scaled_df), size=bg_n)
        background = shap.maskers.Independent(X_scaled_df.values[bg_idx])
        expl_lr = shap.LinearExplainer(model.lr_model, background)
        sv = expl_lr.shap_values(X_scaled_df.values)
        shap_dict["lr"] = pd.DataFrame(sv, index=X_aligned.index, columns=model.feature_names)
    except Exception as exc:
        log.error("LR SHAP failed: %s", exc)

    # Weighted ensemble SHAP (XGB 40%, RF 30%, LR 30%)
    weights = {"xgb": 0.40, "rf": 0.30, "lr": 0.30}
    available = [k for k in weights if k in shap_dict]
    if available:
        total_w = sum(weights[k] for k in available)
        mean_shap = sum(
            shap_dict[k] * (weights[k] / total_w) for k in available
        )
        shap_dict["mean"] = mean_shap

    return shap_dict


def get_top_shap_drivers(
    shap_values: pd.DataFrame,
    n_top: int = 5,
    last_row: bool = True,
) -> list[dict]:
    """
    Extract top N SHAP drivers for the current or average prediction.

    Parameters
    ----------
    shap_values : SHAP DataFrame from compute_shap_values.
    n_top       : Number of top features to return.
    last_row    : True = use last row (current date); False = use time-mean.

    Returns
    -------
    List of dicts sorted by |shap_value| descending:
        {"feature": str, "shap_value": float, "direction": str, "magnitude": float}
    """
    if shap_values is None or shap_values.empty:
        return []
    row = shap_values.iloc[-1] if last_row else shap_values.abs().mean()
    top_features = row.abs().nlargest(n_top).index
    return [
        {
            "feature": feat,
            "shap_value": float(row[feat]),
            "direction": "increases_risk" if row[feat] > 0 else "decreases_risk",
            "magnitude": abs(float(row[feat])),
        }
        for feat in top_features
    ]


def get_shap_summary_data(shap_values: pd.DataFrame) -> pd.DataFrame:
    """
    Return mean absolute SHAP per feature, sorted descending.
    Used by Streamlit horizontal bar chart in Prompt 5.

    Returns DataFrame: columns = [feature, mean_abs_shap].
    """
    if shap_values is None or shap_values.empty:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    summary = shap_values.abs().mean(axis=0).sort_values(ascending=False)
    return summary.reset_index().rename(
        columns={"index": "feature", 0: "mean_abs_shap"}
    )


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_model(model: HorizonModel, metrics_summary: dict) -> Path:
    """
    Save HorizonModel to disk via joblib.

    Filename: {horizon}_{YYYYMMDD_HHMM}_prauc{val}.joblib
    Compression level 3 for reasonable file size.
    """
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    pr_str = f"{metrics_summary.get('pr_auc', 0.0):.3f}".replace(".", "p")
    fname = f"{model.horizon}_{ts}_prauc{pr_str}.joblib"
    path = MODEL_DIR / fname
    # Pin the module so joblib always pickles as 'ml_model.HorizonModel',
    # not '__main__.HorizonModel'. This prevents load failures when app.py
    # is __main__ and calls load_latest_model().
    HorizonModel.__module__ = "ml_model"
    joblib.dump(
        {
            "model": model,
            "metrics_summary": metrics_summary,
            "saved_at": ts,
            "random_seed": RANDOM_SEED,
            "feature_names": model.feature_names,
        },
        path,
        compress=3,
    )
    log.info("Model saved: %s", path)
    return path


def load_latest_model(horizon: str) -> Optional[HorizonModel]:
    """
    Load the most recently saved model for a given horizon.
    Returns None if no saved model exists.

    Ensures HorizonModel is in scope before joblib.load() so Python can
    deserialise the pickled class regardless of which module is __main__.
    """
    candidates = sorted(MODEL_DIR.glob(f"{horizon}_*.joblib"), reverse=True)
    if not candidates:
        log.warning("No saved model for horizon '%s'.", horizon)
        return None

    try:
        # Register this module in sys.modules under 'ml_model' so joblib
        # can resolve HorizonModel even when __main__ is app.py.
        import sys
        import ml_model as _ml_mod  # noqa: F401 — side-effect import needed
        sys.modules.setdefault("ml_model", _ml_mod)

        payload = joblib.load(candidates[0])
        log.info("Loaded model: %s", candidates[0])
        return payload["model"]
    except ModuleNotFoundError:
        # Fallback: patch __main__ so joblib can find HorizonModel there
        import sys as _sys
        _sys.modules["__main__"].HorizonModel = HorizonModel
        try:
            payload = joblib.load(candidates[0])
            log.info("Loaded model (via __main__ patch): %s", candidates[0])
            return payload["model"]
        except Exception as exc2:
            log.error("Failed to load %s even with __main__ patch: %s",
                      candidates[0], exc2)
            return None
    except Exception as exc:
        log.error("Failed to load %s: %s", candidates[0], exc)
        return None


# =============================================================================
# MASTER TRAINING ORCHESTRATOR
# =============================================================================

def train_all_models(
    indicators: pd.DataFrame,
    sp500_prices: pd.Series,
    usrec: Optional[pd.Series] = None,
    composite_score: Optional[pd.Series] = None,
) -> tuple[dict[str, HorizonModel], pd.DataFrame]:
    """
    Full ML training pipeline: targets -> features -> train -> calibrate -> evaluate.

    Parameters
    ----------
    indicators      : Raw 20-indicator monthly DataFrame.
    sp500_prices    : Monthly S&P 500 closing prices.
    usrec           : FRED USREC series (optional secondary labels).
    composite_score : Deterministic composite score (optional hybrid feature).

    Returns
    -------
    (models_dict, backtest_report_df)
        models_dict       : {horizon: HorizonModel} ready for predict_current().
        backtest_report_df: DataFrame from run_backtest().
    """
    log.info("=== AI Bear Risk Dashboard -- ML Training Pipeline ===")
    log.info("Seed=%d | SMOTE=%s | HMM=%s", RANDOM_SEED, _SMOTE_AVAILABLE, _HMM_AVAILABLE)

    # Ensure prices are monthly regardless of whether daily or monthly were passed in.
    # If already monthly (freq ME or MS, or fewer than 600 rows for 40 years of data)
    # resample is idempotent — last() of a single-row month returns that row.
    if isinstance(sp500_prices, pd.DataFrame):
        sp500_prices = sp500_prices.iloc[:, 0]
    sp500_prices = sp500_prices.dropna().astype(float)
    sp500_monthly = sp500_prices.resample("ME").last().dropna()
    targets = build_targets(sp500_monthly, usrec)

    # Features
    features = engineer_features(indicators, composite_score)

    models: dict[str, HorizonModel] = {}
    for horizon in HORIZONS:
        model, _ = train_horizon_model(features, targets, horizon)

        # Calibration: last 20% of in-sample (at least 24 months)
        y_h = targets[horizon].dropna()
        X_h = features.reindex(y_h.index).ffill().fillna(features.median())
        n_cal = max(24, int(0.20 * len(y_h)))
        model = calibrate_model(
            model, X_h.iloc[-n_cal:].values, y_h.iloc[-n_cal:].values
        )

        save_model(model, {k: v for k, v in model.metrics.items()
                           if isinstance(v, (int, float, str))})
        models[horizon] = model

    backtest_df = run_backtest(features, targets, sp500_monthly, models)
    log.info("Training complete.\n%s", backtest_df.to_string(index=False))
    return models, backtest_df


# =============================================================================
# PREDICT CURRENT -- PUBLIC API FOR STREAMLIT (PROMPT 5)
# =============================================================================

def predict_current(
    indicators: pd.DataFrame,
    composite_score: Optional[pd.Series] = None,
    models: Optional[dict[str, HorizonModel]] = None,
) -> Optional[dict]:
    """
    Generate current bear market probability estimates for all three horizons.

    Loads pre-trained models from disk if `models` is not provided.
    Returns None (with logged warning) if prediction is impossible.

    Parameters
    ----------
    indicators      : Raw 20-indicator DataFrame (monthly).
    composite_score : Deterministic composite score (optional).
    models          : Pre-loaded models dict. None -> loads from disk.

    Returns
    -------
    Dict with keys:
        bear_3m_prob    float   P(bear within 3M)
        bear_6m_prob    float   P(bear within 6M)
        bear_12m_prob   float   P(bear within 12M)
        bear_3m_ci      (lo, hi)  95% interval
        bear_6m_ci      (lo, hi)
        bear_12m_ci     (lo, hi)
        top_drivers     {horizon: [driver_dict]}  top 5 SHAP drivers
        shap_history    {horizon: DataFrame}       full SHAP time series
        summary         str  human-readable risk summary
        as_of_date      str  ISO date of latest data
        model_flags     {horizon: calibration_flag}
    Returns None if prediction cannot be made.
    """
    if models is None:
        models = {}
        for h in HORIZONS:
            m = load_latest_model(h)
            if m is not None:
                models[h] = m

    if not models:
        log.warning("predict_current: No trained models available. Returning None.")
        return None

    try:
        features = engineer_features(indicators, composite_score)
    except Exception as exc:
        log.error("predict_current: feature engineering failed: %s", exc)
        return None

    if features.empty or len(features) < 12:
        log.warning("predict_current: Insufficient feature history (%d rows).", len(features))
        return None

    as_of_date = features.index[-1]
    probs:        dict[str, float]           = {}
    cis:          dict[str, tuple]           = {}
    top_drivers:  dict[str, list]            = {}
    shap_history: dict[str, pd.DataFrame]   = {}
    model_flags:  dict[str, str]             = {}

    for horizon, model in models.items():
        if model.xgb_model is None:
            log.warning("predict_current: %s not trained.", horizon)
            probs[horizon] = np.nan
            cis[horizon] = (np.nan, np.nan)
            continue

        X_pred = features.reindex(columns=model.feature_names, fill_value=0.0)
        X_pred = X_pred.ffill().fillna(0.0)
        X_cur_s = model.scaler.transform(X_pred.iloc[[-1]].values)

        try:
            xgb_p = model.xgb_model.predict_proba(X_cur_s)[:, 1][0]
            rf_p  = model.rf_model.predict_proba(X_cur_s)[:, 1][0]
            lr_p  = model.lr_model.predict_proba(X_cur_s)[:, 1][0]
            meta_X = np.array([[xgb_p, rf_p, lr_p]])
            prob = float(model.meta_calibrated.predict_proba(meta_X)[:, 1][0])
        except Exception as exc:
            log.error("predict_current %s prediction failed: %s", horizon, exc)
            prob = np.nan

        probs[horizon] = prob

        # Bootstrap CI from last 36 months of predictions
        n_ci = min(36, len(X_pred))
        try:
            X_ci_s = model.scaler.transform(X_pred.iloc[-n_ci:].values)
            xgb_ci = model.xgb_model.predict_proba(X_ci_s)[:, 1]
            rf_ci  = model.rf_model.predict_proba(X_ci_s)[:, 1]
            lr_ci  = model.lr_model.predict_proba(X_ci_s)[:, 1]
            p_ci   = model.meta_calibrated.predict_proba(
                np.column_stack([xgb_ci, rf_ci, lr_ci])
            )[:, 1]
            cis[horizon] = (float(np.percentile(p_ci, 2.5)), float(np.percentile(p_ci, 97.5)))
        except Exception:
            cis[horizon] = (np.nan, np.nan)

        # SHAP
        try:
            shap_vals = compute_shap_values(model, X_pred)
            if "mean" in shap_vals:
                shap_history[horizon] = shap_vals["mean"]
                top_drivers[horizon]  = get_top_shap_drivers(shap_vals["mean"], n_top=5)
        except Exception as exc:
            log.warning("predict_current %s SHAP failed: %s", horizon, exc)
            top_drivers[horizon] = []

        model_flags[horizon] = model.metrics.get("calibration_flag", "UNKNOWN")

    # Human-readable summary
    p3  = probs.get("bear_3m", np.nan)
    p6  = probs.get("bear_6m", np.nan)
    p12 = probs.get("bear_12m", np.nan)

    valid_probs = [x for x in [p3, p6, p12] if not np.isnan(x)]
    max_prob = max(valid_probs) if valid_probs else np.nan

    if np.isnan(max_prob):      risk_level = "UNKNOWN"
    elif max_prob >= 0.60:      risk_level = "HIGH"
    elif max_prob >= 0.40:      risk_level = "ELEVATED"
    elif max_prob >= 0.20:      risk_level = "GUARDED"
    else:                       risk_level = "LOW"

    primary_drivers = top_drivers.get("bear_6m", [])
    driver_str = ""
    if primary_drivers:
        top = primary_drivers[0]
        driver_str = (
            f" Primary driver: {top['feature']} "
            f"({top['direction'].replace('_', ' ')})."
        )

    def _pct(p: float) -> str:
        return "N/A" if np.isnan(p) else f"{p * 100:.0f}%"

    summary = (
        f"As of {as_of_date.date()} | Risk: {risk_level} | "
        f"P(3M)={_pct(p3)} P(6M)={_pct(p6)} P(12M)={_pct(p12)}.{driver_str}"
    )
    log.info("predict_current: %s", summary)

    # ── Persist prediction to disk for bear_risk_api.py ───────────────────────
    # bear_risk_api.get_ml_probabilities() reads this file (cache-only, no API
    # calls). We convert tuples to lists so json.dump handles them cleanly.
    import json as _json
    _pred_payload = {
        "bear_3m_prob":  p3  if not np.isnan(p3)  else None,
        "bear_6m_prob":  p6  if not np.isnan(p6)  else None,
        "bear_12m_prob": p12 if not np.isnan(p12) else None,
        "bear_3m_ci":    list(cis.get("bear_3m",  (None, None))),
        "bear_6m_ci":    list(cis.get("bear_6m",  (None, None))),
        "bear_12m_ci":   list(cis.get("bear_12m", (None, None))),
        "top_drivers":   {str(k): v for k, v in top_drivers.items()},
        "summary":       summary,
        "as_of_date":    str(as_of_date.date()),
        "model_flags":   model_flags,
    }
    try:
        _pred_path = _CACHE_DIR / "latest_prediction.json"
        _pred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_pred_path, "w", encoding="utf-8") as _pf:
            _json.dump(_pred_payload, _pf, default=str, indent=2)
        log.info("predict_current: prediction saved to %s", _pred_path)
    except Exception as _save_exc:
        log.warning("predict_current: could not save prediction cache: %s", _save_exc)
    # ─────────────────────────────────────────────────────────────────────────

    return {
        "bear_3m_prob":  p3,
        "bear_6m_prob":  p6,
        "bear_12m_prob": p12,
        "bear_3m_ci":    cis.get("bear_3m",  (np.nan, np.nan)),
        "bear_6m_ci":    cis.get("bear_6m",  (np.nan, np.nan)),
        "bear_12m_ci":   cis.get("bear_12m", (np.nan, np.nan)),
        "top_drivers":   top_drivers,
        "shap_history":  shap_history,
        "summary":       summary,
        "as_of_date":    str(as_of_date.date()),
        "model_flags":   model_flags,
    }


def load_models_and_predict(
    indicators: pd.DataFrame,
    composite_score: Optional[pd.Series] = None,
) -> Optional[dict]:
    """
    Primary entry point for Streamlit (Prompt 5).
    Loads latest saved models from disk, then runs predict_current().
    Returns None if no models are saved.
    """
    return predict_current(indicators, composite_score, models=None)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — run directly for training + prediction
# Usage: python ml_model.py
#        python ml_model.py --predict-only
#        python ml_model.py --backtest
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import argparse

    # ── Force all output to stdout (fixes Windows silent output issue) ─────────
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    _stdout_handler = logging.StreamHandler(sys.stdout)
    _stdout_handler.setLevel(logging.INFO)
    _stdout_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(_stdout_handler)

    # ── Parse arguments ────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Bear Risk ML Pipeline")
    parser.add_argument(
        "--predict-only", action="store_true",
        help="Skip training — run prediction with existing saved models only.",
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="Print full backtest performance report after training.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Bear Risk Dashboard — ML Pipeline")
    if args.predict_only:
        print("  Mode: predict only (using existing models)")
    else:
        print("  Mode: full training + prediction")
        print("  Expected time: 15-25 minutes")
    print("=" * 60)
    print()

    # ── Load data ──────────────────────────────────────────────────────────────
    log.info("Loading cached indicator data...")
    try:
        from data_fetcher import fetch_all_data
        from indicator_calculator import calculate_all_indicators
        raw = fetch_all_data()
        indicators_df = calculate_all_indicators(raw)
        log.info("Indicators loaded: %d columns x %d rows", indicators_df.shape[1], indicators_df.shape[0])
    except Exception as e:
        print(f"\nERROR: Could not load indicator data: {e}")
        print("Run this first:  python data_fetcher.py")
        sys.exit(1)

    # ── Get S&P 500 prices ─────────────────────────────────────────────────────
    sp500_df = raw.get("sp500_price")
    if sp500_df is None:
        print("\nERROR: No S&P 500 price data in cache.")
        print("Run this first:  python data_fetcher.py")
        sys.exit(1)
    sp500_prices = sp500_df.iloc[:, 0].dropna()

    # ── NBER recession dates (optional) ───────────────────────────────────────
    usrec_df = raw.get("USREC")
    usrec = usrec_df.iloc[:, 0] if usrec_df is not None else None

    # ── Predict-only mode ──────────────────────────────────────────────────────
    if args.predict_only:
        log.info("Running predict_current() with existing models...")
        result = predict_current(indicators_df)
        if result:
            print()
            print("=" * 60)
            print(result.get("summary", "No summary available."))
            print("=" * 60)
        else:
            print("\nNo trained models found.")
            print("Run without --predict-only to train first.")
        sys.exit(0)

    # ── Full training ──────────────────────────────────────────────────────────
    # train_all_models() builds targets and features internally.
    # Pass the RAW indicators + price series — NOT pre-built features/targets.
    # It returns a (models_dict, backtest_df) TUPLE — unpack both.
    log.info("Starting model training (this takes 15-25 minutes)...")
    models, backtest_df = train_all_models(
        indicators=indicators_df,
        sp500_prices=sp500_prices,
        usrec=usrec,
    )

    # ── Run prediction and save to disk ───────────────────────────────────────
    log.info("Running current prediction...")
    result = predict_current(indicators_df, models=models)

    if result:
        print()
        print("=" * 60)
        print("CURRENT PREDICTION")
        print("=" * 60)
        print(result.get("summary", "No summary available."))
        print("=" * 60)
    else:
        log.warning("Prediction returned None — check logs above.")

    # ── Backtest report ────────────────────────────────────────────────────────
    if args.backtest:
        if backtest_df is not None and not backtest_df.empty:
            print()
            print("=" * 60)
            print("BACKTEST PERFORMANCE REPORT")
            print("=" * 60)
            print(backtest_df.to_string(index=False))
            print("=" * 60)
        else:
            log.warning("Backtest DataFrame is empty — no trained models produced results.")

    print()
    print("Training complete. Models saved to:", MODEL_DIR)
    print("Prediction saved to:", _CACHE_DIR / "latest_prediction.json")
