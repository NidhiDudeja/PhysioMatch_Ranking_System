# ---------------------------------------------------------------
# TIME-AWARE DATASET PIPELINE (FEATURES ONLY — NO MODELING)
#
# Train for week T:    features from history ≤ (T-1), label from week T
# Validate for week V: features from history ≤ (V-1), label from week V
# Serve for week S:    features from history ≤ (S-1), no label (score & rank later)
# ----------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    RobustScaler,
    FunctionTransformer,
    MultiLabelBinarizer,
)

# --------------------------
# Config & small utilities
# --------------------------

@dataclass(frozen=True)
class DatasetConfig:
    user_id: str = "user_id"
    trainer_id: str = "trainer_id"
    week_id: str = "week_id"
    label_col: str = "label"

    # minimally required columns
    user_required: Tuple[str, ...] = ("user_id", "age", "gender", "goal", "fitness_level")
    trainer_required_min: Tuple[str, ...] = ("trainer_id", "specialities")
    interaction_required: Tuple[str, ...] = (
        "user_id", "trainer_id", "week_id",
        "impressions", "clicks", "video_views", "avg_watch_time", "likes",
        "label"
    )

CFG = DatasetConfig()


def _ensure_columns(df: pd.DataFrame, required: Tuple[str, ...], name: str) -> None:
    """Raise if any required column is missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _safe_div(a: float, b: float) -> float:
    """Numerically safe division a / b with 0 guard."""
    return float(a) / float(b) if b not in (None, 0, 0.0) else 0.0


# -------------------------------------------------
# Core: aggregation (weekly) & recency-weighted FE
# -------------------------------------------------

def _weekly_aggregate(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw logs into (user_id, trainer_id, week_id) rows and derive base rates.
    The 'label' for the target week is read from this table (max across events).
    """
    _ensure_columns(interactions, CFG.interaction_required, "interactions")
    df = interactions[list(CFG.interaction_required)].copy()

    wk = (
        df.groupby([CFG.user_id, CFG.trainer_id, CFG.week_id], as_index=False)
          .agg(
              impressions=("impressions", "sum"),
              clicks=("clicks", "sum"),
              video_views=("video_views", "sum"),
              likes=("likes", "sum"),
              avg_watch_time=("avg_watch_time", "mean"),
              label=(CFG.label_col, "max"),
          )
    )
    wk["ctr"]       = wk.apply(lambda r: _safe_div(r["clicks"], r["impressions"]), axis=1)
    wk["view_rate"] = wk.apply(lambda r: _safe_div(r["video_views"], r["impressions"]), axis=1)
    wk["like_rate"] = wk.apply(lambda r: _safe_div(r["likes"], max(r["video_views"], 1)), axis=1)
    for c in ("ctr", "view_rate", "like_rate", "avg_watch_time"):
        wk[c] = wk[c].fillna(0.0)
    return wk


def _recency_aggregate(
    weekly_df: pd.DataFrame,
    history_end_week: int,
    min_history_week: Optional[int] = None,
    decay: float = 0.8,
) -> pd.DataFrame:
    """
    Build recency-weighted aggregates for each (user, trainer) from history ≤ history_end_week.
    Optionally restrict the window to [min_history_week, history_end_week].
    Outputs:
      imp_rw, clk_rw, vv_rw, like_rw  (decayed sums)
      awt_rw                           (decayed mean)
      ctr_rw, vr_rw, lr_rw             (decayed rates)
    """
    if min_history_week is None:
        hist = weekly_df[weekly_df[CFG.week_id] <= history_end_week].copy()
    else:
        hist = weekly_df[
            (weekly_df[CFG.week_id] >= min_history_week) &
            (weekly_df[CFG.week_id] <= history_end_week)
        ].copy()

    if hist.empty:
        # No history: return zero features so the pipeline still works (cold start)
        base = weekly_df[[CFG.user_id, CFG.trainer_id]].drop_duplicates().copy()
        for c in ["imp_rw","clk_rw","vv_rw","like_rw","awt_rw","ctr_rw","vr_rw","lr_rw"]:
            base[c] = 0.0
        return base

    hist["w"] = decay ** (history_end_week - hist[CFG.week_id])  # more recent → higher weight

    g = hist.groupby([CFG.user_id, CFG.trainer_id])
    out = g.apply(lambda d: pd.Series({
        "imp_rw":  (d["impressions"]    * d["w"]).sum(),
        "clk_rw":  (d["clicks"]         * d["w"]).sum(),
        "vv_rw":   (d["video_views"]    * d["w"]).sum(),
        "like_rw": (d["likes"]          * d["w"]).sum(),
        "awt_rw":  (d["avg_watch_time"] * d["w"]).sum() / max(d["w"].sum(), 1e-9),
    })).reset_index()

    out["ctr_rw"] = out.apply(lambda r: _safe_div(r["clk_rw"],  r["imp_rw"]), axis=1)
    out["vr_rw"]  = out.apply(lambda r: _safe_div(r["vv_rw"],   r["imp_rw"]), axis=1)
    out["lr_rw"]  = out.apply(lambda r: _safe_div(r["like_rw"], max(r["vv_rw"], 1)), axis=1)
    return out


# -----------------------------------------
# Trainers: specialities → multi-hot (sp__*)
# -----------------------------------------

def _expand_trainer_specialities(trainers: pd.DataFrame) -> tuple[pd.DataFrame, List[str], List[str]]:
    """
    Convert 'specialities' free text to multi-hot (sp__*) and keep all trainer columns.
    Returns: (expanded_df, speciality_cols, trainer_numeric_cols)
    """
    _ensure_columns(trainers, CFG.trainer_required_min, "trainers")
    tr = trainers.copy()

    tokens = (
        tr["specialities"].astype(str)
        .str.lower()
        .str.replace(r"[;|]", ",", regex=True)
        .str.split(",")
        .apply(lambda lst: [t.strip() for t in lst if t and t.strip()])
    )

    mlb = MultiLabelBinarizer()
    sp_matrix = mlb.fit_transform(tokens)
    sp_cols = [f"sp__{t}" for t in mlb.classes_]
    sp_df = pd.DataFrame(sp_matrix, columns=sp_cols, index=tr.index)

    tr_out = pd.concat([tr.drop(columns=["specialities"]), sp_df], axis=1)

    trainer_numeric_cols = [
        c for c in tr_out.select_dtypes(include=[np.number]).columns
        if c not in (CFG.trainer_id,) and c not in sp_cols
    ]
    return tr_out, sp_cols, trainer_numeric_cols


# ------------------------------------------
# Preprocessor (ColumnTransformer by feature)
# ------------------------------------------

def _build_preprocessor_time_aware(X: pd.DataFrame, trainer_numeric_cols: List[str]) -> Pipeline:
    """
    Build the preprocessing pipeline for HISTORY features (+ user/trainer attributes).
    Scaler choice per feature type (robust to outliers where needed).
    """
    # 1) Counts (heavy-tailed) → log1p + StandardScaler
    count_cols = [c for c in ["imp_rw","clk_rw","vv_rw","like_rw"] if c in X.columns]
    # 2) Rates / bounded + cross flags → passthrough
    rate_cols  = [c for c in ["awt_rw","ctr_rw","vr_rw","lr_rw","goal_match"] if c in X.columns]
    # 3) Small continuous
    small_cols = [c for c in ["age"] if c in X.columns]
    # 4) Trainer numeric (outlier-prone)
    robust_cols = [c for c in trainer_numeric_cols if c in X.columns]
    # 5) Specialities (multi-hot)
    sp_cols_in = [c for c in X.columns if c.startswith("sp__")]
    # 6) Categoricals
    cat_cols   = [c for c in ["goal","fitness_level","gender"] if c in X.columns]

    log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    counts_tf = Pipeline([("log1p", log1p), ("std", StandardScaler(with_mean=False))])
    rates_tf  = "passthrough"
    small_tf  = StandardScaler()
    robust_tf = RobustScaler(with_centering=False)
    sp_tf     = "passthrough"
    cat_tf    = OneHotEncoder(handle_unknown="ignore")

    pre = ColumnTransformer(
        transformers=[
            ("counts", counts_tf, count_cols),
            ("rates",  rates_tf,  rate_cols),
            ("small",  small_tf,  small_cols),
            ("robust", robust_tf, robust_cols),
            ("sp",     sp_tf,     sp_cols_in),
            ("cat",    cat_tf,    cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.2
    )
    return Pipeline([("pre", pre)])


# --------------------------------------------------
# Train/Validation (time-aware, labels)
# --------------------------------------------------

def build_time_aware_dataset(
    users_df: pd.DataFrame,
    trainers_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    *,
    target_week: int,
    min_history_week: Optional[int] = None,
    decay: float = 0.8,
) -> tuple[pd.DataFrame, pd.Series, Pipeline]:
    """
    TRAIN/VALIDATION BUILDER (features + label)
    - Features: recency aggregates from history ≤ (target_week-1)
    - Label:    from target_week (0/1)
    Returns:
      X_raw (DataFrame) — human-readable features
      y     (Series)    — 0/1 labels for target_week
      preproc (Pipeline) — ColumnTransformer to transform X_raw for modeling
    """
    # 1) Aggregate interactions to weekly
    weekly = _weekly_aggregate(interactions_df)

    # 2) Extract labels for the target week (no leakage)
    y_frame = weekly[weekly[CFG.week_id] == target_week][[CFG.user_id, CFG.trainer_id, CFG.label_col]].copy()
    if y_frame.empty:
        raise ValueError(f"No interactions (labels) for target_week={target_week}")

    # 3) Build history features up to (target_week - 1)
    hist = _recency_aggregate(weekly, history_end_week=target_week - 1, min_history_week=min_history_week, decay=decay)

    # 4) Trainer specialities & numeric columns
    trainers_expanded, sp_cols, trainer_numeric_cols = _expand_trainer_specialities(trainers_df)

    # 5) Join: labels ⨝ history ⨝ users ⨝ trainers
    out = (
        y_frame
        .merge(hist, on=[CFG.user_id, CFG.trainer_id], how="left")
        .merge(users_df[list(CFG.user_required)], on=CFG.user_id, how="left")
        .merge(trainers_expanded, on=CFG.trainer_id, how="left")
    )

    # 6) Fill missing history (cold-start pairs → zeros)
    for c in ["imp_rw","clk_rw","vv_rw","like_rw","awt_rw","ctr_rw","vr_rw","lr_rw"]:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)

    # 7) Cross feature: goal_match
    out["goal_token"] = out["goal"].astype(str).str.replace("_", " ", regex=False).str.lower()
    out["goal_match"] = out.apply(
        lambda r: int(f"sp__{r['goal_token']}" in out.columns and r.get(f"sp__{r['goal_token']}", 0) == 1),
        axis=1
    )
    out = out.drop(columns=["goal_token"])

    # 8) Light fills
    out["age"] = out["age"].fillna(out["age"].median())

    # 9) Split X / y and build preprocessor
    y = out[CFG.label_col].astype(int)
    X = out.drop(columns=[CFG.label_col])

    preproc = _build_preprocessor_time_aware(X, trainer_numeric_cols)
    return X, y, preproc


# ---------------------------------------------------------
# Serving candidates (time-aware, no labels)
# ---------------------------------------------------------

def build_candidates_time_aware(
    users_df: pd.DataFrame,
    trainers_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    *,
    serve_week: int,
    min_history_week: Optional[int] = None,
    decay: float = 0.8,
) -> pd.DataFrame:
    """
    SERVING BUILDER (features only; NO labels)
    - Build ALL (user, trainer) candidate rows using history ≤ (serve_week-1).
    - Output schema aligns with build_time_aware_dataset() features (so the same preproc can transform it).
    """
    weekly = _weekly_aggregate(interactions_df)
    hist = _recency_aggregate(weekly, history_end_week=serve_week - 1, min_history_week=min_history_week, decay=decay)

    # Cross-join users × trainers (expanded)
    users_small = users_df[list(CFG.user_required)].copy()
    trainers_expanded, sp_cols, trainer_numeric_cols = _expand_trainer_specialities(trainers_df)
    pairs = users_small.assign(_k=1).merge(trainers_expanded.assign(_k=1), on="_k").drop(columns="_k")

    # Join recency history (cold-start → zeros)
    X = pairs.merge(hist, on=[CFG.user_id, CFG.trainer_id], how="left")
    for c in ["imp_rw","clk_rw","vv_rw","like_rw","awt_rw","ctr_rw","vr_rw","lr_rw"]:
        if c in X.columns:
            X[c] = X[c].fillna(0.0)

    # goal_match
    X["goal_token"] = X["goal"].astype(str).str.replace("_", " ", regex=False).str.lower()
    X["goal_match"] = X.apply(
        lambda r: int(f"sp__{r['goal_token']}" in X.columns and r.get(f"sp__{r['goal_token']}", 0) == 1),
        axis=1
    )
    X = X.drop(columns=["goal_token"])

    # fills
    X["age"] = X["age"].fillna(X["age"].median())

    return X
