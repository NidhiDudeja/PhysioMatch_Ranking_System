from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
import joblib


try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# import time-aware feature builders
from MRDS import (
    build_time_aware_dataset,      # features+label for a target week (history ≤ week-1)
    build_candidates_time_aware,   # features only for serving (history ≤ serve_week-1)
)

# -----------------------------------------------------------------------------
# Ranking metrics (evaluate what users actually see: the ranked list)
# -----------------------------------------------------------------------------
def precision_at_k(y_sorted: np.ndarray, k: int) -> float:
    if k <= 0:
        return 0.0
    return float(np.sum(y_sorted[:k])) / float(k)

def dcg_at_k(y_sorted: np.ndarray, k: int) -> float:
    k = min(k, len(y_sorted))
    gains = (2 ** y_sorted[:k] - 1)  # with binary labels, this is y_sorted itself
    discounts = 1 / np.log2(np.arange(2, k + 2))
    return float(np.sum(gains * discounts))

def ndcg_at_k(y_sorted: np.ndarray, k: int) -> float:
    ideal = np.sort(y_sorted)[::-1]
    dcg = dcg_at_k(y_sorted, k)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0

def per_user_ranking_metrics(
    y_true: np.ndarray, y_score: np.ndarray, user_ids: np.ndarray, ks=(3, 5, 10)
) -> dict[str, float]:
    """
    For each user, sort that user's candidates by score desc, then compute P@K and NDCG@K.
    Return the mean across users — this mirrors the actual UX (lists are per user).
    """
    df = pd.DataFrame({"user_id": user_ids, "y": y_true.astype(int), "s": y_score})
    out = {f"p@{k}": [] for k in ks} | {f"ndcg@{k}": [] for k in ks}
    for _, g in df.groupby("user_id"):
        g = g.sort_values("s", ascending=False)
        y_sorted = g["y"].values
        for k in ks:
            out[f"p@{k}"].append(precision_at_k(y_sorted, k))
            out[f"ndcg@{k}"].append(ndcg_at_k(y_sorted, k))
    # average across users
    return {m: (float(np.mean(v)) if len(v) else 0.0) for m, v in out.items()}


# ---- Candidate filtering utilities (goal_match + backfill) ----

def _trainer_popularity(interactions_df: pd.DataFrame, serve_week: int, decay: float = 0.8) -> pd.Series:
    """
    Global popularity prior per trainer using recency-weighted impressions up to serve_week-1.
    Returns Series(index=trainer_id, values in [0,1]).
    """
    req = ["trainer_id", "week_id", "impressions"]
    w = interactions_df[req].copy()
    w = w[w["week_id"] <= serve_week - 1]
    if w.empty:
        return pd.Series(dtype=float)

    w["w"] = decay ** ((serve_week - 1) - w["week_id"])
    w["imp_w"] = w["impressions"] * w["w"]
    pop = w.groupby("trainer_id", as_index=True)["imp_w"].sum().sort_values(ascending=False)

    # Normalize to [0,1] for neatness
    pop = (pop - pop.min()) / (pop.max() - pop.min() + 1e-9)
    return pop


def filter_candidates_by_goal_with_backfill(
    Xcand_raw: pd.DataFrame,
    interactions_df: pd.DataFrame,
    serve_week: int,
    *,
    min_candidates_per_user: int = 30,
    decay: float = 0.8,
) -> pd.DataFrame:
    """
    Keep only goal-matched candidates per user; if a user has < min_candidates_per_user,
    backfill with globally popular trainers (they don't already have).
    """
    assert "goal_match" in Xcand_raw.columns, "Goal-match must exist (build_candidates_time_aware adds it)."
    assert {"user_id","trainer_id"}.issubset(Xcand_raw.columns)

    pop = _trainer_popularity(interactions_df, serve_week=serve_week, decay=decay).rename("pop_score").reset_index()

    matched = Xcand_raw[Xcand_raw["goal_match"] == 1].copy()
    nonmatched = (Xcand_raw[Xcand_raw["goal_match"] == 0]
                  .merge(pop, on="trainer_id", how="left")
                  .assign(pop_score=lambda d: d["pop_score"].fillna(0.0)))

    out = []

    # Users with at least one matched trainer
    for uid, g in matched.groupby("user_id"):
        need = max(0, min_candidates_per_user - len(g))
        if need > 0:
            pool = nonmatched[nonmatched["user_id"] == uid]
            back = (pool.sort_values("pop_score", ascending=False)
                        .drop(columns=["pop_score"])
                        .head(need))
            out.append(pd.concat([g, back], ignore_index=True))
        else:
            out.append(g)

    # Users with zero matches → all backfill
    unmatched_users = set(Xcand_raw["user_id"].unique()) - set(matched["user_id"].unique())
    if unmatched_users:
        pool = nonmatched[nonmatched["user_id"].isin(unmatched_users)]
        for uid, g in pool.groupby("user_id"):
            back = (g.sort_values("pop_score", ascending=False)
                      .drop(columns=["pop_score"])
                      .head(min_candidates_per_user))
            out.append(back)

    filtered = (pd.concat(out, ignore_index=True)
                  .drop_duplicates(subset=["user_id","trainer_id"])
                  .reset_index(drop=True))
    return filtered

# -----------------------------------------------------------------------------
# Model zoo: start simple then include boosted trees
# -----------------------------------------------------------------------------
def make_models() -> dict[str, object]:
    models = {
        # Balanced class weights help if positives are rare; increase max_iter for convergence.
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        # Strong non-linear baseline; n_estimators moderate to avoid overfit initially.
        "rf": RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1),
    }
    if HAS_LGBM:
        models["lgbm"] = LGBMClassifier(
            n_estimators=800, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, random_state=42
        )
    if HAS_XGB:
        models["xgb"] = XGBClassifier(
            n_estimators=900, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            eval_metric="logloss", tree_method="hist",
            random_state=42, n_jobs=-1
        )
    return models

# -----------------------------------------------------------------------------
# Train on week T (history ≤ T-1), validate on week V (history ≤ V-1)
# -----------------------------------------------------------------------------
def train_and_validate(
    users_df: pd.DataFrame,
    trainers_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    train_week: int,
    val_week: int,
) -> tuple[object, object, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      best_model_fitted, preproc_fitted, leaderboard_df, val_frame_with_scores
    """
    # --- Build train set (labels from train_week, features from ≤ train_week-1)
    Xtr_raw, ytr, preproc = build_time_aware_dataset(
        users_df, trainers_df, interactions_df,
        target_week=train_week, min_history_week=None, decay=0.8
    )

    # --- Build validation set (labels from val_week, features from ≤ val_week-1)
    Xva_raw, yva, _ = build_time_aware_dataset(
        users_df, trainers_df, interactions_df,
        target_week=val_week, min_history_week=None, decay=0.8
    )

    # --- Fit preprocessors on TRAIN ONLY (avoid leakage), transform both
    Xtr = preproc.fit_transform(Xtr_raw)
    Xva = preproc.transform(Xva_raw)

    va_user_ids = Xva_raw["user_id"].values  # needed for per-user ranking metrics

    # --- Train several models and compare AUC + ranking metrics
    models = make_models()
    rows = []
    fitted = {}
    val_frames = []  # keep per-model scores if you want to inspect later

    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        p_tr = mdl.predict_proba(Xtr)[:, 1]
        p_va = mdl.predict_proba(Xva)[:, 1]

        auc_tr = roc_auc_score(ytr, p_tr)
        auc_va = roc_auc_score(yva, p_va)
        rank = per_user_ranking_metrics(y_true=yva.values, y_score=p_va, user_ids=va_user_ids, ks=(3,5,10))

        rows.append({"model": name, "auc_tr": auc_tr, "auc_va": auc_va, **rank})
        fitted[name] = mdl

        tmp = Xva_raw[["user_id","trainer_id"]].copy()
        tmp["y"] = yva.values
        tmp[f"score_{name}"] = p_va
        val_frames.append(tmp)

    leaderboard = pd.DataFrame(rows).sort_values(["ndcg@5","auc_va"], ascending=[False, False])
    best_name = leaderboard.iloc[0]["model"]
    best_model = fitted[best_name]

    # --- bundle validation scores for the winner 
    val_join = Xva_raw[["user_id","trainer_id"]].copy()
    val_join["y"] = yva.values
    val_join["score"] = [df[f"score_{best_name}"] for df in val_frames if f"score_{best_name}" in df.columns][0]

    print("\n=== Validation comparison (train on week", train_week, "→ validate on week", val_week, ") ===")
    print(leaderboard.to_string(index=False))
    print(f"\nSelected model: {best_name}")

    return best_model, preproc, leaderboard, val_join





# -----------------------------------------------------------------------------
#  Refit on (train + val) to use more labels before serving next week
# -----------------------------------------------------------------------------
def refit_on_two_weeks(
    users_df: pd.DataFrame,
    trainers_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    winner_name: str,
    train_week: int,
    val_week: int,
):
    """
    Stack week=train_week and week=val_week training rows, fit a fresh preprocessor on the stack,
    then train a fresh model of the chosen type on the stacked data.
    Use this model to serve week=val_week+1 (features will be history ≤ val_week).
    """
    # Build two labeled datasets
    X1_raw, y1, _ = build_time_aware_dataset(users_df, trainers_df, interactions_df, target_week=train_week)
    X2_raw, y2, _ = build_time_aware_dataset(users_df, trainers_df, interactions_df, target_week=val_week)
    Xs_raw = pd.concat([X1_raw, X2_raw], ignore_index=True)
    ys = pd.concat([y1, y2], ignore_index=True)

    
    from MRDS import _build_preprocessor_time_aware  # internal helper; OK to reuse
    # Identify trainer numeric columns heuristically (columns that existed as numerics in X1/X2 paths were already transformed,
    # but here at this stage we only have Xs_raw. We can infer robust cols by pattern or pass an empty list safely.)
    trainer_numeric_cols = [c for c in Xs_raw.columns if c not in ("user_id","trainer_id","goal","fitness_level","gender","name") and not c.startswith("sp__") and c not in
                            ["goal_match","imp_rw","clk_rw","vv_rw","like_rw","awt_rw","ctr_rw","vr_rw","lr_rw","week_id"]]
    pre = _build_preprocessor_time_aware(Xs_raw, trainer_numeric_cols)
    Xs = pre.fit_transform(Xs_raw)

    # Recreate winner model with the same hyperparams
    mdl = make_models()[winner_name]
    mdl.fit(Xs, ys)
    return mdl, pre

# -----------------------------------------------------------------------------
# Serving: score ALL user×trainer candidates for week S and return top-K/user
# -----------------------------------------------------------------------------
def serve_topk(
    users_df: pd.DataFrame,
    trainers_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    preproc_fitted,
    model_fitted,
    serve_week: int,
    topk: int = 5,
    *,
    min_candidates_per_user: int = 30,   # let the model re-rank a reasonable pool
    decay: float = 0.8                   # for popularity backfill
) -> pd.DataFrame:
    """
    Build candidates from history ≤ serve_week-1, filter to goal_match==1 with backfill,
    transform with the same preprocessor, score, and return top-K trainers per user.
    """
    # 1) Build raw candidates (full cross-join + history features + goal_match)
    Xcand_raw = build_candidates_time_aware(
        users_df, trainers_df, interactions_df,
        serve_week=serve_week, min_history_week=None, decay=decay
    )

    # 2) Filter to goal-matched + backfill (so every user still has a pool to re-rank)
    Xcand_raw = filter_candidates_by_goal_with_backfill(
        Xcand_raw, interactions_df, serve_week,
        min_candidates_per_user=max(min_candidates_per_user, topk),
        decay=decay
    )

    # 3) Transform & score
    Xcand = preproc_fitted.transform(Xcand_raw)
    scores = model_fitted.predict_proba(Xcand)[:, 1]

    scored = Xcand_raw[["user_id","trainer_id"]].copy()
    scored["score"] = scores

    # 4) Top-K per user
    top_rows = []
    for uid, g in scored.groupby("user_id"):
        g = g.sort_values("score", ascending=False).head(topk)
        top_rows.append(g)

    topk_df = (pd.concat(top_rows, ignore_index=True)
                 .sort_values(["user_id","score"], ascending=[True, False])
                 .reset_index(drop=True))
    return topk_df


# -----------------------------------------------------------------------------
#  Model understanding: permutation importance on validation
# -----------------------------------------------------------------------------
def permutation_importance_on_val(model, preproc, Xva_raw: pd.DataFrame, yva: pd.Series, n_repeats: int = 5):
    """
    Quick, model-agnostic importance. Warning: with a ColumnTransformer, importances are
    over *transformed* columns. This still gives a useful signal on which *groups* matter.
    """
    Xva = preproc.transform(Xva_raw)
    r = permutation_importance(model, Xva, yva, n_repeats=n_repeats, random_state=42, n_jobs=-1)
    # The feature names after transformation can be long; we print top 20 indices by importance:
    imp = pd.DataFrame({"idx": np.arange(len(r.importances_mean)),
                        "mean_imp": r.importances_mean,
                        "std_imp": r.importances_std}).sort_values("mean_imp", ascending=False)
    return imp.head(20)
