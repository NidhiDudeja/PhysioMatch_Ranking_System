# Trainer Recommender - Physio Ranking Engine

## 1. What problem are we solving?

We recommend the best trainers/physiotherapists to each user based on their current goal (e.g., weight loss) and their weekly engagement (clicks, video views, watch time, likes).  
The ranked list adapts weekly and works even with small data, handling cold-starts using a popularity backfill.

---

## 2. How the algorithm works (at a glance)

### Step A - Build features (no leakage)
- For target week T, only use weeks ≤ T−1 to build features.
- For each (user, trainer) pair, compute:
  - Sums: impressions, clicks, video_views, likes
  - Mean: avg_watch_time (0–1)
  - Rates: ctr = clicks/impressions, view_rate, like_rate
  - Recency weighting: recent weeks count more than older ones.
  - Trainer specialities → multi-hot flags (e.g., sp__weight loss, sp__rehab)
  - goal_match: 1 if the trainer’s speciality matches the user’s goal.

### Step B - Train & validate
- Example: train on week 3 (features ≤ 2), validate on week 4 (features ≤ 3).
- Try several models: Logistic Regression, Random Forest, optionally LightGBM/XGBoost.
- Pick the best by ranking metrics: Precision@K and NDCG@K (averaged over users).

### Step C - Serve (make weekly recommendations)
- For serve week S (features ≤ S−1):
  - Build all (user, trainer) candidates.
  - Filter to goal_match==1. If too few, backfill with recently popular trainers so everyone gets K results.
  - Score candidates with the trained model and sort.
  - Return Top-K trainers per user.

---

## 3. Project structure

```
├─ Data Generation.ipynb          # Creates realistic synthetic users & interactions
├─ MRDS.py                        # Builds model ready datasets (time-aware, no leakage)
├─ Model_Training_and_Serving.py  # Model training, validation, candidate filter, serving
├─ main.ipynb                     # End to end demo: load → train → validate → serve
└─ requirements.txt               # Dependencies
```

---

## 4. What each file does

**Data Generation.ipynb**  
Creates synthetic data: `users.csv` and `all_weeks_interactions.csv` using an Instagram-like funnel (impressions → clicks → views → likes). Goal-matched content is more likely to be clicked/watched.

**MRDS.py (Model-Ready Dataset, time-aware)**
- `_weekly_aggregate`: roll up events per (user, trainer, week) + compute rates.
- `_recency_aggregate`: apply decay so recent weeks count more.
- `_expand_trainer_specialities`: turn free-text specialities into sp__* columns; keep numeric trainer features.
- `build_time_aware_dataset(target_week)`: returns (X_raw, y, preprocessor) for training/validation.
- `build_candidates_time_aware(serve_week)`: returns features only (no labels) for serving.

**Model_Training_and_Serving.py**
- Metrics: `precision_at_k`, `ndcg_at_k`, `per_user_ranking_metrics`.
- Models: Logistic Regression, Random Forest (LightGBM/XGBoost if installed).
- `train_and_validate(train_week, val_week)`: trains several models and prints a leaderboard.
- Candidate filter utilities: `_trainer_popularity` (recency-weighted), `filter_candidates_by_goal_with_backfill` (keep goal-matched + fill up with popular).
- `serve_topk(serve_week, topk)`: production step — build candidates → filter → score → Top-K per user.

**main.ipynb**  
Example notebook that runs the whole flow end-to-end.

---

## 5. Install & run

### A) Install
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### B) Generate data (or use provided CSV/XLSX)
Open `Data Generation.ipynb` and run all cells. It creates:
- `users.csv`
- `all_weeks_interactions.csv`

### C) Train & validate
```python
import pandas as pd
from Model_Training_and_Serving import train_and_validate

users = pd.read_csv("users.csv")
trainers = pd.read_excel("Trainers_ Data.xlsx")
inter = pd.read_csv("all_weeks_interactions.csv")

best_model, preproc, leaderboard, val_join = train_and_validate(
    users, trainers, inter,
    train_week=3,   # features from weeks ≤ 2, labels from week 3
    val_week=4      # features from weeks ≤ 3, labels from week 4
)
print(leaderboard)  # choose model by NDCG@5 / P@5 (primary), AUC (diagnostic)
```

### D) Serve recommendations (what users actually see)
```python
from Model_Training_and_Serving import serve_topk

topk_df = serve_topk(
    users, trainers, inter,
    preproc_fitted=preproc, model_fitted=best_model,
    serve_week=5,            # features from weeks ≤ 4
    topk=5,                  # show 5 trainers per user
    min_candidates_per_user=30,  # size of candidate pool before ranking
    decay=0.8                # recency weighting for popularity backfill
)
topk_df.head()
```

---

## 6. Why this design works

- **Filter → Rank** is the standard approach in large recommenders: remove obvious mismatches, then let the model decide the order among relevant options.
- **No leakage:** for week T, features are built only from weeks ≤ T−1; week T is used only to read the label.
- **Adapts weekly:** new interactions automatically update features and rankings.
- **Cold-start safe:** popularity backfill guarantees each user still gets K suggestions.

---

## 7. Metrics we care about

- **Precision@K** - fraction of relevant trainers in each user’s top-K.
- **NDCG@K** - gives higher credit when relevant trainers appear earlier in the list.
- **AUC** - general diagnostic; useful but not directly what users see.

---

## 8. Tuning knobs (easy to change)

- `topk` - how many trainers to show (default 5).
- `min_candidates_per_user` - candidate pool size before ranking (e.g., 30–100).
- `decay` - how fast old weeks fade (e.g., 0.8 = faster, 0.9 = slower).
- `goal_match` - start simple; you can add synonyms/substring logic later for more recall.

---

## 9. Limitations & future work

- **Synthetic data:** realistic but not the real world.
- **Serve vs validation:** we validate on the broad pool; serving uses the goal filter. (You can also report a “serve-style” validation to mirror production.)
- **Exploration:** add a small random slice to avoid “filter bubbles”.
- **Smarter retrieval:** try embeddings for candidate generation; try pairwise/listwise ranking losses.

---

## Data Disclaimer

**All data in this project is synthetic and anonymized, generated for demonstration and testing purposes.**

---

Project by  - Nidhi Dudeja