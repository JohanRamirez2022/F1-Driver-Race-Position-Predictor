# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Jupyter notebook-based ML pipeline that predicts F1 race finishing positions using historical race and qualifying data (2020–2025). The model outputs per-race driver finish position rankings.

There are two active notebooks:
- **`F1.ipynb`** — original pipeline (features based on raw position rolling averages)
- **`F1V2.ipynb`** — improved pipeline (features based on positions gained/lost relative to grid; 2016–2024 training data)

## Running the Project

Run either notebook cell-by-cell in order — cells are stateful and depend on prior cells.

```bash
jupyter notebook F1V2.ipynb
# or
jupyter lab F1V2.ipynb
```

Dependencies: `fastf1`, `pandas`, `numpy`, `scikit-learn`, `scipy`, `seaborn`, `matplotlib`

**Python environment:** Always use the `F1` conda environment when running any Python code. Use `conda run -n F1 python <script>` for scripts. For inline commands, write to a temp file first since `conda run` does not support multiline `-c` arguments on Windows.

## Data

- `data/f1_{year}.csv` — per-season raw data (2020–2025), one row per driver per race
- `data/f1_2020_2025.csv` — combined dataset across all seasons
- `f1_2025_predictions.csv` — model output: predicted finish ranks for 2025 races

## F1V2 Notebook Architecture (cell numbers as shown in Jupyter)

| Code Cell | nb_idx | Description |
|-----------|--------|-------------|
| #1  | 1  | Imports |
| #2  | 3  | fastf1 data fetch helpers |
| #3  | 4  | Combine seasons → `f1_2016_2025.csv` (drops legacy FinishPosition) |
| #4  | 6  | Load `f1_2016_2025.csv` |
| #5  | 8  | Raw EDA correlation |
| #6  | 9  | Grid vs finish boxplot |
| #7  | 10 | Per-year team stats |
| #8  | 13 | `positions_delta`, `driver_delta_rolling_5`, `driver_rolling_std_5` |
| #9  | 15 | `team_delta_rolling_5` |
| #10 | 18 | `best_q_time`, `quali_delta_to_teammate` |
| #11 | 20 | `track_overtaking_factor`, `driver_track_delta_avg` |
| #12 | 22 | Rename, TimeDelta, encode, drop, NaN fills |
| #13 | 24 | Feature correlation heatmap |
| #14–17 | 26–29 | EDA: track variance, street vs permanent, track history signal, grid baseline |
| #18 | 31 | Train/eval split (≤2024 train, 2025 eval) |
| #19 | 32 | RF training + hyperparameter search |
| #20 | 34 | RF evaluation metrics |
| #21–22 | 35–36 | RF scatter + Tau bar chart |
| #23 | 37 | Model vs grid-only metric comparison |
| #24 | 38 | RF feature importances |
| #25 | 41 | XGBRanker training |
| #26 | 43 | XGB evaluation metrics |
| #27 | 45 | XGB feature importances |
| #28 | 46 | Winner accuracy (RF vs XGB vs grid baseline) |
| #29 | 47 | Podium accuracy (RF vs XGB vs grid baseline) |

## Key Design Decisions

- **Target variable**: `FinishPosition` (numeric finish order, 1 = winner)
- **Train/eval split**: Season ≤ 2024 for training, Season == 2025 for validation — no random splitting to avoid temporal leakage
- **Rolling features use `.shift(1)`** before `.rolling()` to prevent look-ahead leakage
- **DNFs**: `TimeDelta` is filled with 300 (seconds) as a proxy for non-finishers; NaN fill value is computed from training data only (≤ 2024) to avoid leakage
- **Ranking**: `method='first'` used in `.rank()` to guarantee unique positions per race
- **Evaluation metrics**: Kendall's Tau (overall ranking quality), MAE (average positions off), AP@5 (top-5 ordering quality), Top-5 Precision (top-5 driver identification)
- **F1V2 features** (delta-based, low GridPosition correlation):
  - `positions_delta = GridPosition - RacePosition` — intermediate, dropped before model
  - `driver_delta_rolling_5` — rolling mean of positions gained over last 5 races
  - `driver_rolling_std_5` — rolling std of finish position (consistency)
  - `team_delta_rolling_5` — rolling mean of team positions gained
  - `track_overtaking_factor` — per-circuit std of positions delta (how much shuffling happens)
  - `driver_track_delta_avg` — per-driver per-circuit mean delta (does Hamilton gain places at Silverstone?)
  - `quali_delta_to_teammate` — kept (low GridPosition correlation)
- **F1 original features** (raw position rolling averages — high GridPosition correlation, worse baseline):
  - `driver_rolling_avg_5`, `team_rolling_avg_5`, `driver_last_finish`, `driver_track_history_avg`
- Dropped after EDA: `driver_rolling_avg_timedelta_3`, `team_rolling_avg_3`, `Q1_seconds`, `Q2_seconds`, `Q3_seconds`, `best_q_time`

## Report Convention

Each time `generate_report.py` is run, the generated `report_YYYY-MM-DD_HH-MM.md` should include a summary of what changed in the model since the previous run (e.g. hyperparameter changes, new features, leakage fixes). Add this as a "Model Changes" section in the report writing block so future runs are self-documenting.

## Keeping the Notebook and Script in Sync

`generate_report.py` is a standalone executable version of the notebook pipeline. **Any change made to the pipeline, features, or evaluation metrics must be applied to both `F1.ipynb` and `generate_report.py`.** This includes:
- Feature engineering changes
- New or removed evaluation metrics
- Leakage fixes
- Model parameter changes
