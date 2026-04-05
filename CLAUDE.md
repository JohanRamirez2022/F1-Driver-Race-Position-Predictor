# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Jupyter notebook-based ML pipeline that predicts F1 race finishing positions using historical race and qualifying data (2020–2025). The model outputs per-race driver finish position rankings.

## Running the Project

All work happens inside `F1.ipynb`. Run it cell-by-cell in order — cells are stateful and depend on prior cells.

```bash
jupyter notebook F1.ipynb
# or
jupyter lab F1.ipynb
```

Dependencies: `fastf1`, `pandas`, `numpy`, `scikit-learn`, `scipy`, `seaborn`, `matplotlib`

**Python environment:** Always use the `F1` conda environment when running any Python code. Use `conda run -n F1 python <script>` for scripts. For inline commands, write to a temp file first since `conda run` does not support multiline `-c` arguments on Windows.

## Data

- `data/f1_{year}.csv` — per-season raw data (2020–2025), one row per driver per race
- `data/f1_2020_2025.csv` — combined dataset across all seasons
- `AustralianGpP.csv` — 2026 Australian GP qualifying/results fetched via fastf1
- `f1_2025_predictions.csv` — model output: predicted finish ranks for 2025 races

Raw data is fetched using `fastf1.get_session()` with a resume pattern (skips already-fetched rounds). Data loading is commented out in Cell 2 since CSVs already exist.

## Notebook Architecture

The notebook flows in this order:

1. **Data fetch** (Cell 1–2): Pull 2026 Australian GP via fastf1; commented-out bulk loader for historical seasons.
2. **Combine seasons** (Cell 3): Merge `f1_{year}.csv` files into `f1_2020_2025.csv`.
3. **EDA & feature engineering** (Cells 4–11):
   - Drop leakage columns: `Points`, `TimeDelta`, `Podium`, `Status`, `Time`
   - Engineer rolling averages (3- and 5-race windows) for driver position, team position, and time delta
   - Add `driver_last_finish`, `driver_rolling_std_5`, `quali_delta_to_teammate`
   - One-hot encode `TeamName`, `FullName`, `EventName`
4. **Model training** (Cells 12–13): `RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=42)` trained on 2020–2024, evaluated on 2025
5. **Evaluation** (Cell 14): Per-race Spearman rank correlation between predicted and actual finish positions

## Key Design Decisions

- **Target variable**: `FinishPosition` (numeric finish order, 1 = winner)
- **Train/eval split**: Season ≤ 2024 for training, Season == 2025 for validation — no random splitting to avoid temporal leakage
- **Rolling features use `.shift(1)`** before `.rolling()` to prevent look-ahead leakage
- **DNFs**: `TimeDelta` is filled with 300 (seconds) as a proxy for non-finishers; NaN fill value is computed from training data only (≤ 2024) to avoid leakage
- **Ranking**: `method='first'` used in `.rank()` to guarantee unique positions per race
- **Evaluation metrics**: Kendall's Tau (overall ranking quality), MAE (average positions off), AP@5 (top-5 ordering quality), Top-5 Precision (top-5 driver identification)
- Dropped after EDA: `driver_rolling_avg_timedelta_3`, `team_rolling_avg_3`, `Q1_seconds`, `Q2_seconds`, `Q3_seconds`, `best_q_time`

## Report Convention

Each time `generate_report.py` is run, the generated `report_YYYY-MM-DD_HH-MM.md` should include a summary of what changed in the model since the previous run (e.g. hyperparameter changes, new features, leakage fixes). Add this as a "Model Changes" section in the report writing block so future runs are self-documenting.

## Keeping the Notebook and Script in Sync

`generate_report.py` is a standalone executable version of the notebook pipeline. **Any change made to the pipeline, features, or evaluation metrics must be applied to both `F1.ipynb` and `generate_report.py`.** This includes:
- Feature engineering changes
- New or removed evaluation metrics
- Leakage fixes
- Model parameter changes
