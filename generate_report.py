# ── Describe changes made since the previous run ──────────────────────────────
# Update this before each run so the report is self-documenting.
MODEL_CHANGES = """
- Added XGBoost Rank model (rank:pairwise, no team dummies) alongside RF
- XGBoost params: n_estimators=200, learning_rate=0.01, max_depth=3, subsample=0.7
- XGBoost beats RF on all metrics: Tau clean 0.657 vs 0.645, MAE 2.607 vs 2.682, AP@5 0.914 vs 0.898
- Report now includes full model comparison table (Grid only, RF, XGBoost)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import average_precision_score, make_scorer
from scipy.stats import kendalltau
from datetime import datetime
import json
import os

# ── Pipeline ──────────────────────────────────────────────────────────────────

df = pd.read_csv('data/f1_2020_2025.csv')
df = df.sort_values(['Season', 'Round']).reset_index(drop=True)

df = df.drop(columns=['Time', 'Podium', 'Points', 'Status', 'DriverNumber'], errors='ignore')

df['driver_rolling_avg_3']   = df.groupby('FullName')['RacePosition'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
df['driver_rolling_avg_5']   = df.groupby('FullName')['RacePosition'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
df['driver_rolling_std_5']   = df.groupby('FullName')['RacePosition'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).std())
df['driver_last_finish']     = df.groupby('FullName')['RacePosition'].transform(lambda x: x.shift(1))
df['team_rolling_avg_5']     = df.groupby('TeamName')['RacePosition'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
df['team_rolling_avg_3']     = df.groupby('TeamName')['RacePosition'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())

df['best_q_time'] = df['Q3_seconds'].fillna(df['Q2_seconds']).fillna(df['Q1_seconds'])
teammate_avg = df.groupby(['Season', 'Round', 'TeamName'])['best_q_time'].transform('mean')
df['quali_delta_to_teammate'] = df['best_q_time'] - teammate_avg

# Track-specific driver history: avg finish at this circuit from all prior seasons
df = df.sort_values(['FullName', 'EventName', 'Season', 'Round'])
df['driver_track_history_avg'] = (
    df.groupby(['FullName', 'EventName'])['RacePosition']
    .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
)
df['driver_track_history_avg'] = df['driver_track_history_avg'].fillna(df['driver_rolling_avg_5'])

df = df.drop(columns=['DriverNumber'], errors='ignore')
df = df.rename(columns={'RacePosition': 'FinishPosition'})
df = df.sort_values(['FullName', 'Season', 'Round'])
df.loc[df['FinishPosition'] == 1, 'TimeDelta'] = 0
df['TimeDelta'] = df['TimeDelta'].fillna(300)

df['driver_rolling_avg_timedelta_3'] = df.groupby('FullName')['TimeDelta'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
df['driver_rolling_avg_timedelta_5'] = df.groupby('FullName')['TimeDelta'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

train_timedelta_mean = df[df['Season'] <= 2024]['TimeDelta'].mean()
df['driver_rolling_avg_timedelta_3'] = df['driver_rolling_avg_timedelta_3'].fillna(train_timedelta_mean)
df['driver_rolling_avg_timedelta_5'] = df['driver_rolling_avg_timedelta_5'].fillna(train_timedelta_mean)

# Drop leakage columns
leakage = ['Points', 'TimeDelta', 'TimeDetla', 'Podium', 'Status', 'Time']
df = df.drop(columns=leakage, errors='ignore')

# Save reference columns before encoding
df_ref = df[['Season', 'Round', 'EventName', 'FullName', 'FinishPosition']].copy()

# One-hot encode TeamName only
teamDummy = pd.get_dummies(df['TeamName'], prefix='Team')
df = df.drop(columns=['TeamName', 'FullName', 'EventName'])
df = pd.concat([df, teamDummy], axis=1)

# Drop rows missing qualifying data, then drop intermediate columns
df = df.dropna(subset=['best_q_time'])
df = df.drop(columns=['driver_rolling_avg_timedelta_3', 'team_rolling_avg_3', 'driver_rolling_avg_3',
                       'Q1_seconds', 'Q2_seconds', 'Q3_seconds', 'best_q_time'], errors='ignore')

df_train = df[df['Season'] <= 2024]
df_eval  = df[df['Season'] == 2025]

X_train = df_train.drop(columns=['FinishPosition', 'Season', 'Round'])
y_train = df_train['FinishPosition']
X_val   = df_eval.drop(columns=['FinishPosition', 'Season', 'Round'])

mask = y_train.notna()
X_train = X_train[mask]
y_train = y_train[mask]

X_train = X_train.fillna(0)
X_val   = X_val.fillna(0)

sorted_index = df_ref.loc[X_train.index].sort_values(by=['Season', 'Round']).index
X_train = X_train.loc[sorted_index]
y_train = y_train.loc[sorted_index]

# ── RF Model ──────────────────────────────────────────────────────────────────

def kendalltau_scorer(y_true, y_pred):
    corr, _ = kendalltau(y_true, y_pred)
    return corr

tau_scorer = make_scorer(kendalltau_scorer, greater_is_better=True)
tscv = TimeSeriesSplit(n_splits=5)

path = 'best_params.json'
if os.path.exists(path):
    with open(path) as f:
        best_params = json.load(f)
    rf = RandomForestRegressor(random_state=42, **best_params)
    rf.fit(X_train, y_train)
else:
    rf_grid = {"n_estimators": [50, 100, 150, 200], "max_depth": [5, 10, 15], "min_samples_split": [2, 5, 6, 7]}
    rf = GridSearchCV(RandomForestRegressor(random_state=42), rf_grid, cv=tscv, scoring=tau_scorer)
    rf.fit(X_train, y_train)
    with open(path, 'w') as f:
        json.dump(rf.best_params_, f, indent=2)
    print(f"Best RF params found and saved: {rf.best_params_}")

y_pred = rf.predict(X_val)
df_eval_ref = df_ref.loc[X_val.index].copy()
df_eval_ref['Predicted'] = y_pred
df_eval_ref['Predicted_Rank'] = df_eval_ref.groupby('EventName')['Predicted'].rank(method='first').astype(int)

# ── XGBoost Rank Model ────────────────────────────────────────────────────────

groups_train = df_ref.loc[X_train.index].groupby(['Season', 'Round'], sort=False).size().values
y_relevance  = (y_train.max() + 1) - y_train

team_cols    = [c for c in X_train.columns if c.startswith('Team_')]
X_train_xgb  = X_train.drop(columns=team_cols)
X_val_xgb    = X_val.drop(columns=team_cols)

xgb_model = xgb.XGBRanker(
    objective='rank:pairwise',
    random_state=42,
    n_estimators=200,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.7
)
xgb_model.fit(X_train_xgb, y_relevance, group=groups_train)

xgb_scores = xgb_model.predict(X_val_xgb)
df_eval_ref['XGB_score'] = xgb_scores
df_eval_ref['XGB Rank']  = df_eval_ref.groupby('EventName')['XGB_score'].rank(ascending=False, method='first').astype(int)

# ── Metrics per race ─────────────────────────────────────────────────────────

raw_2025 = pd.read_csv('data/f1_2025.csv')

def finished_drivers(race):
    return raw_2025[
        (raw_2025['EventName'] == race) &
        (raw_2025['Status'].isin(['Finished', 'Lapped']))
    ]['FullName']

def compute_metrics(df_eval_ref, rank_col):
    tau_full_s, tau_clean_s, mae_s, ap5_s, top5_s = {}, {}, {}, {}, {}
    for race in df_eval_ref['EventName'].unique():
        race_df_full  = df_eval_ref[df_eval_ref['EventName'] == race].dropna(subset=['FinishPosition'])
        race_df_clean = race_df_full[race_df_full['FullName'].isin(finished_drivers(race))]

        tau_full, _  = kendalltau(race_df_full['FinishPosition'],  race_df_full[rank_col])
        tau_clean, _ = kendalltau(race_df_clean['FinishPosition'], race_df_clean[rank_col])
        mae = (race_df_clean['FinishPosition'] - race_df_clean[rank_col]).abs().mean()
        actual_top5  = (race_df_clean['FinishPosition'] <= 5).astype(int)
        ap5 = average_precision_score(actual_top5, -race_df_clean[rank_col])
        pred_top5 = set(race_df_clean.nsmallest(5, rank_col)['FullName'])
        act_top5  = set(race_df_clean.nsmallest(5, 'FinishPosition')['FullName'])

        tau_full_s[race]  = round(tau_full, 3)
        tau_clean_s[race] = round(tau_clean, 3)
        mae_s[race]       = round(mae, 3)
        ap5_s[race]       = round(ap5, 3)
        top5_s[race]      = round(len(pred_top5 & act_top5) / 5, 3)

    return tau_full_s, tau_clean_s, mae_s, ap5_s, top5_s

tau_full_scores,  tau_clean_scores,  mae_scores,  ap5_scores,  top5_scores  = compute_metrics(df_eval_ref, 'Predicted_Rank')
xgb_tau_full_scores, xgb_tau_clean_scores, xgb_mae_scores, xgb_ap5_scores, xgb_top5_scores = compute_metrics(df_eval_ref, 'XGB Rank')

mean_tau_full  = round(np.mean(list(tau_full_scores.values())), 3)
mean_tau_clean = round(np.mean(list(tau_clean_scores.values())), 3)
mean_mae       = round(np.mean(list(mae_scores.values())), 3)
mean_ap5       = round(np.mean(list(ap5_scores.values())), 3)
mean_top5      = round(np.mean(list(top5_scores.values())), 3)

xgb_mean_tau_full  = round(np.mean(list(xgb_tau_full_scores.values())), 3)
xgb_mean_tau_clean = round(np.mean(list(xgb_tau_clean_scores.values())), 3)
xgb_mean_mae       = round(np.mean(list(xgb_mae_scores.values())), 3)
xgb_mean_ap5       = round(np.mean(list(xgb_ap5_scores.values())), 3)
xgb_mean_top5      = round(np.mean(list(xgb_top5_scores.values())), 3)

# ── Baseline (grid only) ──────────────────────────────────────────────────────

actual_2025 = pd.read_csv('data/f1_2020_2025.csv')
actual_2025 = actual_2025[actual_2025['Season'] == 2025].copy()
base = actual_2025.dropna(subset=['GridPosition', 'RacePosition']).copy()

tau_b_full, tau_b_clean, mae_b, ap5_b, top5_b = {}, {}, {}, {}, {}
for race in base['EventName'].unique():
    rdf_full  = base[base['EventName'] == race]
    rdf_clean = rdf_full[rdf_full['FullName'].isin(finished_drivers(race))]
    if len(rdf_clean) < 2:
        continue
    tau_f, _ = kendalltau(rdf_full['RacePosition'],  rdf_full['GridPosition'])
    tau_c, _ = kendalltau(rdf_clean['RacePosition'], rdf_clean['GridPosition'])
    mae  = (rdf_clean['RacePosition'] - rdf_clean['GridPosition']).abs().mean()
    at5  = (rdf_clean['RacePosition'] <= 5).astype(int)
    ap5  = average_precision_score(at5, -rdf_clean['GridPosition'])
    pt5  = set(rdf_clean.nsmallest(5, 'GridPosition')['FullName'])
    at5n = set(rdf_clean.nsmallest(5, 'RacePosition')['FullName'])
    tau_b_full[race]  = round(tau_f, 3)
    tau_b_clean[race] = round(tau_c, 3)
    mae_b[race]       = round(mae, 3)
    ap5_b[race]       = round(ap5, 3)
    top5_b[race]      = round(len(pt5 & at5n) / 5, 3)

mean_tau_b_full  = round(np.mean(list(tau_b_full.values())), 3)
mean_tau_b_clean = round(np.mean(list(tau_b_clean.values())), 3)
mean_mae_b       = round(np.mean(list(mae_b.values())), 3)
mean_ap5_b       = round(np.mean(list(ap5_b.values())), 3)
mean_top5_b      = round(np.mean(list(top5_b.values())), 3)

# ── Compare against previous predictions ─────────────────────────────────────

PREV_FILE = 'f1_2025_predictions.csv'
changes = []

if os.path.exists(PREV_FILE):
    prev = pd.read_csv(PREV_FILE)
    for _, row in df_eval_ref.iterrows():
        prev_row = prev[
            (prev['EventName'] == row['EventName']) &
            (prev['FullName']   == row['FullName'])
        ]
        if prev_row.empty:
            changes.append(f"  NEW entry: {row['FullName']} @ {row['EventName']}")
        else:
            old_rank = int(prev_row['Predicted_Rank'].values[0])
            new_rank = int(row['Predicted_Rank'])
            if old_rank != new_rank:
                delta = old_rank - new_rank
                direction = f"+{delta}" if delta > 0 else str(delta)
                changes.append(
                    f"  {row['FullName']:<25} @ {row['EventName']:<30} "
                    f"P{old_rank} -> P{new_rank}  ({direction})"
                )
else:
    changes.append("  No previous predictions file found — this is the first run.")

# ── Save new predictions ──────────────────────────────────────────────────────

df_eval_ref.sort_values(['Round', 'Predicted_Rank']).to_csv(PREV_FILE, index=False)

# ── Write report ──────────────────────────────────────────────────────────────

now = datetime.now()
report_filename = f"report_{now.strftime('%Y-%m-%d_%H-%M')}.md"

lines = []
lines.append(f"# F1 2025 Prediction Report")
lines.append(f"Generated: {now.strftime('%Y-%m-%d %H:%M')}\n")
lines.append(f"## Model Changes This Run\n")
lines.append(MODEL_CHANGES)

lines.append(f"## Model Comparison\n")
lines.append(f"| Model | Tau (Full) | Tau (Clean) | MAE | AP@5 | Top-5 |")
lines.append(f"|-------|------------|-------------|-----|------|-------|")
lines.append(f"| Grid only    | {mean_tau_b_full} | {mean_tau_b_clean} | {mean_mae_b} | {mean_ap5_b} | {mean_top5_b} |")
lines.append(f"| RF           | {mean_tau_full} | {mean_tau_clean} | {mean_mae} | {mean_ap5} | {mean_top5} |")
lines.append(f"| XGBoost Rank | {xgb_mean_tau_full} | {xgb_mean_tau_clean} | {xgb_mean_mae} | {xgb_mean_ap5} | {xgb_mean_top5} |\n")

lines.append(f"## RF Per-Race Metrics\n")
lines.append(f"| Race | Tau (Full) | Tau (Clean) | MAE | AP@5 | Top-5 |")
lines.append(f"|------|------------|-------------|-----|------|-------|")
for race in sorted(tau_clean_scores.keys(), key=lambda r: -tau_clean_scores[r]):
    lines.append(f"| {race} | {tau_full_scores[race]:.3f} | {tau_clean_scores[race]:.3f} | {mae_scores[race]:.3f} | {ap5_scores[race]:.3f} | {top5_scores[race]:.3f} |")

lines.append(f"\n## XGBoost Per-Race Metrics\n")
lines.append(f"| Race | Tau (Full) | Tau (Clean) | MAE | AP@5 | Top-5 |")
lines.append(f"|------|------------|-------------|-----|------|-------|")
for race in sorted(xgb_tau_clean_scores.keys(), key=lambda r: -xgb_tau_clean_scores[r]):
    lines.append(f"| {race} | {xgb_tau_full_scores[race]:.3f} | {xgb_tau_clean_scores[race]:.3f} | {xgb_mae_scores[race]:.3f} | {xgb_ap5_scores[race]:.3f} | {xgb_top5_scores[race]:.3f} |")

lines.append(f"\n## Changes Since Previous RF Predictions\n")
if changes:
    lines.extend(changes)
else:
    lines.append("  No rank changes from previous run.")

lines.append(f"\n## Predictions by Race\n")
for round_num in sorted(df_eval_ref['Round'].unique()):
    race_df   = df_eval_ref[df_eval_ref['Round'] == round_num].sort_values('Predicted_Rank')
    race_name = race_df['EventName'].iloc[0]
    lines.append(f"### Round {int(round_num)}: {race_name}")
    lines.append(f"(RF — Tau full: {tau_full_scores.get(race_name, float('nan')):.3f}, Tau clean: {tau_clean_scores.get(race_name, float('nan')):.3f}, MAE: {mae_scores.get(race_name, float('nan')):.3f})")
    lines.append(f"(XGB — Tau full: {xgb_tau_full_scores.get(race_name, float('nan')):.3f}, Tau clean: {xgb_tau_clean_scores.get(race_name, float('nan')):.3f}, MAE: {xgb_mae_scores.get(race_name, float('nan')):.3f})\n")
    lines.append(f"| RF Predicted | XGB Predicted | Driver | Actual Finish |")
    lines.append(f"|--------------|---------------|--------|---------------|")
    for _, row in race_df.iterrows():
        actual   = int(row['FinishPosition']) if pd.notna(row['FinishPosition']) else 'N/A'
        xgb_rank = int(row['XGB Rank']) if pd.notna(row['XGB Rank']) else 'N/A'
        lines.append(f"| P{int(row['Predicted_Rank'])} | P{xgb_rank} | {row['FullName']} | P{actual} |")
    lines.append("")

with open(report_filename, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Report saved: {report_filename}")
print(f"Predictions saved: {PREV_FILE}")
print(f"\n=== Model Comparison ===")
print(f"{'Model':<15} {'Tau (Full)':>10} {'Tau (Clean)':>12} {'MAE':>7} {'AP@5':>7} {'Top-5':>7}")
print("-" * 65)
print(f"{'Grid only':<15} {mean_tau_b_full:>10} {mean_tau_b_clean:>12} {mean_mae_b:>7} {mean_ap5_b:>7} {mean_top5_b:>7}")
print(f"{'RF':<15} {mean_tau_full:>10} {mean_tau_clean:>12} {mean_mae:>7} {mean_ap5:>7} {mean_top5:>7}")
print(f"{'XGBoost Rank':<15} {xgb_mean_tau_full:>10} {xgb_mean_tau_clean:>12} {xgb_mean_mae:>7} {xgb_mean_ap5:>7} {xgb_mean_top5:>7}")
print(f"\nRF rank changes from previous: {len(changes)}")
