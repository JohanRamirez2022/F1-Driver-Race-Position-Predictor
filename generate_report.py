# ── Describe changes made since the previous run ──────────────────────────────
# Update this before each run so the report is self-documenting.
MODEL_CHANGES = """
- Added driver_track_history_avg feature
- Removed redundant FinishPosition column from CSVs (using RacePosition everywhere)
- Synced generate_report.py with notebook: max_depth=None, dropna on best_q_time, load from combined CSV
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import average_precision_score
from scipy.stats import kendalltau
from datetime import datetime
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

df = df.dropna(subset=['best_q_time'])

drop_cols = ['Points', 'TimeDelta', 'TimeDetla', 'Podium', 'Status', 'Time',
             'Q1_seconds', 'Q2_seconds', 'Q3_seconds', 'best_q_time',
             'driver_rolling_avg_timedelta_3', 'team_rolling_avg_3']
df = df.drop(columns=drop_cols, errors='ignore')

df_ref = df[['Season', 'Round', 'EventName', 'FullName', 'FinishPosition']].copy()

teamDummy   = pd.get_dummies(df['TeamName'],  prefix='Team')
driverDummy = pd.get_dummies(df['FullName'],  prefix='Driver')
eventDummy  = pd.get_dummies(df['EventName'], prefix='Event')
df = df.drop(columns=['TeamName', 'FullName', 'EventName'])
df = pd.concat([df, teamDummy, driverDummy, eventDummy], axis=1)

df_train = df[df['Season'] <= 2024]
df_eval  = df[df['Season'] == 2025]

X_train = df_train.drop(columns=['FinishPosition', 'Season', 'Round'])
y_train = df_train['FinishPosition']
X_val   = df_eval.drop(columns=['FinishPosition', 'Season', 'Round'])

mask = y_train.notna()
X_train = X_train[mask]
y_train = y_train[mask]

rf = RandomForestRegressor(n_estimators=100, max_depth=None, max_features='sqrt', min_samples_split=2, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)
df_eval_ref = df_ref.loc[X_val.index].copy()
df_eval_ref['Predicted'] = y_pred
df_eval_ref['Predicted_Rank'] = df_eval_ref.groupby('EventName')['Predicted'].rank(method='first').astype(int)

# ── Metrics per race ─────────────────────────────────────────────────────────

tau_scores = {}
mae_scores = {}
ap5_scores = {}
top5_scores = {}

for race in df_eval_ref['EventName'].unique():
    race_df = df_eval_ref[df_eval_ref['EventName'] == race].dropna(subset=['FinishPosition'])

    tau, _ = kendalltau(race_df['FinishPosition'], race_df['Predicted_Rank'])
    mae = (race_df['FinishPosition'] - race_df['Predicted_Rank']).abs().mean()

    actual_top5 = (race_df['FinishPosition'] <= 5).astype(int)
    ap5 = average_precision_score(actual_top5, -race_df['Predicted_Rank'])

    predicted_top5 = set(race_df.nsmallest(5, 'Predicted_Rank')['FullName'])
    actual_top5_names = set(race_df.nsmallest(5, 'FinishPosition')['FullName'])
    top5_precision = len(predicted_top5 & actual_top5_names) / 5

    tau_scores[race]  = round(tau, 3)
    mae_scores[race]  = round(mae, 3)
    ap5_scores[race]  = round(ap5, 3)
    top5_scores[race] = round(top5_precision, 3)

mean_tau  = round(np.mean(list(tau_scores.values())), 3)
mean_mae  = round(np.mean(list(mae_scores.values())), 3)
mean_ap5  = round(np.mean(list(ap5_scores.values())), 3)
mean_top5 = round(np.mean(list(top5_scores.values())), 3)

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

lines.append(f"## Evaluation\n")
lines.append(f"- **Mean Kendall's Tau:** {mean_tau}")
lines.append(f"- **Mean MAE:** {mean_mae} positions")
lines.append(f"- **Mean AP@5:** {mean_ap5}")
lines.append(f"- **Mean Top-5 Precision:** {mean_top5}\n")
lines.append(f"| Race | Kendall's Tau | MAE | AP@5 | Top-5 Precision |")
lines.append(f"|------|---------------|-----|------|-----------------|")
for race in sorted(tau_scores.keys(), key=lambda r: -tau_scores[r]):
    lines.append(f"| {race} | {tau_scores[race]:.3f} | {mae_scores[race]:.3f} | {ap5_scores[race]:.3f} | {top5_scores[race]:.3f} |")

lines.append(f"\n## Changes Since Previous Predictions\n")
if changes:
    lines.extend(changes)
else:
    lines.append("  No rank changes from previous run.")

lines.append(f"\n## Predictions by Race\n")
for round_num in sorted(df_eval_ref['Round'].unique()):
    race_df = df_eval_ref[df_eval_ref['Round'] == round_num].sort_values('Predicted_Rank')
    race_name = race_df['EventName'].iloc[0]
    tau = tau_scores.get(race_name, float('nan'))
    mae = mae_scores.get(race_name, float('nan'))
    ap5 = ap5_scores.get(race_name, float('nan'))
    top5 = top5_scores.get(race_name, float('nan'))
    lines.append(f"### Round {int(round_num)}: {race_name}  (Tau: {tau:.3f}, MAE: {mae:.3f}, AP@5: {ap5:.3f}, Top-5: {top5:.3f})\n")
    lines.append(f"| Predicted | Driver | Actual Finish |")
    lines.append(f"|-----------|--------|---------------|")
    for _, row in race_df.iterrows():
        actual = int(row['FinishPosition']) if pd.notna(row['FinishPosition']) else 'N/A'
        lines.append(f"| P{int(row['Predicted_Rank'])} | {row['FullName']} | P{actual} |")
    lines.append("")

with open(report_filename, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Report saved: {report_filename}")
print(f"Predictions saved: {PREV_FILE}")
print(f"Mean Kendall's Tau:   {mean_tau}")
print(f"Mean MAE:             {mean_mae} positions")
print(f"Mean AP@5:            {mean_ap5}")
print(f"Mean Top-5 Precision: {mean_top5}")
print(f"Rank changes: {len(changes)}")
