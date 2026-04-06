# Track Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two EDA cells to `F1.ipynb` — one analyzing position variance by track, one comparing driver performance on street vs. permanent circuits — using 2020–2024 data only.

**Architecture:** Both cells load a fresh copy of `data/f1_2020_2025.csv` filtered to `Season <= 2024`. They are inserted after cell id `2ad44193` (Cell 18, last feature-engineering cell) and before cell id `519483ed` (Cell 19, "EDA IS NOW OVER"). Each cell is fully self-contained and does not depend on the modified `df` state from earlier cells.

**Tech Stack:** pandas, matplotlib, numpy — all already imported in the notebook.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `F1.ipynb` | Insert 2 cells after cell id `2ad44193` | Add position variance cell and street/track scatter cell |

---

## Task 1: Position Variance by Track Cell

**Files:**
- Modify: `F1.ipynb` — insert new code cell after cell id `2ad44193`

- [ ] **Step 1: Insert the position variance cell**

Use NotebookEdit with `edit_mode=insert`, `cell_id=2ad44193`, `cell_type=code`:

```python
# === EDA: Position Variance by Track (2020–2024) ===
import matplotlib.cm as cm

_raw = pd.read_csv('data/f1_2020_2025.csv')
_raw = _raw[_raw['Season'] <= 2024].copy()

# Positive delta = gained positions, negative = lost
_raw['positions_delta'] = _raw['GridPosition'] - _raw['FinishPosition']

def _avg_gained(x):
    gains = x[x > 0]
    return gains.mean() if len(gains) > 0 else 0.0

def _avg_lost(x):
    losses = x[x < 0]
    return losses.mean() if len(losses) > 0 else 0.0

track_stats = _raw.groupby('EventName')['positions_delta'].agg(
    mean_abs_delta=lambda x: x.abs().mean(),
    std_delta='std',
    median_delta='median',
    avg_gained=_avg_gained,
    avg_lost=_avg_lost
).reset_index().sort_values('mean_abs_delta', ascending=True)

# Horizontal bar chart colored by std deviation
fig, ax = plt.subplots(figsize=(10, 10))
norm = plt.Normalize(track_stats['std_delta'].min(), track_stats['std_delta'].max())
colors = cm.RdYlGn_r(norm(track_stats['std_delta']))
ax.barh(track_stats['EventName'], track_stats['mean_abs_delta'], color=colors)
sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Std Dev of Delta (higher = more unpredictable)')
ax.set_xlabel('Mean Absolute Position Delta')
ax.set_title('Position Variance by Track (2020–2024)\nSorted by avg positions moved from grid')
plt.tight_layout()
plt.show()

print(track_stats.sort_values('mean_abs_delta', ascending=False).to_string(index=False))
```

- [ ] **Step 2: Run the cell and verify output**

Run the new cell in the notebook. Expected output:
- A horizontal bar chart appears with tracks sorted from lowest to highest variance
- Monaco Grand Prix and Singapore Grand Prix should appear near the bottom (low variance)
- A printed table of all tracks with their stats follows the chart

---

## Task 2: Street vs. Permanent Track Scatter Cell

**Files:**
- Modify: `F1.ipynb` — insert new code cell after the cell inserted in Task 1 (which will now be cell id `2ad44193`'s immediate successor)

**Note:** After Task 1 inserts a cell after `2ad44193`, the new cell will have a fresh auto-generated id. Insert this second cell after that new cell by running the notebook and checking the new cell id, OR insert it as a second insert after `2ad44193` — NotebookEdit inserts after the specified cell, so inserting twice after `2ad44193` will place Task 2's cell between Task 1's cell and the model training section. Insert this cell after the cell created in Task 1.

- [ ] **Step 1: Insert the street vs. track scatter cell**

Use NotebookEdit with `edit_mode=insert`, `cell_id=<id of cell inserted in Task 1>`, `cell_type=code`:

```python
# === EDA: Driver Performance — Street vs. Permanent Tracks (2020–2024) ===

_street_circuits = {
    'Monaco Grand Prix',
    'Singapore Grand Prix',
    'Azerbaijan Grand Prix',
    'Las Vegas Grand Prix',
    'Miami Grand Prix',
    'Saudi Arabian Grand Prix',
}

_raw2 = pd.read_csv('data/f1_2020_2025.csv')
_raw2 = _raw2[_raw2['Season'] <= 2024].copy()
_raw2['track_type'] = _raw2['EventName'].apply(
    lambda e: 'Street' if e in _street_circuits else 'Permanent'
)

_street_avg = (
    _raw2[_raw2['track_type'] == 'Street']
    .groupby('FullName')['FinishPosition']
    .agg(street_avg='mean', street_count='count')
)
_perm_avg = (
    _raw2[_raw2['track_type'] == 'Permanent']
    .groupby('FullName')['FinishPosition']
    .agg(perm_avg='mean', perm_count='count')
)

_driver_stats = _street_avg.join(_perm_avg, how='inner')
_driver_stats = _driver_stats[
    (_driver_stats['street_count'] >= 3) & (_driver_stats['perm_count'] >= 3)
]

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(_driver_stats['perm_avg'], _driver_stats['street_avg'], s=80, alpha=0.8)

for name, row in _driver_stats.iterrows():
    ax.annotate(
        name.split()[-1],
        (row['perm_avg'], row['street_avg']),
        fontsize=8, ha='left', va='bottom',
        xytext=(3, 3), textcoords='offset points'
    )

ax.plot([1, 20], [1, 20], 'k--', alpha=0.4, label='Parity (equal on both)')
ax.set_xlim(1, 20)
ax.set_ylim(1, 20)
ax.set_xlabel('Avg Finish — Permanent Tracks (lower = better)')
ax.set_ylabel('Avg Finish — Street Circuits (lower = better)')
ax.set_title('Driver Performance: Street vs. Permanent Tracks (2020–2024)\nBelow diagonal = better on streets; Above diagonal = better on permanent')
ax.legend()
plt.tight_layout()
plt.show()

print(_driver_stats[['street_avg', 'perm_avg', 'street_count', 'perm_count']].sort_values('street_avg').to_string())
```

- [ ] **Step 2: Run the cell and verify output**

Run the new cell. Expected output:
- A scatter plot with one point per driver, labeled by last name
- A dashed diagonal parity line
- Drivers below the diagonal are relatively stronger on street circuits
- A printed table sorted by street circuit average follows the chart

---

## Self-Review

**Spec coverage:**
- ✅ Cell 1: position variance by track with mean abs delta, std dev, median delta, avg gained, avg lost
- ✅ Cell 1: horizontal bar chart sorted by mean abs delta, color-encoded by std dev
- ✅ Cell 2: street vs. permanent track taxonomy defined
- ✅ Cell 2: scatter plot with diagonal parity line, driver labels
- ✅ Both cells filtered to Season <= 2024 only
- ✅ Both cells load fresh data (not dependent on modified `df`)
- ✅ Cells inserted after EDA, before model training

**Placeholder scan:** No TBDs, no vague steps, all code is complete.

**Type consistency:** `_raw` and `_raw2` are local variables prefixed with `_` to avoid polluting the notebook namespace. `FinishPosition` from the raw CSV is the correct column (the raw CSV has both `FinishPosition` and `RacePosition`; after Cell 12 the notebook renames `RacePosition` → `FinishPosition`, but since we load fresh here we use the original `FinishPosition` column which is the actual finishing order).
