# Track-Specific Driver History Signal Investigation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one EDA cell to `F1.ipynb` that validates whether a driver's historical average finish at a specific circuit predicts their future finish there (pooled Spearman correlation + scatter plot).

**Architecture:** A single self-contained notebook cell loads fresh data from CSV (2020–2024 only), computes per-driver per-circuit historical averages using only strictly prior seasons, pools all pairs, and visualizes via scatter plot with Spearman ρ annotation. Decision threshold: ρ > 0.35 means the signal is worth adding as a feature.

**Tech Stack:** pandas, scipy.stats.spearmanr, matplotlib — all already available in the notebook environment.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `F1.ipynb` | Insert 1 cell after cell id `1b141ed1` (Cell 20, street vs. track scatter) | Track history signal EDA |

---

## Task 1: Insert Track-Specific History Signal Cell

**Files:**
- Modify: `F1.ipynb` — insert new code cell after cell id `1b141ed1`

- [ ] **Step 1: Insert the cell**

Use NotebookEdit with `edit_mode=insert`, `cell_id=1b141ed1`, `cell_type=code`:

```python
# === EDA: Track-Specific Driver History Signal (2020–2024) ===
from scipy.stats import spearmanr

_raw3 = pd.read_csv('data/f1_2020_2025.csv')
_raw3 = _raw3[_raw3['Season'] <= 2024].copy()

# For each driver x circuit x season, compute historical avg finish from prior seasons
_pairs = []
for (driver, circuit), group in _raw3.groupby(['FullName', 'EventName']):
    group = group.sort_values('Season')
    seasons = group['Season'].values
    finishes = group['FinishPosition'].values
    for i in range(1, len(seasons)):
        historical_avg = finishes[:i].mean()
        actual_finish = finishes[i]
        if not np.isnan(historical_avg) and not np.isnan(actual_finish):
            _pairs.append({
                'driver': driver,
                'circuit': circuit,
                'season': seasons[i],
                'historical_avg': historical_avg,
                'actual_finish': actual_finish,
            })

_pairs_df = pd.DataFrame(_pairs)

# Pooled Spearman correlation
_rho, _pval = spearmanr(_pairs_df['historical_avg'], _pairs_df['actual_finish'])

# Scatter plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(_pairs_df['historical_avg'], _pairs_df['actual_finish'], s=30, alpha=0.4)
ax.plot([1, 20], [1, 20], 'k--', alpha=0.4, label='Parity (y = x)')
ax.set_xlim(1, 20)
ax.set_ylim(1, 20)
ax.set_xlabel('Historical Avg Finish at This Circuit (prior seasons)')
ax.set_ylabel('Actual Finish at This Circuit (this season)')
ax.set_title(f'Track-Specific Driver History Signal (2020-2024)\nSpearman rho = {_rho:.3f}, p = {_pval:.2e} | Threshold: rho > 0.35')
ax.legend()
plt.tight_layout()
plt.show()

print(f'Pooled Spearman rho: {_rho:.3f}  (p-value: {_pval:.2e})')
print(f'Number of driver-circuit-season pairs: {len(_pairs_df)}')
print(f'\nSignal assessment: {"WORTH ADDING as feature" if _rho > 0.35 else "Too weak to add"}')
```

- [ ] **Step 2: Commit**

```bash
git add F1.ipynb
git commit -m "feat: add track-specific driver history signal EDA cell (2020-2024)"
```

- [ ] **Step 3: Verify the cell runs**

Write the cell logic to a temp file and run with `conda run -n F1 python test_cell3.py`. Expected output:
- Spearman rho value printed (likely in the 0.3–0.6 range)
- p-value printed (should be very small if signal is real)
- Number of pairs printed (should be several hundred)
- Signal assessment printed based on 0.35 threshold

Clean up: `rm test_cell3.py`

---

## Self-Review

**Spec coverage:**
- ✅ Loads `data/f1_2020_2025.csv` filtered to Season <= 2024
- ✅ Computes historical_avg from strictly prior seasons only (temporal safety via `finishes[:i]`)
- ✅ Requires at least 1 prior race (loop starts at `i=1`)
- ✅ Pools all driver-circuit-season pairs
- ✅ Spearman correlation with ρ and p-value
- ✅ Scatter plot with parity line, annotated with ρ and threshold
- ✅ Decision threshold at 0.35
- ✅ Inserted after street vs. track cell, before "EDA IS NOW OVER"
- ✅ No 2025 data used
- ✅ EDA only, no model changes

**Placeholder scan:** No TBDs, all code is complete.

**Type consistency:** `_raw3`, `_pairs`, `_pairs_df`, `_rho`, `_pval` — all underscore-prefixed to avoid polluting notebook namespace. Consistent with `_raw` and `_raw2` naming from prior cells.
