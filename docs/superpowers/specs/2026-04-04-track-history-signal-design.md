# Track-Specific Driver History Signal Investigation

**Date:** 2026-04-04
**Scope:** One new EDA cell in `F1.ipynb` to validate whether a driver's historical finish at a specific circuit predicts their future finish there.
**Data:** 2020–2024 only (2025 is the held-out validation set).

---

## Goal

Determine whether "driver's average historical finish at this circuit" is a strong enough signal to justify adding as a model feature. If drivers consistently finish in similar positions at the same circuit year over year, the signal is real and worth encoding.

---

## Cell: Pooled Track-Specific History Correlation

### Data Source
`data/f1_2020_2025.csv` filtered to `Season <= 2024`.

### Computation

For every driver × circuit × season combination:
1. Compute `historical_avg` — the driver's average `FinishPosition` at that specific circuit across all seasons strictly before this one
2. Record `actual_finish` — the driver's `FinishPosition` at that circuit in this season
3. Only include rows where the driver has at least 1 prior race at that circuit (otherwise there's no history to use)

Pool all valid pairs across every driver, circuit, and season. Compute a single Spearman correlation (ρ) between `historical_avg` and `actual_finish`.

### Decision Threshold
- ρ > 0.35 — signal is worth adding as a feature (even moderate signals stack with existing strong features)
- ρ 0.2–0.35 — weak, unlikely to help given existing feature set
- ρ < 0.2 — noise

### Visualization
Scatter plot:
- X-axis: historical avg finish at this circuit (from prior seasons)
- Y-axis: actual finish at this circuit (this season)
- One dot per driver-circuit-season pair
- Dots colored by circuit for visual grouping
- Diagonal parity line (y = x)
- Annotate with overall Spearman ρ and p-value in the title

### Placement
Insert after the street vs. permanent track cell (the last EDA cell before "EDA IS NOW OVER").

---

## Constraints
- **No 2025 data.** 2025 is the held-out validation set.
- **Temporal safety:** Only use seasons strictly before the current one to compute historical avg (no look-ahead).
- **EDA only** — no model changes, no new features added. Feature addition is a separate decision after reviewing results.
- **Minimum history:** Require at least 1 prior race at that circuit per driver to include in the pool.
