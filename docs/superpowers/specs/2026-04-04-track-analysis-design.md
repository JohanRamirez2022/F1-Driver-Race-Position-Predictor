# Track Analysis Design

**Date:** 2026-04-04  
**Scope:** Two new EDA cells added to `F1.ipynb` analyzing position variance by track and driver performance by track type.  
**Data:** 2020–2024 only (2025 is the held-out validation set and must not influence any analysis or feature decisions).

---

## Goal

Explore two related questions:
1. How much do drivers move from their grid position at each circuit, and which tracks have the most position variance?
2. Do drivers show systematic performance differences between street circuits and permanent tracks?

---

## Cell 1: Position Variance by Track

### Data Source
`data/f1_2020_2025.csv` filtered to `Season <= 2024`.

### Computation
For each race (grouped by `EventName`), compute per-driver `positions_delta = GridPosition - FinishPosition` (positive = gained positions, negative = lost positions).

Aggregate per track across all races (2020–2024):

| Metric | Description |
|--------|-------------|
| **Mean absolute delta** | On average, how many positions does a driver move from their starting grid spot? High = lots of overtaking/chaos. Low = grid order holds (e.g. Monaco). |
| **Std deviation of delta** | How inconsistent is position movement across drivers at that track? High = some move a lot, others barely move (unpredictable). Low = uniform movement. |
| **Median delta** | Whether the typical driver tends to gain or lose positions. Near 0 = neutral. |
| **Avg positions gained** | Among drivers who moved forward, how many spots did they gain on average? |
| **Avg positions lost** | Among drivers who moved backward (or DNF'd), how many spots did they lose on average? |

### Visualization
Horizontal bar chart sorted descending by mean absolute delta. High-variance tracks at top, low-variance (Monaco, Singapore) at bottom. Color-encode by std deviation.

---

## Cell 2: Driver Performance by Track Type (Street vs. Track)

### Track Taxonomy
Two categories — manually defined, stable F1 knowledge:

- **Street circuits:** Monaco, Singapore, Baku (Azerbaijan), Las Vegas, Miami, Saudi Arabia (Jeddah)
- **Permanent tracks:** All other circuits

### Data Source
`data/f1_2020_2025.csv` filtered to `Season <= 2024`.

### Computation
For each driver, compute average `FinishPosition` on street circuits and permanent tracks separately. Only include drivers with at least 3 races in each category to avoid noise from limited appearances.

### Visualization
Scatter plot:
- X-axis: avg finish position on permanent tracks
- Y-axis: avg finish position on street circuits
- Each point = one driver, labeled by name
- Diagonal line (y = x) as parity reference — drivers above the diagonal are relatively better on permanent tracks; drivers below are relatively better on streets
- Lower values = better (1st place = best)

---

## Constraints
- **No 2025 data in either cell.** 2025 is the held-out validation set. Including it — even descriptively — could skew feature engineering decisions.
- Both cells are purely exploratory EDA — no model changes, no new features added as part of this spec. Feature additions informed by findings are a separate task.
- Cells are inserted after the existing EDA section in `F1.ipynb`, before model training cells.
