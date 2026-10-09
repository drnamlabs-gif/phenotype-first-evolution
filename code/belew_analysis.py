"""
Belew-style exact analysis of the compositional experiments.

The generative encoding used by baldwin_compositional.py /
phenotype_first_compositional.py has a tiny reachable phenotype space:
5 shape primitives, each in one of 9 states (off, or 2x2 offsets x 2 scales).
We enumerate it exactly and compute:

  (1) the coverable cell set and the resulting HARD CEILING F* per task
      -> predicted Phenopoiesis plateau (the ratchet converges to F*)
  (2) the exact distribution of F(X) for a random genome
      -> E[F(X)]  (predicted Baldwin recorded-final-fitness, since the
         recorded Baldwin phenotype is a drifting genome independent of
         the learned fitness)
      -> E[max of 20 draws]   (per-organism learned fitness, flat in t)
      -> E[max of 1000 draws] (population-best learned fitness, flat in t)
  (3) bootstrap condition for the Pheno ratchet: can any single draw
      exceed the 0.5 storage threshold?  If not, Pheno degenerates to
      Baldwin exactly (prediction for L+Plus).

Compare all of it against the published numbers:
  L+T:    Baldwin 0.254+-0.196, Pheno 0.772+-0.075
  T+Plus: Baldwin 0.086+-0.123, Pheno 0.517+-0.000
  L+Plus: Baldwin 0.177+-0.149, Pheno 0.189+-0.156
"""
import itertools
import numpy as np
from compositional_environments_extended import get_task

PRIMS = [
    [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],  # L
    [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)],  # T
    [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],  # Cross
    [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],  # Line
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],  # Block
]

def placement_cells(prim, ox, oy, scale):
    cells = set()
    for dx, dy in prim:
        x, y = ox + dx * scale, oy + dy * scale
        if 0 <= x < 10 and 0 <= y < 10:
            cells.add((x, y))
    return frozenset(cells)

# Per-primitive states: index 0 = off; 1..8 = placements
# Random-genome probabilities: P(off)=1/2; each placement 1/16
STATES = []
PROBS = []
for prim in PRIMS:
    states = [frozenset()]
    probs = [0.5]
    for ox in (0, 5):
        for oy in (0, 5):
            for scale in (1, 3):
                states.append(placement_cells(prim, ox, oy, scale))
                probs.append(0.5 / 8)
    STATES.append(states)
    PROBS.append(probs)

coverable = set()
for states in STATES:
    for s in states:
        coverable |= s

def to_grid(cellset):
    g = np.zeros((10, 10), dtype=int)
    for (x, y) in cellset:
        g[x, y] = 1
    return g

tasks = {name: get_task(name) for name in ["L+T", "T+Plus", "L+Plus"]}

print("=" * 88)
print("(1) HARD CEILING F* (recall of each target against the coverable cell set)")
print("=" * 88)
cov_grid = to_grid(coverable)
for name, task in tasks.items():
    s = task.get_component_scores(cov_grid)
    fstar = task.evaluate_fitness(cov_grid)
    print(f"{name:8s}: F* = {fstar:.3f}   components: {s}")

print()
print("=" * 88)
print("(2) EXACT distribution of F(X) for a random genome (full 9^5 enumeration)")
print("=" * 88)

# Enumerate all 9^5 = 59049 combinations with exact probabilities.
for name, task in tasks.items():
    t1 = task.pattern1_target.astype(bool)
    t2 = task.pattern2_target.astype(bool)
    n1 = t1.sum()
    n2 = t2.sum()

    values = []   # fitness values
    weights = []  # probabilities
    best_single = 0.0
    for combo in itertools.product(*(range(9) for _ in range(5))):
        cells = set()
        p = 1.0
        for i, s in enumerate(combo):
            cells |= STATES[i][s]
            p *= PROBS[i][s]
        g = to_grid(cells).astype(bool)
        f = ((g & t1).sum() / n1 + (g & t2).sum() / n2) / 2
        values.append(f)
        weights.append(p)
        if f > best_single:
            best_single = f

    values = np.array(values)
    weights = np.array(weights)
    EF = float((values * weights).sum())

    # Exact E[max of k iid draws] from the finite distribution
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    cdf = np.cumsum(w_sorted)
    uniq_v, idx = np.unique(v_sorted, return_index=True)
    # P(F <= v) at each unique value
    cdf_at = np.array([cdf[np.searchsorted(v_sorted, v, side="right") - 1] for v in uniq_v])

    def E_max(k):
        # E[max] = sum_v v * (P(F<=v)^k - P(F<v)^k)
        p_le = cdf_at ** k
        p_lt = np.concatenate([[0.0], cdf_at[:-1]]) ** k
        return float((uniq_v * (p_le - p_lt)).sum())

    print(f"{name:8s}: E[F(X)] = {EF:.3f}   E[max20] = {E_max(20):.3f}   "
          f"E[max1000] = {E_max(1000):.3f}   max single-draw = {best_single:.3f}   "
          f"P(F > 0.5) = {float(weights[values > 0.5].sum()):.5f}")

print()
print("=" * 88)
print("(3) OBSERVED (from the published 540-run dataset)")
print("=" * 88)
print("L+T:    Baldwin 0.254+-0.196   Pheno 0.772+-0.075")
print("T+Plus: Baldwin 0.086+-0.123   Pheno 0.517+-0.000")
print("L+Plus: Baldwin 0.177+-0.149   Pheno 0.189+-0.156")
