"""
Generate the corrected Figure 1 (both experimental environments, drawn from the
real code) and the new Figure 6 (analytical predictions vs observed results).
"""
import sys
import itertools
sys.path.insert(0, "..")   # for environment.py / metrics.py (experiment 1)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from environment import ShapeEnvironment
from metrics import translation_invariant_iou
from compositional_environments_extended import L_target, T_target, Plus_target

GRID = 10

# ---------------------------------------------------------------- helpers
def draw_grid(ax, filled, title, colors=None, missed=None, fontsize=8):
    """filled: dict (x,y)->color or set of (row,col); draws 10x10 grid"""
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 9.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(range(10), fontsize=5)
    ax.set_yticklabels(range(10), fontsize=5)
    ax.grid(True, color="gray", linewidth=0.4, alpha=0.3)
    for r in range(GRID):
        for c in range(GRID):
            cell = (r, c)
            if missed is not None and cell in missed:
                rect = Rectangle((c - 0.4, r - 0.4), 0.8, 0.8, linewidth=1.8,
                                  edgecolor="red", facecolor="#ffd5d5", alpha=0.95)
                ax.add_patch(rect)
                ax.plot([c - 0.28, c + 0.28], [r - 0.28, r + 0.28], "r-", lw=1.6)
                ax.plot([c - 0.28, c + 0.28], [r + 0.28, r - 0.28], "r-", lw=1.6)
            elif isinstance(filled, dict) and cell in filled:
                rect = Rectangle((c - 0.4, r - 0.4), 0.8, 0.8, linewidth=1.2,
                                  edgecolor="black", facecolor=filled[cell], alpha=0.85)
                ax.add_patch(rect)
            elif not isinstance(filled, dict) and cell in filled:
                rect = Rectangle((c - 0.4, r - 0.4), 0.8, 0.8, linewidth=1.2,
                                  edgecolor="darkblue", facecolor="lightblue", alpha=0.85)
                ax.add_patch(rect)
            else:
                rect = Rectangle((c - 0.4, r - 0.4), 0.8, 0.8, linewidth=0.3,
                                  edgecolor="lightgray", facecolor="white", alpha=0.3)
                ax.add_patch(rect)
    ax.set_title(title, fontsize=fontsize, fontweight="bold")


def cells_of(arr):
    return {(r, c) for r in range(GRID) for c in range(GRID) if arr[r][c]}

# ------------------------------------------- experiment-1 patterns (list grids)
env = ShapeEnvironment()
exp1 = {}
for s in ["L", "T", "Plus"]:
    env.set_target(s)
    exp1[s] = env.current_target

# worked IoU example: L shifted by (+2,-1) plus one stray cell
L_cells = cells_of(exp1["L"])
org1 = {(r - 1, c + 2) for (r, c) in L_cells} | {(0, 0)}
org1_grid = [[1 if (r, c) in org1 else 0 for c in range(10)] for r in range(10)]
iou, dx, dy = translation_invariant_iou(org1_grid, exp1["L"], search_range=5, use_recall=False)

# ------------------------------------------- experiment-2 reachable set
PRIMS = [
    [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
    [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)],
    [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
]
coverable = set()
for prim in PRIMS:
    for ox in (0, 5):
        for oy in (0, 5):
            for sc in (1, 3):
                for dx_, dy_ in prim:
                    x, y = ox + dx_ * sc, oy + dy_ * sc
                    if 0 <= x < 10 and 0 <= y < 10:
                        coverable.add((x, y))

tgt = {"L": cells_of(L_target), "T": cells_of(T_target), "Plus": cells_of(Plus_target)}

# ================================================================ FIGURE 1
fig = plt.figure(figsize=(15, 7.6))
gs = fig.add_gridspec(2, 4, hspace=0.5, wspace=0.38)

for col, s in enumerate(["L", "T", "Plus"]):
    ax = fig.add_subplot(gs[0, col])
    n = len(cells_of(exp1[s]))
    draw_grid(ax, cells_of(exp1[s]), f"({chr(65+col)}) Exp. 1 target: {s} ({n} cells)")

ax = fig.add_subplot(gs[0, 3])
draw_grid(ax, org1, f"(D) Exp. 1 worked example\nIoU best shift ({dx:+d},{dy:+d}): fitness {iou*100:.1f}%")

# (E) experiment-2 fixed targets, colour-coded
colmap = {}
for cell in tgt["L"]:
    colmap[cell] = "#9ecae9"
for cell in tgt["T"]:
    colmap[cell] = "#a8dba8"
for cell in tgt["Plus"]:
    colmap[cell] = "#f5c48f"
ax = fig.add_subplot(gs[1, 0])
draw_grid(ax, colmap, "(E) Exp. 2 fixed targets\nL (blue, 5), T (green, 6), Plus (orange, 5)")

# (F) reachable set
colmap2 = {c: "#d9d9d9" for c in coverable}
all_t = tgt["L"] | tgt["T"] | tgt["Plus"]
missed = all_t - coverable
for cell in all_t & coverable:
    colmap2[cell] = "#6baed6"
ax = fig.add_subplot(gs[1, 1])
draw_grid(ax, colmap2, "(F) Exp. 2 reachable cells (grey),\ncovered target cells (blue), unreachable (red X)", missed=missed)

ax = fig.add_subplot(gs[1, 2:4])
ax.axis("off")
rl = len(tgt["L"] & coverable) / len(tgt["L"])
rt = len(tgt["T"] & coverable) / len(tgt["T"])
rp = len(tgt["Plus"] & coverable) / len(tgt["Plus"])
ax.text(0.02, 0.95, "(G) Reachability ceilings of the Exp. 2 encoding", fontsize=11,
        fontweight="bold", va="top")
ax.text(0.02, 0.80,
        f"max recall(L)    = {len(tgt['L'] & coverable)}/{len(tgt['L'])} = {rl:.3f}\n"
        f"max recall(T)    = {len(tgt['T'] & coverable)}/{len(tgt['T'])} = {rt:.3f}\n"
        f"max recall(Plus) = {len(tgt['Plus'] & coverable)}/{len(tgt['Plus'])} = {rp:.3f}",
        fontsize=10, family="monospace", va="top")
ax.text(0.02, 0.48,
        f"F*(L+T)    = {(rl+rt)/2:.3f}\n"
        f"F*(T+Plus) = {(rt+rp)/2:.3f}\n"
        f"F*(L+Plus) = {(rl+rp)/2:.3f}",
        fontsize=10, family="monospace", va="top")
ax.text(0.02, 0.16,
        "The Plus target sits at the grid centre; the encoding's offsets {0,5}\n"
        "cannot reach 4 of its 5 cells.",
        fontsize=9, style="italic", va="top")

fig.suptitle("Task environments and fitness calculation (all panels computed from the project code)",
             fontsize=11, fontweight="bold", y=0.99)
fig.savefig("../biosystems_submission/figures/fig1_task_environment.png", dpi=200, bbox_inches="tight")
fig.savefig("../arxiv_version/figures/fig1_task_environment.png", dpi=200, bbox_inches="tight")
print("fig1 saved")

# ================================================================ FIGURE 6
# analytical numbers from belew_analysis.py (exact enumeration)
tasks = ["L+T", "T+Plus", "L+Plus"]
pred_baldwin = [0.204, 0.086, 0.130]          # E[F(X)]
pred_pheno   = [0.817, 0.517, 0.130]          # F*; L+Plus: ratchet disengaged -> E[F(X)]
obs_baldwin  = [0.254, 0.086, 0.177]
obs_baldwin_sd = [0.196, 0.123, 0.149]
obs_pheno    = [0.772, 0.517, 0.189]
obs_pheno_sd = [0.075, 0.000, 0.156]

x = np.arange(len(tasks))
w = 0.19
fig2, ax = plt.subplots(figsize=(8.6, 4.6))
ax.bar(x - 1.5 * w, pred_baldwin, w, label="Baldwin, predicted  E[F(X)]",
       color="#f5c48f", edgecolor="black", hatch="//")
ax.bar(x - 0.5 * w, obs_baldwin, w, yerr=obs_baldwin_sd, capsize=3,
       label="Baldwin, observed", color="#e08214", edgecolor="black")
ax.bar(x + 0.5 * w, pred_pheno, w, label="Phenopoiesis, predicted  $F^*$",
       color="#a8dba8", edgecolor="black", hatch="//")
ax.bar(x + 1.5 * w, obs_pheno, w, yerr=obs_pheno_sd, capsize=3,
       label="Phenopoiesis, observed", color="#2b8c5a", edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=11)
ax.set_ylabel("Final fitness", fontsize=11)
ax.set_ylim(0, 1.0)
ax.legend(fontsize=8.5, loc="upper right")
ax.set_title("Exact analytical predictions vs observed results (30 runs per condition)",
             fontsize=11, fontweight="bold")
ax.annotate("ratchet\ndisengaged", xy=(2 + 0.5 * w, 0.14), xytext=(2.05, 0.42),
            fontsize=8, style="italic", ha="center",
            arrowprops=dict(arrowstyle="->", lw=0.9))
for xi, (p, o) in enumerate(zip(pred_pheno, obs_pheno)):
    if xi == 1:
        ax.annotate("exact match,\ns.d. = 0", xy=(xi + 1.5 * w, o + 0.02), xytext=(xi + 0.45, 0.72),
                    fontsize=8, style="italic", ha="center",
                    arrowprops=dict(arrowstyle="->", lw=0.9))
fig2.tight_layout()
fig2.savefig("../biosystems_submission/figures/fig6_theory_vs_observed.png", dpi=200, bbox_inches="tight")
fig2.savefig("../arxiv_version/figures/fig6_theory_vs_observed.png", dpi=200, bbox_inches="tight")
print("fig6 saved")
