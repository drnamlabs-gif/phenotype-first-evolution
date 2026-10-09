"""
Regenerate Figure 1 (task environment) using the REAL target patterns and the
REAL fitness functions from environment.py / metrics.py, instead of the
hand-drawn mock patterns the old figure used.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from environment import ShapeEnvironment
from metrics import translation_invariant_iou, evaluate_shape_similarity

GRID = 10


def grid_from_env(shape_name):
    env = ShapeEnvironment()
    env.set_target(shape_name)
    return env.current_target  # list of lists, grid[y][x]


def new_grid():
    return [[0] * GRID for _ in range(GRID)]


def set_cells(grid, cells):
    g = [row[:] for row in grid]
    for (x, y) in cells:
        g[y][x] = 1
    return g


def draw_grid(ax, pattern, title, diff=None):
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 9.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(range(10), fontsize=6)
    ax.set_yticklabels(range(10), fontsize=6)
    ax.grid(True, color='gray', linewidth=0.5, alpha=0.3)
    for y in range(GRID):
        for x in range(GRID):
            if diff is not None and diff[y][x] == 1:
                rect = Rectangle((x - 0.4, y - 0.4), 0.8, 0.8, linewidth=2,
                                  edgecolor='red', facecolor='#ffcccc', alpha=0.9)
                ax.add_patch(rect)
            elif pattern[y][x] == 1:
                rect = Rectangle((x - 0.4, y - 0.4), 0.8, 0.8, linewidth=1.5,
                                  edgecolor='darkblue', facecolor='lightblue', alpha=0.85)
                ax.add_patch(rect)
            else:
                rect = Rectangle((x - 0.4, y - 0.4), 0.8, 0.8, linewidth=0.4,
                                  edgecolor='lightgray', facecolor='white', alpha=0.3)
                ax.add_patch(rect)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel('x', fontsize=7)
    ax.set_ylabel('y', fontsize=7)


fig = plt.figure(figsize=(15, 7.5))
gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.4)

# --- Row 1: the three real target patterns -----------------------------
L = grid_from_env('L')
T = grid_from_env('T')
P = grid_from_env('Plus')

for col, (name, g) in enumerate([('L', L), ('T', T), ('Plus', P)]):
    n_cells = sum(sum(row) for row in g)
    ax = fig.add_subplot(gs[0, col])
    draw_grid(ax, g, f"({chr(65+col)}) {name} pattern ({n_cells} cells)")

ax = fig.add_subplot(gs[0, 3])
ax.axis('off')
ax.text(0.0, 0.9, "Patterns", fontsize=11, fontweight='bold')
ax.text(0.0, 0.70, "L, T, Plus each occupy\n5 fixed cells on a\n10x10 grid.", fontsize=9)
ax.text(0.0, 0.40, "Defined in\nenvironment.py::\n_create_target_shape()", fontsize=8, style='italic')
ax.text(0.0, 0.10, "(D)", fontsize=9, fontweight='bold')

# --- Row 2, panel 1: simple task, fitness = translation-invariant IoU ---
# organism: the L shape shifted by (+2,-1), plus one stray off-target cell
L_shifted = [(x + 2, y - 1) for (x, y) in [(3, 5), (3, 6), (3, 7), (4, 7), (5, 7)]]
organism_simple = set_cells(new_grid(), L_shifted + [(0, 0)])

score_simple, dx, dy = translation_invariant_iou(organism_simple, L, search_range=5, use_recall=False)

ax = fig.add_subplot(gs[1, 0])
draw_grid(ax, organism_simple, "(B) Organism (simple task)")

ax = fig.add_subplot(gs[1, 1])
ax.axis('off')
ax.text(0.0, 0.95, "Simple-task fitness", fontsize=10, fontweight='bold')
ax.text(0.0, 0.75, "Metric: translation-invariant IoU\n(search range ±5 cells)", fontsize=9)
ax.text(0.0, 0.45,
        f"Best alignment: shift ({dx:+d},{dy:+d})\n"
        f"IoU = intersection / union\n"
        f"Fitness = {score_simple*100:.1f}%",
        fontsize=9)
ax.text(0.0, 0.08, "Note: unlike compositional tasks,\nextra off-target cells REDUCE\nIoU (no free preadaptation here).",
        fontsize=8, style='italic')

# --- Row 2, panel 2-3: compositional task, fitness = mean of two independent recalls ---
# organism: full T at its canonical location, L missing one cell (translated to overlap less)
T_full = [(3, 3), (4, 3), (5, 3), (4, 4), (4, 5)]
L_partial = [(3, 5), (3, 6), (3, 7), (4, 7)]  # 4 of 5 L cells (last cell (5,7) dropped)
organism_comp = set_cells(new_grid(), T_full + L_partial)

score_L = evaluate_shape_similarity(organism_comp, L, translation_invariant=True, search_range=7, use_recall=True) / 100.0
score_T = evaluate_shape_similarity(organism_comp, T, translation_invariant=True, search_range=7, use_recall=True) / 100.0
fitness_comp = (score_L + score_T) / 2.0

ax = fig.add_subplot(gs[1, 2])
draw_grid(ax, organism_comp, "(C1) Organism (L+T task)")

ax = fig.add_subplot(gs[1, 3])
ax.axis('off')
ax.text(0.0, 0.95, "Compositional-task fitness", fontsize=10, fontweight='bold')
ax.text(0.0, 0.72,
        "Each component pattern is searched\nfor independently, anywhere in the\norganism's grid (search range ±7).",
        fontsize=8.5)
ax.text(0.0, 0.42,
        f"recall(L) = {score_L*100:.0f}%  (4/5 L-cells present)\n"
        f"recall(T) = {score_T*100:.0f}%  (T fully present)\n"
        f"Fitness = mean = {fitness_comp*100:.1f}%",
        fontsize=9)
ax.text(0.0, 0.05, "Extra cells never penalised here\n(this is where preadaptation applies).",
        fontsize=8, style='italic')

fig.suptitle("Task environment and fitness calculation (patterns and scores computed with the actual project code)",
             fontsize=11, fontweight='bold', y=0.98)

fig.savefig("fig1_task_environment_NEW.png", dpi=200, bbox_inches='tight')
print("saved fig1_task_environment_NEW.png")
print(f"simple score={score_simple*100:.2f}% dx={dx} dy={dy}")
print(f"comp: recall_L={score_L*100:.2f}% recall_T={score_T*100:.2f}% fitness={fitness_comp*100:.2f}%")
