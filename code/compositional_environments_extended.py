"""
Three Compositional Task Environments
Tests: L+T, L+Plus, T+Plus (30 runs each)
180 total experiments: 3 tasks × 2 methods × 30 runs
"""

import numpy as np


class CompositionTask:
    """Base class for compositional tasks"""

    def __init__(self, task_name, pattern1, pattern2, p1_name, p2_name):
        self.task_name = task_name
        self.pattern1_target = pattern1
        self.pattern2_target = pattern2
        self.p1_name = p1_name
        self.p2_name = p2_name
        self.composite_target = self.pattern1_target | self.pattern2_target

    def evaluate_fitness(self, phenotype):
        """Fitness = average of both pattern recalls"""
        p1_intersection = np.sum(phenotype & self.pattern1_target)
        p2_intersection = np.sum(phenotype & self.pattern2_target)

        p1_recall = p1_intersection / np.sum(self.pattern1_target)
        p2_recall = p2_intersection / np.sum(self.pattern2_target)

        return (p1_recall + p2_recall) / 2

    def get_component_scores(self, phenotype):
        """Get separate scores for each pattern"""
        p1_intersection = np.sum(phenotype & self.pattern1_target)
        p2_intersection = np.sum(phenotype & self.pattern2_target)

        p1_recall = p1_intersection / np.sum(self.pattern1_target)
        p2_recall = p2_intersection / np.sum(self.pattern2_target)

        return {
            f'{self.p1_name}_score': p1_recall,
            f'{self.p2_name}_score': p2_recall,
            'both_present': (p1_recall > 0.5) and (p2_recall > 0.5),
        }

    def print_info(self):
        """Print task visualization"""
        print("\n" + "="*80)
        print(f"COMPOSITIONAL TASK: {self.task_name}")
        print("="*80)
        print(f"\nTarget: {self.p1_name}+{self.p2_name} pattern")
        print(f"{self.p1_name} target cells: {np.sum(self.pattern1_target)}")
        print(f"{self.p2_name} target cells: {np.sum(self.pattern2_target)}")
        print(f"Composite target cells: {np.sum(self.composite_target)}")
        print("\nVisualization:")
        print(f"\n{self.p1_name} pattern:")
        self._print_grid(self.pattern1_target)
        print(f"\n{self.p2_name} pattern:")
        self._print_grid(self.pattern2_target)
        print(f"\n{self.p1_name}+{self.p2_name} Composite (target):")
        self._print_grid(self.composite_target)
        print("="*80 + "\n")

    @staticmethod
    def _print_grid(grid):
        """Print grid visualization"""
        for row in grid:
            print("  " + "".join(["█" if cell else "·" for cell in row]))


# Define patterns
L_target = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])

T_target = np.array([
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])

# Plus pattern (centered cross): 5 cells
Plus_target = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])


def get_task(task_name):
    """Factory to create compositional tasks"""
    if task_name == 'L+T':
        return CompositionTask('L+T', L_target, T_target, 'L', 'T')
    elif task_name == 'L+Plus':
        return CompositionTask('L+Plus', L_target, Plus_target, 'L', 'Plus')
    elif task_name == 'T+Plus':
        return CompositionTask('T+Plus', T_target, Plus_target, 'T', 'Plus')
    else:
        raise ValueError(f"Unknown task: {task_name}")


if __name__ == "__main__":
    # Show all 3 tasks
    for task_name in ['L+T', 'L+Plus', 'T+Plus']:
        task = get_task(task_name)
        task.print_info()
