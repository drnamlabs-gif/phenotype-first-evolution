"""
Baldwin Effect on Compositional Task
Learning L+T pattern (must learn BOTH together)
"""

import random
import numpy as np
from copy import deepcopy
from compositional_environment import CompositionTask


class BaldwinCompositionOrganism:
    """Baldwin Effect organism on compositional L+T task"""

    def __init__(self):
        self.genome = np.random.randint(0, 2, 100)
        self.learned_patterns = []
        self.fitness = 0.0

    @staticmethod
    def _random_genome():
        return np.random.randint(0, 2, 100)

    def phenotype_from_genome(self):
        """Decode genome to 10x10 grid"""
        grid = np.zeros((10, 10), dtype=int)

        primitives = [
            [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],  # L
            [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)],  # T
            [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],  # Cross
            [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],  # Line
            [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],  # Block
        ]

        for prim_idx in range(5):
            start_bit = prim_idx * 20
            offset_x = int(self.genome[start_bit] * 5)
            offset_y = int(self.genome[start_bit + 1] * 5)
            scale = 1 + int(self.genome[start_bit + 2] * 2)
            active = self.genome[start_bit + 3] == 1

            if active:
                for dx, dy in primitives[prim_idx]:
                    x, y = offset_x + dx * scale, offset_y + dy * scale
                    if 0 <= x < 10 and 0 <= y < 10:
                        grid[x, y] = 1

        return grid

    def learn(self, env, num_trials=20):
        """
        Learning phase: Try random compositions to find L+T pattern
        Challenge: Must find BOTH L and T together
        """
        best_fitness = 0.0
        best_pattern = None

        for trial in range(num_trials):
            test_genome = self._random_genome()
            old_genome = self.genome.copy()
            self.genome = test_genome
            phenotype = self.phenotype_from_genome()

            # Evaluate on compositional task (must have L AND T)
            fitness = env.evaluate_fitness(phenotype)

            if fitness > best_fitness:
                best_fitness = fitness
                best_pattern = phenotype

        self.genome = old_genome

        # Store learned pattern
        if best_pattern is not None and best_fitness > 0.3:
            self.learned_patterns.append({
                'pattern': best_pattern,
                'fitness': best_fitness,
            })

        self.fitness = best_fitness

    def copy(self):
        """
        Reproduction (Baldwin: NO inheritance of learned patterns)
        Offspring inherit genes but must learn from scratch
        """
        child_genome = self.genome.copy()

        # Mutation
        mutation_rate = 0.02
        for i in range(len(child_genome)):
            if random.random() < mutation_rate:
                child_genome[i] = 1 - child_genome[i]

        # KEY: learned_patterns NOT inherited!
        child = BaldwinCompositionOrganism()
        child.genome = child_genome

        return child


def run_baldwin_compositional(max_generations=150, population_size=50):
    """Run Baldwin on compositional L+T task"""

    env = CompositionTask()
    population = [BaldwinCompositionOrganism() for _ in range(population_size)]

    print("\n" + "="*80)
    print("BALDWIN EFFECT ON COMPOSITIONAL L+T TASK")
    print("="*80)
    print("Challenge: Must learn BOTH L and T together (fitness = avg(L, T))")
    print("Difficulty: Only L or only T → 50% fitness (incomplete)")
    print("="*80 + "\n")

    convergence_gen = None

    for generation in range(max_generations):
        # Learning phase
        for organism in population:
            organism.learn(env, num_trials=20)

        # Check convergence
        best_fitness = max(org.fitness for org in population)
        avg_fitness = np.mean([org.fitness for org in population])

        # Component analysis
        best_org = max(population, key=lambda x: x.fitness)
        best_phenotype = best_org.phenotype_from_genome()
        components = env.get_component_scores(best_phenotype)

        if generation % 10 == 0 or generation < 5:
            print(f"Gen {generation:3d}: Best={best_fitness:.3f}, Avg={avg_fitness:.3f}, "
                  f"L={components['L_score']:.2f}, T={components['T_score']:.2f}")

        if best_fitness >= 0.95 and convergence_gen is None:
            convergence_gen = generation
            print(f"  ✓ Converged at generation {generation}!")

        # Selection and reproduction
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        elite = sorted_pop[:10]

        new_population = elite[:]
        while len(new_population) < population_size:
            parent = random.choice(sorted_pop[:25])
            child = parent.copy()
            new_population.append(child)

        population = new_population[:population_size]

    # Final results
    best_organism = max(population, key=lambda x: x.fitness)
    best_phenotype = best_organism.phenotype_from_genome()
    final_fitness = env.evaluate_fitness(best_phenotype)
    final_components = env.get_component_scores(best_phenotype)

    if convergence_gen is None:
        convergence_gen = max_generations

    print(f"\n{'='*80}")
    print("BALDWIN RESULTS")
    print(f"{'='*80}")
    print(f"Final fitness: {final_fitness:.3f}")
    print(f"L score: {final_components['L_score']:.3f}")
    print(f"T score: {final_components['T_score']:.3f}")
    print(f"Both present: {final_components['both_present']}")
    print(f"Convergence: Generation {convergence_gen}")
    print(f"{'='*80}\n")

    return {
        'method': 'Baldwin',
        'final_fitness': final_fitness,
        'convergence_gen': convergence_gen,
        'L_score': final_components['L_score'],
        'T_score': final_components['T_score'],
    }


if __name__ == "__main__":
    result = run_baldwin_compositional()
    print(f"\nResult: {result}")
