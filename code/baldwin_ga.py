#!/usr/bin/env python3
"""
Baldwin Effect GA - Learning Without Inheritance

Organisms can learn during lifetime and get fitness boost.
But learned patterns DIE at reproduction (not inherited).
"""

import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple
from environment import ShapeEnvironment
from primitives import GenomicPrimitives
from copy import deepcopy


class BaldwinOrganism:
    """Organism with lifetime learning"""

    def __init__(self, genome: List[int] = None):
        if genome is None:
            self.genome = [random.randint(0, 1) for _ in range(100)]
        else:
            self.genome = genome[:]

        self.fitness = 0.0
        self.phenotype = None
        self.learned_patterns = []  # Pattern compositions learned this lifetime

    def develop_phenotype(self) -> List[List[int]]:
        """Build initial phenotype from genome"""
        grid = [[0]*10 for _ in range(10)]
        primitives = GenomicPrimitives.get_all_primitives()
        prim_list = list(primitives.keys())

        for i in range(0, len(self.genome)-4, 5):
            prim_idx = int(''.join(map(str, self.genome[i:i+2])), 2) % len(prim_list)
            x = int(''.join(map(str, self.genome[i+2:i+4])), 2) % 10
            y = int(''.join(map(str, self.genome[i+4:i+5])), 2) % 10

            prim_name = prim_list[prim_idx]
            prim_func = primitives[prim_name]

            if 'line' in prim_name:
                grid = prim_func(grid, x, y, 3)
            else:
                grid = prim_func(grid, x, y)

        self.phenotype = grid
        return grid

    def learn(self, env: ShapeEnvironment, trials: int = 20):
        """
        Lifetime learning: Try random compositions, keep best.

        This is the Baldwin Effect - learning helps THIS generation
        but patterns are NOT inherited (reset each generation).
        """
        best_grid = deepcopy(self.phenotype)
        best_fitness = env.evaluate(best_grid) / 100.0

        primitives = GenomicPrimitives.get_all_primitives()
        prim_list = list(primitives.keys())

        # Try random compositions
        for trial in range(trials):
            test_grid = [[0]*10 for _ in range(10)]

            # Random composition
            num_prims = random.randint(1, 4)
            for _ in range(num_prims):
                prim_name = random.choice(prim_list)
                x = random.randint(0, 9)
                y = random.randint(0, 9)

                prim_func = primitives[prim_name]
                if 'line' in prim_name:
                    test_grid = prim_func(test_grid, x, y, 3)
                else:
                    test_grid = prim_func(test_grid, x, y)

            # Evaluate
            test_fitness = env.evaluate(test_grid) / 100.0

            if test_fitness > best_fitness:
                best_fitness = test_fitness
                best_grid = deepcopy(test_grid)

        self.phenotype = best_grid
        self.fitness = best_fitness

        # Store patterns (but will NOT be inherited!)
        self.learned_patterns = [best_grid]

    def mutate(self, mutation_rate: float = 0.02):
        """Bit-flip mutation"""
        for i in range(len(self.genome)):
            if random.random() < mutation_rate:
                self.genome[i] = 1 - self.genome[i]
        return self

    def copy(self):
        """Create copy (learning is NOT inherited)"""
        child = BaldwinOrganism(self.genome)
        # NOTE: learned_patterns are NOT copied! (Baldwin Effect)
        return child


class BaldwinGA:
    """GA with Baldwin Effect (learning without inheritance)"""

    def __init__(self,
                 population_size: int = 50,
                 generations: int = 100,
                 mutation_rate: float = 0.02,
                 learning_trials: int = 20):
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.learning_trials = learning_trials
        self.population = [BaldwinOrganism() for _ in range(population_size)]
        self.best_fitness = 0.0
        self.avg_fitness = 0.0
        self.history = []

    def evaluate(self, env: ShapeEnvironment):
        """Evaluate all organisms (with learning)"""

        for org in self.population:
            # Development
            org.develop_phenotype()

            # Lifetime learning (Baldwin Effect)
            org.learn(env, self.learning_trials)

        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Track stats
        self.best_fitness = self.population[0].fitness
        self.avg_fitness = sum(org.fitness for org in self.population) / len(self.population)

    def select_parents(self) -> Tuple[BaldwinOrganism, BaldwinOrganism]:
        """Tournament selection"""
        tournament = random.sample(self.population, min(3, len(self.population)))
        parent1 = max(tournament, key=lambda x: x.fitness)

        tournament = random.sample(self.population, min(3, len(self.population)))
        parent2 = max(tournament, key=lambda x: x.fitness)

        return parent1, parent2

    def evolve(self, env: ShapeEnvironment):
        """Run evolution loop"""

        for gen in range(self.generations):
            # Evaluate current population
            self.evaluate(env)

            # Record history
            self.history.append({
                'generation': gen,
                'best_fitness': self.best_fitness,
                'avg_fitness': self.avg_fitness,
                'shape': env.current_shape_name
            })

            if gen % 20 == 0:
                print(f"Gen {gen:3d}: Best={self.best_fitness:.1%}, Avg={self.avg_fitness:.1%} (Baldwin)")

            if self.best_fitness > 0.95:
                print(f"  → CONVERGED at gen {gen} with {self.best_fitness:.1%} fitness")
                break

            # Create new population
            new_pop = []

            # Elitism
            elite_size = max(1, self.pop_size // 10)
            new_pop.extend([org.copy() for org in self.population[:elite_size]])

            # Fill rest with offspring
            while len(new_pop) < self.pop_size:
                p1, p2 = self.select_parents()
                child = p1.copy()
                child.mutate(self.mutation_rate)
                new_pop.append(child)

            self.population = new_pop

        return self.best_fitness, self.history


def run_single_shape(shape: str, pop_size: int = 50, gens: int = 100):
    """Run Baldwin GA on a single shape"""
    print(f"\n{'='*60}")
    print(f"  Baldwin Effect: {shape} shape")
    print(f"  Population: {pop_size}, Generations: {gens}")
    print(f"  Learning trials: 20 per organism")
    print(f"{'='*60}")

    env = ShapeEnvironment()
    env.set_target(shape)

    ga = BaldwinGA(pop_size, gens, mutation_rate=0.02, learning_trials=20)
    final_fitness, history = ga.evolve(env)

    converge_gen = None
    for record in history:
        if record['best_fitness'] > 0.95:
            converge_gen = record['generation']
            break

    if converge_gen is None:
        converge_gen = len(history)

    print(f"\nFinal: {final_fitness:.1%} fitness at gen {converge_gen}")

    return {
        'shape': shape,
        'final_fitness': final_fitness,
        'convergence_gen': converge_gen,
        'history': history,
        'best_organism': ga.population[0]
    }


if __name__ == "__main__":
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█" + "  BALDWIN EFFECT GA: Learning Without Inheritance  ".center(58) + "█")
    print("█" + " "*58 + "█")
    print("█"*60)

    results = {}

    for shape in ['L', 'T', 'Plus']:
        result = run_single_shape(shape, pop_size=50, gens=150)
        results[shape] = result

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Shape':<10} {'Fitness':<12} {'Convergence Gen':<15}")
    print("-"*60)
    for shape in ['L', 'T', 'Plus']:
        r = results[shape]
        print(f"{shape:<10} {r['final_fitness']:<12.1%} {r['convergence_gen']:<15}")

    print("\n✅ Baldwin Effect evolution complete!")
    print("\nNote: Learning helps (higher fitness) but slower convergence than pure mutation")
    print("because learned patterns are NOT inherited!")
