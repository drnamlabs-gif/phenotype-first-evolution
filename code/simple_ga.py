#!/usr/bin/env python3
"""
Simple Genetic Algorithm for Shape Evolution

Solves the substrate problem using basic evolutionary algorithm.
No learning, no epigenome - just genes + mutation + selection.
"""

import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple
from environment import ShapeEnvironment
from primitives import GenomicPrimitives


class SimpleOrganism:
    """Simple organism: genome only"""

    def __init__(self, genome: List[int] = None):
        """
        Args:
            genome: 100-bit genome encoding primitive positions
        """
        if genome is None:
            self.genome = [random.randint(0, 1) for _ in range(100)]
        else:
            self.genome = genome[:]

        self.fitness = 0.0
        self.phenotype = None

    def develop_phenotype(self) -> List[List[int]]:
        """
        Interpret genome and build phenotype.
        Simple rule: every 5 bits encode (primitive_id, x, y)
        """
        grid = [[0]*10 for _ in range(10)]
        primitives = GenomicPrimitives.get_all_primitives()
        prim_list = list(primitives.keys())

        # Parse genome into instructions
        for i in range(0, len(self.genome)-4, 5):
            # Bits [i:i+5] encode one primitive placement
            prim_idx = int(''.join(map(str, self.genome[i:i+2])), 2) % len(prim_list)
            x = int(''.join(map(str, self.genome[i+2:i+4])), 2) % 10
            y = int(''.join(map(str, self.genome[i+4:i+5])), 2) % 10

            prim_name = prim_list[prim_idx]
            prim_func = primitives[prim_name]

            # Apply primitive
            if 'line' in prim_name:
                length = 3
                grid = prim_func(grid, x, y, length)
            else:
                grid = prim_func(grid, x, y)

        self.phenotype = grid
        return grid

    def mutate(self, mutation_rate: float = 0.02):
        """Bit-flip mutation"""
        for i in range(len(self.genome)):
            if random.random() < mutation_rate:
                self.genome[i] = 1 - self.genome[i]
        return self

    def copy(self):
        """Create copy"""
        return SimpleOrganism(self.genome)


class SimpleGeneticAlgorithm:
    """Simple GA for shape evolution"""

    def __init__(self,
                 population_size: int = 50,
                 generations: int = 100,
                 mutation_rate: float = 0.02):
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = [SimpleOrganism() for _ in range(population_size)]
        self.best_fitness = 0.0
        self.avg_fitness = 0.0
        self.history = []

    def evaluate(self, env: ShapeEnvironment):
        """Evaluate all organisms"""
        for org in self.population:
            org.develop_phenotype()
            org.fitness = env.evaluate(org.phenotype) / 100.0  # Normalize to [0,1]

        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Track stats
        self.best_fitness = self.population[0].fitness
        self.avg_fitness = sum(org.fitness for org in self.population) / len(self.population)

    def select_parents(self) -> Tuple[SimpleOrganism, SimpleOrganism]:
        """Tournament selection (pick 2 random, return best)"""
        # Tournament for first parent
        tournament = random.sample(self.population, min(3, len(self.population)))
        parent1 = max(tournament, key=lambda x: x.fitness)

        # Tournament for second parent
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
                print(f"Gen {gen:3d}: Best={self.best_fitness:.1%}, Avg={self.avg_fitness:.1%} ({env.current_shape_name})")

            # Check if converged
            if self.best_fitness > 0.95:
                print(f"  → CONVERGED at gen {gen} with {self.best_fitness:.1%} fitness")
                break

            # Create new population
            new_pop = []

            # Elitism: keep best 10%
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
    """Run GA on a single shape"""
    print(f"\n{'='*60}")
    print(f"  Evolving {shape} shape")
    print(f"  Population: {pop_size}, Generations: {gens}")
    print(f"{'='*60}")

    env = ShapeEnvironment()
    env.set_target(shape)

    ga = SimpleGeneticAlgorithm(pop_size, gens, mutation_rate=0.02)
    final_fitness, history = ga.evolve(env)

    # Find convergence generation
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
    print("█" + "  SIMPLE GENETIC ALGORITHM: SHAPE EVOLUTION  ".center(58) + "█")
    print("█" + " "*58 + "█")
    print("█"*60)

    results = {}

    # Test on single shapes
    for shape in ['L', 'T', 'Plus']:
        result = run_single_shape(shape, pop_size=50, gens=150)
        results[shape] = result

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Shape':<10} {'Fitness':<12} {'Convergence Gen':<15}")
    print("-"*60)
    for shape in ['L', 'T', 'Plus']:
        r = results[shape]
        print(f"{shape:<10} {r['final_fitness']:<12.1%} {r['convergence_gen']:<15}")

    print("\n✅ Evolution complete!")
