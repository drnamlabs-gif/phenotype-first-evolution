"""
Comprehensive Multi-Task Experiment
3 Compositional Tasks × 30 runs × 2 algorithms = 180 total experiments
Tasks: L+T, L+Plus, T+Plus
"""

import random
import numpy as np
import csv
from compositional_environments_extended import get_task
from baldwin_compositional import run_baldwin_compositional as run_baldwin_base
from phenotype_first_compositional import run_phenotype_first_compositional as run_phenotype_first_base


def run_baldwin_on_task(task, max_generations=150, population_size=50):
    """Run Baldwin on specific task"""
    from baldwin_compositional import BaldwinCompositionOrganism

    population = [BaldwinCompositionOrganism() for _ in range(population_size)]
    best_fitness_overall = 0.0
    convergence_gen = None

    for generation in range(max_generations):
        for organism in population:
            organism.learn(task, num_trials=20)

        best_fitness = max(org.fitness for org in population)
        best_org = max(population, key=lambda x: x.fitness)
        best_phenotype = best_org.phenotype_from_genome()
        components = task.get_component_scores(best_phenotype)

        if best_fitness >= best_fitness_overall:
            best_fitness_overall = best_fitness

        if best_fitness >= 0.95 and convergence_gen is None:
            convergence_gen = generation

        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        elite = sorted_pop[:10]
        new_population = elite[:]
        while len(new_population) < population_size:
            parent = random.choice(sorted_pop[:25])
            child = parent.copy()
            new_population.append(child)
        population = new_population[:population_size]

    best_organism = max(population, key=lambda x: x.fitness)
    best_phenotype = best_organism.phenotype_from_genome()
    final_fitness = task.evaluate_fitness(best_phenotype)
    final_components = task.get_component_scores(best_phenotype)

    if convergence_gen is None:
        convergence_gen = max_generations

    p1_key = list(final_components.keys())[0]
    p2_key = list(final_components.keys())[1]

    return {
        'method': 'Baldwin',
        'final_fitness': final_fitness,
        'convergence_gen': convergence_gen,
        f'{p1_key}': final_components[p1_key],
        f'{p2_key}': final_components[p2_key],
    }


def run_phenotype_on_task(task, max_generations=150, population_size=50):
    """Run Phenotype-First on specific task"""
    from phenotype_first_compositional import PhenotypeFirstCompositionOrganism
    from copy import deepcopy

    population = [PhenotypeFirstCompositionOrganism() for _ in range(population_size)]
    best_fitness_overall = 0.0
    convergence_gen = None

    for generation in range(max_generations):
        for organism in population:
            organism.learn(task, num_trials=20)

        best_fitness = max(org.fitness for org in population)
        best_org = max(population, key=lambda x: x.fitness)
        best_phenotype = best_org.phenotype_from_genome()

        inherited_phenotype = np.zeros((10, 10), dtype=int)
        for pattern in best_org.epigenome_patterns:
            inherited_phenotype |= pattern
        best_phenotype = best_phenotype | inherited_phenotype

        components = task.get_component_scores(best_phenotype)

        if best_fitness >= best_fitness_overall:
            best_fitness_overall = best_fitness

        if best_fitness >= 0.95 and convergence_gen is None:
            convergence_gen = generation

        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        elite = sorted_pop[:10]
        new_population = elite[:]
        while len(new_population) < population_size:
            parent = random.choice(sorted_pop[:25])
            child = parent.copy()
            new_population.append(child)
        population = new_population[:population_size]

    best_organism = max(population, key=lambda x: x.fitness)
    best_phenotype = best_organism.phenotype_from_genome()

    inherited_phenotype = np.zeros((10, 10), dtype=int)
    for pattern in best_organism.epigenome_patterns:
        inherited_phenotype |= pattern
    best_phenotype = best_phenotype | inherited_phenotype

    final_fitness = task.evaluate_fitness(best_phenotype)
    final_components = task.get_component_scores(best_phenotype)

    if convergence_gen is None:
        convergence_gen = max_generations

    p1_key = list(final_components.keys())[0]
    p2_key = list(final_components.keys())[1]

    return {
        'method': 'Phenotype-First',
        'final_fitness': final_fitness,
        'convergence_gen': convergence_gen,
        f'{p1_key}': final_components[p1_key],
        f'{p2_key}': final_components[p2_key],
        'patterns_inherited': len(best_organism.epigenome_patterns),
    }


def run_comprehensive_multi_task_experiment(num_runs=30):
    """Run 30 experiments per task × 2 methods × 3 tasks = 180 total"""

    tasks = ['L+T', 'L+Plus', 'T+Plus']
    all_results = []

    print("\n" + "="*80)
    print("COMPREHENSIVE MULTI-TASK COMPOSITIONAL EXPERIMENT")
    print("="*80)
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Methods: Baldwin, Phenotype-First")
    print(f"Runs: {num_runs} per (task, method) pair")
    print(f"Total experiments: {len(tasks)} × 2 × {num_runs} = {len(tasks)*2*num_runs}")
    print("="*80 + "\n")

    for task_name in tasks:
        print(f"\n{'='*80}")
        print(f"TASK: {task_name}")
        print(f"{'='*80}\n")

        task = get_task(task_name)
        task.print_info()

        # Get pattern names
        import numpy as np
        if task_name == 'L+T':
            p1_key, p2_key = 'L_score', 'T_score'
        elif task_name == 'L+Plus':
            p1_key, p2_key = 'L_score', 'Plus_score'
        else:  # T+Plus
            p1_key, p2_key = 'T_score', 'Plus_score'

        # Run Baldwin 30 times
        print(f"\n▶ Running BALDWIN on {task_name} (30 runs)...\n")
        for run_id in range(1, num_runs + 1):
            seed = run_id * 42
            random.seed(seed)
            np.random.seed(seed)

            try:
                result = run_baldwin_on_task(task)
                result['task'] = task_name
                result['run'] = run_id
                result['seed'] = seed
                all_results.append(result)

                fitness = result['final_fitness']
                p1 = result.get(p1_key, 0)
                p2 = result.get(p2_key, 0)
                print(f"  Run {run_id:2d}/30: Fitness={fitness:.3f}, {p1_key}={p1:.3f}, {p2_key}={p2:.3f}")
            except Exception as e:
                print(f"  Run {run_id:2d}/30: ERROR - {str(e)}")

        # Run Phenotype-First 30 times
        print(f"\n▶ Running PHENOTYPE-FIRST on {task_name} (30 runs)...\n")
        for run_id in range(1, num_runs + 1):
            seed = run_id * 42
            random.seed(seed)
            np.random.seed(seed)

            try:
                result = run_phenotype_on_task(task)
                result['task'] = task_name
                result['run'] = run_id
                result['seed'] = seed
                all_results.append(result)

                fitness = result['final_fitness']
                p1 = result.get(p1_key, 0)
                p2 = result.get(p2_key, 0)
                print(f"  Run {run_id:2d}/30: Fitness={fitness:.3f}, {p1_key}={p1:.3f}, {p2_key}={p2:.3f}")
            except Exception as e:
                print(f"  Run {run_id:2d}/30: ERROR - {str(e)}")

    # Save results
    csv_file = 'compositional_results_all_tasks.csv'
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['task', 'method', 'run', 'seed', 'final_fitness', 'convergence_gen',
                     'pattern1_score', 'pattern2_score', 'patterns_inherited']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            # Get pattern scores
            keys = list(result.keys())
            pattern1_key = [k for k in keys if '_score' in k][0] if any('_score' in k for k in keys) else None
            pattern2_key = [k for k in keys if '_score' in k and k != pattern1_key][0] if len([k for k in keys if '_score' in k]) > 1 else None

            row = {
                'task': result.get('task', ''),
                'method': result.get('method', ''),
                'run': result.get('run', ''),
                'seed': result.get('seed', ''),
                'final_fitness': result.get('final_fitness', ''),
                'convergence_gen': result.get('convergence_gen', ''),
                'pattern1_score': result.get(pattern1_key, '') if pattern1_key else '',
                'pattern2_score': result.get(pattern2_key, '') if pattern2_key else '',
                'patterns_inherited': result.get('patterns_inherited', ''),
            }
            writer.writerow(row)

    print(f"\n✅ All results saved to: {csv_file}")
    print(f"Total experiments completed: {len(all_results)}")

    return all_results


if __name__ == "__main__":
    results = run_comprehensive_multi_task_experiment(num_runs=30)
    print("\n✅ Experiment complete!")
