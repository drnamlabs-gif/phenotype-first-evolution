#!/usr/bin/env python3
"""
Comprehensive 30-run experiment for paper
Tests: Gene-Centric, Baldwin, Lamarckian, Phenotype-First
Shapes: L, T, Plus
Total: 30 runs × 4 methods × 3 shapes = 360 experiments
"""

import sys
import os
import csv
import json
import random
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_ga import run_single_shape as run_gene_centric
from baldwin_ga import run_single_shape as run_baldwin
from phenotype_first_ga import run_single_shape as run_phenotype_first
from lamarckian_ga import run_lamarckian_ga as run_lamarckian_orig


def run_lamarckian(shape, pop_size=50, gens=150):
    """Wrapper to match interface"""
    return run_lamarckian_orig(shape, max_generations=gens, population_size=pop_size)


class ExperimentRunner:
    def __init__(self, num_runs=30):
        self.num_runs = num_runs
        self.shapes = ['L', 'T', 'Plus']
        self.methods = {
            'Gene-Centric': run_gene_centric,
            'Baldwin': run_baldwin,
            'Lamarckian': run_lamarckian,
            'Phenotype-First': run_phenotype_first,
        }
        self.results = []
        self.start_time = None

    def run_all_experiments(self):
        """Run all experiments with consistent seeds"""
        self.start_time = datetime.now()
        total_experiments = len(self.methods) * len(self.shapes) * self.num_runs
        current = 0

        print("\n" + "="*80)
        print("COMPREHENSIVE EVOLUTIONARY ALGORITHMS EXPERIMENT")
        print("="*80)
        print(f"Methods: {list(self.methods.keys())}")
        print(f"Shapes: {self.shapes}")
        print(f"Runs per condition: {self.num_runs}")
        print(f"Total experiments: {total_experiments}")
        print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")

        for method_name, method_func in self.methods.items():
            print(f"\n{'='*80}")
            print(f"METHOD: {method_name}")
            print(f"{'='*80}")

            for shape in self.shapes:
                print(f"\n{'-'*80}")
                print(f"Shape: {shape}")
                print(f"{'-'*80}")

                for run_id in range(1, self.num_runs + 1):
                    current += 1

                    # Set seed for reproducibility
                    seed = hash((method_name, shape, run_id)) % (2**32)
                    random.seed(seed)
                    np.random.seed(seed)

                    # Run experiment
                    try:
                        result = method_func(shape, pop_size=50, gens=150)

                        # Store result
                        self.results.append({
                            'method': method_name,
                            'shape': shape,
                            'run': run_id,
                            'fitness': result['final_fitness'],
                            'convergence_gen': result['convergence_gen'],
                        })

                        # Progress
                        pct = (current / total_experiments) * 100
                        print(f"  Run {run_id:2d}/30: Fitness={result['final_fitness']:.3f}, "
                              f"Conv={result['convergence_gen']:3d} gens [{pct:5.1f}%]")

                    except Exception as e:
                        print(f"  Run {run_id:2d}/30: ERROR - {str(e)}")
                        self.results.append({
                            'method': method_name,
                            'shape': shape,
                            'run': run_id,
                            'fitness': np.nan,
                            'convergence_gen': np.nan,
                        })

        end_time = datetime.now()
        elapsed = (end_time - self.start_time).total_seconds()

        print(f"\n{'='*80}")
        print(f"EXPERIMENT COMPLETE")
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"Avg time per experiment: {elapsed/total_experiments:.1f} seconds")
        print(f"{'='*80}\n")

    def save_results(self):
        """Save results to CSV and JSON"""
        # Save detailed results to CSV
        csv_file = 'experiment_results_detailed.csv'
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['method', 'shape', 'run', 'fitness', 'convergence_gen'])
            writer.writeheader()
            writer.writerows(self.results)

        print(f"✅ Detailed results saved to: {csv_file}")

        # Save summary statistics to CSV
        summary_file = 'experiment_results_summary.csv'
        summary_data = []

        for method_name in self.methods.keys():
            for shape in self.shapes:
                subset = [r for r in self.results
                         if r['method'] == method_name and r['shape'] == shape]

                if subset:
                    fitness_vals = [r['fitness'] for r in subset if not np.isnan(r['fitness'])]
                    conv_vals = [r['convergence_gen'] for r in subset if not np.isnan(r['convergence_gen'])]

                    if fitness_vals:
                        summary_data.append({
                            'method': method_name,
                            'shape': shape,
                            'n_runs': len(fitness_vals),
                            'fitness_mean': np.mean(fitness_vals),
                            'fitness_std': np.std(fitness_vals),
                            'fitness_min': np.min(fitness_vals),
                            'fitness_max': np.max(fitness_vals),
                            'convergence_mean': np.mean(conv_vals),
                            'convergence_std': np.std(conv_vals),
                            'convergence_min': np.min(conv_vals),
                            'convergence_max': np.max(conv_vals),
                        })

        with open(summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'method', 'shape', 'n_runs',
                'fitness_mean', 'fitness_std', 'fitness_min', 'fitness_max',
                'convergence_mean', 'convergence_std', 'convergence_min', 'convergence_max'
            ])
            writer.writeheader()
            writer.writerows(summary_data)

        print(f"✅ Summary statistics saved to: {summary_file}")

        # Save to JSON for Python analysis
        json_file = 'experiment_results.json'
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"✅ Raw results saved to: {json_file}")

        return summary_data

    def print_summary(self, summary_data):
        """Print summary statistics"""
        print("\n" + "="*120)
        print("SUMMARY STATISTICS")
        print("="*120)

        # Group by method
        methods_data = {}
        for row in summary_data:
            method = row['method']
            if method not in methods_data:
                methods_data[method] = []
            methods_data[method].append(row)

        for method in self.methods.keys():
            if method not in methods_data:
                continue

            print(f"\n{method}")
            print("-" * 120)
            print(f"{'Shape':<10} {'Fitness':<20} {'Convergence (gens)':<25} {'N':<5}")
            print("-" * 120)

            method_fitness = []
            method_convergence = []

            for row in methods_data[method]:
                shape = row['shape']
                fitness_str = f"{row['fitness_mean']:.3f} ± {row['fitness_std']:.3f}"
                conv_str = f"{row['convergence_mean']:.1f} ± {row['convergence_std']:.1f}"

                print(f"{shape:<10} {fitness_str:<20} {conv_str:<25} {row['n_runs']:<5}")

                method_fitness.extend([r['fitness_mean'] for r in methods_data[method]])
                method_convergence.extend([r['convergence_mean'] for r in methods_data[method]])

            avg_fitness = np.mean([r['fitness_mean'] for r in methods_data[method]])
            avg_conv = np.mean([r['convergence_mean'] for r in methods_data[method]])

            print("-" * 120)
            print(f"{'AVERAGE':<10} {avg_fitness:<20.3f} {avg_conv:<25.1f}")

        print("\n" + "="*120)


def main():
    runner = ExperimentRunner(num_runs=30)
    runner.run_all_experiments()
    summary = runner.save_results()
    runner.print_summary(summary)

    print("\n" + "="*80)
    print("READY FOR STATISTICAL ANALYSIS")
    print("="*80)
    print("Files created:")
    print("  1. experiment_results_detailed.csv - All 360 runs (for detailed analysis)")
    print("  2. experiment_results_summary.csv - Summary statistics (for plots)")
    print("  3. experiment_results.json - Raw data (for Python analysis)")
    print("\nNext steps:")
    print("  1. Statistical tests (ANOVA, t-tests)")
    print("  2. Create plots (matplotlib/seaborn)")
    print("  3. Effect size analysis")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
