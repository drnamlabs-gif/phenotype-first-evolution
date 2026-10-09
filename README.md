# Organisms Write Their Own Evolution

Code, data, and LaTeX source for *"Organisms Write Their Own Evolution: Direct
Epigenomic Inheritance Accelerates Learning on Compositional Tasks"* (Nam H.
Le, Vermont Complexity Center, University of Vermont).

The paper implements **Phenopoiesis**, a computational model in which
organisms carry two independent heritable layers (genetic and epigenetic), and
tests Denis Noble's hypothesis that organisms can write learned knowledge
directly into an inheritable epigenome. It is compared against Selection Only
(Gene-Centric GA) and the Baldwin Effect across six learning tasks (three
simple, three compositional), 30 replicates each, 540 runs total.

## Structure

```
code/     Gene-Centric GA, Baldwin Effect, Phenopoiesis implementations,
          the full experiment runner, and the statistical analysis script
data/     raw results for all 540 runs (CSV) and the computed statistics (JSON)
figures/  the five figures used in the manuscript
paper/    LaTeX source (main.tex, main.bbl, references.bib)
```

## Running

```bash
pip install -r code/requirements.txt
python code/run_full_experiment.py
python code/statistical_analysis.py
```

## Algorithms

- `code/simple_ga.py` -- Gene-Centric GA (no learning; baseline)
- `code/baldwin_ga.py` -- Baldwin Effect (learning influences selection, not inherited)
- `code/phenotype_first_ga.py` -- Phenopoiesis (learned patterns written to an inherited epigenetic layer)

## Data

- `data/experiment_results_detailed.csv` -- simple tasks (L, T, Plus), per-run fitness and convergence
- `data/experiment_results_summary.csv` -- simple tasks, aggregated statistics
- `data/compositional_results_all_tasks.csv` -- compositional tasks (L+T, L+Plus, T+Plus), per-run results
- `data/statistical_results.json`, `data/compositional_statistical_results.json` -- ANOVA / Welch's t-test / Cohen's d output

## Citation

If you use this code or data, please cite the preprint (details to follow once
posted to arXiv).
