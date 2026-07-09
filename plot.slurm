#!/bin/bash
#SBATCH --job-name=spo_plot
#SBATCH --output=logs/plot_%j.out
#SBATCH --error=logs/plot_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

# ---------------------------------------------------------------------------
# Aggregate all results/trial_*.npz and render the final boxplot figures into
# figures/. Run this AFTER the run_spo.slurm array has finished, e.g.:
#
#     jid=$(sbatch --parsable run_spo.slurm)
#     sbatch --dependency=afterok:${jid} plot.slurm
# ---------------------------------------------------------------------------

set -euo pipefail

mkdir -p logs figures

# module load python/3.13
# source activate spo

srun python aggregate_plot.py
