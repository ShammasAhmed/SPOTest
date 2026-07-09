#!/bin/bash
#SBATCH --job-name=spo_plot
#SBATCH --account=guptavis_1305
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

# Activate the same venv the trial array used (provisioned by submit.sh).
module load python/3.11   # same module used to build the venv in submit.sh
if [ ! -d venv ]; then
    echo "ERROR: venv/ not found. Create it first (bash submit.sh, or:" >&2
    echo "       python3 -m venv venv && ./venv/bin/pip install -r requirements.txt)" >&2
    exit 1
fi
source venv/bin/activate

srun python aggregate_plot.py
