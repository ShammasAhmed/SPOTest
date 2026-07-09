#!/bin/bash
#SBATCH --job-name=spo_trials
#SBATCH --account=guptavis_1305
#SBATCH --output=logs/spo_%A_%a.out
#SBATCH --error=logs/spo_%A_%a.err
#SBATCH --array=0-499%20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00

# ---------------------------------------------------------------------------
# SPO+ shortest-path Monte-Carlo experiment.
#
# 500 independent trials = 5 degrees x 2 noise regimes x 50 trials.
# Each array task runs exactly one trial (run_trial.py picks it from
# SLURM_ARRAY_TASK_ID) and writes results/trial_<id>.npz.
#
# The %20 suffix caps the number of SIMULTANEOUSLY RUNNING array tasks at 20;
# the remaining tasks queue and start as slots free up.
#
# When every task has finished, run the aggregation job to build the boxplots:
#     sbatch --dependency=afterok:<this_job_id> plot.slurm
# (or just: sbatch plot.slurm  once this array is done.)
# ---------------------------------------------------------------------------

set -euo pipefail

mkdir -p logs results

# --- Environment ---------------------------------------------------------
# Activate the venv provisioned by submit.sh. This is what was missing before:
# without it the job ran the bare system python and crashed with
# "ModuleNotFoundError: No module named 'numpy'".
# module load python/3.11   # uncomment / adjust if your cluster needs it
if [ ! -d venv ]; then
    echo "ERROR: venv/ not found. Create it first (bash submit.sh, or:" >&2
    echo "       python3 -m venv venv && ./venv/bin/pip install -r requirements.txt)" >&2
    exit 1
fi
source venv/bin/activate

python --version

srun python run_trial.py "${SLURM_ARRAY_TASK_ID}"
