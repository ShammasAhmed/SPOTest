#!/bin/bash
# ---------------------------------------------------------------------------
# Interactive PILOT test -- run a handful of representative trials on an salloc
# node BEFORE committing the full 500-task array. It:
#   1. provisions the venv (same one run_spo.slurm / plot.slurm activate),
#   2. times a single trial so you can sanity-check the array's --time budget,
#   3. runs 5 trials spanning the degree extremes and both noise regimes,
#   4. renders the figures from whatever results exist (aggregate tolerates a
#      partial set), to exercise the plotting path too.
#
# Grab an interactive node first, e.g.:
#     salloc --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:30:00
#     # add --partition=<name> / --account=<name> if your cluster requires them
# then, on the node:
#     cd ~/SPOTest && bash pilot.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# module load python/3.11   # uncomment / adjust if python3 isn't on PATH

# --- 1. venv -------------------------------------------------------------
if [ ! -d venv ]; then
    echo ">> creating venv and installing requirements ..."
    python3 -m venv venv
    ./venv/bin/pip install --upgrade pip
    ./venv/bin/pip install -r requirements.txt
fi
source venv/bin/activate
python --version

mkdir -p results figures

# Representative slice of the 500-task grid (see table in the README/PR):
PILOT_TASKS=(0 50 200 450 499)

# --- 2. time one trial ---------------------------------------------------
echo ">> timing a single trial (task ${PILOT_TASKS[0]}) ..."
SECONDS=0
python run_trial.py "${PILOT_TASKS[0]}"
one=$SECONDS
echo ">> one trial took ${one}s."
echo ">> full array is 500 trials, 20 at a time => ~$(( (500 * one) / 20 ))s wall-clock (very rough)."
echo ">> per-task --time=01:00:00 in run_spo.slurm should comfortably cover ${one}s."

# --- 3. run the remaining pilot trials -----------------------------------
for t in "${PILOT_TASKS[@]:1}"; do
    echo ">> running task ${t} ..."
    python run_trial.py "${t}"
done

echo ">> pilot results:"
ls -1 results/

# --- 4. exercise the aggregation/plot path -------------------------------
echo ">> rendering figures from the pilot results (expect a partial-set WARNING) ..."
python aggregate_plot.py
ls -1 figures/

echo ">> PILOT OK. If the numbers look sane, submit the full run with: bash submit.sh"
