#!/bin/bash
# Convenience launcher: submit the trial array, then the plot job chained to it.
set -euo pipefail

# --- Provision the Python environment ONCE, on the login node ---------------
# The array tasks and the plot job all activate this venv (see the .slurm files).
# Building it here (instead of inside each task) avoids a 20-way race from the
# concurrently-running array tasks and the earlier "No module named 'numpy'" crash.
module load python/3.11   # needed so `python3` is 3.11, not the broken system 3.6
if [ ! -d venv ]; then
    echo "creating venv and installing requirements ..."
    python3 -m venv venv
    ./venv/bin/pip install --upgrade pip
    ./venv/bin/pip install -r requirements.txt
else
    echo "venv already exists, skipping install"
fi

jid=$(sbatch --parsable run_spo.slurm)
echo "submitted trial array as job ${jid} (array 0-499%20)"

pid=$(sbatch --parsable --dependency=afterok:${jid} plot.slurm)
echo "submitted plot job as ${pid} (runs after ${jid} completes)"
