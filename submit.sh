#!/bin/bash
# Convenience launcher: submit the trial array, then the plot job chained to it.
set -euo pipefail

jid=$(sbatch --parsable run_spo.slurm)
echo "submitted trial array as job ${jid} (array 0-499%20)"

pid=$(sbatch --parsable --dependency=afterok:${jid} plot.slurm)
echo "submitted plot job as ${pid} (runs after ${jid} completes)"
