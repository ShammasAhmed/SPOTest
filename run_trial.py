"""SLURM array worker: run exactly one trial and write its result to disk.

The trial to run is selected by the ``SLURM_ARRAY_TASK_ID`` environment variable
(0 .. 499). A single task id maps deterministically to a (degree, regime, trial)
triple via ``spo_core.task_to_params``. The result is written to
``results/trial_<id>.npz`` so the aggregation step can rebuild the full data set.

Can also be run locally for a single task:  ``python run_trial.py 0``
"""

import os
import sys

import numpy as np

from spo_core import run_one_trial, task_to_params, NUM_TASKS

RESULTS_DIR = "results"


def main():
    # Prefer an explicit CLI argument, fall back to the SLURM array env var.
    if len(sys.argv) > 1:
        task_id = int(sys.argv[1])
    elif "SLURM_ARRAY_TASK_ID" in os.environ:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    else:
        raise SystemExit("No task id: pass one on the CLI or set SLURM_ARRAY_TASK_ID.")

    deg, regime, trial = task_to_params(task_id)
    print(f"[task {task_id}/{NUM_TASKS}] degree={deg} regime={regime} trial={trial}",
          flush=True)

    l2_pct, spo_pct = run_one_trial(deg, regime, trial)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"trial_{task_id:04d}.npz")
    np.savez(out_path,
             task_id=task_id, degree=deg, regime=regime, trial=trial,
             l2_pct=l2_pct, spo_pct=spo_pct)
    print(f"[task {task_id}] wrote {out_path}  l2={l2_pct:.4f}%  spo={spo_pct:.4f}%",
          flush=True)


if __name__ == "__main__":
    main()
