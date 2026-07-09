# SPO+ Shortest-Path Experiment (SLURM)

Monte-Carlo comparison of **L2 (Lasso) regression** vs. the **SPO+ decision-aware loss**
on a 4×4 grid shortest-path problem, across increasing model misspecification
(polynomial degree) and two noise regimes.

The experiment is **500 independent trials** = 5 degrees × 2 regimes × 50 trials.
Each trial trains, tunes over a λ-path, and tests on freshly generated data, then
records the normalized operational regret (%) for each method.

## Layout

| File | Purpose |
|------|---------|
| `spo_core.py` | Topology, LP-based SPO+ solver, single-trial driver, task↔params mapping |
| `run_trial.py` | Array worker — runs **one** trial (`SLURM_ARRAY_TASK_ID`), writes `results/trial_<id>.npz` |
| `aggregate_plot.py` | Reads all results, renders the two boxplot PNGs into `figures/` |
| `run_spo.slurm` | SLURM **array job** (`--array=0-499%20`) — the compute |
| `plot.slurm` | Aggregation/plot job (run after the array finishes) |
| `submit.sh` | Submits both, chaining the plot job to the array via a dependency |
| `Sanity_SPO_Gemini.py` | Original single-process reference script |

## Running on the cluster

`submit.sh` builds a local `venv/` and installs `requirements.txt` once (on the
login node), then both `.slurm` files activate it. If your cluster hides `python3`
behind a module, uncomment the `module load` line near the top of `submit.sh` and
both `.slurm` files. Then:

```bash
bash submit.sh
```

This submits the 500-task array (max **20** running at once, per the `%20` cap) and
a plot job that fires automatically once every trial succeeds. Results land in
`figures/regret_boxplot_noisy.png` and `figures/regret_boxplot_noiseless.png`.

To submit manually instead (create the venv first — the `.slurm` files require it):

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
jid=$(sbatch --parsable run_spo.slurm)
sbatch --dependency=afterok:${jid} plot.slurm
```

## Running locally (no SLURM)

```bash
pip install -r requirements.txt
for i in $(seq 0 499); do python run_trial.py $i; done   # or run Sanity_SPO_Gemini.py directly
python aggregate_plot.py
```

## Reproducibility

Each trial owns a `numpy` generator seeded deterministically from
`(degree, regime, trial)`, so results are independent of execution order and
identical whether run on 1 node or 20.
