"""Shared SPO+ shortest-path experiment core.

Factored out of the original monolithic ``Sanity_SPO_Gemini.py`` so that a single
trial can be run in isolation by a SLURM array task (see ``run_trial.py``) and the
results aggregated afterwards (see ``aggregate_plot.py``).

The numerics are unchanged from the original script; the only structural change is
that every trial now owns its own ``numpy`` random generator, seeded
deterministically from ``(degree, regime, trial)``. This makes each array task fully
reproducible and independent of execution order, which the original sequential
single-``rng`` design could not provide.
"""

import warnings
from itertools import combinations

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning
from scipy import sparse
from scipy.optimize import linprog

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------
# 1. Global Setup & Topology Enumeration
# ---------------------------------------------------------
D = 40   # 40 edges
P = 5    # 5 features
H = 0.5
NUM_TRIALS = 50
NUM_TRAIN = 1000
NUM_VAL = int(NUM_TRAIN / 4)
NUM_TEST = 1000

# Tuning parameters -- log-spaced lambda grid matching the Julia experiment.
lambda_max = 100.0
lambda_min_ratio = 1e-8
num_lambda = 10
lambdas = np.exp(np.linspace(np.log(lambda_max * lambda_min_ratio),
                             np.log(lambda_max), num_lambda))

degrees_to_test = [1, 2, 4, 6, 8]
noise_settings = {"noiseless": 0.0, "noisy": 0.5}

# Enumerate all 70 unique paths on the 4x4 grid (8 moves, choose 4 downs).
paths = []
for positions in combinations(range(8), 4):
    path = ['R'] * 8
    for pos in positions:
        path[pos] = 'D'
    paths.append(''.join(path))

# Pre-build path-edge incidence matrix (70 x 40) for fast vector operations.
path_matrix = np.zeros((len(paths), D))
for idx, path in enumerate(paths):
    row, col = 0, 0
    for move in path:
        if move == 'R':
            path_matrix[idx, row * 4 + col] = 1
            col += 1
        elif move == 'D':
            path_matrix[idx, 20 + row * 5 + col] = 1
            row += 1


def solve_shortest_path(edge_costs):
    """Vectorized oracle returning the binary decision vector for the shortest path."""
    costs = path_matrix @ edge_costs
    best_idx = np.argmin(costs)
    return path_matrix[best_idx]


def evaluate_path_cost(path_vector, edge_costs):
    return np.dot(path_vector, edge_costs)


# ---------------------------------------------------------
# 2. Exact SPO+ Reformulation Solver Core
# ---------------------------------------------------------
def spo_plus_reform_path(X, C, lambdas_grid):
    """Exact SPO+ regularized path via LP reformulation. Returns a list of (d x p+1) B
    matrices, one per lambda in ``lambdas_grid``, matching sp_reformulation_path_jump."""
    n, p = X.shape
    d = D
    pp = p + 1
    K = path_matrix.shape[0]

    X_aug = np.hstack([np.ones((n, 1)), X])            # (n, p+1); column 0 = intercept
    W_star = np.array([solve_shortest_path(C[i]) for i in range(n)])  # (n, d)

    nB = d * pp                                        # B variables (B[j, l] -> j*pp + l)
    nvar = 2 * nB + n                                  # [ B | theta | t ]

    coeffB = 2.0 * (W_star.T @ X_aug)                  # (d, p+1)

    # Constraint block A: t_i >= (c_i - 2 B x_i)' path_k for all i, k.
    M = np.einsum('kj,il->ikjl', path_matrix, X_aug)   # (n, K, d, p+1)
    Bblock = sparse.csr_matrix((-2.0 * M).reshape(n * K, nB))
    row_idx = np.arange(n * K)
    t_sel = sparse.coo_matrix((-np.ones(n * K), (row_idx, row_idx // K)), shape=(n * K, n))
    A_t = sparse.hstack([Bblock, sparse.csr_matrix((n * K, nB)), t_sel])
    b_t = -(C @ path_matrix.T).reshape(n * K)          # -c_i' path_k

    # Constraint blocks B/C: theta >= B and theta >= -B.
    I_nB = sparse.identity(nB, format='csr')
    zeros_nB_n = sparse.csr_matrix((nB, n))
    A_pos = sparse.hstack([I_nB, -I_nB, zeros_nB_n])   #  B - theta <= 0
    A_neg = sparse.hstack([-I_nB, -I_nB, zeros_nB_n])  # -B - theta <= 0

    A_ub = sparse.vstack([A_t, A_pos, A_neg]).tocsr()
    b_ub = np.concatenate([b_t, np.zeros(nB), np.zeros(nB)])

    bounds = [(None, None)] * nB + [(0, None)] * nB + [(None, None)] * n

    B_soln_list = []
    for lmbda in lambdas_grid:
        c_obj = np.zeros(nvar)
        c_obj[:nB] = coeffB.reshape(nB)
        theta_pen = np.full((d, pp), n * lmbda)
        theta_pen[:, 0] = 0.0                          # intercept column unregularized
        c_obj[nB:2 * nB] = theta_pen.reshape(nB)
        c_obj[2 * nB:] = 1.0                           # sum_i t_i

        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            B = res.x[:nB].reshape(d, pp)
        else:
            print(f"SPO+ LP did not solve (lambda={lmbda}): {res.message}")
            B = np.zeros((d, pp))
        B_soln_list.append(B)
    return B_soln_list


def predict_spo(W, x):
    """SPO+ linear prediction with the prepended intercept feature (B x_aug)."""
    return np.concatenate(([1.0], x)) @ W.T


def generate_costs(X_mat, deg, h, B, rng):
    """Polynomial-kernel cost generation matching generate_poly_kernel_data_simple:
    c = ((1/sqrt(P)) * (B @ x) + 3)^deg + 1, with multiplicative noise ~ U[1-h, 1+h]
    applied only when h > 0."""
    n = X_mat.shape[0]
    C = np.zeros((n, D))
    for t in range(n):
        c = (((1 / np.sqrt(P)) * np.dot(B, X_mat[t]) + 3) ** deg + 1)
        if h > 0:
            c = c * rng.uniform(1 - h, 1 + h, size=D)
        C[t] = c
    return C


# ---------------------------------------------------------
# 3. Single-trial driver (one SLURM array task == one call)
# ---------------------------------------------------------
def trial_seed(deg, regime, trial):
    """Deterministic, order-independent seed for a single trial."""
    regime_idx = list(noise_settings).index(regime)
    return np.random.SeedSequence([42, deg, regime_idx, trial])


def run_one_trial(deg, regime, trial):
    """Run one complete trial (train -> tune -> test) and return the two normalized
    regret percentages (L2, SPO+)."""
    h = noise_settings[regime]
    rng = np.random.default_rng(trial_seed(deg, regime, trial))

    # New random B for this run.
    B = rng.binomial(1, 0.5, size=(D, P))

    # Data generation: train + validation at this regime's noise level.
    X_train = rng.standard_normal(size=(NUM_TRAIN, P))
    C_train = generate_costs(X_train, deg, h, B, rng)

    X_val = rng.standard_normal(size=(NUM_VAL, P))
    C_val = generate_costs(X_val, deg, h, B, rng)

    # SPO+ full regularization path, solved exactly for the whole lambda grid.
    spo_candidates = spo_plus_reform_path(X_train, C_train, lambdas)

    l2_candidates = []
    l2_val_err, spo_val_err = [], []

    for idx, lmbda in enumerate(lambdas):
        m_l2 = Lasso(alpha=lmbda, fit_intercept=True, max_iter=100000, tol=1e-9)
        m_l2.fit(X_train, C_train)
        l2_candidates.append(m_l2)

        w_spo = spo_candidates[idx]

        reg_l2, reg_spo = 0, 0
        for v_idx in range(NUM_VAL):
            c_v_true = C_val[v_idx]
            s_v_true = solve_shortest_path(c_v_true)
            z_v_star = evaluate_path_cost(s_v_true, c_v_true)

            p_l2 = m_l2.predict(X_val[v_idx].reshape(1, -1)).flatten()
            p_spo = predict_spo(w_spo, X_val[v_idx])

            reg_l2 += evaluate_path_cost(solve_shortest_path(p_l2), c_v_true) - z_v_star
            reg_spo += evaluate_path_cost(solve_shortest_path(p_spo), c_v_true) - z_v_star

        l2_val_err.append(reg_l2)
        spo_val_err.append(reg_spo)

    best_l2 = l2_candidates[np.argmin(l2_val_err)]
    best_spo_w = spo_candidates[np.argmin(spo_val_err)]

    # Testing at the same regime noise level.
    X_test = rng.standard_normal(size=(NUM_TEST, P))
    C_test = generate_costs(X_test, deg, h, B, rng)

    sum_l2_reg, sum_spo_reg, sum_z_star = 0, 0, 0
    for i in range(NUM_TEST):
        x_inst = X_test[i]
        c_true = C_test[i]

        s_star = solve_shortest_path(c_true)
        z_star = evaluate_path_cost(s_star, c_true)

        pred_l2 = best_l2.predict(x_inst.reshape(1, -1)).flatten()
        pred_spo = predict_spo(best_spo_w, x_inst)

        sum_l2_reg += evaluate_path_cost(solve_shortest_path(pred_l2), c_true) - z_star
        sum_spo_reg += evaluate_path_cost(solve_shortest_path(pred_spo), c_true) - z_star
        sum_z_star += z_star

    l2_pct = (sum_l2_reg / sum_z_star) * 100
    spo_pct = (sum_spo_reg / sum_z_star) * 100
    return l2_pct, spo_pct


# ---------------------------------------------------------
# 4. Task-index <-> (degree, regime, trial) mapping
# ---------------------------------------------------------
REGIMES = list(noise_settings)          # ["noiseless", "noisy"]
TASKS_PER_REGIME = NUM_TRIALS           # 50
TASKS_PER_DEGREE = len(REGIMES) * NUM_TRIALS  # 100
NUM_TASKS = len(degrees_to_test) * TASKS_PER_DEGREE  # 500


def task_to_params(task_id):
    """Map a flat SLURM array index (0 .. NUM_TASKS-1) to (deg, regime, trial)."""
    if not (0 <= task_id < NUM_TASKS):
        raise ValueError(f"task_id {task_id} out of range [0, {NUM_TASKS})")
    deg = degrees_to_test[task_id // TASKS_PER_DEGREE]
    rem = task_id % TASKS_PER_DEGREE
    regime = REGIMES[rem // TASKS_PER_REGIME]
    trial = rem % TASKS_PER_REGIME
    return deg, regime, trial
