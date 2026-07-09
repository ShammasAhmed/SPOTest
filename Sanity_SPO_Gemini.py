import numpy as np
import warnings
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.linear_model import Lasso  # QuantileRegressor used only for L1 loss
# from sklearn.multioutput import MultiOutputRegressor  # used only for L1 loss
from sklearn.exceptions import ConvergenceWarning
from scipy import sparse
from scipy.optimize import linprog  # exact SPO+ reformulation LP (stands in for Gurobi)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------
# 1. Global Setup & Topology Enumeration
# ---------------------------------------------------------
rng = np.random.default_rng(seed=42)
D = 40   # 40 edges
P = 5    # 5 features
H = 0.5
NUM_TRIALS = 50
NUM_TRAIN = 1000
NUM_VAL = int(NUM_TRAIN / 4) # 62 validation samples
NUM_TEST = 1000

B = rng.binomial(1, 0.5, size=(D, P))

# Enumerate all 70 unique paths
paths = []
for positions in combinations(range(8), 4):
    path = ['R'] * 8
    for pos in positions:
        path[pos] = 'D'
    paths.append(''.join(path))

# Pre-build path-edge incidence matrix (70 x 40) for fast vector operations
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

# Tuning Parameters
# Match the Julia experiment's regularization grid exactly:
#   lambda_max = 100, lambda_min_ratio = 1e-8, num_lambda = 10, log-spaced.
# This same lambda is used directly (no rescaling) by BOTH methods, mirroring how
# the reformulation code applies n*lambda (SPO+) / 2n*lambda (LS) penalties.
lambda_max = 100.0
lambda_min_ratio = 1e-8
num_lambda = 10
lambdas = np.exp(np.linspace(np.log(lambda_max * lambda_min_ratio), np.log(lambda_max), num_lambda))
degrees_to_test = [1, 2, 4, 6, 8]

# Data structures to collect raw trial statistics for final boxplots
raw_trials_l2 = {"noiseless": {d: [] for d in degrees_to_test}, "noisy": {d: [] for d in degrees_to_test}}
# raw_trials_l1 = {"noiseless": {d: [] for d in degrees_to_test}, "noisy": {d: [] for d in degrees_to_test}}
raw_trials_spo = {"noiseless": {d: [] for d in degrees_to_test}, "noisy": {d: [] for d in degrees_to_test}}

# ---------------------------------------------------------
# 2. Exact SPO+ Reformulation Solver Core
# ---------------------------------------------------------
# The original Julia code (sp_reformulation_path_jump in reformulation.jl) solves the
# regularized SPO+ empirical risk problem EXACTLY as a linear program with Gurobi, using
# the LP dual of the inner problem max_{w in S} (c - 2Bx)'w. Because the vertices of the
# shortest-path polytope S are exactly the enumerated paths, that inner max equals a max
# over the 70 paths, so we can write the same LP directly on the path-incidence matrix and
# solve it with HiGHS (scipy.linprog) instead of Gurobi. This reproduces the Julia optimum
# rather than approximating it with subgradient descent.
#
# For a d x (p+1) model B (column 0 is the unregularized intercept, feature row of ones),
# the per-sample SPO+ loss is  t_i + 2 (B x_i)' w*(c_i) - z*(c_i),  where
#   t_i = max_k (c_i - 2 B x_i)' path_k  =  -min_w (2 B x_i - c_i)' w  (matches util.jl).
# Julia's objective (lasso, intercept excluded) is
#   sum_i [ t_i + 2 (B x_i)' w*_i - z*_i ]  +  n * lambda * sum |B[:, 1:]|.
# We drop the constant -sum z*_i (does not change argmin) and solve for the whole lambda
# path, rebuilding only the objective's L1 weight between solves.
def spo_plus_reform_path(X, C, lambdas_grid):
    """Exact SPO+ regularized path via LP reformulation. Returns a list of (d x p+1) B
    matrices, one per lambda in `lambdas_grid`, matching sp_reformulation_path_jump."""
    n, p = X.shape
    d = D
    pp = p + 1
    K = path_matrix.shape[0]

    X_aug = np.hstack([np.ones((n, 1)), X])           # (n, p+1); column 0 = intercept
    W_star = np.array([solve_shortest_path(C[i]) for i in range(n)])  # (n, d): w*(c_i)

    nB = d * pp                                        # B variables (row-major B[j, l] -> j*pp + l)
    nvar = 2 * nB + n                                  # [ B | theta | t ]

    # --- Objective piece that does not depend on lambda ---
    # Linear coefficient on B[j, l] from the sum_i 2 (B x_i)' w*_i term.
    coeffB = 2.0 * (W_star.T @ X_aug)                  # (d, p+1)

    # --- Constraint block A: t_i >= (c_i - 2 B x_i)' path_k  for all i, k ---
    # Rewritten as  -2 * sum_{j,l} path_k[j] x_i[l] B[j,l]  -  t_i  <=  -c_i' path_k.
    M = np.einsum('kj,il->ikjl', path_matrix, X_aug)   # (n, K, d, p+1)
    Bblock = sparse.csr_matrix((-2.0 * M).reshape(n * K, nB))
    row_idx = np.arange(n * K)
    t_sel = sparse.coo_matrix((-np.ones(n * K), (row_idx, row_idx // K)), shape=(n * K, n))
    A_t = sparse.hstack([Bblock, sparse.csr_matrix((n * K, nB)), t_sel])
    b_t = -(C @ path_matrix.T).reshape(n * K)          # -c_i' path_k

    # --- Constraint blocks B/C: theta >= B and theta >= -B ---
    I_nB = sparse.identity(nB, format='csr')
    zeros_nB_n = sparse.csr_matrix((nB, n))
    A_pos = sparse.hstack([I_nB, -I_nB, zeros_nB_n])   #  B - theta <= 0
    A_neg = sparse.hstack([-I_nB, -I_nB, zeros_nB_n])  # -B - theta <= 0

    A_ub = sparse.vstack([A_t, A_pos, A_neg]).tocsr()
    b_ub = np.concatenate([b_t, np.zeros(nB), np.zeros(nB)])

    # Bounds: B free, theta >= 0, t free.
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

# =========================================================
# 3. CORE EXPERIMENTAL LOOP
# =========================================================
print("Beginning unified trial execution matrix...")

# The original sweeps polykernel_noise_half_width in {0, 0.5}, regenerating train/
# validation/test together at the chosen noise level for each setting (see
# shortest_path_run.jl: polykernel_noise_half_width_vec = [0; 0.5], and the nested loop in
# shortest_path_multiple_replications). We mirror that here: each regime trains, tunes, and
# tests entirely at its own noise level rather than reusing one noisy-trained model.
noise_settings = {"noiseless": 0.0, "noisy": 0.5}


def generate_costs(X_mat, deg, h, B):
    """Polynomial-kernel cost generation matching generate_poly_kernel_data_simple:
    c = ((1/sqrt(P)) * (B @ x) + 3)^deg + 1, with multiplicative noise ~ U[1-h, 1+h]
    applied only when h > 0 (noise switched off entirely for h == 0)."""
    n = X_mat.shape[0]
    C = np.zeros((n, D))
    for t in range(n):
        c = (((1 / np.sqrt(P)) * np.dot(B, X_mat[t]) + 3) ** deg + 1)
        if h > 0:
            c = c * rng.uniform(1 - h, 1 + h, size=D)
        C[t] = c
    return C


for deg in degrees_to_test:
    print(f"Processing environment landscape context: DEGREE = {deg}")

    for regime, h in noise_settings.items():
        for trial in range(NUM_TRIALS):
            # New B for each run
            B = rng.binomial(1, 0.5, size=(D, P))

            # Data generation: train + validation at this regime's noise level
            X_train = rng.standard_normal(size=(NUM_TRAIN, P))
            C_train = generate_costs(X_train, deg, h, B)

            X_val = rng.standard_normal(size=(NUM_VAL, P))
            C_val = generate_costs(X_val, deg, h, B)

            # Julia fits every method on the RAW cost vectors (no rescaling, no clipping);
            # generate_poly_kernel_data_simple passes normalize_c = false. We match that.

            # Candidate arrays and error trackers for hyperparameter evaluation
            l2_candidates, spo_candidates = [], []
            l2_val_err, spo_val_err = [], []
            # l1_candidates = []
            # l1_val_err = []

            # SPO+ full regularization path, solved exactly (LP reformulation), once for
            # the whole lambda grid -- mirrors sp_reformulation_path_jump.
            spo_candidates = spo_plus_reform_path(X_train, C_train, lambdas)

            # Grid search sweep
            for idx, lmbda in enumerate(lambdas):
                # L2 loss + Lasso penalty. sklearn's objective is
                #   (1/2n)||C - X B||^2 + alpha*||B||_1  (intercept unpenalized),
                # which is the Julia LS reformulation objective ||C - B X||^2 + 2n*lambda*||B||_1
                # scaled by 1/2n -- same argmin. So alpha = lambda reproduces the Julia solution.
                m_l2 = Lasso(alpha=lmbda, fit_intercept=True, max_iter=100000, tol=1e-9)
                m_l2.fit(X_train, C_train)
                l2_candidates.append(m_l2)

                # L1 (absolute) loss + L1 penalty
                # m_l1 = MultiOutputRegressor(QuantileRegressor(quantile=0.5, alpha=lmbda, solver='highs'))
                # m_l1.fit(X_train, C_train)
                # l1_candidates.append(m_l1)

                w_spo = spo_candidates[idx]

                # Record validation SPO regrets (every method tuned on SPO loss, matching
                # different_validation_losses = false in the original)
                reg_l2, reg_spo = 0, 0
                # reg_l1 = 0
                for v_idx in range(NUM_VAL):
                    c_v_true = C_val[v_idx]
                    s_v_true = solve_shortest_path(c_v_true)
                    z_v_star = evaluate_path_cost(s_v_true, c_v_true)

                    p_l2 = m_l2.predict(X_val[v_idx].reshape(1, -1)).flatten()
                    # p_l1 = m_l1.predict(X_val[v_idx].reshape(1, -1)).flatten()
                    p_spo = predict_spo(w_spo, X_val[v_idx])

                    reg_l2 += evaluate_path_cost(solve_shortest_path(p_l2), c_v_true) - z_v_star
                    # reg_l1 += evaluate_path_cost(solve_shortest_path(p_l1), c_v_true) - z_v_star
                    reg_spo += evaluate_path_cost(solve_shortest_path(p_spo), c_v_true) - z_v_star

                l2_val_err.append(reg_l2)
                # l1_val_err.append(reg_l1)
                spo_val_err.append(reg_spo)

            best_l2 = l2_candidates[np.argmin(l2_val_err)]
            # best_l1 = l1_candidates[np.argmin(l1_val_err)]
            best_spo_w = spo_candidates[np.argmin(spo_val_err)]

            # --- Testing: test set generated at the same regime noise level ---
            X_test = rng.standard_normal(size=(NUM_TEST, P))
            C_test = generate_costs(X_test, deg, h, B)

            sum_l2_reg, sum_spo_reg, sum_z_star = 0, 0, 0
            # sum_l1_reg = 0
            for i in range(NUM_TEST):
                x_inst = X_test[i]
                c_true = C_test[i]

                s_star = solve_shortest_path(c_true)
                z_star = evaluate_path_cost(s_star, c_true)

                pred_l2 = best_l2.predict(x_inst.reshape(1, -1)).flatten()
                # pred_l1 = best_l1.predict(x_inst.reshape(1, -1)).flatten()
                pred_spo = predict_spo(best_spo_w, x_inst)

                sum_l2_reg += evaluate_path_cost(solve_shortest_path(pred_l2), c_true) - z_star
                # sum_l1_reg += evaluate_path_cost(solve_shortest_path(pred_l1), c_true) - z_star
                sum_spo_reg += evaluate_path_cost(solve_shortest_path(pred_spo), c_true) - z_star
                sum_z_star += z_star

            # Normalized SPO loss (regret / optimal cost), matching the R plot script
            raw_trials_l2[regime][deg].append((sum_l2_reg / sum_z_star) * 100)
            # raw_trials_l1[regime][deg].append((sum_l1_reg / sum_z_star) * 100)
            raw_trials_spo[regime][deg].append((sum_spo_reg / sum_z_star) * 100)

# =========================================================
# 4. Multi-Method Grouped Plotting Interface
# =========================================================
def render_unified_figure(l2_dict, spo_dict, title_string):
    plt.figure(figsize=(13, 6.5))

    positions = np.arange(len(degrees_to_test)) * 4.0
    colors = ['#3498db', '#e74c3c', '#2ecc71'] # Blue = L2, Red = L1, Green = SPO+

    for idx, deg in enumerate(degrees_to_test):
        b2 = plt.boxplot(l2_dict[deg], positions=[positions[idx] - 0.5], widths=0.6,
                         patch_artist=True, boxprops=dict(facecolor=colors[0], color='#2c3e50'), whis=1.5)
        # b1 = plt.boxplot(l1_dict[deg], positions=[positions[idx]], widths=0.6,
        #                  patch_artist=True, boxprops=dict(facecolor=colors[1], color='#2c3e50'), whis=1.5)
        b_spo = plt.boxplot(spo_dict[deg], positions=[positions[idx] + 0.5], widths=0.6,
                            patch_artist=True, boxprops=dict(facecolor=colors[2], color='#2c3e50'), whis=1.5)

        if idx == 0:
            legend_handles = [b2["boxes"][0], b_spo["boxes"][0]]

    plt.xticks(positions, [f"Degree {d}" for d in degrees_to_test], fontsize=10)
    plt.xlabel("Model Misspecification Complexity (True Polynomial Degree)", fontsize=11, fontweight='bold')
    plt.ylabel("Normalized Operational Regret Percentage (%)", fontsize=11, fontweight='bold')
    plt.title(title_string, fontsize=13, fontweight='bold', pad=15)
    plt.legend(legend_handles, ['L2 Loss (Lasso Regression)', 'SPO+ Loss (Decision-Aware Regularized)'], loc='upper left', fontsize=10)
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()

# Render Figure 1: Noisy Regrets
render_unified_figure(raw_trials_l2["noisy"], raw_trials_spo["noisy"],
                       "Figure 1: Side-by-Side Operational Regret Under Noisy Scoring Environments (H = 0.5)")

# Render Figure 2: Noiseless Regrets
render_unified_figure(raw_trials_l2["noiseless"], raw_trials_spo["noiseless"],
                       "Figure 2: Side-by-Side Operational Regret Under Perfect Noiseless Environments")

plt.show()