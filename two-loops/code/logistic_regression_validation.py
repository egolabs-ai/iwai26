"""
Tier-2 validation: small-N logistic regression.

A representative small-but-real Bayesian inference problem. Tests the
multivariate generalisation of the loop expansion against MCMC ground
truth (via Laplace + importance sampling, an MCMC-equivalent estimator
that is robust and reproducible at K = 3, N up to a few hundred).

Model:
  β ~ N(0, I_K)
  y_n | β ~ Bernoulli(σ(x_n^T β))   for n = 1, ..., N
  X is a fixed N × K design matrix (drawn once, frozen).

Action:
  A[β] = -log p(β, y)
       = (K/2) log(2π) + (1/2) β^T β
         + Σ_n [log(1 + exp(x_n^T β)) - y_n · x_n^T β]

We compute:
  tree   : A[β*]                                 (MAP value)
  1-loop : tree + (1/2) log det H_β* - (K/2) log(2π)
  2-loop : 1-loop + (1/8) W_ijkl G_ij G_kl
                   - (5/24) V_ijk V_lmn G_il G_jm G_kn
  exact  : -log p(y) by Laplace + importance sampling, M = 5·10^5

Reports tree, 1-loop, 2-loop, exact, and the absolute errors of each,
for several values of N at fixed K = 3.
"""
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize


# ----------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------
def generate_data(N, K, seed=0, intercept_shift=0.0):
    """Generate synthetic logistic-regression data.

    intercept_shift: positive values shift the systematic logit toward +∞,
    producing class-imbalanced labels (most y_n = 1). This drives σ_n away
    from 1/2 systematically, which is the regime where the spectacles
    contribution u_i G_ij u_j is non-negligible.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, K))
    # Always include intercept-like first column
    X[:, 0] = 1.0
    beta_true = rng.standard_normal(K)
    beta_true[0] += intercept_shift
    p = expit(X @ beta_true)
    y = (rng.random(N) < p).astype(float)
    return X, y, beta_true


# ----------------------------------------------------------------------
# Action and derivatives
# ----------------------------------------------------------------------
def neg_log_joint(beta, X, y):
    K = beta.shape[0]
    logits = X @ beta
    log_lik = np.sum(np.logaddexp(0.0, logits) - y * logits)
    log_prior = 0.5 * K * np.log(2 * np.pi) + 0.5 * beta @ beta
    return log_prior + log_lik


def grad_action(beta, X, y):
    return beta + X.T @ (expit(X @ beta) - y)


def find_map(X, y, K):
    res = minimize(
        neg_log_joint, x0=np.zeros(K), args=(X, y),
        jac=grad_action, method="Newton-CG",
        options={"xtol": 1e-12},
    )
    return res.x


def hessian(beta, X, y):
    sig = expit(X @ beta)
    W_diag = sig * (1 - sig)                              # N
    return np.eye(beta.shape[0]) + X.T @ (W_diag[:, None] * X)


def cubic_tensor(beta, X):
    """V_ijk = Σ_n σ(1-σ)(1-2σ) x_ni x_nj x_nk at β."""
    sig = expit(X @ beta)
    w = sig * (1 - sig) * (1 - 2 * sig)                   # N
    return np.einsum("n,ni,nj,nk->ijk", w, X, X, X)


def quartic_tensor(beta, X):
    """W_ijkl = Σ_n σ(1-σ)[1 - 6σ(1-σ)] x_ni x_nj x_nk x_nl at β."""
    sig = expit(X @ beta)
    s1ms = sig * (1 - sig)
    w = s1ms * (1 - 6 * s1ms)                             # N
    return np.einsum("n,ni,nj,nk,nl->ijkl", w, X, X, X, X)


# ----------------------------------------------------------------------
# Loop expansion predictions for -log p(y)
# ----------------------------------------------------------------------
def loop_predictions(X, y):
    """Compute tree, one-loop, and two TWO-LOOP variants:

    F2_naive = (1/8) W:GG - (5/24) V V:GGG   [1D-correct, multivariate-wrong]
    F2_correct = (1/8) W:GG - (1/12) V V:GGG - (1/8) u G u
        where u_i := V_{ijk} G_{jk}, the tadpole vector.

    In 1D both reduce to (1/8) W G² - (5/24) V² G³ (the sunset and
    spectacles contractions collapse). In multivariate they differ.
    """
    K = X.shape[1]
    beta_star = find_map(X, y, K)
    A_star = neg_log_joint(beta_star, X, y)
    H = hessian(beta_star, X, y)
    G = np.linalg.inv(H)
    sign, logdetH = np.linalg.slogdet(H)
    assert sign > 0, "Hessian should be positive definite at MAP"

    # 1-loop (Laplace)
    one_loop_correction = 0.5 * logdetH - 0.5 * K * np.log(2 * np.pi)
    one_loop = A_star + one_loop_correction

    # Two-loop ingredients
    V = cubic_tensor(beta_star, X)
    Wq = quartic_tensor(beta_star, X)
    figure_eight = (1.0 / 8.0) * np.einsum("ijkl,ij,kl->", Wq, G, G)
    sunset_contraction = np.einsum("ijk,lmn,il,jm,kn->", V, V, G, G, G)
    # Tadpole vector u_i = V_{ijk} G_{jk}
    u = np.einsum("ijk,jk->i", V, G)
    spectacles_contraction = np.einsum("i,ij,j->", u, G, u)

    # Naive (5/24-sunset) version — wrong in multivariate
    F2_naive = figure_eight - (5.0 / 24.0) * sunset_contraction

    # Correct (1/12-sunset + 1/8-spectacles) version
    F2_correct = (
        figure_eight
        - (1.0 / 12.0) * sunset_contraction
        - (1.0 / 8.0) * spectacles_contraction
    )

    return {
        "tree": A_star,
        "one_loop": one_loop,
        "two_loop_naive": one_loop + F2_naive,
        "two_loop_correct": one_loop + F2_correct,
        "fig8": figure_eight,
        "sunset_contraction": sunset_contraction,
        "spectacles_contraction": spectacles_contraction,
        "u_norm": np.linalg.norm(u),
        "beta_star": beta_star,
        "H": H,
        "G": G,
    }


# ----------------------------------------------------------------------
# Ground-truth: Laplace + importance sampling
# ----------------------------------------------------------------------
def neg_log_joint_batch(betas, X, y):
    """Vectorised neg_log_joint over a batch of (M, K) beta values."""
    K = betas.shape[1]
    logits = betas @ X.T                                  # (M, N)
    # log(1+e^z) - y z, summed over n
    log_lik = np.sum(np.logaddexp(0.0, logits) - y[None, :] * logits, axis=1)  # (M,)
    log_prior = 0.5 * K * np.log(2 * np.pi) + 0.5 * np.sum(betas * betas, axis=1)
    return log_prior + log_lik


def ground_truth(X, y, M=500_000, seed=1, batch=50_000):
    """Estimate -log p(y) by importance sampling from the Laplace-approx posterior,
    chunked to keep memory bounded.
    """
    rng = np.random.default_rng(seed)
    K = X.shape[1]
    beta_star = find_map(X, y, K)
    H = hessian(beta_star, X, y)
    sign, logdetH = np.linalg.slogdet(H)
    L = np.linalg.cholesky(np.linalg.inv(H))

    log_w_chunks = []
    M_remaining = M
    while M_remaining > 0:
        m = min(batch, M_remaining)
        Z = rng.standard_normal((m, K))
        samples = beta_star[None, :] + Z @ L.T
        log_p_joint = -neg_log_joint_batch(samples, X, y)
        log_q = (
            -0.5 * K * np.log(2 * np.pi)
            + 0.5 * logdetH
            - 0.5 * np.einsum("mi,mi->m", Z, Z)
        )
        log_w_chunks.append(log_p_joint - log_q)
        M_remaining -= m

    log_w = np.concatenate(log_w_chunks)
    log_w_max = log_w.max()
    log_p_y = log_w_max + np.log(np.mean(np.exp(log_w - log_w_max)))
    weights = np.exp(log_w - log_w_max)
    ess = (weights.sum()) ** 2 / (weights ** 2).sum()
    return -log_p_y, ess


# ----------------------------------------------------------------------
# Run comparison across N and across label-balance regimes
# ----------------------------------------------------------------------
def run_sweep(intercept_shift, title):
    print()
    print("=" * 122)
    print(title)
    print("=" * 122)
    print(
        f"{'N':>4} | {'mean y':>7} {'|u|':>7} | "
        f"{'tree':>10} {'1-loop':>10} {'2L naive':>10} {'2L correct':>11} {'exact':>12} | "
        f"{'err 1L':>8} {'err 2Ln':>8} {'err 2Lc':>8}"
    )
    print("-" * 122)

    K = 3
    for N in [20, 30, 50, 100, 200]:
        X, y, beta_true = generate_data(N, K, seed=42, intercept_shift=intercept_shift)
        preds = loop_predictions(X, y)
        M = 500_000
        exact, ess = ground_truth(X, y, M=M, seed=N * 7)

        err_1L = abs(preds["one_loop"] - exact)
        err_2Ln = abs(preds["two_loop_naive"] - exact)
        err_2Lc = abs(preds["two_loop_correct"] - exact)

        print(
            f"{N:>4} | {y.mean():>7.3f} {preds['u_norm']:>7.3f} | "
            f"{preds['tree']:>10.4f} {preds['one_loop']:>10.4f} "
            f"{preds['two_loop_naive']:>10.4f} {preds['two_loop_correct']:>11.4f} "
            f"{exact:>12.6f} | "
            f"{err_1L:>8.5f} {err_2Ln:>8.5f} {err_2Lc:>8.5f}"
        )


# Balanced labels: mean y ≈ 0.5; |u| should be small.
run_sweep(intercept_shift=0.0, title="Balanced labels (intercept shift = 0.0)")

# Strongly imbalanced labels: most y = 1; (1-2σ) is systematically nonzero,
# so the spectacles tadpole vector u_i has a measurable norm.
run_sweep(intercept_shift=3.0, title="Imbalanced labels (intercept shift = +3.0)")

print()
print("=" * 122)
print("Interpretation:")
print("  tree        = A[β*]")
print("  1-loop      = Laplace approximation")
print("  2L naive    = 1-loop + (1/8) W:GG - (5/24) V V:GGG          [1D-correct, multivariate-WRONG]")
print("  2L correct  = 1-loop + (1/8) W:GG - (1/12) V V:GGG - (1/8) u G u  [multivariate-correct]")
print("  |u|         = ||V_{ijk} G_{jk}|| — spectacles tadpole vector norm")
print()
print("In the balanced regime, |u| is small (each σ_n(1-σ_n)(1-2σ_n) is small)")
print("and the two formulae agree, so the 1D-derived (5/24) sunset coefficient appears")
print("to fit the multivariate data 'by accident'. In the imbalanced regime, |u| is large,")
print("the spectacles contribution is non-negligible, and the corrected formula tracks")
print("the exact ground truth while the naive (5/24) formula degrades.")
