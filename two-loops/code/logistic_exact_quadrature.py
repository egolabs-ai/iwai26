"""
Exact ground truth for the logistic-regression convergence test (paper Table 2),
by whitened tensor-product Gauss-Hermite quadrature instead of importance
sampling.

Why this exists. The submitted paper estimates -log p(y) by importance sampling
from the Laplace posterior at M = 5e5 draws. That estimator has a Monte Carlo
standard error of order 1e-3 nats, which is the same size as the two-loop
residuals it is being used to measure. This script removes the sampling error
entirely: for K = 3 the evidence integral is three-dimensional, and after
whitening by the Laplace covariance the integrand is smooth and Gauss-Hermite
converges to machine precision (verified by node-count refinement below).

Method. Write beta = beta* + L z with L L^T = H^-1. Then
    p(y) = |det L| * integral exp(-A(beta* + L z)) dz
         = |det L| * integral exp(-z'z/2) * [exp(-A + z'z/2)] dz,
and the bracket is slowly varying, so Gauss-Hermite quadrature against the
exp(-z'z/2) weight is very accurate. Reported spreads across node counts are
~1e-13 nats, i.e. ten orders of magnitude below the quantities of interest.

Run:  uv run python logistic_exact_quadrature.py
"""

import itertools
import math

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import minimize
from scipy.special import expit

K = 3
NODES = (30, 45, 60)


def generate_data(N, K, seed=42, intercept_shift=0.0):
    """Identical to logistic_regression_validation.generate_data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, K))
    X[:, 0] = 1.0
    beta_true = rng.standard_normal(K)
    beta_true[0] += intercept_shift
    y = (rng.random(N) < expit(X @ beta_true)).astype(float)
    return X, y, beta_true


def make_action(X, y):
    K = X.shape[1]

    def A(b):
        z = X @ b
        return (0.5 * K * math.log(2 * math.pi) + 0.5 * b @ b
                + np.sum(np.logaddexp(0.0, z) - y * z))
    return A


def loop_terms(X, y):
    """MAP, tensors, the three diagrams, and the tree / 1-loop / 2-loop values."""
    Kd = X.shape[1]
    A = make_action(X, y)
    b = minimize(A, np.zeros(Kd), method="BFGS",
                 options=dict(gtol=1e-12, maxiter=20000)).x
    s = expit(X @ b)
    H = np.eye(Kd) + np.einsum("n,ni,nj->ij", s * (1 - s), X, X)
    V = np.einsum("n,ni,nj,nk->ijk", s * (1 - s) * (1 - 2 * s), X, X, X)
    W = np.einsum("n,ni,nj,nk,nl->ijkl",
                  s * (1 - s) * (1 - 6 * s * (1 - s)), X, X, X, X)
    G = np.linalg.inv(H)
    fig8 = 0.125 * np.einsum("ijkl,ij,kl->", W, G, G)
    sun_c = np.einsum("ijk,lmn,il,jm,kn->", V, V, G, G, G)
    u = np.einsum("ijk,jk->i", V, G)
    spec_c = u @ G @ u
    tree = A(b)
    one = tree + 0.5 * np.linalg.slogdet(H)[1] - 0.5 * Kd * math.log(2 * math.pi)
    F2 = fig8 - sun_c / 12 - spec_c / 8
    F2_naive = fig8 - (5 / 24) * sun_c
    return dict(beta=b, H=H, G=G, u=u, unorm=float(np.linalg.norm(u)),
                fig8=fig8, sunset=-sun_c / 12, spect=-spec_c / 8,
                tree=tree, one=one, two=one + F2, two_naive=one + F2_naive,
                F2=F2, F2_naive=F2_naive)


def evidence_gh(X, y, n):
    """-log p(y) by whitened tensor-product Gauss-Hermite with n nodes per axis."""
    Kd = X.shape[1]
    A = make_action(X, y)
    b = minimize(A, np.zeros(Kd), method="BFGS",
                 options=dict(gtol=1e-12, maxiter=20000)).x
    s = expit(X @ b)
    H = np.eye(Kd) + np.einsum("n,ni,nj->ij", s * (1 - s), X, X)
    L = np.linalg.cholesky(np.linalg.inv(H))
    x, w = hermegauss(n)
    idx = np.array(list(itertools.product(range(n), repeat=Kd)))
    Z = x[idx]
    Wt = np.prod(w[idx], axis=1)
    B = b[None, :] + Z @ L.T
    logits = B @ X.T
    ll = np.sum(np.logaddexp(0.0, logits) - y[None, :] * logits, axis=1)
    lp = 0.5 * Kd * math.log(2 * math.pi) + 0.5 * np.sum(B * B, axis=1)
    expo = -(lp + ll) + 0.5 * np.sum(Z * Z, axis=1)
    m = expo.max()
    log_p = (math.log(np.sum(Wt * np.exp(expo - m))) + m
             + math.log(abs(np.linalg.det(L))))
    return -log_p


def importance_sampling(X, y, M=500_000, seed=1):
    """The estimator used in the submitted paper, for comparison."""
    rng = np.random.default_rng(seed)
    Kd = X.shape[1]
    A = make_action(X, y)
    b = minimize(A, np.zeros(Kd), method="BFGS",
                 options=dict(gtol=1e-12, maxiter=20000)).x
    s = expit(X @ b)
    H = np.eye(Kd) + np.einsum("n,ni,nj->ij", s * (1 - s), X, X)
    logdetH = np.linalg.slogdet(H)[1]
    L = np.linalg.cholesky(np.linalg.inv(H))
    Z = rng.standard_normal((M, Kd))
    B = b[None, :] + Z @ L.T
    logits = B @ X.T
    ll = np.sum(np.logaddexp(0.0, logits) - y[None, :] * logits, axis=1)
    lp = 0.5 * Kd * math.log(2 * math.pi) + 0.5 * np.sum(B * B, axis=1)
    lw = -(lp + ll) - (-0.5 * Kd * math.log(2 * math.pi) + 0.5 * logdetH
                       - 0.5 * np.einsum("mi,mi->m", Z, Z))
    m = lw.max()
    log_p = m + math.log(np.mean(np.exp(lw - m)))
    wt = np.exp(lw - m)
    ess = wt.sum() ** 2 / (wt ** 2).sum()
    # delta-method standard error of log(mean(w))
    se = math.sqrt(np.var(wt, ddof=1) / M) / wt.mean()
    return -log_p, ess / M, se


PAPER = {
    ("Balanced", 20):  (13.059, 12.129, 12.071, 12.074, 0.055, 0.003),
    ("Balanced", 50):  (27.980, 28.079, 28.043, 28.044, 0.035, 0.001),
    ("Balanced", 100): (47.201, 48.256, 48.225, 48.226, 0.030, 0.001),
    ("Balanced", 200): (107.002, 109.288, 109.273, 109.273, 0.015, 5e-5),
    ("Imbalanced", 20):  (10.318, 9.072, 9.039, 9.038, 0.035, 0.001),
    ("Imbalanced", 50):  (19.718, 19.361, 19.326, 19.329, 0.033, 0.003),
    ("Imbalanced", 100): (22.034, 22.229, 22.195, 22.197, 0.032, 0.002),
    ("Imbalanced", 200): (38.896, 39.752, 39.729, 39.730, 0.022, 0.001),
}

if __name__ == "__main__":
    print("=" * 118)
    print("Quadrature convergence check (spread across node counts "
          f"{NODES}, in nats)")
    print("=" * 118)
    exact = {}
    for (regime, N) in PAPER:
        shift = 0.0 if regime == "Balanced" else 3.0
        X, y, _ = generate_data(N, K, seed=42, intercept_shift=shift)
        vals = [evidence_gh(X, y, n) for n in NODES]
        exact[(regime, N)] = vals[-1]
        print(f"  {regime:>10} N={N:>3}: {vals[-1]:.9f}   "
              f"spread = {max(vals)-min(vals):.2e}")

    print()
    print("=" * 118)
    print("Table 2 recomputed against EXACT ground truth")
    print("=" * 118)
    print(f"{'regime':>10} {'N':>4} {'ybar':>6} {'|u|':>6} | "
          f"{'tree':>9} {'1-loop':>9} {'2-loop':>9} {'exact(quad)':>12} "
          f"{'IS estimate':>12} | {'err 1L':>8} {'err 2L':>9} {'ratio':>7}")
    print("-" * 118)
    rows = []
    for (regime, N), pv in PAPER.items():
        shift = 0.0 if regime == "Balanced" else 3.0
        X, y, _ = generate_data(N, K, seed=42, intercept_shift=shift)
        t = loop_terms(X, y)
        ex = exact[(regime, N)]
        is_val, essf, se = importance_sampling(X, y, M=500_000, seed=N * 7)
        e1, e2 = abs(t["one"] - ex), abs(t["two"] - ex)
        rows.append((regime, N, y.mean(), t, ex, is_val, essf, se, e1, e2))
        print(f"{regime:>10} {N:>4} {y.mean():>6.3f} {t['unorm']:>6.2f} | "
              f"{t['tree']:>9.3f} {t['one']:>9.3f} {t['two']:>9.3f} "
              f"{ex:>12.6f} {is_val:>12.6f} | "
              f"{e1:>8.5f} {e2:>9.6f} {e1/e2:>7.1f}")

    print()
    print("=" * 118)
    print("Diagnosis of the importance-sampling ground truth used in the paper")
    print("=" * 118)
    print(f"{'regime':>10} {'N':>4} {'ESS/M':>7} {'IS s.e.':>9} "
          f"{'|IS - exact|':>13} {'paper |d2L|':>12} {'true |d2L|':>11}")
    print("-" * 118)
    for regime, N, ybar, t, ex, is_val, essf, se, e1, e2 in rows:
        print(f"{regime:>10} {N:>4} {essf:>7.3f} {se:>9.2e} "
              f"{abs(is_val-ex):>13.2e} {PAPER[(regime,N)][5]:>12} {e2:>11.6f}")

    print()
    print("=" * 118)
    print("Diagram breakdown, and the naive-versus-correct comparison")
    print("=" * 118)
    print(f"{'regime':>10} {'N':>4} {'|u|':>6} | {'fig-8':>9} {'sunset':>9} "
          f"{'spect':>9} {'F2':>9} {'F2 naive':>9} | "
          f"{'err 2L':>9} {'err naive':>10}")
    print("-" * 118)
    for regime, N, ybar, t, ex, is_val, essf, se, e1, e2 in rows:
        en = abs(t["two_naive"] - ex)
        print(f"{regime:>10} {N:>4} {t['unorm']:>6.2f} | {t['fig8']:>9.4f} "
              f"{t['sunset']:>9.4f} {t['spect']:>9.4f} {t['F2']:>9.4f} "
              f"{t['F2_naive']:>9.4f} | {e2:>9.6f} {en:>10.6f}")
