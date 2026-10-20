"""
Minimum viable validation experiment for the two-loop correction.

The integral being expanded is the marginal likelihood,
  p(o) = ∫ p(s, o) ds = ∫ exp(-A[s]) ds,   with A[s] := -log p(s, o).

The saddle s* is the MAP. The loop expansion of -log p(o) is:
  tree    : A[s*]                          (just the MAP value)
  1-loop  : tree + (1/2) log det H_s - (N/2) log(2π)   (Laplace approximation)
  2-loop  : 1-loop + F_2  with  F_2 = (1/8) W G² - (5/24) V² G³
where H_s = A''[s*], V = A'''[s*], W = A''''[s*], G = H_s^{-1}.

This is the version of the loop expansion that DIRECTLY approximates
-log p(o), and that the standard variational-Laplace literature (MacKay
1992; Friston et al. 2007) reproduces at one loop. Going to two loops
is what our paper does.

Concrete model: Poisson likelihood with Gaussian prior on the log-rate.
  p(s)    = N(0, 1)
  p(o|s)  = Poisson(exp(s))
The exact marginal is a 1D integral, computable to machine precision.

We report tree, 1-loop, 2-loop, exact, and the errors of each, for
several observed counts o.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from math import lgamma


# ----------------------------------------------------------------------
# The action A[s] = -log p(s, o) and its derivatives at the MAP
# ----------------------------------------------------------------------
def neg_log_joint(s, o):
    """A[s] = -log p(s, o) = e^s - s o + log o! + s²/2 + (1/2) log 2π."""
    return np.exp(s) - s * o + lgamma(o + 1) + 0.5 * s**2 + 0.5 * np.log(2 * np.pi)


def map_estimate(o):
    """Solve dA/ds = e^s - o + s = 0 for the MAP."""
    return brentq(lambda s: np.exp(s) - o + s, -10, 30)


def action_derivatives_at_map(o):
    """Return (s*, A[s*], H_s, V_s, W_s)."""
    s_star = map_estimate(o)
    A_star = neg_log_joint(s_star, o)
    exp_s = np.exp(s_star)
    H = exp_s + 1     # A''[s*]
    V = exp_s         # A'''[s*]
    W = exp_s         # A''''[s*]
    return s_star, A_star, H, V, W


# ----------------------------------------------------------------------
# Loop expansion terms
# ----------------------------------------------------------------------
def laplace_approximation(o):
    """Tree + 1-loop = standard Laplace approximation to -log p(o)."""
    s_star, A_star, H, _, _ = action_derivatives_at_map(o)
    return A_star + 0.5 * np.log(H) - 0.5 * np.log(2 * np.pi)


def two_loop_correction(o):
    """F_2 = (1/8) W G² - (5/24) V² G³.

    Derivation: F_2 = ⟨X_4⟩ - (1/2)⟨X_3²⟩_connected with X_4 = (W/24)η⁴,
    X_3 = (V/6)η³, and Gaussian moments ⟨η²⟩ = G, ⟨η⁴⟩ = 3G², ⟨η⁶⟩ = 15G³.
    The figure-eight gives (1/24)W · 3G² = WG²/8.
    The sunset gives (1/2)(1/36)V² · 15G³ = 5V²G³/24.
    The (5/24) coefficient is the connected-cumulant value, not (1/12).
    """
    _, _, H, V, W = action_derivatives_at_map(o)
    G = 1.0 / H
    fig8 = (1.0 / 8.0) * W * G**2
    sunset = (5.0 / 24.0) * V**2 * G**3
    return fig8 - sunset


# ----------------------------------------------------------------------
# Exact marginal log-likelihood by 1D quadrature
# ----------------------------------------------------------------------
def neg_log_marginal_exact(o, half_width=15.0):
    """-log p(o) via adaptive Gauss-Kronrod quadrature, with log-sum-exp shifting."""
    s_star = map_estimate(o)
    integrand = lambda s: np.exp(-(neg_log_joint(s, o) - neg_log_joint(s_star, o)))
    integral, _ = quad(integrand, s_star - half_width, s_star + half_width,
                       epsabs=1e-14, epsrel=1e-14)
    log_p_o = -neg_log_joint(s_star, o) + np.log(integral)
    return -log_p_o


# ----------------------------------------------------------------------
# Run comparison
# ----------------------------------------------------------------------
print("=" * 102)
print("Poisson(exp(s)) with N(0,1) prior on s — Laplace expansion of -log p(o)")
print("=" * 102)
print(f"{'o':>4} | {'tree (A*)':>11} {'+ 1-loop':>11} {'+ 2-loop':>11} {'exact':>11} | "
      f"{'err tree':>9} {'err 1L':>9} {'err 2L':>9} | {'ratio 2L/1L':>11}")
print("-" * 102)

for o in [0, 1, 2, 3, 5, 10, 20, 50]:
    s_star, A_star, H, V, W = action_derivatives_at_map(o)
    one_loop = laplace_approximation(o)
    F2 = two_loop_correction(o)
    two_loop = one_loop + F2
    exact = neg_log_marginal_exact(o)

    err_tree = abs(A_star - exact)
    err_1L = abs(one_loop - exact)
    err_2L = abs(two_loop - exact)
    ratio = err_2L / err_1L if err_1L > 0 else float("nan")

    print(f"{o:>4} | {A_star:>11.6f} {one_loop:>11.6f} {two_loop:>11.6f} {exact:>11.6f} | "
          f"{err_tree:>9.5f} {err_1L:>9.5f} {err_2L:>9.5f} | {ratio:>11.4f}")

print("=" * 102)
print()
print("Interpretation:")
print("  tree     = A[s*] = -log p(s*, o)  (MAP value, ignores curvature)")
print("  1-loop   = tree + (1/2) log H_s - (1/2) log(2π)   (Laplace, Friston 2007)")
print("  2-loop   = 1-loop + (1/8) W G² - (5/24) V² G³     (this paper)")
print("  exact    = -log p(o) by 1D Gauss-Kronrod quadrature (machine precision)")
print()
print("Expected pattern: tree is bad (no curvature), 1-loop is much better (Laplace),")
print("2-loop further reduces error in the small-o regime where the posterior is")
print("strongly non-Gaussian. At large o (Gaussian limit), 1-loop and 2-loop converge.")
