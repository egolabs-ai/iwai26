"""
Worked examples for the primer, "Feynman Diagrams for the Variational Free
Energy" (VFE_Loop_Expansion.pdf).

Each numbered exercise corresponds to a section of the primer and prints the
numbers quoted there. Every numerical claim the primer makes is computed here
from scratch, so a reader can check the algebra they did on the whiteboard
against ground truth.

  Exercise 1  Gaussian integrals and Wick's theorem, 1-D and multivariate,
              verified to machine precision by Gauss-Hermite quadrature.
  Exercise 2  Explicit enumeration of the 15 pairings of two cubic vertices and
              their split into 6 "sunset" and 9 "spectacles" topologies.
  Exercise 3  Zero-dimensional field theory, quartic only: isolates the
              figure-eight diagram (W != 0, V = 0) and its 1/8 coefficient.
  Exercise 4  Zero-dimensional field theory, cubic and quartic: isolates the
              combined 5/24 coefficient of the 1-D sunset+spectacles collapse.
  Exercise 5  Two-dimensional exact test: separates the 1/12 sunset coefficient
              from the 1/8 spectacles coefficient against machine-precision
              quadrature; also the one-loop tadpole shift of the mean.
  Exercise 6  Poisson-Gaussian model: every intermediate number at o = 1, then
              the full sweep that reproduces Table 1 of the paper.
  Exercise 7  Three-state categorical with exact cumulant tensors: the full
              four-index contraction versus the diagonal-only shortcut, and
              the approach to the simplex boundary.

All integrals are done in the rescaled fluctuation coordinate xi = sqrt(beta) *
(phi - phi*), which is the same change of variables the primer performs
analytically. That keeps the integrand O(1)-wide for every beta and avoids the
quadrature roundoff that a fixed integration box would suffer at large beta.

Run:  uv run python primer_examples.py
"""

import itertools
import math

import numpy as np
from scipy.integrate import quad, dblquad
from scipy.optimize import brentq, minimize
from numpy.polynomial.hermite_e import hermegauss

RULE = "=" * 78
SUB = "-" * 78


def head(n, title):
    print()
    print(RULE)
    print(f"Exercise {n}.  {title}")
    print(RULE)


# ======================================================================
# Exercise 1: Gaussian integrals and Wick's theorem
# ======================================================================
def gauss_hermite_moment(G, idx, n_nodes=6):
    """
    Exact Gaussian moment E[ xi_{i1} ... xi_{ik} ] for xi ~ N(0, G), by
    tensor-product Gauss-Hermite quadrature after a Cholesky whitening.

    Gauss-Hermite with n nodes integrates polynomials of degree <= 2n-1 exactly
    against the standard normal weight, so for k <= 11 this is exact up to
    floating-point roundoff -- not a Monte Carlo estimate.
    """
    G = np.asarray(G, dtype=float)
    d = G.shape[0]
    L = np.linalg.cholesky(G)
    x, w = hermegauss(n_nodes)          # weight exp(-x^2/2), sum(w) = sqrt(2 pi)
    w = w / w.sum()                     # normalise to a probability measure
    total = 0.0
    for nodes in itertools.product(range(n_nodes), repeat=d):
        z = np.array([x[m] for m in nodes])
        weight = math.prod(w[m] for m in nodes)
        xi = L @ z
        total += weight * math.prod(xi[i] for i in idx)
    return total


def exercise_1():
    head(1, "Gaussian integrals and Wick's theorem")

    print("A. One dimension.  x ~ N(0, G), G = 2.0.")
    print("   Wick's theorem says <x^(2n)> = (number of pairings) * G^n,")
    print("   and the number of pairings of 2n objects is (2n-1)!! .")
    G1 = 2.0
    norm = math.sqrt(2 * math.pi * G1)
    print()
    print(f"   {'2n':>4} {'(2n-1)!!':>10} {'Wick':>16} {'quadrature':>16} {'rel err':>10}")
    for n in (1, 2, 3, 4):
        k = 2 * n
        dfact = math.prod(range(1, k, 2))
        pred = dfact * G1**n
        val, _ = quad(lambda x: x**k * math.exp(-x * x / (2 * G1)) / norm,
                      -25, 25, epsabs=1e-11, epsrel=1e-12, limit=400)
        print(f"   {k:>4} {dfact:>10d} {pred:>16.8f} {val:>16.8f} "
              f"{abs(val-pred)/pred:>10.1e}")
    print("   The 15 pairings of six objects are the ones enumerated in Exercise 2.")

    print()
    print("B. Multivariate.  xi ~ N(0, G) in 3 dimensions, G symmetric positive")
    print("   definite. Wick: <xi_i xi_j xi_k xi_l> = Gij Gkl + Gik Gjl + Gil Gjk.")
    G = np.array([[2.0, 0.6, -0.4],
                  [0.6, 1.5, 0.3],
                  [-0.4, 0.3, 1.2]])
    worst_4, worst_6 = 0.0, 0.0
    for i, j, k, l in itertools.product(range(3), repeat=4):
        wick = G[i, j] * G[k, l] + G[i, k] * G[j, l] + G[i, l] * G[j, k]
        exact = gauss_hermite_moment(G, (i, j, k, l))
        worst_4 = max(worst_4, abs(exact - wick))
    print(f"   worst absolute error over all 81 four-index moments: {worst_4:.2e}")

    # six-index: sum over all 15 pairings, built from the enumeration routine
    for idx in [(0, 0, 1, 1, 2, 2), (0, 1, 2, 0, 1, 2), (0, 0, 0, 1, 1, 1)]:
        wick = 0.0
        for m in perfect_matchings(range(6)):
            wick += math.prod(G[idx[p], idx[q]] for p, q in m)
        exact = gauss_hermite_moment(G, idx)
        worst_6 = max(worst_6, abs(exact - wick))
        print(f"   <xi_{''.join(str(i) for i in idx)}>: 15-pairing sum "
              f"{wick:>12.8f}   quadrature {exact:>12.8f}")
    print(f"   worst absolute error on the six-index moments: {worst_6:.2e}")

    print()
    print("C. The contraction that becomes the figure-eight.")
    print("   For a fully symmetric W the three pairings of <xi_i xi_j xi_k xi_l>")
    print("   all collapse onto the same scalar W_ijkl Gij Gkl, so")
    print("      (1/24) W_ijkl <xi_i xi_j xi_k xi_l> = (3/24) W_ijkl Gij Gkl")
    print("                                         = (1/8)  W_ijkl Gij Gkl.")
    rng = np.random.default_rng(0)
    W = rng.normal(size=(3, 3, 3, 3))
    W = sum(np.transpose(W, p) for p in itertools.permutations(range(4))) / 24
    direct = sum(W[i, j, k, l] * gauss_hermite_moment(G, (i, j, k, l))
                 for i, j, k, l in itertools.product(range(3), repeat=4)) / 24
    formula = np.einsum("ijkl,ij,kl->", W, G, G) / 8
    print(f"   (1/24) sum_ijkl W_ijkl <xi xi xi xi> = {direct:.10f}")
    print(f"   (1/8) W_ijkl Gij Gkl                 = {formula:.10f}")
    print(f"   difference: {abs(direct-formula):.2e}")


# ======================================================================
# Exercise 2: enumerating the 15 pairings
# ======================================================================
def perfect_matchings(items):
    """All perfect matchings (Wick pairings) of an even-length sequence."""
    items = list(items)
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for idx in range(len(rest)):
        pair = (first, rest[idx])
        for tail in perfect_matchings(rest[:idx] + rest[idx + 1:]):
            yield [pair] + tail


def exercise_2():
    head(2, "The 15 pairings of two cubic vertices: 6 sunset + 9 spectacles")

    legs = ["a1", "a2", "a3", "b1", "b2", "b3"]
    print("Vertex A has legs a1 a2 a3; vertex B has legs b1 b2 b3.")
    print("A Wick pairing is a perfect matching of these six legs.")
    matchings = list(perfect_matchings(legs))
    print(f"Number of perfect matchings: {len(matchings)}   (= 5!! = 5*3*1 = 15)")

    sunset, spectacles, other = [], [], []
    for m in matchings:
        crossing = sum(1 for (x, y) in m if x[0] != y[0])
        (sunset if crossing == 3 else
         spectacles if crossing == 1 else other).append(m)

    def show(m):
        return "  ".join(f"({x}-{y})" for x, y in m)

    print()
    print(f"SUNSET topology: all 3 legs cross A -> B.  {len(sunset)} pairings = 3!")
    for m in sunset:
        print("   ", show(m))
    print("   Each contributes V_ijk V_lmn G_il G_jm G_kn.")
    print()
    print(f"SPECTACLES topology: 1 leg crosses, 1 self-pair inside each vertex.")
    print(f"   {len(spectacles)} pairings = 3 (self-pair in A) x 3 (self-pair in B).")
    for m in spectacles:
        print("   ", show(m))
    print("   Each contributes u_i G_il u_l with the tadpole vector u_i = V_ijk G_jk.")
    print()
    print(f"Pairings with any other crossing number: {len(other)}.")
    print("   This must be zero. A cubic vertex has 3 legs, an odd number, so the")
    print("   number of its legs that cross is 3 - 2*(self-pairs) which is 3 or 1.")
    print("   There is no 2-crossing topology, hence exactly two diagrams.")
    print()
    print("Prefactor. The second cumulant term is (1/2!) <X3^2>, and each X3")
    print("carries 1/3!, so the common prefactor is (1/2)(1/6)^2 = 1/72.")
    print(f"   sunset coefficient      = 6/72 = {6/72:.10f} = 1/12")
    print(f"   spectacles coefficient  = 9/72 = {9/72:.10f} = 1/8")
    print(f"   total pairings check    : 6 + 9 = {6+9}")
    print()
    print("In ONE dimension u G u = (VG) G (VG) = V^2 G^3 and the sunset")
    print("contraction is also V^2 G^3, so the two collapse into a single scalar")
    print(f"with coefficient 1/12 + 1/8 = {1/12 + 1/8:.10f} = 5/24.")
    print("This is exactly why a 1-D test cannot separate them.")


# ======================================================================
# Shared machinery for the zero- and two-dimensional exact tests
# ======================================================================
def exact_neglogZ_1d(lam, g, H, beta, L=14.0):
    """
    -(1/beta) log Z for  S(x) = (H/2) x^2 + (lam/6) x^3 + (g/24) x^4,  x* = 0.

    Integrate in the rescaled coordinate xi = sqrt(beta) x, in which
        S_beta(xi) = (H/2) xi^2 + lam xi^3/(6 sqrt(beta)) + g xi^4/(24 beta),
        Z = beta^(-1/2) * integral exp(-S_beta(xi)) dxi.
    """
    def S_beta(xi):
        return (0.5 * H * xi * xi
                + lam * xi**3 / (6.0 * math.sqrt(beta))
                + g * xi**4 / (24.0 * beta))

    I, _ = quad(lambda xi: math.exp(-S_beta(xi)), -L, L,
                epsabs=1e-13, epsrel=1e-13, limit=500)
    return 0.5 * math.log(beta) / beta - math.log(I) / beta


def zero_dim_report(name, lam, g, betas, H=1.0):
    """Tree / one-loop / two-loop versus exact for the 0-D model."""
    G = 1.0 / H
    fig8 = 0.125 * g * G**2
    cub = (5.0 / 24.0) * lam**2 * G**3
    F2 = fig8 - cub
    print()
    print(f"  {name}")
    print(f"    S(x) = ({H}/2) x^2 + ({lam}/6) x^3 + ({g}/24) x^4")
    print(f"    x* = 0,  S(x*) = 0,  H = {H},  G = 1/H = {G},  V = {lam},  W = {g}")
    print(f"    figure-eight   +(1/8) W G^2     = {fig8:+.8f}")
    print(f"    sunset+spect   -(5/24) V^2 G^3  = {-cub:+.8f}")
    print(f"    F2                              = {F2:+.8f}")
    print()
    print(f"    {'beta':>7} {'1-loop':>12} {'2-loop':>12} {'exact':>12}"
          f" {'err 1L':>10} {'err 2L':>10} {'err1L*b^2':>10} {'err2L*b^3':>10}")
    for beta in betas:
        exact = exact_neglogZ_1d(lam, g, H, beta)
        one = (0.5 * math.log(H) - 0.5 * math.log(2 * math.pi / beta)) / beta
        two = one + F2 / beta**2
        e1, e2 = abs(one - exact), abs(two - exact)
        print(f"    {beta:>7.0f} {one:>12.6f} {two:>12.6f} {exact:>12.6f}"
              f" {e1:>10.2e} {e2:>10.2e} {e1*beta**2:>10.5f} {e2*beta**3:>10.5f}")
    print(f"    err1L * beta^2 should approach |F2| = {abs(F2):.5f}:  that is the")
    print(f"    quantitative statement that the one-loop truncation error IS the")
    print(f"    two-loop term. err2L * beta^3 approaches the (uncomputed) F3.")


# ======================================================================
# Exercise 3: quartic only
# ======================================================================
def exercise_3():
    head(3, "Zero-dimensional field theory, quartic only: the figure-eight alone")
    print("The simplest non-trivial 'field theory' has a single real variable, so")
    print("there is no space and no time -- only the algebra of the loop expansion.")
    print("Setting V = 0 kills both cubic diagrams, leaving the figure-eight as the")
    print("only two-loop object, so its coefficient 1/8 is tested on its own.")

    print()
    print("  Sanity check first: a purely Gaussian action (V = W = 0) must be exact")
    print("  at one loop, with no two-loop correction at all.")
    for beta in (1, 5, 20):
        exact = exact_neglogZ_1d(0.0, 0.0, 1.3, beta)
        one = (0.5 * math.log(1.3) - 0.5 * math.log(2 * math.pi / beta)) / beta
        print(f"    beta = {beta:>3}:  one-loop {one:+.12f}   exact {exact:+.12f}"
              f"   difference {abs(one-exact):.2e}")
    print("  This is the statement 'variational Laplace is exact for linear-Gaussian")
    print("  models'. Everything below measures departures from that.")

    zero_dim_report("Pure quartic, g = 1", lam=0.0, g=1.0,
                    betas=[1, 2, 5, 10, 20, 50, 100])
    zero_dim_report("Pure quartic, g = 3 (stronger non-Gaussianity)", lam=0.0, g=3.0,
                    betas=[1, 2, 5, 10, 20, 50, 100])


# ======================================================================
# Exercise 4: cubic and quartic
# ======================================================================
def exercise_4():
    head(4, "Zero-dimensional field theory, cubic and quartic: the 5/24 collapse")
    print("With V != 0 the sunset and spectacles diagrams both switch on. In one")
    print("dimension they are the same scalar V^2 G^3, so this tests only their")
    print("sum, 1/12 + 1/8 = 5/24. Separating them needs Exercise 5.")
    print()
    print("GOTCHA. A pure cubic action is not integrable: exp(-S) blows up as")
    print("x -> -infinity. We always keep g > 0 so that Z exists. This is the")
    print("0-D shadow of the fact that the loop expansion is an expansion of a")
    print("well-defined integral, not a formal manipulation.")

    zero_dim_report("Cubic and quartic, V = 0.5, W = 1", lam=0.5, g=1.0,
                    betas=[1, 2, 5, 10, 20, 50, 100])
    zero_dim_report("Cubic dominant, V = 0.8, W = 0.5", lam=0.8, g=0.5,
                    betas=[1, 2, 5, 10, 20, 50, 100])

    print()
    print("  What a wrong coefficient would cost. Suppose you had derived only the")
    print("  sunset diagram and used coefficient 1/12 in one dimension:")
    print(f"    {'V':>5} {'W':>5} {'F2 correct (5/24)':>19} {'F2 wrong (1/12)':>17}"
          f" {'gap':>9}")
    for lam, g in [(0.5, 1.0), (0.8, 0.5), (1.0, 1.0)]:
        good = g / 8 - (5 / 24) * lam**2
        bad = g / 8 - (1 / 12) * lam**2
        print(f"    {lam:>5.2f} {g:>5.2f} {good:>19.6f} {bad:>17.6f} "
              f"{abs(good-bad):>9.6f}")
    print("  The gap is (1/8) V^2 G^3, i.e. exactly the spectacles diagram. The 1-D")
    print("  test cannot tell you how to split 5/24, but it does tell you that the")
    print("  total must be 5/24 -- which is already a strong constraint.")


# ======================================================================
# Exercise 5: 2-D exact test, sunset vs spectacles
# ======================================================================
def two_d_tensors(a, b, c, d):
    """
    S(phi) = (1/2) phi^T H phi + (a/6) phi_1^3 + (b/6) phi_2^3
                               + (c/24) phi_1^4 + (d/24) phi_2^4

    Third and fourth derivative tensors at phi* = 0. These are automatically
    fully symmetric: V[0,0,0] = a, V[1,1,1] = b, every mixed entry is 0.
    """
    V = np.zeros((2, 2, 2))
    V[0, 0, 0], V[1, 1, 1] = a, b
    W = np.zeros((2, 2, 2, 2))
    W[0, 0, 0, 0], W[1, 1, 1, 1] = c, d
    return V, W


def two_loop_pieces(H, V, W):
    """(figure-eight, sunset contraction, spectacles, tadpole u) for F2."""
    G = np.linalg.inv(H)
    fig8 = 0.125 * np.einsum("ijkl,ij,kl->", W, G, G)
    sun_contraction = np.einsum("ijk,lmn,il,jm,kn->", V, V, G, G, G)
    u = np.einsum("ijk,jk->i", V, G)
    return fig8, sun_contraction, u @ G @ u, u


def make_S(H, a, b, c, d):
    def S(x, y):
        return (0.5 * (H[0, 0] * x * x + 2 * H[0, 1] * x * y + H[1, 1] * y * y)
                + (a / 6) * x**3 + (b / 6) * y**3
                + (c / 24) * x**4 + (d / 24) * y**4)
    return S


def exact_2d(H, a, b, c, d, beta, L=14.0, want_mean=False):
    """
    -(1/beta) log Z in rescaled coordinates xi = sqrt(beta) phi, and optionally
    the exact mean <phi> = <xi>/sqrt(beta).
    """
    sb = math.sqrt(beta)

    def S_beta(xi, eta):
        return (0.5 * (H[0, 0] * xi * xi + 2 * H[0, 1] * xi * eta + H[1, 1] * eta * eta)
                + (a * xi**3 + b * eta**3) / (6 * sb)
                + (c * xi**4 + d * eta**4) / (24 * beta))

    I, _ = dblquad(lambda eta, xi: math.exp(-S_beta(xi, eta)),
                   -L, L, lambda xi: -L, lambda xi: L,
                   epsabs=1e-13, epsrel=1e-13)
    neglogZ = (2 / 2) * math.log(beta) / beta - math.log(I) / beta
    if not want_mean:
        return neglogZ
    m = []
    for coord in (0, 1):
        num, _ = dblquad(
            lambda eta, xi, coord=coord: (xi if coord == 0 else eta)
            * math.exp(-S_beta(xi, eta)),
            -L, L, lambda xi: -L, lambda xi: L,
            epsabs=1e-13, epsrel=1e-13)
        m.append(num / I / sb)
    return neglogZ, np.array(m)


def exercise_5():
    head(5, "Two-dimensional exact test: 1/12 sunset versus 1/8 spectacles")
    print("Model:  S(phi) = (1/2) phi^T H phi + (a/6) phi_1^3 + (b/6) phi_2^3")
    print("                                   + (c/24) phi_1^4 + (d/24) phi_2^4")
    print()
    print("This is the smallest model in which the sunset and spectacles diagrams")
    print("are different numbers. The exact answer comes from two-dimensional")
    print("adaptive quadrature at machine precision -- no Monte Carlo, so there is")
    print("no sampling error to hide behind. The cubic couplings a, b drive the")
    print("tadpole vector u_i = V_ijk G_jk, which is what the spectacles diagram")
    print("feeds on; a large ||u|| is what makes the two coefficients separable.")

    cases = [
        ("A  moderate cubic", np.array([[1.0, 0.5], [0.5, 1.0]]), (1.2, -0.9, 1.5, 1.2)),
        ("B  strong cubic, positive off-diagonal",
         np.array([[1.0, 0.6], [0.6, 1.1]]), (1.5, 1.0, 2.0, 1.5)),
    ]
    for label, H, (a, b, c, d) in cases:
        V, W = two_d_tensors(a, b, c, d)
        G = np.linalg.inv(H)
        fig8, sun_contraction, spect_contraction, u = two_loop_pieces(H, V, W)
        sunset = sun_contraction / 12
        spect = spect_contraction / 8
        F2_correct = fig8 - sunset - spect
        F2_naive = fig8 - (5 / 24) * sun_contraction

        # verify phi* = 0 is the global minimum, not just a local one
        S = make_S(H, a, b, c, d)
        best = min((minimize(lambda z: S(z[0], z[1]), x0).fun
                    for x0 in [(-6, -6), (-6, 6), (6, -6), (6, 6), (0.1, 0.1)]))
        print()
        print(SUB)
        print(f"  Case {label}")
        print(f"    H = [[{H[0,0]}, {H[0,1]}], [{H[1,0]}, {H[1,1]}]],  "
              f"det H = {np.linalg.det(H):.6f}")
        print(f"    a = {a}, b = {b}, c = {c}, d = {d}")
        print(f"    global-minimum check: lowest S found = {best:+.6e} "
              f"(must be 0, i.e. phi* = 0 is global)")
        print(f"    G = [[{G[0,0]:.4f}, {G[0,1]:.4f}], [{G[1,0]:.4f}, {G[1,1]:.4f}]]")
        print(f"    tadpole u_i = V_ijk G_jk = ({u[0]:+.4f}, {u[1]:+.4f}), "
              f"||u|| = {np.linalg.norm(u):.4f}")
        print()
        print(f"    figure-eight      +(1/8) W:GG        = {fig8:+.6f}")
        print(f"    sunset            -(1/12) VV:GGG     = {-sunset:+.6f}"
              f"   (contraction = {sun_contraction:+.6f})")
        print(f"    spectacles        -(1/8) uGu         = {-spect:+.6f}"
              f"   (contraction = {spect_contraction:+.6f})")
        print(f"    F2 correct                           = {F2_correct:+.6f}")
        print(f"    F2 naive (1-D formula, 5/24 on VV:GGG) = {F2_naive:+.6f}")
        print(f"    gap between the two                  = "
              f"{abs(F2_correct-F2_naive):.6f}")
        print(f"    (the gap is exactly (1/8)[uGu - VV:GGG] = "
              f"{(spect_contraction - sun_contraction)/8:+.6f})")
        print()
        print(f"    {'beta':>6} {'1-loop':>11} {'2L naive':>11} {'2L correct':>11}"
              f" {'exact':>11} {'err 1L':>9} {'errNv':>9} {'errCo':>9}"
              f" {'errNv*b^2':>10} {'errCo*b^3':>10}")
        for beta in (5, 10, 20, 50, 100, 200):
            exact = exact_2d(H, a, b, c, d, beta)
            one = (0.5 * math.log(np.linalg.det(H))
                   - math.log(2 * math.pi / beta)) / beta
            tn, tc = one + F2_naive / beta**2, one + F2_correct / beta**2
            e1, en, ec = abs(one - exact), abs(tn - exact), abs(tc - exact)
            print(f"    {beta:>6.0f} {one:>11.6f} {tn:>11.6f} {tc:>11.6f}"
                  f" {exact:>11.6f} {e1:>9.2e} {en:>9.2e} {ec:>9.2e}"
                  f" {en*beta**2:>10.5f} {ec*beta**3:>10.4f}")
        print("    Read the last two columns, not the raw errors. The naive error")
        print("    times beta^2 settles on the gap above: the naive formula gets the")
        print("    beta^-2 coefficient wrong, so its error never falls below")
        print("    O(beta^-2). The correct error times beta^3 settles on a constant,")
        print("    which is the signature of a correct beta^-2 coefficient and an")
        print("    O(beta^-3) truncation error.")

    print()
    print(SUB)
    print("  Bonus: the tadpole shift of the posterior mean.")
    print("  Diagrams do not only correct the evidence. The saddle puts the mean at")
    print("  phi* exactly; the first correction is one power of 1/beta down:")
    print("      <phi_i> - phi*_i = -(1/(2 beta)) G_ij u_j + O(beta^-2).")
    print("  Derivation: <xi_i> = -<xi_i X3>_0 = -(1/(6 sqrt(beta))) * 3 G_ij u_j,")
    print("  then <phi> = <xi>/sqrt(beta).")
    H = np.array([[1.0, 0.5], [0.5, 1.0]])
    a, b, c, d = 1.2, -0.9, 1.5, 1.2
    V, _ = two_d_tensors(a, b, c, d)
    G = np.linalg.inv(H)
    u = np.einsum("ijk,jk->i", V, G)
    pred_coef = -(G @ u) / 2
    print()
    print(f"  u = ({u[0]:+.4f}, {u[1]:+.4f}),  -(1/2) G u = "
          f"({pred_coef[0]:+.4f}, {pred_coef[1]:+.4f})")
    print(f"    {'beta':>6} {'pred phi_1':>12} {'exact phi_1':>12} {'ratio':>8}"
          f"   {'pred phi_2':>12} {'exact phi_2':>12} {'ratio':>8}")
    for beta in (10, 20, 50, 100, 200):
        _, m = exact_2d(H, a, b, c, d, beta, want_mean=True)
        p = pred_coef / beta
        print(f"    {beta:>6.0f} {p[0]:>12.6f} {m[0]:>12.6f} {m[0]/p[0]:>8.4f}"
              f"   {p[1]:>12.6f} {m[1]:>12.6f} {m[1]/p[1]:>8.4f}")
    print("  Both ratios approach 1 as beta grows, confirming the coefficient.")


# ======================================================================
# Exercise 6: Poisson-Gaussian
# ======================================================================
def poisson_row(o):
    """All quantities for the Poisson-Gaussian model at observed count o."""
    s_star = brentq(lambda s: math.exp(s) + s - o, -80, 25, xtol=1e-15, rtol=8.9e-16)
    e = math.exp(s_star)
    H = e + 1.0
    V = W = e
    G = 1.0 / H

    def A(s):
        return (math.exp(s) - s * o + 0.5 * s * s
                + 0.5 * math.log(2 * math.pi) + math.lgamma(o + 1))

    tree = A(s_star)
    laplace_term = 0.5 * math.log(H) - 0.5 * math.log(2 * math.pi)
    one = tree + laplace_term
    fig8 = 0.125 * W * G**2
    cub = (5 / 24) * V**2 * G**3
    two = one + fig8 - cub
    width = 25.0 / math.sqrt(H)
    Zval, _ = quad(lambda s: math.exp(-A(s)), s_star - width, s_star + width,
                   epsabs=1e-14, epsrel=1e-13, limit=500)
    exact = -math.log(Zval)
    return dict(o=o, s_star=s_star, e=e, H=H, G=G, V=V, W=W, tree=tree,
                laplace_term=laplace_term, one=one, fig8=fig8, cub=cub,
                F2=fig8 - cub, two=two, exact=exact)


def exercise_6():
    head(6, "Poisson-Gaussian model: every intermediate number, then Table 1")
    print("Model:  s ~ N(0, 1),  o | s ~ Poisson(e^s).")
    print("Action (negative log joint, so that Z = p(o) exactly at beta = 1):")
    print("   A(s) = e^s - s o + s^2/2 + (1/2) log(2 pi) + log(o!)")
    print("Saddle:  A'(s) = e^s - o + s = 0,  i.e.  e^s* + s* = o.")
    print("Derivatives at the saddle: H = e^s* + 1,  V = e^s*,  W = e^s*.")
    print()
    print("GOTCHA 1.  The saddle equation is  e^s + s = o,  NOT  s e^s = o.")
    print("   At o = 1 the solution is exactly s* = 0, because e^0 + 0 = 1.")
    print("   It is not 0.5671 (the Omega constant, which solves s e^s = 1).")
    print("GOTCHA 2.  Here beta = 1 and the 'field' is the latent state s, not a")
    print("   variational parameter. The loop counting is identical; what plays the")
    print("   role of a large beta is the curvature H at the saddle. See the")
    print("   primer, 'Why the fourth derivative is the last one needed'.")

    r = poisson_row(1)
    print()
    print("  o = 1, in full:")
    print(f"    s*                                = {r['s_star']:.15f}")
    print(f"    e^s*                              = {r['e']:.10f}")
    print(f"    H = e^s* + 1                      = {r['H']:.10f}")
    print(f"    G = 1/H                           = {r['G']:.10f}")
    print(f"    V = W = e^s*                      = {r['V']:.10f}")
    print(f"    tree  = A(s*)                     = {r['tree']:.6f}")
    print(f"    (1/2) log H - (1/2) log(2 pi)     = {r['laplace_term']:+.6f}")
    print(f"    1-loop                            = {r['one']:.6f}")
    print(f"    figure-eight  +(1/8) W G^2        = {r['fig8']:+.8f}")
    print(f"    sunset+spect  -(5/24) V^2 G^3     = {-r['cub']:+.8f}")
    print(f"    F2                                = {r['F2']:+.8f}")
    print(f"    2-loop                            = {r['two']:.6f}")
    print(f"    exact -log p(o) by quadrature     = {r['exact']:.6f}")
    print(f"    one-loop error                    = {abs(r['one']-r['exact']):.6f}")
    print(f"    two-loop error                    = {abs(r['two']-r['exact']):.6f}")
    print()
    print("  Note the near-cancellation: figure-eight +0.03125 against")
    print("  sunset+spectacles -0.02604 leaves only +0.00521. The two-loop term is")
    print("  a difference of two comparable diagrams, which is why getting their")
    print("  coefficients right matters more than their individual sizes suggest.")

    print()
    print("  Full sweep (this is Table 1 of the paper):")
    print(f"    {'o':>3} {'s*':>9} {'H':>9} {'tree':>9} {'1-loop':>9} {'2-loop':>9}"
          f" {'exact':>9} {'err 1L':>9} {'err 2L':>10}")
    for o in (0, 1, 2, 3, 5, 10, 20, 50):
        r = poisson_row(o)
        print(f"    {o:>3} {r['s_star']:>9.4f} {r['H']:>9.4f} {r['tree']:>9.4f}"
              f" {r['one']:>9.4f} {r['two']:>9.4f} {r['exact']:>9.4f}"
              f" {abs(r['one']-r['exact']):>9.5f} {abs(r['two']-r['exact']):>10.7f}")
    print()
    print("  GOTCHA 3.  o = 2 is not a counterexample. There the one-loop error")
    print("  happens to pass through zero because F2 changes sign, so the ratio")
    print("  err2L/err1L is meaningless at that single point. Look at the neighbours.")


# ======================================================================
# Exercise 7: categorical with exact cumulant tensors
# ======================================================================
class CumulantTerm:
    """
    One term of a derivative of the categorical log-partition function.

    A term is  coef * prod(delta-groups) * prod(q_a for a in qs), where a
    delta-group is a set of index labels constrained to be equal. Everything is
    a polynomial in the q_i, so this representation is exact.
    """

    __slots__ = ("coef", "groups", "qs")

    def __init__(self, coef, groups, qs):
        self.coef = coef
        self.groups = tuple(frozenset(g) for g in groups)
        self.qs = tuple(qs)

    def differentiate(self, new):
        """d/dphi_new, using  dq_a/dphi_l = q_a (delta_{al} - q_l)."""
        out = []
        for pos, a in enumerate(self.qs):
            # branch 1: delta_{a,new}, q-factors unchanged
            out.append(CumulantTerm(self.coef, self.groups + ((a, new),), self.qs))
            # branch 2: -q_new appended
            out.append(CumulantTerm(-self.coef, self.groups, self.qs + (new,)))
        return out

    def evaluate(self, idx_map, q):
        """idx_map: label -> concrete index. q: array of q_i values."""
        for g in self.groups:
            vals = {idx_map[lab] for lab in g}
            if len(vals) > 1:
                return 0.0
        val = self.coef
        for a in self.qs:
            val *= q[idx_map[a]]
        return val


def categorical_tensors_exact(q):
    """
    Exact H, V, W for the gauge-fixed categorical.

    Coordinates: phi in R^(K-1) with q_i = e^{phi_i} / (1 + sum_j e^{phi_j}) for
    i = 1..K-1, and q_K = 1 - sum q_i. The log-partition function is
    A(phi) = log(1 + sum_j e^{phi_j}), whose derivatives are the categorical
    cumulant tensors. Starting from
        H_ij = q_i delta_ij - q_i q_j
    we differentiate symbolically using dq_a/dphi_l = q_a(delta_al - q_l).
    """
    q = np.asarray(q, dtype=float)
    n = len(q) - 1
    qs = q[:n]

    base = [CumulantTerm(+1.0, [("i", "j")], ["i"]),
            CumulantTerm(-1.0, [], ["i", "j"])]

    def grow(terms, new):
        out = []
        for t in terms:
            out.extend(t.differentiate(new))
        return out

    v_terms = grow(base, "k")
    w_terms = grow(v_terms, "l")

    def build(terms, labels):
        shape = (n,) * len(labels)
        T = np.zeros(shape)
        for idx in itertools.product(range(n), repeat=len(labels)):
            m = dict(zip(labels, idx))
            T[idx] = sum(t.evaluate(m, qs) for t in terms)
        return T

    H = build(base, ["i", "j"])
    V = build(v_terms, ["i", "j", "k"])
    W = build(w_terms, ["i", "j", "k", "l"])
    return H, V, W


def categorical_tensors_fd(q, h=2e-2):
    """Finite-difference tensors, as an independent check on the exact ones."""
    n = len(q) - 1
    p = np.asarray(q[:n], dtype=float)
    phi = np.log(p / q[-1])

    def A(x):
        return np.log1p(np.sum(np.exp(x)))

    def d(idx):
        order = len(idx)
        total = 0.0
        for signs in itertools.product((+1, -1), repeat=order):
            x = phi.astype(float).copy()
            for s, i in zip(signs, idx):
                x[i] += s * h
            total += math.prod(signs) * A(x)
        return total / (2 * h) ** order

    H = np.array([[d((i, j)) for j in range(n)] for i in range(n)])
    V = np.zeros((n,) * 3)
    for idx in itertools.product(range(n), repeat=3):
        V[idx] = d(idx)
    W = np.zeros((n,) * 4)
    for idx in itertools.product(range(n), repeat=4):
        W[idx] = d(idx)
    return H, V, W


def categorical_F2(q):
    H, V, W = categorical_tensors_exact(q)
    G = np.linalg.inv(H)
    fig8 = 0.125 * np.einsum("ijkl,ij,kl->", W, G, G)
    sunset = (1 / 12) * np.einsum("ijk,lmn,il,jm,kn->", V, V, G, G, G)
    u = np.einsum("ijk,jk->i", V, G)
    spect = 0.125 * u @ G @ u
    return H, V, W, G, u, fig8, sunset, spect, fig8 - sunset - spect


def exercise_7():
    head(7, "Three-state categorical: exact cumulant tensors, and the boundary")
    q = (0.5, 0.3, 0.2)
    print(f"Belief q* = {q} on the 2-simplex.")
    print("Coordinates: gauge-fixed logits phi_1, phi_2 with")
    print("   q_i = e^{phi_i} / (1 + e^{phi_1} + e^{phi_2}),  q_3 = 1 - q_1 - q_2.")
    print("The log-partition function A(phi) = log(1 + e^{phi_1} + e^{phi_2})")
    print("generates the cumulants: dA/dphi_i = q_i, and the Hessian of A is the")
    print("Fisher information H_ij = q_i delta_ij - q_i q_j.")
    print()
    print("GOTCHA.  Gauge fixing is not optional. Without it the three q_i are")
    print("linearly dependent, the 3x3 Fisher matrix is singular, det H = 0, and")
    print("the one-loop term is -infinity. Dropping one coordinate is what makes")
    print("the propagator exist.")

    H, V, W, G, u, fig8, sunset, spect, F2 = categorical_F2(q)
    n = len(q) - 1
    qa = np.array(q[:n])
    H_closed = np.diag(qa) - np.outer(qa, qa)
    Hf, Vf, Wf = categorical_tensors_fd(q)

    print()
    print("  Hessian, closed form q_i delta_ij - q_i q_j:")
    print("   ", np.array2string(H_closed, precision=6, prefix="    "))
    print(f"  agrees with the symbolic construction to {np.abs(H-H_closed).max():.1e}")
    print(f"  agrees with finite differences of A to    {np.abs(H-Hf).max():.1e}")
    print(f"  det H = {np.linalg.det(H):.8f}")
    print(f"  one-loop  F1 = (1/2) log det H = {0.5*math.log(np.linalg.det(H)):+.6f}")
    print("  Propagator G = H^{-1}:")
    print("   ", np.array2string(G, precision=4, prefix="    "))
    print()
    print("  This F1 is the Fisher-Rao log-volume, NOT the Shannon entropy.")
    sh = -sum(x * math.log(x) for x in q)
    print(f"    (1/2) log det H at q* = {0.5*math.log(np.linalg.det(H)):+.6f}")
    print(f"    Shannon entropy at q* = {sh:+.6f}")
    print("  They are both maximal at the uniform belief but behave oppositely near")
    print("  a corner: the log-volume diverges to -infinity, the entropy stays finite.")
    unif = (1 / 3, 1 / 3, 1 / 3)
    Hu, *_ = categorical_tensors_exact(unif)
    print(f"    at uniform: (1/2) log det H = "
          f"{0.5*math.log(np.linalg.det(Hu)):+.6f},  entropy = "
          f"{-sum(x*math.log(x) for x in unif):+.6f}")

    print()
    print("  Exact cumulant tensors. Diagonal entries match the closed forms")
    print("  V_iii = q_i(1-q_i)(1-2q_i) and W_iiii = q_i(1-q_i)[1-6q_i(1-q_i)]:")
    for i in range(n):
        vc = q[i] * (1 - q[i]) * (1 - 2 * q[i])
        wc = q[i] * (1 - q[i]) * (1 - 6 * q[i] * (1 - q[i]))
        print(f"    V_{i+1}{i+1}{i+1} closed {vc:+.8f}  symbolic {V[i,i,i]:+.8f}"
              f"  fd {Vf[i,i,i]:+.6f}")
        print(f"    W_{i+1}{i+1}{i+1}{i+1} closed {wc:+.8f}  symbolic {W[i,i,i,i]:+.8f}"
              f"  fd {Wf[i,i,i,i]:+.6f}")
    print()
    print("  The off-diagonal entries are not zero, and they are not negligible:")
    print(f"    V_112 = {V[0,0,1]:+.8f}   V_122 = {V[0,1,1]:+.8f}")
    print(f"    W_1112 = {W[0,0,0,1]:+.8f}  W_1122 = {W[0,0,1,1]:+.8f}")

    fig8_diag = 0.125 * sum(W[i, i, i, i] * G[i, i] ** 2 for i in range(n))
    print()
    print("  GOTCHA.  Keeping only the diagonal of W is a shortcut, not the answer.")
    print(f"    diagonal-only figure-eight   = {fig8_diag:+.6f}"
          "   <- the -1.24 quoted in the paper")
    print(f"    full four-index figure-eight = {fig8:+.6f}")
    print(f"    ratio                        = {fig8/fig8_diag:.3f}")
    print()
    print(f"    tadpole u = ({u[0]:+.4f}, {u[1]:+.4f}),  ||u|| = "
          f"{np.linalg.norm(u):.4f}")
    print(f"    figure-eight  +(1/8) W:GG    = {fig8:+.6f}")
    print(f"    sunset        -(1/12) VV:GGG = {-sunset:+.6f}")
    print(f"    spectacles    -(1/8) uGu     = {-spect:+.6f}")
    print(f"    full F2                      = {F2:+.6f}")
    print(f"    one-loop F1                  = "
          f"{0.5*math.log(np.linalg.det(H)):+.6f}")
    print(f"    |F2 / F1|                    = "
          f"{abs(F2/(0.5*math.log(np.linalg.det(H)))):.3f}")
    print("  The two-loop total is the same order as the one-loop term. That is the")
    print("  quantitative version of 'variational Laplace cannot be trusted for")
    print("  concentrated discrete beliefs'.")

    print()
    print("  Approach to the simplex boundary: q* = (1 - 2 eps, eps, eps).")
    print(f"    {'eps':>7} {'det H':>11} {'F1':>10} {'F2':>13} {'|F2/F1|':>10}"
          f" {'||u||':>10}")
    for eps in (0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005):
        qq = (1 - 2 * eps, eps, eps)
        Hh, _, _, Gg, uu, f8, sn, sp, f2 = categorical_F2(qq)
        f1 = 0.5 * math.log(np.linalg.det(Hh))
        print(f"    {eps:>7.3f} {np.linalg.det(Hh):>11.3e} {f1:>10.4f} {f2:>13.4f}"
              f" {abs(f2/f1):>10.3f} {np.linalg.norm(uu):>10.3f}")
    print("  |F2/F1| grows without bound: the expansion is not merely inaccurate")
    print("  near a corner, it is not an expansion at all. The primer section")
    print("  'Discrete beliefs: the simplex' says what to do instead.")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    exercise_6()
    exercise_7()
    print()
    print(RULE)
    print("All exercises complete.")
    print(RULE)
