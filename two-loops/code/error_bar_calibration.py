"""Is |F2| a usable error bar on the one-loop (variational Laplace) answer?

RESULT, up front.  Across sg in [0.5, 3] and y in [0, 50] -- more than an order
of magnitude in error size -- the ratio of the true one-loop error to |F2| has
median 0.98 and lies in [0.82, 1.00] in 19 of 20 cases.  The one outlier is a
point where the one-loop error happens to pass through zero, so |F2|
OVERestimates it, which is the safe direction.

That is the calibration quoted in section 18 of the primer.  Note what it does
not need: no ground truth is required to compute |F2|, only two more
derivatives of an action already differentiated twice.  The exact column below
exists so that the calibration can be checked, not so that it can be used.

Model: the paper's Poisson-Gaussian evidence,

    s ~ N(mu, sg^2),    y ~ Poisson(e^{s+d}),    want -log q(y|d),

with mu = d = 0 throughout.  The three estimates are

    1 loop    A(s*) + 1/2 log(H*/2pi)                   expand at the mode
    2 loops   + F2,  F2 = 1/8 W G^2 - 5/24 V^2 G^3      the next term
    exact     quadrature

Run:  uv run python error_bar_calibration.py
"""
import numpy as np
from scipy.special import gammaln, logsumexp

MU = 0.0


def exact_neglogq(y, d, sg, width=13.0, npts=40001):
    """-log q(y|d) by quadrature on a grid wide enough to be exact to ~1e-12."""
    s = np.linspace(MU - width * sg, MU + width * sg, npts)
    lw = -0.5 * ((s - MU) / sg) ** 2 - 0.5 * np.log(2 * np.pi * sg**2) \
        + np.log(s[1] - s[0])
    ll = y * (s + d) - np.exp(s + d) - gammaln(y + 1.0)
    return -logsumexp(ll + lw)


def loops(y, d, sg):
    """(one loop, two loops, |F2|) for -log q(y|d).

    Newton on e^{s+d} - y + (s - mu)/sg^2 = 0 gives the saddle s*; then
    H = A''(s*) = e^{s*+d} + 1/sg^2, and V = W = e^{s*+d}, so that
    G = 1/H and F2 = 1/8 W G^2 - 5/24 V^2 G^3 (the one-dimensional collapse
    of the sunset and spectacles diagrams; see section 12 of the primer).
    """
    s = max(np.log(y + 1.0) - d, MU)
    for _ in range(200):
        e = np.exp(s + d)
        s = s - (e - y + (s - MU) / sg**2) / (e + 1.0 / sg**2)
    e = np.exp(s + d)
    A = e - y * (s + d) + gammaln(y + 1.0) + (s - MU) ** 2 / (2 * sg**2) \
        + 0.5 * np.log(2 * np.pi * sg**2)
    H = e + 1.0 / sg**2
    G = 1.0 / H
    one = A + 0.5 * np.log(H) - 0.5 * np.log(2 * np.pi)
    F2 = (1.0 / 8.0) * e * G**2 - (5.0 / 24.0) * e**2 * G**3
    return one, one + F2, abs(F2)


if __name__ == "__main__":
    print("Does |F2| predict the error of the one-loop answer?\n")
    print(f"{'sg':>5} {'y':>4} {'|F2|':>11} {'actual |1L err|':>17} {'ratio':>9}")
    print("-" * 52)
    rat = []
    for sg in (0.5, 1.0, 2.0, 3.0):
        for y in (0, 1, 3, 10, 50):
            ex = exact_neglogq(y, 0.0, sg)
            one, _, f2 = loops(y, 0.0, sg)
            a = abs(one - ex)
            rat.append(a / f2)
            print(f"{sg:>5} {y:>4} {f2:>11.3e} {a:>17.3e} {a/f2:>9.3f}")
    rat = np.array(rat)
    within = int(np.sum((rat >= 0.82) & (rat <= 1.0)))
    print(f"\n  ratio actual/|F2|:  min {rat.min():.3f}  median {np.median(rat):.3f}"
          f"  max {rat.max():.3f}")
    print(f"  in [0.82, 1.00] in {within}/{len(rat)} cases")
    print("  A ratio near 1 across orders of magnitude means |F2| is the error bar.")
