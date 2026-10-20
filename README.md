# AIM Research Lab at IWAI 2026

Materials from the **7th International Workshop on Active Inference**,
Madrid, 14–16 October 2026.

Adding here material on three
contributions: two posters, and a primer on the methods introduced in a - heavy - paper, each in
its own directory. Items marked *link to follow* are not here yet; they are added
as each piece is cleared for release.

```
rebundling/            The Rebundling Hypothesis — poster
return-is-not-enough/  Return Is Not Enough — poster
two-loops/             Variational Laplace at Two Loops
  primer.pdf             the long-form companion to the paper, 54 pp
  code/                  five scripts that reproduce every number in it
```

Posters are A0 portrait PDFs, as printed. The code is plain Python whose only
dependencies are `numpy` and `scipy`.

---

## The Rebundling Hypothesis

### What LLMs unbundle, and what active inference could restore

**Stefano Goria** · poster

Linguistic fluency, long treated as a signal of intelligence, is separable from
understanding. Read through Tomasello's four-level hierarchy of agency, an LLM is
**inverted**: it produces rational and normative *discourse* while the embodied,
goal-directed substrate biology had to build first is absent. The claim is that the
training objective determines the bundle — so an agent trained and deployed under a
**free-energy objective**, with homeostatic priors and embodied sensorimotor coupling,
should recover it. The hypothesis is testable by ablation.

- [Poster (A0, PDF)](rebundling/poster.pdf)
- Extended abstract — *link to follow*
- Code — *link to follow*

---

## Return Is Not Enough

### Instruments for dissecting what each term of the expected free energy actually does

**Stefano Goria** · poster

Return is a scalar, and a scalar collapses the trajectory. Where a reward admits many
equally-scoring solutions, return cannot register what the **epistemic term** of the
expected free energy changed, and a line-up of agents differing only in that term
reports that the term costs points and should be dropped. Four other instruments
disagree. Path signatures separate the gaits at **matched return**; mechanics price the
difference in effort; held-out model error and the bonus term dissociate; and tilting
gravity pushes the agent off the distribution it trained on, where the deficit reverses
sign. Every instrument that fires on a degenerate reward goes quiet, or reverses, on a
non-degenerate one, which makes **reward degeneracy the gating condition**. The poster
stops short of claiming the mechanism.

This is the [rebundling](#the-rebundling-hypothesis) protocol — drop a component, lose
a signature — run on one term of the expected free energy, in continuous control.

- [Poster (A0, PDF)](return-is-not-enough/poster.pdf)
- Extended abstract — *link to follow*
- Code — *link to follow*

---

## Variational Laplace at Two Loops

### Feynman diagrams for the variational free energy

**Stefano Goria** · full paper and oral presentation

Variational Laplace returns a log evidence and no error estimate, and the log evidence
is what model comparison rests on. The number it returns is the **first two terms of an
asymptotic series**. The third term is a sum of three Feynman diagrams — *figure-eight*,
*sunset* and *spectacles* — with a closed form in the third and fourth derivatives of
the action at the saddle. Those are two further derivatives of a function the pipeline
has already differentiated twice, at a point it has already located, and their
magnitude estimates the error of the two-term truncation. A pipeline that computes them
reports an evidence together with a bar on that evidence.

- [**Primer** (PDF, 54 pp)](two-loops/primer.pdf) — the long-form companion to the
  paper, and an expansion of the talk, written for a reader who can compute a free
  energy and has never drawn a Feynman diagram. One worked example carries the
  derivation, every symbol is defined where it first appears, each part closes with an
  annotated reading list, and every number is reproduced by one of the scripts below.
- [**Code**](two-loops/code) — see the table.
- Paper — *link to follow*
- Slides — *link to follow*

### What is in `two-loops/code`

Deterministic, with exact quadrature in place of sampling wherever that is
possible, so that a disagreement between your algebra and the output is an
algebra error rather than Monte Carlo noise. With
[uv](https://docs.astral.sh/uv/):

```sh
cd two-loops/code
uv sync
uv run python primer_examples.py
```

| Script | What it computes | Runtime |
| --- | --- | --- |
| `primer_examples.py` | The primer's seven exercises. Wick's theorem to 1e-14 by Gauss–Hermite quadrature; the 15 pairings enumerated and split 6 + 9; the figure-eight coefficient 1/8, and the 5/24 that the two cubic diagrams collapse to in one dimension; a two-dimensional test that separates the 1/12 sunset from the 1/8 spectacles against exact quadrature; the Poisson–Gaussian numbers; the categorical cumulant tensors and the divergence at the simplex boundary. | ~2 s |
| `two_loop_validation.py` | Table 1 of the paper. Tree, one-loop and two-loop estimates of −log p(o) for a Poisson likelihood with a Gaussian prior on the log rate, against Gauss–Kronrod quadrature. | seconds |
| `logistic_exact_quadrature.py` | Table 2 with exact ground truth, by whitened tensor-product Gauss–Hermite quadrature, plus the per-diagram breakdown that settles 1/12 against 1/8 on a model not built for the purpose. | ~10 s |
| `logistic_regression_validation.py` | The same table by importance sampling from the Laplace posterior. Kept because the primer uses it to make a point: the sampler's standard error is the size of the effect being measured, so it is not a usable ground truth here. | under a minute |
| `error_bar_calibration.py` | The calibration behind the claim that \|F₂\| is an error bar. Across a sweep spanning more than an order of magnitude in error size, the ratio of the true one-loop error to \|F₂\| has median 0.98 and lies in [0.82, 1.00] in 19 of 20 cases. | seconds |

Appendix E of the primer maps each of its sections to the script that produces
its numbers.

---

## Also at IWAI 2026

*To be added.*

---

## Contact

Stefano Goria · AIM Research Lab · <stefano@aimresearchlab.com>
