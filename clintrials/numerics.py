import warnings

import numpy as np
from scipy.special import logsumexp


def integrate_posterior_1d(
    logpost,
    f,
    lo,
    hi,
    *,
    method="grid",
    n_points=2001,
    adaptive_limits=True,
    edge_frac=0.02,
    tail_mass_tol=1e-3,
    expand_factor=1.0,
    max_expansions=6,
    warn_on_max=True,
    return_diagnostics=False,
):
    """Integrate a 1D posterior density with optional adaptive bounds.

    Parameters
    ----------
    logpost : callable
        Log posterior density function evaluated on a numpy array.
    f : callable
        Function of the parameter to integrate with respect to the
        posterior density.
    lo, hi : float
        Initial lower and upper bounds of integration.
    method : str, optional
        Only ``"grid"`` is implemented. A linspace grid of ``n_points``
        is used.
    n_points : int, optional
        Number of grid points in the integration domain.
    adaptive_limits : bool, optional
        If ``True`` iteratively expand the integration limits when tail
        mass near the boundary exceeds ``tail_mass_tol``.
    edge_frac : float, optional
        Fraction of the width considered to be the edge for the tail mass
        heuristic.
    tail_mass_tol : float, optional
        Maximum tolerated posterior mass in the edges before expansion is
        triggered.
    expand_factor : float, optional
        Fractional increase of the current width when expanding limits.
    max_expansions : int, optional
        Maximum number of expansions before giving up.
    warn_on_max : bool, optional
        Issue a warning when expansion hits ``max_expansions``.
    return_diagnostics : bool, optional
        If ``True`` return a dict with diagnostics in addition to the
        integral value.
    """

    expansions = 0
    while True:
        xs = np.linspace(lo, hi, n_points)
        lp = logpost(xs)
        w = np.exp(lp - logsumexp(lp))
        val = np.sum(w * f(xs))

        width = hi - lo
        left = xs <= lo + edge_frac * width
        right = xs >= hi - edge_frac * width
        tail_mass = w[left].sum() + w[right].sum()
        max_at_edge = np.argmax(lp) == 0 or np.argmax(lp) == len(xs) - 1

        if not adaptive_limits or ((tail_mass < tail_mass_tol) and (not max_at_edge)):
            diag = {
                "expansions": expansions,
                "tail_mass": float(tail_mass),
                "max_at_edge": bool(max_at_edge),
            }
            return (val, diag) if return_diagnostics else val

        if expansions >= max_expansions:
            if warn_on_max:
                warnings.warn(
                    "Posterior mass remains near bounds after max expansions; results may be biased.",
                    RuntimeWarning,
                )
            diag = {
                "expansions": expansions,
                "tail_mass": float(tail_mass),
                "max_at_edge": bool(max_at_edge),
                "hit_cap": True,
            }
            return (val, diag) if return_diagnostics else val

        lo -= expand_factor * width
        hi += expand_factor * width
        expansions += 1
