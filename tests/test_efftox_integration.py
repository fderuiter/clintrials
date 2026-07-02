import logging
import numpy as np
from scipy.stats import norm
from clintrials.dosefinding.efftox import (
    EffTox,
    LpNormCurve,
    efftox_get_posterior_probs,
)


def test_adaptive_integration_extreme_case(caplog):
    # Prior: N(0, 1) for all parameters
    priors = [norm(0, 1) for _ in range(6)]
    scaled_doses = [-1.0, 0.0, 1.0]

    # 20 patients at dose 3, all had Tox and no Eff
    cases = [(3, 1, 0)] * 20
    tox_cutoff = 0.3
    eff_cutoff = 0.5

    # Run with small max_iter and large mass_threshold to force a warning or expansion
    # We use a smaller n for speed in tests
    with caplog.at_level(logging.WARNING):
        probs, pds = efftox_get_posterior_probs(
            cases,
            priors,
            scaled_doses,
            tox_cutoff,
            eff_cutoff,
            n=10**4,
            max_iter=1,
            mass_threshold=0.999999,
        )

    # Since we set max_iter=1, it should warn if it doesn't meet the threshold
    # But for N(0,1) prior and epsilon=1e-6, the initial limits are [-4.75, 4.75]
    # Posterior mean was around 1.4-1.5, SD will be smaller.
    # Let's check if expansion actually happens when we allow more iterations

    probs2, pds2 = efftox_get_posterior_probs(
        cases,
        priors,
        scaled_doses,
        tox_cutoff,
        eff_cutoff,
        n=10**4,
        max_iter=5,
        mass_threshold=0.999999,
    )

    # Compare limits
    [(p.ppf(1e-6), p.ppf(1 - 1e-6)) for p in priors]
    # We can't easily get the final limits from pds without modifying the return
    # but we can check if results differ significantly if the first one was truncated.
    # In my repro, it wasn't truncated much by [-4.75, 4.75].

    # Let's try a REALLY extreme case.
    priors_extreme = [norm(10, 0.1) for _ in range(6)]
    # Initial limits will be around [9.5, 10.5]
    # But if data suggests parameters should be at 0
    cases_extreme = [(1, 0, 1)] * 50  # Dose 1 is scaled dose -1.0.
    # If parameters are 10, prob tox is high. If 0, it's lower.

    # This should definitely trigger expansion if we start far away.
    probs_exp, pds_exp = efftox_get_posterior_probs(
        cases_extreme,
        priors_extreme,
        scaled_doses,
        tox_cutoff,
        eff_cutoff,
        n=10**4,
        max_iter=5,
        mass_threshold=0.9999,
    )

    # Check if we can find evidence of expansion.
    # The final sample points in pds_exp should be outside the original prior-based limits
    orig_low = norm(10, 0.1).ppf(1e-6)
    orig_high = norm(10, 0.1).ppf(1 - 1e-6)

    assert np.any(pds_exp._samp < orig_low) or np.any(pds_exp._samp > orig_high)


def test_efftox_class_propagation():
    real_doses = [1, 2, 3]
    priors = [norm(0, 1) for _ in range(6)]
    metric = LpNormCurve(0.4, 0.7, 0.5, 0.4)

    # Test that we can pass the new parameters to EffTox
    trial = EffTox(
        real_doses,
        priors,
        0.3,
        0.5,
        0.9,
        0.9,
        metric,
        30,
        k_sd=8.0,
        max_iter=5,
        mass_threshold=0.9999999,
    )

    assert trial.k_sd == 8.0
    assert trial.max_iter == 5
    assert trial.mass_threshold == 0.9999999

    # Update and check if it uses them (hard to check directly, but check no crash)
    trial.update([(1, 0, 1)])
    assert len(trial.prob_tox) == 3


def test_boundary_mass_warning(caplog):
    # Create a situation where it's impossible to cover the mass (e.g. max_iter=1 and very tight threshold)
    priors = [norm(0, 1) for _ in range(6)]
    scaled_doses = [0.0]
    cases = [(1, 1, 1)]

    with caplog.at_level(logging.WARNING):
        efftox_get_posterior_probs(
            cases,
            priors,
            scaled_doses,
            0.3,
            0.5,
            n=1000,
            max_iter=1,
            mass_threshold=1.0,  # threshold of 1.0 is impossible
        )

    assert "Monte Carlo integration limits did not cover mass threshold" in caplog.text
