import pytest

def test_crm_example():
    """
    Tests the CRM example from the README.md file.
    """
    from clintrials.dosefinding.crm import CRM

    # Define the prior probabilities of toxicity
    prior_tox_probs = [0.025, 0.05, 0.1, 0.25]
    # Define the target toxicity rate
    tox_target = 0.35
    # Define the starting dose
    first_dose = 3
    # Define the maximum number of patients
    trial_size = 30

    # Create a CRM trial object
    trial = CRM(prior_tox_probs, tox_target, first_dose, trial_size)

    # Get the next recommended dose
    next_dose = trial.next_dose()
    assert next_dose == 3

    # Update the trial with new patient data
    trial.update([(3, 0), (3, 0), (3, 0)])
    next_dose = trial.next_dose()
    assert next_dose == 4

def test_efftox_example():
    """
    Tests the EffTox example from the README.md file.
    """
    from clintrials.dosefinding.efftox import EffTox, LpNormCurve
    from scipy.stats import norm

    # Define the real dose levels
    real_doses = [1.0, 2.0, 4.0, 8.0]

    # Define priors for the model parameters
    theta_priors = [
        norm(-1.386, 1.732),  # mu_T
        norm(0, 1.732),       # beta_T
        norm(-1.386, 1.732),  # mu_E
        norm(0, 1.732),       # beta1_E
        norm(0, 1.732),       # beta2_E
        norm(0, 1),           # psi
    ]

    # Define the utility metric
    metric = LpNormCurve(
        minimum_tolerable_efficacy=0.2,
        maximum_tolerable_toxicity=0.4,
        hinge_prob_eff=0.5,
        hinge_prob_tox=0.2,
    )

    # Create an EffTox trial object
    trial = EffTox(
        real_doses=real_doses,
        theta_priors=theta_priors,
        tox_cutoff=0.4,
        eff_cutoff=0.2,
        tox_certainty=0.8,
        eff_certainty=0.8,
        metric=metric,
        max_size=30,
    )

    # Get the next recommended dose
    assert trial.next_dose() == 1

def test_gsd_example():
    """
    Tests the GroupSequentialDesign example from the README.md file.
    """
    from clintrials.phase3.gsd import GroupSequentialDesign, spending_function_obrien_fleming
    import numpy as np

    # Create a 4-look GSD with an O'Brien-Fleming spending function
    gsd = GroupSequentialDesign(
        k=4,
        alpha=0.025,
        sfu=spending_function_obrien_fleming
    )

    # Check the efficacy boundaries
    assert len(gsd.efficacy_boundaries) == 4
    assert np.all(np.isclose(gsd.efficacy_boundaries, [4.3326, 2.9631, 2.3591, 2.0141], atol=1e-4))
