import numpy as np
import pandas as pd
from scipy.stats import norm

from clintrials.core.math import inverse_logit1, logit1
from clintrials.dosefinding.crm import crm


def test_CRM_with_bcrm_fixtures():
    # Load the fixtures
    expected_probs = pd.read_csv("tests/fixtures/expected_posterior_dlt_probs.csv")
    expected_doses = pd.read_csv("tests/fixtures/next_dose_recommendations.csv")

    # Scenario 1
    p_tox_prior_1 = [0.1, 0.2, 0.3, 0.4]
    target_tox_1 = 0.3
    doses_1 = [1, 1, 2, 2, 3, 3]
    dlt_1 = [0, 0, 0, 1, 1, 1]

    _, _, _, prob_tox_1 = crm(
        prior=p_tox_prior_1,
        target=target_tox_1,
        toxicities=dlt_1,
        dose_levels=doses_1,
        F_func=logit1,
        inverse_F=inverse_logit1,
        beta_dist=norm(loc=0, scale=np.sqrt(1.34)),
        method="bayes",
        plugin_mean=False,
    )
    recommended_dose_1 = np.argmin(np.abs(np.array(prob_tox_1) - target_tox_1)) + 1

    expected_prob_tox_1 = expected_probs[expected_probs.scenario == 1].prob.values
    expected_next_dose_1 = expected_doses[
        expected_doses.scenario == 1
    ].next_dose.values[0]

    assert np.allclose(prob_tox_1, expected_prob_tox_1, atol=1e-2)
    assert recommended_dose_1 == expected_next_dose_1

    # Scenario 2
    p_tox_prior_2 = [0.05, 0.1, 0.2, 0.35, 0.5]
    target_tox_2 = 0.2
    doses_2 = [1, 1, 1, 2, 2, 2]
    dlt_2 = [0, 0, 0, 0, 0, 0]

    _, _, _, prob_tox_2 = crm(
        prior=p_tox_prior_2,
        target=target_tox_2,
        toxicities=dlt_2,
        dose_levels=doses_2,
        F_func=logit1,
        inverse_F=inverse_logit1,
        beta_dist=norm(loc=0, scale=np.sqrt(1.34)),
        method="bayes",
        plugin_mean=False,
    )
    recommended_dose_2 = np.argmin(np.abs(np.array(prob_tox_2) - target_tox_2)) + 1

    expected_prob_tox_2 = expected_probs[expected_probs.scenario == 2].prob.values
    expected_next_dose_2 = expected_doses[
        expected_doses.scenario == 2
    ].next_dose.values[0]

    assert np.allclose(prob_tox_2, expected_prob_tox_2, atol=1e-1)
    assert recommended_dose_2 == expected_next_dose_2
