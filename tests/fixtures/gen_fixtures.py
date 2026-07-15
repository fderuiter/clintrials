import os
import numpy as np
import pandas as pd
from scipy.stats import norm

from clintrials.core.math import inverse_logit1, logit1
from clintrials.dosefinding.crm import crm

def generate_fixtures():
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

    # Combine the results into data frames
    posterior_dlt_probs_df = pd.DataFrame({
        "scenario": [1] * len(prob_tox_1) + [2] * len(prob_tox_2),
        "dose": list(range(1, len(prob_tox_1) + 1)) + list(range(1, len(prob_tox_2) + 1)),
        "prob": list(prob_tox_1) + list(prob_tox_2)
    })

    next_dose_df = pd.DataFrame({
        "scenario": [1, 2],
        "next_dose": [recommended_dose_1, recommended_dose_2]
    })

    # Write the data frames to CSV files
    out_probs = os.path.join("tests", "fixtures", "expected_posterior_dlt_probs.csv")
    out_next = os.path.join("tests", "fixtures", "next_dose_recommendations.csv")
    
    os.makedirs(os.path.dirname(out_probs), exist_ok=True)
    
    posterior_dlt_probs_df.to_csv(out_probs, index=False)
    next_dose_df.to_csv(out_next, index=False)
    
    print("Fixtures generated successfully.")  # noqa: T201

if __name__ == "__main__":
    generate_fixtures()
