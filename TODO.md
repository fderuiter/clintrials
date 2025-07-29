# Outstanding TODOs

The following tasks were extracted from TODO comments removed across the codebase.

- **clintrials/dosefinding/watu.py**
  - Parameter `must_try_lowest_dose` in `__init__` is currently unused.
  - Replace the Monte Carlo approach in `prob_eff_exceeds` with a proper posterior integral.
  - Allow the sample size `n=10**5` used in `_stage_one_next_dose` to be configurable.
  - Allow the sample size `n=10**5` used in `_stage_two_next_dose` to be configurable.
- **clintrials/dosefinding/crm/design.py**
  - Investigate how variance estimation is possible in `_get_beta_hat_mle`.
- **clintrials/dosefinding/efftox/design.py**
  - Ensure integration limits fully cover the posterior range for accurate estimation (appears in two functions).
  - Implement the `solve` method that should return the un-specified probability for a given `delta`.
  - Replace the placeholder parameter means and standard deviations currently taken from MD Anderson's EffTox software.
- **clintrials/recruitment.py**
  - Clarify whether zero or negative values are allowed for `initial_intensity` in `Recruitment`
    initialization.
- **clintrials/phase2/bebop/peps2v1.py**
  - Complete the docstring for `get_posterior_probs` to describe the trial correctly.
  - Implement confidence interval calculation for correlation in the commented `correlation_effect` method.
- **clintrials/phase2/bebop/__init__.py**
  - Write a full docstring for `update` describing parameters and behaviour.
- **clintrials/simulation.py**
  - Provide handling for non-pandas outputs in `extract_sim_data` (`TODO` placeholders currently return raw tuples and lists).
  - Review whether the helper functions at the end of the module belong in a general package.
- **tests/test_crm.py**
  - Add tests of the full Bayesian CRM verified against the `bcrm` R package.
- **tests/test_efftox.py**
  - Add dedicated tests for the EffTox implementation.
- **tutorials/CRM.ipynb**
  - Replace the placeholder markdown cell containing "TODO" with actual content.
