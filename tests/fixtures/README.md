# Test Fixtures for Bayesian CRM

This directory contains test fixtures for the Bayesian CRM tests. The fixtures are CSV files that contain the expected output of the `bcrm` R package for a set of predefined clinical trial scenarios.

## Fixture Generation

The fixtures were generated using the `gen_fixtures.R` script in this directory. This script uses the `bcrm` R package to calculate the expected posterior probabilities of dose-limiting toxicity (DLT) and the next recommended dose for a set of predefined clinical trial scenarios.

To run this script, you need to have R installed with the `bcrm` and `mvtnorm` packages.

To install dependencies (in R):
```R
install.packages(c("bcrm", "mvtnorm"))
```

To run the script (in bash):
```bash
Rscript tests/fixtures/gen_fixtures.R
```

The script will generate the following files:
- `expected_posterior_dlt_probs.csv`: Contains the posterior probabilities of DLT for each dose level in each scenario.
- `next_dose_recommendations.csv`: Contains the recommended next dose for each scenario.

These files are used by the `pytest` suite to validate the `clintrials` Python implementation.
