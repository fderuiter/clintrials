# Test Fixtures for Bayesian CRM

This directory contains test fixtures for the Bayesian CRM tests. The fixtures are CSV files that contain the expected output of the Bayesian CRM implementation for a set of predefined clinical trial scenarios.

## Fixture Generation

The fixtures were generated using the `gen_fixtures.py` script in this directory. This script uses the internal Python implementation of `crm` to calculate the expected posterior probabilities of dose-limiting toxicity (DLT) and the next recommended dose for a set of predefined clinical trial scenarios.

To generate the fixtures, you only need the Python development environment set up via Poetry.

To run the script (in bash):
```bash
./tests/fixtures/generate_fixtures.sh
```
or directly with Poetry:
```bash
poetry run python tests/fixtures/gen_fixtures.py
```

The script will generate the following files:
- `expected_posterior_dlt_probs.csv`: Contains the posterior probabilities of DLT for each dose level in each scenario.
- `next_dose_recommendations.csv`: Contains the recommended next dose for each scenario.

These files are used by the `pytest` suite to validate the `clintrials` Python implementation against the expected benchmarks.
