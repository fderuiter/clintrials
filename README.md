# clintrials

`clintrials` is a Python library for designing and simulating clinical trials. It provides implementations of various trial designs, with a focus on early-phase oncology trials.

## Features

*   **Dose-Finding Designs:**
    *   Continual Reassessment Method (CRM)
    *   Efficacy-Toxicity (EffTox) design by Thall & Cook
    *   Efficacy-Toxicity design by Wages & Tait
    *   BEBOP design for bivariate binary outcomes with predictive variables
*   **Phase II Designs:**
    *   Two-stage Bayesian design for dichotomous endpoints
    *   Chi-squared test for two-arm comparison
*   **Phase III Designs:**
    *   Group Sequential Designs (GSDs) with Pocock and O'Brien-Fleming-like spending functions
*   **Simulation Tools:**
    *   Tools for simulating patient recruitment and trial outcomes
    *   Parameter space exploration for simulation studies
*   **Interactive Dashboard:**
    *   A web-based dashboard for visualizing simulation results.

## Project Structure

The repository is organized as follows:

*   `clintrials/`: The main Python package containing the library's source code.
    *   `core/`: Core components like math functions, simulation tools, and statistical utilities.
    *   `dosefinding/`: Implementations of various dose-finding trial designs.
    *   `phase2/`: Implementations of Phase II trial designs.
    *   `phase3/`: Implementations of Phase III trial designs.
    *   `winratio/`: Tools for win-ratio analysis.
    *   `dashboard/`: The source code for the interactive web dashboard.
*   `docs/`: Documentation files, including tutorials and user guides.
*   `tests/`: Unit tests for the library.

## Installation

To get started with `clintrials`, you can install it using Poetry. If you don't have Poetry installed, you can install it with pip:

```bash
pip install poetry
```

Then, to install the project and its dependencies, run:

```bash
poetry install
```

This will create a virtual environment with all the necessary packages. To activate the environment, run:

```bash
poetry shell
```

## Usage

Here is a simple example of how to use the `CRM` class to simulate a dose-finding trial:

```python
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
print(f"Next dose: {next_dose}")

# Update the trial with new patient data
trial.update([(3, 0), (3, 0), (3, 0)])
next_dose = trial.next_dose()
print(f"Next dose after update: {next_dose}")
```

## Interactive Dashboard

This project includes an interactive web-based dashboard for visualizing simulation results. To run the dashboard, make sure you have installed the project dependencies with Poetry, as described above. Then, run the following command:

```bash
poetry run dashboard
```

This will start a local web server and open the dashboard in your browser.

## Documentation

The full documentation, including tutorials and API reference, is hosted on GitHub Pages and can be found at:

<https://fderuiter.github.io/clintrials/>

## Contributing

Contributions are welcome! Please see the [contributing guide](docs/contributing.md) for more details on how to get involved.

## Development

To set up a development environment, install the development requirements and run the style checks using pre-commit:

```bash
pip install -e .[dev]
pre-commit install
pre-commit run --all-files
```

Run the test suite with:

```bash
pytest -q
```

### Building the Documentation

To build the documentation locally, install the optional docs dependencies and run Sphinx:

```bash
pip install -e .[docs]
make -C docs html
```

The documentation will be generated in the `docs/_build/html` directory.

## Contact

The repo owner is Kristian Brock, @brockk. Feel free to get in contact through GitHub.
