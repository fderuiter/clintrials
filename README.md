# clintrials

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://fderuiter.github.io/clintrials/badge.svg)](https://fderuiter.github.io/clintrials)

`clintrials` is a Python library for designing and simulating clinical trials. It provides implementations of various trial designs, with a focus on early-phase oncology trials.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Trial Simulation Hub](#trial-simulation-hub)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Development](#development)
- [Contact](#contact)

## Features

`clintrials` provides a comprehensive suite of tools for clinical trial design and simulation, with a focus on early-phase oncology trials.

### Dose-Finding Designs

-   **Continual Reassessment Method (CRM):** A Bayesian design for dose-finding based on toxicity. It adapts to the observed outcomes to find the maximum tolerable dose (MTD).
-   **Efficacy-Toxicity (EffTox) by Thall & Cook:** A design that models both efficacy and toxicity to find the optimal dose, balancing the trade-off between the two.
-   **Efficacy-Toxicity by Wages & Tait:** An adaptive Bayesian design for seamless Phase I/II trials that models efficacy and toxicity separately, allowing for more flexible decision-making.

### Phase III Designs

-   **Group Sequential Designs (GSDs):** Flexible designs that allow for interim analyses to stop for efficacy or futility, with support for Pocock and O'Brien-Fleming-like spending functions.

### Simulation Tools

-   **Patient Recruitment:** Simulate patient arrival times with constant or time-varying intensity to model realistic trial conditions.
-   **Parameter Space Exploration:** Tools for running simulations over a grid of parameters to evaluate the operating characteristics of a design under various scenarios.

### Trial Simulation Hub

-   A web-based Trial Simulation Hub for visualizing simulation results, making it easy to explore and understand the performance of different trial designs.

## Getting Started

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

Here are a few examples of how to use the `clintrials` library.

### Example 1: Continual Reassessment Method (CRM)

This example shows how to use the `CRM` class to simulate a simple dose-finding trial.

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
assert next_dose == 3
print(f"Next dose: {next_dose}")

# Update the trial with new patient data
trial.update([(3, 0), (3, 0), (3, 0)])
next_dose = trial.next_dose()
assert next_dose == 4
print(f"Next dose after update: {next_dose}")
```

### Example 2: EffTox Design

This example demonstrates how to set up and use the `EffTox` design.

```python
from clintrials.dosefinding.efftox import EffTox, LpNormCurve
from scipy.stats import norm

# Define the real dose levels
real_doses = [1.0, 2.0, 4.0, 8.0]

# Define prior probabilities for elicitation (skeleton)
prior_tox_probs = [0.05, 0.1, 0.2, 0.4]
prior_eff_probs = [0.2, 0.4, 0.6, 0.7]

# Define the utility metric
metric = LpNormCurve(
    minimum_tolerable_efficacy=0.2,
    maximum_tolerable_toxicity=0.4,
    hinge_prob_eff=0.5,
    hinge_prob_tox=0.2,
)

# Create an EffTox trial object (priors elicited from skeleton)
trial = EffTox(
    real_doses=real_doses,
    prior_tox_probs=prior_tox_probs,
    prior_eff_probs=prior_eff_probs,
    tox_cutoff=0.4,
    eff_cutoff=0.2,
    tox_certainty=0.8,
    eff_certainty=0.8,
    metric=metric,
    max_size=30,
)

# Get the next recommended dose
next_dose = trial.next_dose()
assert next_dose == 1
print(f"Next dose: {next_dose}")
```

### Example 3: Group Sequential Design (GSD)

This example shows how to create a group sequential design with an O'Brien-Fleming-like spending function.

```python
from clintrials.phase3.gsd import GroupSequentialDesign, spending_function_obrien_fleming
import numpy as np

# Create a 4-look GSD with an O'Brien-Fleming spending function
gsd = GroupSequentialDesign(
    k=4,
    alpha=0.025,
    sfu=spending_function_obrien_fleming
)

# Verify the efficacy boundaries
assert len(gsd.efficacy_boundaries) == 4
assert np.all(np.isclose(gsd.efficacy_boundaries, [4.3326, 2.9631, 2.3591, 2.0141], atol=1e-4))
print("Efficacy Boundaries:", gsd.efficacy_boundaries)
```

## Project Structure

The repository is organized as follows:

-   **`clintrials/`**: The main Python package.
    -   **`core/`**: Core components like math functions, numerical integration routines, patient recruitment models, simulation tools, statistical utilities, and time-to-event models.
    -   **`dosefinding/`**: Implementations of various dose-finding trial designs, including CRM, EffTox, and Wages & Tait.
    -   **`phase3/`**: Implementations of Phase III trial designs, including Group Sequential Designs.
    -   **`winratio/`**: Tools for win-ratio analysis.
    -   **`clintrials/visualization/dashboard/main.py`**: The Trial Simulation Hub entry point and source code for the interactive web dashboard.
-   **`docs/`**: Documentation files, including tutorials and API reference.
-   **`tests/`**: Unit tests for the library.

## Trial Simulation Hub

This project includes an interactive Trial Simulation Hub for visualizing simulation results. You can access the hosted version online at:

<https://fderuiter.github.io/clintrials/hub/>

To run the Trial Simulation Hub locally, make sure you have installed the project with the `viz` extra:

```bash
poetry install -E viz
```

Then, run the following command:

```bash
poetry run dashboard
```

This will start a local web server and open the Trial Simulation Hub in your browser.

## Documentation

The full documentation, including tutorials and API reference, is hosted on GitHub Pages and can be found at:

<https://fderuiter.github.io/clintrials/>

## Contributing

Contributions are welcome! Please see the [contributing guide](CONTRIBUTING.md) for more details on how to get involved.

## Development

We recommend using the provided Dev Container for a standardized development environment. When you open this repository in a supported editor (like VS Code), you will be prompted to reopen it in a container. The Dev Container automatically installs all necessary system and Python dependencies using Poetry.

To set up a development environment locally without a container, use Poetry:

```bash
poetry install --all-extras
poetry run pre-commit install
poetry run pre-commit run --all-files
```

### Testing

Run the test suite with:

```bash
poetry run pytest -q
```

### Building the Documentation

To build the documentation locally with Poetry, run:

```bash
poetry run make -C docs html SPHINXOPTS="-W"
```

The documentation will be generated in the `docs/_build/html` directory.

## Contact

The repo owner is Kristian Brock, @brockk. Feel free to get in contact through GitHub.
