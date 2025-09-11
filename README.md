# clintrials

clintrials is a library of clinical trial designs and methods in Python. This library is intended to facilitate research and is provided "as-is".

## Description

This library implements some designs used in clinical trials. It has implementations of O'Quigley's CRM design, Thall & Cook's EffTox design, and Wages & Tait's efficacy+toxicity design. There is also an implementation of the BEBOP trial design for the simultaneous study of bivariate binary outcomes (like efficacy and toxicity) in the presence of predictive variables, both continuous and binary. A win-ratio simulation module estimates the power of hierarchical composite endpoints.

## Implemented Designs

*   **Continual Reassessment Method (CRM):** A dose-finding method that uses a statistical model to determine the maximum tolerated dose (MTD) of a new drug.
*   **Efficacy-Toxicity (EffTox) Design:** A dose-finding method that considers both efficacy and toxicity outcomes to find the optimal dose.
*   **Wages & Tait's Design:** A seamless Phase I/II adaptive design for oncology trials of molecularly targeted agents.
*   **BEBOP Design:** A design for the simultaneous study of bivariate binary outcomes (e.g., efficacy and toxicity) in the presence of predictive variables.
*   **Win-Ratio Simulation:** A module for estimating the statistical power of hierarchical composite endpoints using the win-ratio method.

## Installation

There are two ways to install clintrials.

### Using pip

To install the latest stable version from PyPI, run:

```bash
pip install clintrials
```

### From source

To install the latest development version from the git repository, first clone the repository:

```bash
git clone https://github.com/brockk/clintrials.git
cd clintrials
```

Then, install the project and its dependencies using Poetry:

```bash
poetry install
```

## Usage

### Interactive Dashboard

This project includes an interactive web-based dashboard for visualizing simulation results. To run the dashboard, make sure you have installed the project dependencies with Poetry, as described above. Then, run the following command:

```bash
poetry run dashboard
```

This will start a local web server and open the dashboard in your browser.

The dashboard supports tailored visualizations for different trial designs. Use the sidebar to select the appropriate trial design (e.g., CRM, EffTox, Win Ratio). For CRM and EffTox, upload your simulation results in JSON format to explore them interactively. The Win Ratio option lets you configure parameters and run simulations directly in the browser.

### Library Usage

Here is a simple example of how to use the `CRM` class to conduct a dose-finding trial:

```python
from clintrials.dosefinding.crm import CRM

# Define the prior probabilities of toxicity for each dose level
prior_tox_probs = [0.025, 0.05, 0.1, 0.25]

# Define the target toxicity rate
tox_target = 0.35

# Define the starting dose level
first_dose = 3

# Define the maximum number of patients in the trial
trial_size = 30

# Create a CRM trial instance
trial = CRM(prior_tox_probs, tox_target, first_dose, trial_size)

# Get the next recommended dose
next_dose = trial.next_dose()
print(f"Next recommended dose: {next_dose}")

# Update the trial with new patient outcomes
# (dose, toxicity)
cases = [(3, 0), (3, 0), (3, 0)]
next_dose = trial.update(cases)
print(f"Next recommended dose after {len(cases)} patients: {next_dose}")
```

### Tutorials

Tutorials are provided in the `docs/tutorials` directory. You can run them using Jupyter notebooks.

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for more details on how to get involved.

## License

This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for more details.
