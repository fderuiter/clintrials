# clintrials

clintrials is a library of clinical trial designs and methods in Python. This library is intended to facilitate research and is provided "as-is".

## Description

This library implements some designs used in clinical trials. It has implementations of O'Quigley's CRM design, Thall & Cook's EffTox design, and Wages & Tait's efficacy+toxicity design. There is also an implementation of the BEBOP trial design for the simultaneous study of bivariate binary outcomes (like efficacy and toxicity) in the presence of predictive variables, both continuous and binary. A win-ratio simulation module estimates the power of hierarchical composite endpoints.

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

### Tutorials

Tutorials are provided in the `docs/tutorials` directory. You can run them using Jupyter notebooks.

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for more details on how to get involved.

## License

This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for more details.
