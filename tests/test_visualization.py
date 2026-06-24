import pandas as pd
import numpy as np
import plotly.graph_objects as go
from clintrials.visualization import (
    plot_dose_finding_outcomes,
    plot_crm_toxicity_probabilities,
    plot_efftox_utility_contours,
    plot_efftox_density,
    plot_crm_simulation_recommendation,
    plot_efftox_simulation_recommendation,
    plot_efftox_simulation_acceptability,
)
from clintrials.dosefinding.crm import CRM
from clintrials.dosefinding.efftox import EffTox, LpNormCurve


def test_plot_dose_finding_outcomes():
    # empty trial
    trial = CRM(prior=[0.1, 0.2, 0.3], target=0.2, first_dose=1, max_size=30)
    fig = plot_dose_finding_outcomes(trial)
    assert isinstance(fig, go.Figure)

    # trial with patients
    trial = CRM(prior=[0.1, 0.2, 0.3], target=0.2, first_dose=1, max_size=30)
    trial.update([(1, 0), (2, 0), (3, 1)])
    fig = plot_dose_finding_outcomes(trial)
    assert isinstance(fig, go.Figure)


def test_plot_crm_toxicity_probabilities():
    trial = CRM(prior=[0.1, 0.2, 0.3], target=0.2, first_dose=1, max_size=30)
    fig = plot_crm_toxicity_probabilities(trial)
    assert isinstance(fig, go.Figure)


def test_plot_efftox_utility_contours():
    metric = LpNormCurve(0.5, 0.65, 0.7, 0.25)
    fig = plot_efftox_utility_contours(metric=metric, prob_eff=[0.1], prob_tox=[0.1])
    assert isinstance(fig, go.Figure)


from scipy.stats import norm


def test_plot_efftox_density():
    metric = LpNormCurve(0.5, 0.65, 0.7, 0.25)
    trial = EffTox(
        real_doses=[1, 2, 3],
        theta_priors=[
            norm(-7.9593, 3.5487),
            norm(1.5482, 3.5018),
            norm(0.7367, 2.5423),
            norm(3.4181, 2.4406),
            norm(0, 0.2),
            norm(0, 1),
        ],
        tox_cutoff=0.65,
        eff_cutoff=0.5,
        tox_certainty=0.1,
        eff_certainty=0.1,
        metric=metric,
        max_size=30,
    )

    # Mock pds
    class DummyPDS:
        def __init__(self):
            self._samp = [1.0, 2.0]
            self._probs = np.array([0.5, 0.5])

    trial.pds = DummyPDS()

    def data_func(x, samp):
        return np.random.normal(0, 1, size=len(samp))

    fig = plot_efftox_density(data_func=data_func, trial=trial, include_doses=[1, 2])
    assert isinstance(fig, go.Figure)


def test_plot_crm_simulation_recommendation():
    summary_df = pd.DataFrame(
        {
            "true_tox": [0.1, 0.2],
            "recommended_dose_prob": [{"1": 0.5, "2": 0.5}, {"1": 0.2, "2": 0.8}],
        }
    ).set_index("true_tox")
    fig = plot_crm_simulation_recommendation(summary_df)
    assert isinstance(fig, go.Figure)

    empty_df = pd.DataFrame()
    fig = plot_crm_simulation_recommendation(empty_df)
    assert isinstance(fig, go.Figure)


def test_plot_efftox_simulation_recommendation():
    summary_df = pd.DataFrame(
        {
            "true_prob_tox": [0.1, 0.2],
            "true_prob_eff": [0.5, 0.6],
            "recommended_dose_prob": [{"1": 0.5, "2": 0.5}, {"1": 0.2, "2": 0.8}],
        }
    ).set_index(["true_prob_tox", "true_prob_eff"])
    fig = plot_efftox_simulation_recommendation(summary_df)
    assert isinstance(fig, go.Figure)

    empty_df = pd.DataFrame()
    fig = plot_efftox_simulation_recommendation(empty_df)
    assert isinstance(fig, go.Figure)


def test_plot_efftox_simulation_acceptability():
    summary_df = pd.DataFrame(
        {
            "true_prob_tox": [0.1, 0.2],
            "true_prob_eff": [0.5, 0.6],
            "prob_accept_tox": [0.9, 0.8],
            "prob_accept_eff": [0.8, 0.9],
        }
    ).set_index(["true_prob_tox", "true_prob_eff"])
    fig = plot_efftox_simulation_acceptability(summary_df)
    assert isinstance(fig, go.Figure)

    empty_df = pd.DataFrame()
    fig = plot_efftox_simulation_acceptability(empty_df)
    assert isinstance(fig, go.Figure)
