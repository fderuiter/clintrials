__author__ = "Kristian Brock"
__contact__ = "kristian.brock@gmail.com"

from collections import OrderedDict
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy.stats import norm

from clintrials.dosefinding.efftox import (
    EffTox,
    LpNormCurve,
    _L_n,
    _pi_ab,
    _pi_E,
    _pi_T,
    scale_doses,
)


def assess_efftox_trial(et):
    to_return = OrderedDict()
    to_return["NextDose"] = et.next_dose()
    to_return["ProbEff"] = et.prob_eff
    to_return["ProbTox"] = et.prob_tox
    to_return["ProbAccEff"] = et.prob_acc_eff
    to_return["ProbAccTox"] = et.prob_acc_tox
    to_return["Utility"] = et.utility
    return to_return


def run_trial(trial, cases, summary_func, **kwargs):
    trial.reset()
    trial.update(cases, **kwargs)
    return summary_func(trial)


def test_thall2014_efftox():

    # Recreate all params in a hypothetical path of the
    # trial described in Thall et al, 2014

    real_doses = [1, 2, 4, 6.6, 10]
    trial_size = 39
    first_dose = 1

    # Model params
    tox_cutoff = 0.3
    eff_cutoff = 0.5
    tox_certainty = 0.1
    eff_certainty = 0.1

    efftox_priors = [
        norm(loc=-7.9593, scale=3.5487),
        norm(loc=1.5482, scale=3.5018),
        norm(loc=0.7367, scale=2.5423),
        norm(loc=3.4181, scale=2.4406),
        norm(loc=0.0, scale=0.2),
        norm(loc=0.0, scale=1.0),
    ]

    hinge_points = [(0.5, 0), (1, 0.65), (0.7, 0.25)]
    metric = LpNormCurve(
        hinge_points[0][0], hinge_points[1][1], hinge_points[2][0], hinge_points[2][1]
    )

    et = EffTox(
        real_doses,
        efftox_priors,
        tox_cutoff,
        eff_cutoff,
        tox_certainty,
        eff_certainty,
        metric,
        trial_size,
        first_dose,
    )

    epsilon1 = 0.05
    epsilon2 = 0.05

    # Conduct a hypothetical trial and match the output to the official software

    # Cohort 1 - No responses or tox at dose 1
    cases = [(1, 0, 0), (1, 0, 0), (1, 0, 0)]
    trial_outcomes = [
        run_trial(et, cases, assess_efftox_trial, n=10**6) for i in range(5)
    ]

    assert np.all([o["NextDose"] == 2 for o in trial_outcomes])
    assert np.all(
        np.array([list(o["ProbEff"]) for o in trial_outcomes]).mean(axis=0)
        - [0.04, 0.19, 0.57, 0.78, 0.87]
        < epsilon1
    )
    assert np.all(
        np.array([list(o["ProbTox"]) for o in trial_outcomes]).mean(axis=0)
        - [0.01, 0.01, 0.02, 0.07, 0.13]
        < epsilon1
    )
    assert np.all(
        np.array([list(o["Utility"]) for o in trial_outcomes]).mean(axis=0)
        - [-0.93, -0.62, 0.11, 0.46, 0.53]
        < epsilon1
    )
    assert np.all(
        np.array([list(o["ProbAccEff"]) for o in trial_outcomes]).mean(axis=0)
        - [0.01, 0.12, 0.59, 0.82, 0.89]
        < epsilon1
    )
    assert np.all(
        np.array([list(o["ProbAccTox"]) for o in trial_outcomes]).mean(axis=0)
        - [1.00, 0.99, 0.98, 0.93, 0.85]
        < epsilon1
    )

    # Cohort 2 - Singled response but no tox at dose 2
    cases = cases + [(2, 0, 1), (2, 0, 0), (2, 0, 0)]
    trial_outcomes = [
        run_trial(et, cases, assess_efftox_trial, n=10**6) for i in range(5)
    ]

    assert np.all([o["NextDose"] == 3 for o in trial_outcomes])
    assert np.all(
        np.array([list(o["ProbEff"]) for o in trial_outcomes]).mean(axis=0)
        - [0.05, 0.26, 0.72, 0.86, 0.91]
        < epsilon1
    )
    assert np.all(
        np.array([list(o["ProbTox"]) for o in trial_outcomes]).mean(axis=0)
        - [0.01, 0.01, 0.02, 0.06, 0.12]
        < epsilon1
    )
    assert np.all(
        np.array([list(o["Utility"]) for o in trial_outcomes]).mean(axis=0)
        - [-0.91, -0.47, 0.42, 0.64, 0.64]
        < epsilon1
    )
    assert np.all(
        np.array([list(o["ProbAccEff"]) for o in trial_outcomes]).mean(axis=0)
        - [0.01, 0.13, 0.80, 0.91, 0.94]
        < epsilon1
    )
    assert np.all(
        np.array([list(o["ProbAccTox"]) for o in trial_outcomes]).mean(axis=0)
        - [1.00, 1.00, 0.98, 0.93, 0.86]
        < epsilon1
    )

    # Cohort 3 - Eff, Tox and a Both at dose level 3
    cases = cases + [(3, 0, 1), (3, 1, 0), (3, 1, 1)]
    trial_outcomes = [
        run_trial(et, cases, assess_efftox_trial, n=10**6) for i in range(10)
    ]

    assert np.all([o["NextDose"] == 3 for o in trial_outcomes])
    assert np.all(
        np.array([list(o["ProbEff"]) for o in trial_outcomes]).mean(axis=0)
        - [0.06, 0.24, 0.71, 0.89, 0.94]
        < epsilon1
    )
    assert np.all(
        np.array([list(o["ProbTox"]) for o in trial_outcomes]).mean(axis=0)
        - [0.02, 0.06, 0.41, 0.77, 0.87]
        < epsilon1
    )
    assert np.all(
        np.array([list(o["Utility"]) for o in trial_outcomes]).mean(axis=0)
        - [-0.92, -0.63, -0.24, -0.41, -0.47]
        < epsilon2
    )
    assert np.all(
        np.array([list(o["ProbAccEff"]) for o in trial_outcomes]).mean(axis=0)
        - [0.01, 0.07, 0.84, 0.97, 0.98]
        < epsilon1
    )
    assert np.all(
        np.array([list(o["ProbAccTox"]) for o in trial_outcomes]).mean(axis=0)
        - [1.00, 0.98, 0.36, 0.08, 0.05]
        < epsilon2
    )


def test_matchpoint_efftox():
    mp_real_doses = [7.5, 15, 30, 45]
    mp_trial_size = 30
    mp_first_dose = 3
    mp_tox_cutoff = 0.40
    mp_eff_cutoff = 0.45
    mp_hinge_points = [(0.4, 0), (1, 0.7), (0.5, 0.4)]
    mp_metric = LpNormCurve(
        mp_hinge_points[0][0],
        mp_hinge_points[1][1],
        mp_hinge_points[2][0],
        mp_hinge_points[2][1],
    )
    mp_tox_certainty = 0.05
    mp_eff_certainty = 0.05
    mp_mu_t_mean, mp_mu_t_sd = -5.4317, 2.7643
    mp_beta_t_mean, mp_beta_t_sd = 3.1761, 2.7703
    mp_mu_e_mean, mp_mu_e_sd = -0.8442, 1.9786
    mp_beta_e_1_mean, mp_beta_e_1_sd = 1.9857, 1.9820
    mp_beta_e_2_mean, mp_beta_e_2_sd = 0, 0.2
    mp_psi_mean, mp_psi_sd = 0, 1
    mp_efftox_priors = [
        norm(loc=mp_mu_t_mean, scale=mp_mu_t_sd),
        norm(loc=mp_beta_t_mean, scale=mp_beta_t_sd),
        norm(loc=mp_mu_e_mean, scale=mp_mu_e_sd),
        norm(loc=mp_beta_e_1_mean, scale=mp_beta_e_1_sd),
        norm(loc=mp_beta_e_2_mean, scale=mp_beta_e_2_sd),
        norm(loc=mp_psi_mean, scale=mp_psi_sd),
    ]
    et = EffTox(
        mp_real_doses,
        mp_efftox_priors,
        mp_tox_cutoff,
        mp_eff_cutoff,
        mp_tox_certainty,
        mp_eff_certainty,
        mp_metric,
        mp_trial_size,
        mp_first_dose,
    )
    epsilon1 = 0.05
    epsilon2 = 0.05
    cases = [(3, 0, 0), (3, 1, 0), (3, 1, 0)]
    trial_outcomes = [
        run_trial(et, cases, assess_efftox_trial, n=10**6) for i in range(10)
    ]
    assert np.all(
        np.array([list(o["ProbEff"]) for o in trial_outcomes]).mean(axis=0)
        - [0.11, 0.10, 0.16, 0.25]
        < epsilon1
    )
    assert np.all(
        np.array([list(o["ProbTox"]) for o in trial_outcomes]).mean(axis=0)
        - [0.06, 0.12, 0.52, 0.80]
        < epsilon1
    )
    assert np.all(
        np.array([list(o["Utility"]) for o in trial_outcomes]).mean(axis=0)
        - [-0.49, -0.50, -0.57, -0.67]
        < epsilon1
    )
    assert np.all(
        np.array([list(o["ProbAccEff"]) for o in trial_outcomes]).mean(axis=0)
        - [0.08, 0.04, 0.06, 0.20]
        < epsilon1
    )
    assert np.all(
        np.array([list(o["ProbAccTox"]) for o in trial_outcomes]).mean(axis=0)
        - [0.95, 0.92, 0.33, 0.07]
        < epsilon2
    )


def test_thall2014_efftox_v2():
    real_doses = [1, 2, 4, 6.6, 10]
    trial_size = 39
    first_dose = 1
    tox_cutoff = 0.3
    eff_cutoff = 0.5
    tox_certainty = 0.1
    eff_certainty = 0.1
    efftox_priors = [
        norm(loc=-7.9593, scale=3.5487),
        norm(loc=1.5482, scale=3.5018),
        norm(loc=0.7367, scale=2.5423),
        norm(loc=3.4181, scale=2.4406),
        norm(loc=0.0, scale=0.2),
        norm(loc=0.0, scale=1.0),
    ]
    hinge_points = [(0.5, 0), (1, 0.65), (0.7, 0.25)]
    metric = LpNormCurve(
        hinge_points[0][0], hinge_points[1][1], hinge_points[2][0], hinge_points[2][1]
    )
    et = EffTox(
        real_doses,
        efftox_priors,
        tox_cutoff,
        eff_cutoff,
        tox_certainty,
        eff_certainty,
        metric,
        trial_size,
        first_dose,
    )
    epsilon1 = 0.10
    epsilon2 = 0.10
    cases = [(4, 1, 0)]
    trial_outcomes = [
        run_trial(et, cases, assess_efftox_trial, n=10**6) for i in range(10)
    ]
    assert np.all([o["NextDose"] == 1 for o in trial_outcomes])
    assert np.all(
        np.array([list(o["ProbEff"]) for o in trial_outcomes]).mean(axis=0)
        - [0.16, 0.18, 0.26, 0.40, 0.51]
        < epsilon1
    )
    assert np.all(
        np.array([list(o["ProbTox"]) for o in trial_outcomes]).mean(axis=0)
        - [0.07, 0.10, 0.26, 0.58, 0.79]
        < epsilon1
    )
    assert np.all(
        np.array([list(o["Utility"]) for o in trial_outcomes]).mean(axis=0)
        - [-0.78, -0.80, -0.88, -1.10, -1.18]
        < epsilon1
    )
    assert np.all(
        np.array([list(o["ProbAccEff"]) for o in trial_outcomes]).mean(axis=0)
        - [0.14, 0.13, 0.19, 0.37, 0.52]
        < epsilon2
    )
    assert np.all(
        np.array([list(o["ProbAccTox"]) for o in trial_outcomes]).mean(axis=0)
        - [0.92, 0.88, 0.69, 0.26, 0.11]
        < epsilon1
    )
    cases = cases + [(2, 0, 0)]
    trial_outcomes = [
        run_trial(et, cases, assess_efftox_trial, n=10**6) for i in range(10)
    ]
    assert np.all([o["NextDose"] == 1 for o in trial_outcomes])
    cases = cases + [(1, 0, 0)]
    trial_outcomes = [
        run_trial(et, cases, assess_efftox_trial, n=10**6) for i in range(10)
    ]
    assert np.all([o["NextDose"] in [3, 4] for o in trial_outcomes])
    cases = cases + [(5, 1, 1)]
    trial_outcomes = [
        run_trial(et, cases, assess_efftox_trial, n=10**6) for i in range(5)
    ]
    assert np.all([o["NextDose"] == 3 for o in trial_outcomes])


@pytest.fixture
def lp_norm_curve():
    return LpNormCurve(
        minimum_tolerable_efficacy=0.4,
        maximum_tolerable_toxicity=0.7,
        hinge_prob_eff=0.5,
        hinge_prob_tox=0.4,
    )


class TestLpNormCurve:
    def test_constructor_invalid_hinge_points(self):
        with pytest.raises(ValueError, match="Probability of toxicity at hinge point"):
            LpNormCurve(0.4, 0.7, 0.5, 0.8)
        with pytest.raises(ValueError, match="Probability of efficacy at hinge point"):
            LpNormCurve(0.4, 0.7, 0.3, 0.5)

    def test_call(self, lp_norm_curve):
        assert np.isclose(lp_norm_curve(0.5, 0.4), 0)
        assert lp_norm_curve(0.6, 0.3) > 0
        assert lp_norm_curve(0.4, 0.5) < 0
        assert lp_norm_curve(0.001, 0.999) < 0
        assert np.isnan(lp_norm_curve(0, 0.5))
        assert np.isnan(lp_norm_curve(1, 0.5))
        assert np.isnan(lp_norm_curve(0.5, 0))
        assert np.isnan(lp_norm_curve(0.5, 1))

    def test_solve(self, lp_norm_curve):
        with pytest.raises(Exception):
            lp_norm_curve.solve()
        with pytest.raises(Exception):
            lp_norm_curve.solve(prob_eff=0.5, prob_tox=0.4)
        assert np.isclose(lp_norm_curve.solve(prob_eff=0.5), 0.4)
        assert np.isclose(lp_norm_curve.solve(prob_tox=0.4), 0.5)
        prob_eff = lp_norm_curve.solve(prob_tox=0.2)
        assert np.isclose(lp_norm_curve(prob_eff, 0.2), 0)

    def test_get_tox(self, lp_norm_curve):
        assert np.isclose(lp_norm_curve.get_tox(eff=0.5, util=0), 0.4)
        assert lp_norm_curve.get_tox(eff=0.6, util=0.1) < lp_norm_curve.get_tox(
            eff=0.6, util=0
        )
        assert lp_norm_curve.get_tox(eff=0.6, util=-0.1) > lp_norm_curve.get_tox(
            eff=0.6, util=0
        )

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        st.floats(0.01, 0.99),
        st.floats(0.01, 0.99),
        st.floats(0.01, 0.99),
        st.floats(0.01, 0.99),
    )
    def test_utility_monotonicity(
        self, lp_norm_curve, prob_eff1, prob_eff2, prob_tox1, prob_tox2
    ):
        u1 = lp_norm_curve(prob_eff1, prob_tox1)
        if prob_eff2 > prob_eff1:
            assert lp_norm_curve(prob_eff2, prob_tox1) >= u1
        else:
            assert lp_norm_curve(prob_eff2, prob_tox1) <= u1
        if prob_tox2 > prob_tox1:
            assert lp_norm_curve(prob_eff1, prob_tox2) <= u1
        else:
            assert lp_norm_curve(prob_eff1, prob_tox2) >= u1

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(prob_eff=st.floats(0.01, 0.99))
    def test_solve_consistency(self, lp_norm_curve, prob_eff):
        prob_tox = lp_norm_curve.solve(prob_eff=prob_eff)
        if np.iscomplex(prob_tox).any():
            return
        assert np.isclose(lp_norm_curve(prob_eff, prob_tox), 0)

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(eff=st.floats(0.01, 0.99), util=st.floats(-0.5, 0.5))
    def test_get_tox_consistency(self, lp_norm_curve, eff, util):
        tox = lp_norm_curve.get_tox(eff=eff, util=util)
        if np.iscomplex(tox).any():
            return
        if 0 < tox < 1:
            assert np.isclose(lp_norm_curve(eff, tox), util)


class TestCoreMaths:
    def test_scale_doses(self):
        scaled = scale_doses([10, 20, 40])
        assert np.allclose(scaled, [-0.693147, 0.0, 0.693147])
        assert np.isclose(np.mean(scaled), 0)

    def test_pi_T(self):
        assert np.isclose(_pi_T(1.0, -2.0, 2.0), 0.5)
        assert _pi_T(1.0, 2.0, 2.0) > 0.9
        assert _pi_T(1.0, -2.0, -2.0) < 0.1

    def test_pi_E(self):
        assert np.isclose(_pi_E(1.0, -2.0, 1.0, 1.0), 0.5)
        assert _pi_E(1.0, 2.0, 1.0, 1.0) > 0.9
        assert _pi_E(1.0, -2.0, -1.0, -1.0) < 0.1

    def test_pi_ab(self):
        p_ab = _pi_ab(0, 1, 1, 0, 1, 0, 1, 0, 0)
        p_T = _pi_T(0, 0, 1)
        p_E = _pi_E(0, 0, 1, 0)
        assert np.isclose(p_ab, p_T * p_E)
        p_ab_psi = _pi_ab(0, 1, 1, 0, 1, 0, 1, 0, 1)
        assert not np.isclose(p_ab_psi, p_T * p_E)

    def test_L_n(self):
        cases = [(0, 1, 1), (0.5, 0, 1)]
        params = np.array([[-1, 1, 1, 1, 0, 0], [0, 1, 0, 1, 0, 0]])
        likelihood = _L_n(
            cases,
            params[:, 0],
            params[:, 1],
            params[:, 2],
            params[:, 3],
            params[:, 4],
            params[:, 5],
        )
        assert likelihood.shape == (2,)
        p1 = _pi_ab(0, 1, 1, -1, 1, 1, 1, 0, 0) * _pi_ab(0.5, 0, 1, -1, 1, 1, 1, 0, 0)
        p2 = _pi_ab(0, 1, 1, 0, 1, 0, 1, 0, 0) * _pi_ab(0.5, 0, 1, 0, 1, 0, 1, 0, 0)
        assert np.isclose(likelihood[0], p1)
        assert np.isclose(likelihood[1], p2)


@pytest.fixture
def efftox_trial():
    return EffTox(
        real_doses=[1, 2, 4, 6.6, 10],
        theta_priors=[norm()] * 6,
        tox_cutoff=0.3,
        eff_cutoff=0.5,
        tox_certainty=0.9,
        eff_certainty=0.9,
        metric=LpNormCurve(0.5, 0.3, 0.7, 0.25),
        max_size=30,
    )


class TestEffToxAdmissibleSet:
    @patch("clintrials.dosefinding.efftox.efftox_get_posterior_probs")
    def test_admissible_set_logic(self, mock_post_probs, efftox_trial):
        prob_acc_tox = [0.8, 0.95, 0.8, 0.8, 0.8]
        prob_acc_eff = [0.8, 0.95, 0.8, 0.8, 0.8]
        mock_post_probs.return_value = (
            list(zip([0.1] * 5, [0.6] * 5, prob_acc_tox, prob_acc_eff)),
            None,
        )
        efftox_trial._update_integrals()
        assert efftox_trial.admissable_set() == [2]

    @patch("clintrials.dosefinding.efftox.efftox_get_posterior_probs")
    def test_admissible_set_special_rule(self, mock_post_probs, efftox_trial):
        with patch.object(efftox_trial, "maximum_dose_given", return_value=1):
            prob_acc = [0.8, 0.95, 0.8, 0.8, 0.8]
            mock_post_probs.return_value = (
                list(zip([0.1] * 5, [0.6] * 5, prob_acc, [0.8] * 5)),
                None,
            )
            efftox_trial._update_integrals()
            assert efftox_trial.admissable_set() == [2]


def test_myeloma_integration_deterministic(mocker):
    real_doses = [25, 50, 75, 100, 125]
    priors = [
        norm(loc=-8, scale=3.5),
        norm(loc=1.5, scale=3.5),
        norm(loc=0.7, scale=2.5),
        norm(loc=3.4, scale=2.4),
        norm(loc=0, scale=0.2),
        norm(loc=0, scale=1),
    ]
    metric = LpNormCurve(0.2, 0.3, 0.5, 0.15)
    trial = EffTox(real_doses, priors, 0.3, 0.2, 0.9, 0.9, metric, 30, 1, True)
    mock_post_probs = mocker.patch(
        "clintrials.dosefinding.efftox.efftox_get_posterior_probs"
    )
    assert trial.next_dose() == 1

    prob_tox = [0.1, 0.12, 0.15, 0.2, 0.25]
    prob_eff = [0.3, 0.5, 0.6, 0.55, 0.5]
    acc_tox = [0.95] * 5
    acc_eff = [0.95] * 5
    mock_post_probs.return_value = (
        list(zip(prob_tox, prob_eff, acc_tox, acc_eff)),
        None,
    )
    trial.update([(1, 0, 0)] * 3)
    assert trial.next_dose() == 2

    prob_eff = [0.3, 0.6, 0.5, 0.55, 0.5]
    mock_post_probs.return_value = (
        list(zip(prob_tox, prob_eff, acc_tox, acc_eff)),
        None,
    )
    trial.update([(2, 1, 0), (2, 0, 0), (2, 0, 0)])
    assert trial.next_dose() == 2

    prob_eff = [0.1, 0.5, 0.6, 0.55, 0.5]
    mock_post_probs.return_value = (
        list(zip(prob_tox, prob_eff, acc_tox, acc_eff)),
        None,
    )
    trial.update([(2, 0, 0)] * 3)
    assert trial.next_dose() == 3
