import numpy as np

from clintrials.core.recruitment import ConstantRecruitmentStream
from clintrials.core.tte import BayesianTimeToEvent, matrix_cohort_analysis


def test_bayesian_time_to_event_update_and_test():
    trial = BayesianTimeToEvent(alpha_prior=2, beta_prior=2)
    trial.update([(5, 0), (10, 0)])

    res = trial.test(time=5, cutoff=8, probability=0.8, less_than=True)
    assert res["Patients"] == 2
    assert res["Events"] == 0
    assert abs(res["Probability"] - 0.69301655) < 1e-6
    assert not res["Stop"]

    res = trial.test(time=12, cutoff=8, probability=0.5, less_than=False)
    assert res["Events"] == 2
    assert res["Stop"]


def test_matrix_cohort_analysis_deterministic():
    np.random.seed(0)
    stream = ConstantRecruitmentStream(1)  # type: ignore
    report = matrix_cohort_analysis(
        n_simulations=1,
        n_patients=2,
        true_median=10,
        alpha_prior=2,
        beta_prior=2,
        lower_cutoff=5,
        upper_cutoff=15,
        interim_certainty=0.6,
        final_certainty=0.6,
        interim_analysis_after_patients=[1],
        interim_analysis_time_delta=0,
        final_analysis_time_delta=0,
        recruitment_stream=stream,
    )
    assert report["Decision"] == "StopAtInterim"
    assert report["FinalPatients"] == 1


def test_matrix_cohort_analysis_multiple_runs():
    np.random.seed(1)
    stream = ConstantRecruitmentStream(1)  # type: ignore
    reports = matrix_cohort_analysis(
        n_simulations=2,
        n_patients=1,
        true_median=5,
        alpha_prior=2,
        beta_prior=2,
        lower_cutoff=2,
        upper_cutoff=10,
        interim_certainty=0.6,
        final_certainty=0.6,
        interim_analysis_after_patients=[1],
        interim_analysis_time_delta=0,
        final_analysis_time_delta=0,
        recruitment_stream=stream,
    )
    assert isinstance(reports, list)
    assert len(reports) == 2


def test_matrix_cohort_analysis_go_at_final():
    np.random.seed(0)
    stream = ConstantRecruitmentStream(1)  # type: ignore
    report = matrix_cohort_analysis(
        n_simulations=1,
        n_patients=2,
        true_median=5,
        alpha_prior=2,
        beta_prior=2,
        lower_cutoff=2,
        upper_cutoff=4,
        interim_certainty=0.2,
        final_certainty=0.1,
        interim_analysis_after_patients=[],
        interim_analysis_time_delta=0,
        final_analysis_time_delta=0,
        recruitment_stream=stream,
    )
    assert report["Decision"] == "GoAtFinal"


def test_matrix_cohort_analysis_early_stopping_and_multirun():
    np.random.seed(42)
    stream = ConstantRecruitmentStream(1)  # type: ignore

    # Run with 3 simulations, 10 patients, with interim analyses configured
    # at 3 and 6 patients. A high lower_cutoff (15.0) relative to true_median (1.0)
    # and a low interim_certainty (0.5) will trigger early stopping.
    reports = matrix_cohort_analysis(
        n_simulations=3,
        n_patients=10,
        true_median=1.0,
        alpha_prior=2.0,
        beta_prior=2.0,
        lower_cutoff=15.0,
        upper_cutoff=2.0,
        interim_certainty=0.5,
        final_certainty=0.5,
        interim_analysis_after_patients=[3, 6],
        interim_analysis_time_delta=0.0,
        final_analysis_time_delta=0.0,
        recruitment_stream=stream,
    )

    # Verify that we get a list of results equal to n_simulations even when stopping early
    assert isinstance(reports, list)
    assert len(reports) == 3

    # Verify that each simulation stopped at the interim and has the correct decision and interim details
    for report in reports:
        assert report["Decision"] == "StopAtInterim"
        # Since we stopped at interim, the number of patients in the final analysis should be less than MaxPatients
        assert report["FinalPatients"] < 10
        # Check that interim analyses were actually run and stored
        assert len(report["InterimAnalyses"]) > 0
        assert any(interim["Stop"] for interim in report["InterimAnalyses"])

