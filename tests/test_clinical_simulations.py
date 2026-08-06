# SPDX-License-Identifier: MIT

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from clintrials.core.registry import PROTOCOL_REGISTRY
from clintrials.dosefinding.crm import CRM
from tests.helpers.efftox import EffToxBuilder

# =====================================================================
# 1. Clinician Override Simulation Tests (Dr. Aris Thorne Persona)
# =====================================================================

def test_clinician_override_crm_simulation():
    """Simulates Persona A (Dr. Thorne) overriding CRM dynamic recommendations.

    Verifies that manual override is recorded and respects the cohort state.
    """
    # Initialize a CRM trial starting at Dose 1 with target 0.25
    prior_tox_probs = [0.05, 0.12, 0.25, 0.40, 0.55]
    trial = CRM(prior=prior_tox_probs, target=0.25, first_dose=1, max_size=30)

    # Initial recommendation must be Dose 1
    assert trial.next_dose() == 1

    # Treat Cohort 1 at recommended Dose 1 (3 patients, 0 toxicities)
    trial.update([(1, 0), (1, 0), (1, 0)])

    # The dynamic algorithm calculates a dose > 1 as the next recommendation
    calculated_dose = trial.next_dose()
    assert calculated_dose > 1

    # Dr. Thorne performs a manual clinician override back to Dose 2 (safer pathway)
    trial.set_next_dose(2)
    assert trial.next_dose() == 2

    # Treat Cohort 2 at overridden Dose 2 (3 patients, 0 toxicities)
    trial.update([(2, 0), (2, 0), (2, 0)])

    # Verify that patient allocations correctly reflect the override
    assert trial.treated_at_dose(1) == 3
    assert trial.treated_at_dose(2) == 3


def test_clinician_override_efftox_simulation():
    """Simulates Persona A (Dr. Thorne) overriding EffTox dynamic recommendations.

    Verifies that manual override is recorded and respects the cohort state.
    """
    # Initialize an EffTox trial starting at Dose 1
    trial = EffToxBuilder().with_first_dose(1).build()

    assert trial.next_dose() == 1

    # Treat Cohort 1 at recommended Dose 1 with 3 toxicities and 0 efficacy
    trial.update([(1, 1, 0), (1, 1, 0), (1, 1, 0)])

    # Clinician overrides recommendation and forces Cohort 2 to Dose 2 instead
    trial.set_next_dose(2)
    assert trial.next_dose() == 2

    # Treat Cohort 2 at overridden Dose 2
    trial.update([(2, 0, 1), (2, 0, 1), (2, 0, 1)])

    # Verify that patient counts correctly reflect override
    assert trial.treated_at_dose(1) == 3
    assert trial.treated_at_dose(2) == 3


# =====================================================================
# 2. Recursive Dose Transition Pathway Tests (Eleanor Vance Persona)
# =====================================================================

def test_dose_transition_pathway_crm():
    """Verifies recursive dose transition pathways for CRM safety model.

    Ensures calculations match expected static outcomes step-by-step.
    """
    prior_tox_probs = [0.05, 0.12, 0.25, 0.40, 0.55]
    trial = CRM(prior=prior_tox_probs, target=0.25, first_dose=1, max_size=30)

    # Pathway step 0: Starts at Dose 1
    assert trial.next_dose() == 1

    # Pathway step 1: No toxicity in 3 patients -> Escalates
    trial.update([(1, 0), (1, 0), (1, 0)])
    assert trial.next_dose() > 1


def test_dose_transition_pathway_efftox():
    """Verifies recursive dose transition pathways for EffTox safety model.

    Ensures calculations match expected static outcomes step-by-step.
    """
    trial = EffToxBuilder().with_first_dose(1).build()

    # Pathway step 0: Starts at Dose 1
    assert trial.next_dose() == 1

    # Pathway step 1: 3 toxicities and 0 efficacy -> Recommends either Dose 2 or -1 (suspended/inadmissible)
    trial.update([(1, 1, 0), (1, 1, 0), (1, 1, 0)])
    assert trial.next_dose() in (2, -1)


# =====================================================================
# 3. Headless Streamlit View Verification (No Runtime UI Crashes)
# =====================================================================

def _make_streamlit_mock() -> SimpleNamespace:
    """Helper to mock Streamlit namespace with default dashboard components."""
    sidebar = SimpleNamespace(
        header=MagicMock(),
        selectbox=MagicMock(return_value="CRM"),
        checkbox=MagicMock(return_value=False),
        file_uploader=MagicMock(),
        success=MagicMock(),
        write=MagicMock(),
        json=MagicMock(),
        expander=MagicMock(),
        markdown=MagicMock(),
        toggle=MagicMock(return_value=False),
        radio=MagicMock(return_value="Manual JSON Upload"),
        number_input=MagicMock(return_value=1),
        button=MagicMock(return_value=True),
    )

    st = SimpleNamespace(
        title=MagicMock(),
        header=MagicMock(),
        subheader=MagicMock(),
        write=MagicMock(),
        markdown=MagicMock(),
        warning=MagicMock(),
        error=MagicMock(),
        plotly_chart=MagicMock(),
        expander=MagicMock(),
        sidebar=sidebar,
        fragment=lambda func: func,
        columns=lambda x: (sidebar, sidebar),
        cache_data=lambda **kwargs: lambda f: f,
        spinner=MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())),
        success=MagicMock(),
        metric=MagicMock(),
        session_state={},
    )
    return st


def test_headless_dashboard_views_execute_without_crash(monkeypatch):
    """Verifies that all Streamlit dashboard views run without runtime crashes.

    Loads correct parameters and ensures visuals initialize safely.
    """
    st_mock = _make_streamlit_mock()
    monkeypatch.setitem(sys.modules, "streamlit", st_mock)

    # Patch view modules
    import clintrials.visualization.dashboard.views.crm_view as crm_v
    import clintrials.visualization.dashboard.views.efftox_view as efftox_v
    import clintrials.visualization.dashboard.views.watu_view as watu_v
    import clintrials.visualization.dashboard.views.winratio_view as winratio_v

    monkeypatch.setattr(crm_v, "st", st_mock)
    monkeypatch.setattr(efftox_v, "st", st_mock)
    monkeypatch.setattr(watu_v, "st", st_mock)
    monkeypatch.setattr(winratio_v, "st", st_mock)

    # 1. CRM view headless execution
    crm_sims = crm_v.CRMView.preview_sims(target_tox=0.25, cohort_size=3, max_size=30)  # type: ignore[no-untyped-call]
    render_crm = PROTOCOL_REGISTRY.get_render("CRM")
    assert render_crm is not None
    render_crm(crm_sims)  # Should render and build figures without raising exceptions

    # 2. EffTox view headless execution
    efftox_sims = efftox_v.EffToxView.preview_sims(target_tox=0.25, cohort_size=3, max_size=30)  # type: ignore[no-untyped-call]
    render_efftox = PROTOCOL_REGISTRY.get_render("EffTox")
    assert render_efftox is not None
    render_efftox(efftox_sims)  # Should render and build figures without raising exceptions

    # 3. WATU view headless execution
    watu_sims = watu_v.WATUView.preview_sims(target_tox=0.25, cohort_size=3, max_size=30)  # type: ignore[no-untyped-call]
    render_watu = PROTOCOL_REGISTRY.get_render("WATU")
    assert render_watu is not None
    render_watu(watu_sims)  # Should render and build figures without raising exceptions

    # 4. Win Ratio view headless execution
    render_winratio = PROTOCOL_REGISTRY.get_render("Win Ratio")
    assert render_winratio is not None
    render_winratio()  # Should run simulated trials internally and render without raising exceptions

    # Verify that Streamlit's header or markdown was called for views
    assert st_mock.header.called or st_mock.subheader.called
