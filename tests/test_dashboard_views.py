import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

from clintrials.visualization.dashboard import main
from clintrials.visualization.dashboard.views import (
    crm_view,
    efftox_view,
    watu_view,
    winratio_view,
)


def _make_streamlit_mock(selectbox_return="CRM", file_data=None):  # type: ignore
    """Create a minimal mock of the streamlit module."""

    class DummyFile:
        def __init__(self, data):  # type: ignore
            self._data = data

        def getvalue(self):  # type: ignore
            return self._data

    if file_data is None:
        file_data = json.dumps([{"foo": "bar"}]).encode("utf-8")

    sidebar = SimpleNamespace(
        header=MagicMock(),
        selectbox=MagicMock(return_value=selectbox_return),
        checkbox=MagicMock(return_value=False),
        file_uploader=MagicMock(return_value=DummyFile(file_data)),  # type: ignore
        success=MagicMock(),
        write=MagicMock(),
        json=MagicMock(),
        expander=MagicMock(),
        markdown=MagicMock(),
        toggle=MagicMock(return_value=False),
        radio=MagicMock(return_value="Manual JSON Upload"),
        number_input=MagicMock(return_value=1),
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
        cache_data=lambda **kwargs: lambda f: f,
        spinner=MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())),
        session_state={},
    )
    return st


def _make_winratio_streamlit_mock():  # type: ignore
    """Create a minimal mock for the Win Ratio view."""

    sidebar = SimpleNamespace(
        header=MagicMock(),
        number_input=MagicMock(
            side_effect=[100, 50, 1000, 0.5, 0.5, 0.75, 0.25, 0.43, 0.27, 0.05]
        ),
        button=MagicMock(return_value=True),
    )

    class DummySpinner:
        def __enter__(self):  # type: ignore
            return None

        def __exit__(self, exc_type, exc, tb):  # type: ignore
            return None

    st = SimpleNamespace(
        header=MagicMock(),
        sidebar=sidebar,
        spinner=lambda msg: DummySpinner(),
        success=MagicMock(),
        subheader=MagicMock(),
        write=MagicMock(),
        fragment=lambda func: func,
        metric=MagicMock(),
        plotly_chart=MagicMock(),
        markdown=MagicMock(),
        expander=MagicMock(),
        session_state={},
    )
    return st


def test_dashboard_main_routes_to_crm(monkeypatch):  # type: ignore
    """main() should invoke crm_view.render when CRM is selected."""
    st_mock = _make_streamlit_mock(selectbox_return="CRM")  # type: ignore
    monkeypatch.setattr(main, "st", st_mock)

    called = {}

    def fake_render(data):  # type: ignore
        called["data"] = data

    monkeypatch.setitem(main.PROTOCOL_REGISTRY._designs["CRM"], "render", fake_render)  # type: ignore
    main.main()  # type: ignore
    assert called["data"] == [{"foo": "bar"}]


def test_dashboard_main_routes_to_efftox(monkeypatch):  # type: ignore
    """main() should invoke efftox_view.render when EffTox is selected."""
    st_mock = _make_streamlit_mock(selectbox_return="EffTox")  # type: ignore
    monkeypatch.setattr(main, "st", st_mock)

    called = {}

    def fake_render(data):  # type: ignore
        called["data"] = data

    monkeypatch.setitem(main.PROTOCOL_REGISTRY._designs["EffTox"], "render", fake_render)  # type: ignore
    main.main()  # type: ignore
    assert called["data"] == [{"foo": "bar"}]


def test_dashboard_main_routes_to_watu(monkeypatch):  # type: ignore
    """main() should invoke watu_view.render when WATU is selected."""
    st_mock = _make_streamlit_mock(selectbox_return="WATU")  # type: ignore
    monkeypatch.setattr(main, "st", st_mock)

    called = {}

    def fake_render(data):  # type: ignore
        called["data"] = data

    monkeypatch.setitem(main.PROTOCOL_REGISTRY._designs["WATU"], "render", fake_render)  # type: ignore
    main.main()  # type: ignore
    assert called["data"] == [{"foo": "bar"}]


def test_dashboard_main_routes_to_winratio(monkeypatch):  # type: ignore
    """main() should invoke winratio_view.render when Win Ratio is selected."""
    st_mock = _make_streamlit_mock(selectbox_return="Win Ratio")  # type: ignore
    monkeypatch.setattr(main, "st", st_mock)

    called = {}

    def fake_render():  # type: ignore
        called["called"] = True

    monkeypatch.setitem(main.PROTOCOL_REGISTRY._designs["Win Ratio"], "render", fake_render)  # type: ignore
    main.main()  # type: ignore
    assert called["called"]


def test_crm_view_render_success(monkeypatch):  # type: ignore
    """render() should summarise simulations and plot results when data is valid."""
    import sys

    from clintrials.core.registry import PROTOCOL_REGISTRY

    st_mock = _make_streamlit_mock()  # type: ignore
    monkeypatch.setitem(sys.modules, "streamlit", st_mock)

    import clintrials.core.simulation as sim
    import clintrials.utils as utils
    monkeypatch.setattr(utils, "ParameterSpace", MagicMock())
    monkeypatch.setattr(crm_view, "st", st_mock)

    summary_df = pd.DataFrame(
        {
            "N": [2],
            "recommended_dose_prob": [{1: 0.5, 2: 0.5}],
        },
        index=pd.Index([0.1], name="true_tox"),
    )
    summarise_mock = MagicMock(return_value=summary_df)
    monkeypatch.setattr(sim, "extract_sim_data", summarise_mock)

    bar_fig = object()
    import clintrials.visualization as viz

    monkeypatch.setattr(
        viz, "plot_crm_simulation_recommendation", MagicMock(return_value=bar_fig)
    )

    sims = [{"recommended_dose": 1}, {"recommended_dose": 2}]
    render_func = PROTOCOL_REGISTRY.get_render("CRM")
    render_func(sims)

    summarise_mock.assert_called_once()
    viz.plot_crm_simulation_recommendation.assert_called_once()  # type: ignore
    st_mock.plotly_chart.assert_called_with(bar_fig)


def test_crm_view_warns_without_recommended(monkeypatch):  # type: ignore
    """If the summary lacks recommendation information a warning is shown."""
    import sys

    import clintrials.visualization.dashboard.views.crm_view as crm_view
    from clintrials.core.registry import PROTOCOL_REGISTRY

    st_mock = _make_streamlit_mock()
    monkeypatch.setitem(sys.modules, "streamlit", st_mock)
    import importlib
    importlib.reload(crm_view)
    monkeypatch.setattr(crm_view, "st", st_mock)
    import clintrials.core.simulation as sim
    import clintrials.utils as utils
    monkeypatch.setattr(utils, "ParameterSpace", MagicMock())

    summary_df = pd.DataFrame({"N": [1]}, index=pd.Index([0.1], name="true_tox"))
    monkeypatch.setattr(sim, "extract_sim_data", MagicMock(return_value=summary_df))

    render_func = PROTOCOL_REGISTRY.get_render("CRM")
    render_func([{}])
    st_mock.warning.assert_called_once()


def test_efftox_view_render_success(monkeypatch):  # type: ignore
    """EffTox view should plot recommendation and acceptability probabilities."""
    import sys

    from clintrials.core.registry import PROTOCOL_REGISTRY

    st_mock = _make_streamlit_mock()  # type: ignore
    monkeypatch.setitem(sys.modules, "streamlit", st_mock)

    import clintrials.core.simulation as sim
    import clintrials.utils as utils
    monkeypatch.setattr(utils, "ParameterSpace", MagicMock())
    monkeypatch.setattr(efftox_view, "st", st_mock)

    index = pd.MultiIndex.from_tuples(
        [(0.1, 0.2)], names=["true_prob_tox", "true_prob_eff"]
    )
    summary_df = pd.DataFrame(
        {
            "N": [1],
            "recommended_dose_prob": [{1: 1.0}],
            "prob_accept_tox": [0.6],
            "prob_accept_eff": [0.7],
        },
        index=index,
    )
    monkeypatch.setattr(
        sim, "extract_sim_data", MagicMock(return_value=summary_df)
    )

    import clintrials.visualization as viz

    bar_mock = MagicMock(return_value="fig_bar")
    line_mock = MagicMock(return_value="fig_line")
    monkeypatch.setattr(viz, "plot_bivariate_simulation_recommendation", bar_mock)
    monkeypatch.setattr(viz, "plot_efftox_simulation_acceptability", line_mock)

    render_func = PROTOCOL_REGISTRY.get_render("EffTox")
    render_func([{}])

    bar_mock.assert_called_once()
    line_mock.assert_called_once()
    # st.plotly_chart called for both figures
    assert st_mock.plotly_chart.call_count == 2


def test_efftox_view_warns_when_empty(monkeypatch):  # type: ignore
    """If the summary dataframe is empty a warning is shown."""
    import sys

    import clintrials.visualization.dashboard.views.efftox_view as efftox_view
    from clintrials.core.registry import PROTOCOL_REGISTRY

    st_mock = _make_streamlit_mock()
    monkeypatch.setitem(sys.modules, "streamlit", st_mock)
    import importlib
    importlib.reload(efftox_view)
    monkeypatch.setattr(efftox_view, "st", st_mock)
    import clintrials.core.simulation as sim
    import clintrials.utils as utils
    monkeypatch.setattr(utils, "ParameterSpace", MagicMock())
    monkeypatch.setattr(
        sim, "extract_sim_data", MagicMock(return_value=pd.DataFrame())
    )

    render_func = PROTOCOL_REGISTRY.get_render("EffTox")
    render_func([{}])
    st_mock.warning.assert_called_once()


def test_winratio_view_render_success(monkeypatch):  # type: ignore
    """Win Ratio view should run the simulation and display results."""
    import importlib
    import sys

    st_mock = _make_winratio_streamlit_mock()  # type: ignore
    monkeypatch.setitem(sys.modules, "streamlit", st_mock)
    importlib.reload(winratio_view)
    monkeypatch.setattr(winratio_view, "st", st_mock)
    run_sim = MagicMock()
    run_sim.return_value = {
        "power": 0.8,
        "average_ci": (0.1, 0.2),
        "results": []
    }
    monkeypatch.setattr(winratio_view, "run_winratio_simulations", run_sim)

    winratio_view.render()

    run_sim.assert_called_once_with(
        num_subjects_A=100,
        num_subjects_B=50,
        num_simulations=1000,
        p_y1_A=0.5,
        p_y1_B=0.5,
        p_y2_A=0.75,
        p_y2_B=0.25,
        p_y3_A=0.43,
        p_y3_B=0.27,
        significance_level=0.05,
    )
    st_mock.success.assert_called_once()
    st_mock.metric.assert_any_call(label="Power", value="0.8000")
    st_mock.metric.assert_any_call(
        label="Average 95% Confidence Interval", value="(0.1000, 0.2000)"
    )


def test_watu_view_render_success(monkeypatch):  # type: ignore
    """WATU view should plot recommendation probabilities."""
    import sys

    from clintrials.core.registry import PROTOCOL_REGISTRY

    st_mock = _make_streamlit_mock()  # type: ignore
    monkeypatch.setitem(sys.modules, "streamlit", st_mock)

    import clintrials.core.simulation as sim
    import clintrials.utils as utils
    monkeypatch.setattr(utils, "ParameterSpace", MagicMock())
    monkeypatch.setattr(watu_view, "st", st_mock)

    index = pd.MultiIndex.from_tuples(
        [(0.1, 0.2)], names=["true_prob_tox", "true_prob_eff"]
    )
    summary_df = pd.DataFrame(
        {
            "N": [1],
            "recommended_dose_prob": [{1: 1.0}],
        },
        index=index,
    )
    monkeypatch.setattr(
        sim, "extract_sim_data", MagicMock(return_value=summary_df)
    )

    import clintrials.visualization as viz

    bar_mock = MagicMock(return_value="fig_bar")
    monkeypatch.setattr(viz, "plot_bivariate_simulation_recommendation", bar_mock)

    render_func = PROTOCOL_REGISTRY.get_render("WATU")
    render_func([{}])

    bar_mock.assert_called_once()
    # st.plotly_chart called once
    assert st_mock.plotly_chart.call_count == 1


def test_watu_view_warns_when_empty(monkeypatch):  # type: ignore
    """If the summary dataframe is empty a warning is shown."""
    import sys

    import clintrials.visualization.dashboard.views.watu_view as watu_view
    from clintrials.core.registry import PROTOCOL_REGISTRY

    st_mock = _make_streamlit_mock()
    monkeypatch.setitem(sys.modules, "streamlit", st_mock)
    import importlib
    importlib.reload(watu_view)
    monkeypatch.setattr(watu_view, "st", st_mock)
    import clintrials.core.simulation as sim
    import clintrials.utils as utils
    monkeypatch.setattr(utils, "ParameterSpace", MagicMock())
    monkeypatch.setattr(
        sim, "extract_sim_data", MagicMock(return_value=pd.DataFrame())
    )

    render_func = PROTOCOL_REGISTRY.get_render("WATU")
    render_func([{}])
    st_mock.warning.assert_called_once()


def test_main_preview_mode_crm(monkeypatch):  # type: ignore
    st_mock = _make_streamlit_mock(selectbox_return="CRM")  # type: ignore
    st_mock.sidebar.radio.return_value = "Preview Mode"
    monkeypatch.setattr(main, "st", st_mock)

    monkeypatch.setitem(main.PROTOCOL_REGISTRY._designs["CRM"], "render", MagicMock())  # type: ignore
    monkeypatch.setattr(main, "get_preview_sims", MagicMock(return_value=[{"preview": True}]))
    main.main()  # type: ignore
    main.get_preview_sims.assert_called_once()
    main.PROTOCOL_REGISTRY.get_render("CRM").assert_called_once_with([{"preview": True}])  # type: ignore


def test_main_preview_mode_exception(monkeypatch):  # type: ignore
    st_mock = _make_streamlit_mock(selectbox_return="CRM")  # type: ignore
    st_mock.sidebar.radio.return_value = "Preview Mode"
    monkeypatch.setattr(main, "st", st_mock)

    def raise_err(*args, **kwargs):  # type: ignore
        raise ValueError("Sim error")

    monkeypatch.setattr(main, "get_preview_sims", raise_err)
    main.main()  # type: ignore
    st_mock.error.assert_called_once()


def test_get_preview_sims_crm(monkeypatch):  # type: ignore
    import clintrials.dosefinding as df
    func = getattr(main.get_preview_sims, "__wrapped__", main.get_preview_sims)

    mock_sim = MagicMock(side_effect=lambda *args, **kwargs: {"recommended_dose": 1})
    monkeypatch.setattr(df, "simulate_dose_finding_trial", mock_sim)

    sims = func("CRM", target_tox=0.25, cohort_size=3, max_size=10)

    assert len(sims) == 40
    assert mock_sim.call_count == 40
    assert sims[0]["true_tox"] == (0.05, 0.1, 0.2, 0.3, 0.4)


def test_get_preview_sims_efftox(monkeypatch):  # type: ignore
    import clintrials.dosefinding.efficacytoxicity as et
    func = getattr(main.get_preview_sims, "__wrapped__", main.get_preview_sims)

    mock_sim = MagicMock(side_effect=lambda *args, **kwargs: {"recommended_dose": 2})
    monkeypatch.setattr(et, "simulate_trial", mock_sim)

    sims = func("EffTox", target_tox=0.25, cohort_size=3, max_size=10)

    assert len(sims) == 10
    assert mock_sim.call_count == 10


def test_get_preview_sims_watu(monkeypatch):  # type: ignore
    import clintrials.dosefinding.efficacytoxicity as et
    func = getattr(main.get_preview_sims, "__wrapped__", main.get_preview_sims)

    mock_sim = MagicMock(side_effect=lambda *args, **kwargs: {"recommended_dose": 3})
    monkeypatch.setattr(et, "simulate_trial", mock_sim)

    sims = func("WATU", target_tox=0.25, cohort_size=3, max_size=10)

    assert len(sims) == 10
    assert mock_sim.call_count == 10
