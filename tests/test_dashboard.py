import importlib
import json
import sys
import types
from unittest.mock import MagicMock

import pandas as pd
import pytest


@pytest.fixture
def fake_streamlit(monkeypatch):
    st = types.SimpleNamespace()
    st.title = MagicMock()
    st.header = MagicMock()
    st.subheader = MagicMock()
    st.write = MagicMock()
    st.warning = MagicMock()
    st.error = MagicMock()
    st.plotly_chart = MagicMock()
    st.session_state = {}
    st.columns = lambda x: (MagicMock(), MagicMock())
    st.download_button = MagicMock()
    st.expander = MagicMock()
    st.markdown = MagicMock()
    st.session_state = {}
    st.columns = lambda x: (MagicMock(), MagicMock())
    st.download_button = MagicMock()
    st.session_state = {}
    st.columns = MagicMock(return_value=(MagicMock(), MagicMock()))
    st.download_button = MagicMock()
    sidebar = types.SimpleNamespace()
    sidebar.header = MagicMock()
    sidebar.write = MagicMock()
    sidebar.json = MagicMock()
    sidebar.selectbox = MagicMock()
    sidebar.file_uploader = MagicMock()
    sidebar.success = MagicMock()
    sidebar.expander = MagicMock()
    sidebar.markdown = MagicMock()
    sidebar.toggle = MagicMock(return_value=False)
    sidebar.checkbox = MagicMock(return_value=False)
    sidebar.radio = MagicMock(return_value="Manual JSON Upload")
    sidebar.number_input = MagicMock(return_value=1)
    st.sidebar = sidebar
    st.cache_data = lambda **kwargs: lambda f: f
    st.spinner = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    st.fragment = lambda func: func
    monkeypatch.setitem(sys.modules, "streamlit", st)
    return st


@pytest.fixture
def fake_plotly(monkeypatch):
    px = types.SimpleNamespace()
    px.bar = MagicMock(return_value="bar_fig")
    px.line = MagicMock(return_value="line_fig")
    go = types.SimpleNamespace()
    plotly = types.SimpleNamespace(express=px, graph_objects=go)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", go)
    monkeypatch.setitem(sys.modules, "plotly", plotly)
    monkeypatch.setitem(sys.modules, "plotly.express", px)
    return px


def reload_module(name):
    if name in sys.modules:
        importlib.reload(sys.modules[name])
    return importlib.import_module(name)


def test_main_dispatches_to_crm(fake_streamlit, fake_plotly, monkeypatch):
    fake_streamlit.sidebar.selectbox.return_value = "CRM"
    sims = [{"recommended_dose": 1}]

    class DummyFile:
        def getvalue(self):
            return json.dumps(sims).encode("utf-8")

    fake_streamlit.sidebar.file_uploader.return_value = DummyFile()

    main = reload_module("clintrials.visualization.dashboard.main")
    monkeypatch.setitem(main.PROTOCOL_REGISTRY._designs["CRM"], "render", MagicMock())
    monkeypatch.setitem(main.PROTOCOL_REGISTRY._designs["EffTox"], "render", MagicMock())
    main.main()
    main.PROTOCOL_REGISTRY.get_render("CRM").assert_called_once_with(sims)
    main.PROTOCOL_REGISTRY.get_render("EffTox").assert_not_called()


def test_main_dispatches_to_efftox(fake_streamlit, fake_plotly, monkeypatch):
    fake_streamlit.sidebar.selectbox.return_value = "EffTox"
    sims = [{"recommended_dose": 1}]

    class DummyFile:
        def getvalue(self):
            return json.dumps(sims).encode("utf-8")

    fake_streamlit.sidebar.file_uploader.return_value = DummyFile()

    main = reload_module("clintrials.visualization.dashboard.main")
    monkeypatch.setitem(main.PROTOCOL_REGISTRY._designs["CRM"], "render", MagicMock())
    monkeypatch.setitem(main.PROTOCOL_REGISTRY._designs["EffTox"], "render", MagicMock())
    main.main()
    main.PROTOCOL_REGISTRY.get_render("EffTox").assert_called_once_with(sims)
    main.PROTOCOL_REGISTRY.get_render("CRM").assert_not_called()

def test_main_dispatches_to_watu(fake_streamlit, fake_plotly, monkeypatch):
    fake_streamlit.sidebar.selectbox.return_value = "WATU"
    sims = [{"recommended_dose": 1}]

    class DummyFile:
        def getvalue(self):
            return json.dumps(sims).encode("utf-8")

    fake_streamlit.sidebar.file_uploader.return_value = DummyFile()

    main = reload_module("clintrials.visualization.dashboard.main")
    monkeypatch.setitem(main.PROTOCOL_REGISTRY._designs["WATU"], "render", MagicMock())
    main.main()
    main.PROTOCOL_REGISTRY.get_render("WATU").assert_called_once_with(sims)


def test_main_dispatches_to_winratio(fake_streamlit, fake_plotly, monkeypatch):
    fake_streamlit.sidebar.selectbox.return_value = "Win Ratio"

    main = reload_module("clintrials.visualization.dashboard.main")
    monkeypatch.setitem(main.PROTOCOL_REGISTRY._designs["CRM"], "render", MagicMock())
    monkeypatch.setitem(main.PROTOCOL_REGISTRY._designs["EffTox"], "render", MagicMock())
    monkeypatch.setitem(main.PROTOCOL_REGISTRY._designs["Win Ratio"], "render", MagicMock())
    main.main()
    main.PROTOCOL_REGISTRY.get_render("Win Ratio").assert_called_once()
    main.PROTOCOL_REGISTRY.get_render("CRM").assert_not_called()
    main.PROTOCOL_REGISTRY.get_render("EffTox").assert_not_called()


def test_main_preview_mode_crm(fake_streamlit, fake_plotly, monkeypatch):
    fake_streamlit.sidebar.selectbox.return_value = "CRM"
    fake_streamlit.sidebar.radio.return_value = "Preview Mode"

    main = reload_module("clintrials.visualization.dashboard.main")
    monkeypatch.setitem(main.PROTOCOL_REGISTRY._designs["CRM"], "render", MagicMock())
    monkeypatch.setattr(main, "get_preview_sims", MagicMock(return_value=[{"preview": True}]))
    main.main()
    main.get_preview_sims.assert_called_once()
    main.PROTOCOL_REGISTRY.get_render("CRM").assert_called_once_with([{"preview": True}])

def test_main_preview_mode_exception(fake_streamlit, fake_plotly, monkeypatch):
    fake_streamlit.sidebar.selectbox.return_value = "CRM"
    fake_streamlit.sidebar.radio.return_value = "Preview Mode"
    
    main = reload_module("clintrials.visualization.dashboard.main")
    
    def raise_err(*args, **kwargs):
        raise ValueError("Sim error")
        
    monkeypatch.setattr(main, "get_preview_sims", raise_err)
    main.main()
    fake_streamlit.error.assert_called_once()


def test_get_preview_sims_crm(monkeypatch):
    main = reload_module("clintrials.visualization.dashboard.main")
    # Bypass the cache decorator by calling __wrapped__ if present, else call directly
    func = getattr(main.get_preview_sims, "__wrapped__", main.get_preview_sims)
    
    # Mock simulate_dose_finding_trial to speed up the test
    import clintrials.dosefinding as df
    mock_sim = MagicMock(side_effect=lambda *args, **kwargs: {"recommended_dose": 1})
    monkeypatch.setattr(df, "simulate_dose_finding_trial", mock_sim)
    
    # Call the original wrapped function
    sims = func("CRM", target_tox=0.25, cohort_size=3, max_size=10)
    
    assert len(sims) == 40
    assert mock_sim.call_count == 40
    assert sims[0]["true_tox"] == (0.05, 0.1, 0.2, 0.3, 0.4)

def test_get_preview_sims_efftox(monkeypatch):
    main = reload_module("clintrials.visualization.dashboard.main")
    func = getattr(main.get_preview_sims, "__wrapped__", main.get_preview_sims)
    
    # Mock simulate_trial
    import clintrials.dosefinding.efficacytoxicity as et
    mock_sim = MagicMock(side_effect=lambda *args, **kwargs: {"recommended_dose": 2})
    monkeypatch.setattr(et, "simulate_trial", mock_sim)
    
    sims = func("EffTox", target_tox=0.25, cohort_size=3, max_size=10)
    
    assert len(sims) == 10
    assert mock_sim.call_count == 10

def test_get_preview_sims_watu(monkeypatch):
    main = reload_module("clintrials.visualization.dashboard.main")
    func = getattr(main.get_preview_sims, "__wrapped__", main.get_preview_sims)
    
    import clintrials.dosefinding.efficacytoxicity as et
    mock_sim = MagicMock(side_effect=lambda *args, **kwargs: {"recommended_dose": 3})
    monkeypatch.setattr(et, "simulate_trial", mock_sim)
    
    sims = func("WATU", target_tox=0.25, cohort_size=3, max_size=10)
    
    assert len(sims) == 10
    assert mock_sim.call_count == 10



def test_crm_render_creates_plot(fake_streamlit, monkeypatch):
    crm_view = reload_module("clintrials.visualization.dashboard.views.crm_view")
    monkeypatch.setattr(crm_view, "st", fake_streamlit)

    import clintrials.visualization as viz

    bar_mock = MagicMock(return_value="bar_fig")
    monkeypatch.setattr(viz, "plot_crm_simulation_recommendation", bar_mock)

    class DummyPS:
        def __init__(self, *args, **kwargs):
            self.add = lambda *a, **k: None
            pass

    monkeypatch.setattr(crm_view, "ParameterSpace", DummyPS)

    summary_df = pd.DataFrame(
        {
            "N": [2],
            "recommended_dose_prob": [{1: 0.5, 2: 0.5}],
        },
        index=pd.Index([0.1], name="true_tox"),
    )
    summarise = MagicMock(return_value=summary_df)
    monkeypatch.setattr(crm_view, "extract_sim_data", summarise)

    sims = [{"recommended_dose": 1}, {"recommended_dose": 2}]
    crm_view.render(sims)
    summarise.assert_called_once()
    assert bar_mock.called
    fake_streamlit.plotly_chart.assert_called_once_with("bar_fig")


def test_efftox_render_creates_plots(fake_streamlit, monkeypatch):
    efftox_view = reload_module("clintrials.visualization.dashboard.views.efftox_view")
    monkeypatch.setattr(efftox_view, "st", fake_streamlit)

    import clintrials.visualization as viz

    bar_mock = MagicMock(return_value="bar_fig")
    line_mock = MagicMock(return_value="line_fig")
    monkeypatch.setattr(viz, "plot_efftox_simulation_recommendation", bar_mock)
    monkeypatch.setattr(viz, "plot_efftox_simulation_acceptability", line_mock)

    class DummyPS:
        def __init__(self, *args, **kwargs):
            self.add = lambda *a, **k: None
            pass

    monkeypatch.setattr(efftox_view, "ParameterSpace", DummyPS)

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
    summarise = MagicMock(return_value=summary_df)
    monkeypatch.setattr(efftox_view, "extract_sim_data", summarise)

    sims = [{"recommended_dose": 1, "prob_accept_tox": 0.6, "prob_accept_eff": 0.7}]
    efftox_view.render(sims)
    summarise.assert_called_once()
    assert bar_mock.called
    assert line_mock.called
    assert fake_streamlit.plotly_chart.call_count == 2


def test_crm_render_warning_branch(fake_streamlit, monkeypatch):
    crm_view = reload_module("clintrials.visualization.dashboard.views.crm_view")
    monkeypatch.setattr(crm_view, "st", fake_streamlit)

    class DummyPS:
        def __init__(self, *args, **kwargs):
            self.add = lambda *a, **k: None
            pass

    monkeypatch.setattr(crm_view, "ParameterSpace", DummyPS)
    summary_df = pd.DataFrame({"N": [2]}, index=pd.Index([0.1], name="true_tox"))
    summarise = MagicMock(return_value=summary_df)
    monkeypatch.setattr(crm_view, "extract_sim_data", summarise)
    sims = [{"recommended_dose": 1}]
    crm_view.render(sims)
    fake_streamlit.warning.assert_called_once()
