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
    st.columns = lambda x: (MagicMock(), MagicMock())
    st.download_button = MagicMock()
    st.rerun = MagicMock()
    st.experimental_rerun = MagicMock()
    sidebar = types.SimpleNamespace()
    sidebar.header = MagicMock()
    sidebar.write = MagicMock()
    sidebar.json = MagicMock()
    sidebar.selectbox = MagicMock()
    sidebar.file_uploader = MagicMock()
    sidebar.success = MagicMock()
    sidebar.expander = MagicMock()
    sidebar.button = MagicMock(return_value=False)
    st.sidebar = sidebar
    st.fragment = lambda func: func
    monkeypatch.setitem(sys.modules, "streamlit", st)
    return st


@pytest.fixture
def fake_plotly(monkeypatch):
    px = types.SimpleNamespace()
    px.bar = MagicMock(return_value="bar_fig")
    px.line = MagicMock(return_value="line_fig")
    plotly = types.SimpleNamespace(express=px)
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
    monkeypatch.setattr(main.crm_view, "render", MagicMock())
    monkeypatch.setattr(main.efftox_view, "render", MagicMock())
    main.main()
    main.crm_view.render.assert_called_once_with(sims)
    main.efftox_view.render.assert_not_called()


def test_main_dispatches_to_efftox(fake_streamlit, fake_plotly, monkeypatch):
    fake_streamlit.sidebar.selectbox.return_value = "EffTox"
    sims = [{"recommended_dose": 1}]

    class DummyFile:
        def getvalue(self):
            return json.dumps(sims).encode("utf-8")

    fake_streamlit.sidebar.file_uploader.return_value = DummyFile()

    main = reload_module("clintrials.visualization.dashboard.main")
    monkeypatch.setattr(main.crm_view, "render", MagicMock())
    monkeypatch.setattr(main.efftox_view, "render", MagicMock())
    main.main()
    main.efftox_view.render.assert_called_once_with(sims)
    main.crm_view.render.assert_not_called()


def test_main_dispatches_to_winratio(fake_streamlit, fake_plotly, monkeypatch):
    fake_streamlit.sidebar.selectbox.return_value = "Win Ratio"

    main = reload_module("clintrials.visualization.dashboard.main")
    monkeypatch.setattr(main.crm_view, "render", MagicMock())
    monkeypatch.setattr(main.efftox_view, "render", MagicMock())
    monkeypatch.setattr(main.winratio_view, "render", MagicMock())
    main.main()
    main.winratio_view.render.assert_called_once()
    main.crm_view.render.assert_not_called()
    main.efftox_view.render.assert_not_called()


def test_crm_render_creates_plot(fake_streamlit, monkeypatch):
    crm_view = reload_module("clintrials.visualization.dashboard.views.crm_view")
    monkeypatch.setattr(crm_view, "st", fake_streamlit)

    import clintrials.visualization as viz

    bar_mock = MagicMock(return_value="bar_fig")
    monkeypatch.setattr(viz, "plot_crm_simulation_recommendation", bar_mock)

    class DummyPS:
        def __init__(self, config):
            self.config = config

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
        def __init__(self, config):
            self.config = config

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
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(crm_view, "ParameterSpace", DummyPS)
    summary_df = pd.DataFrame({"N": [2]}, index=pd.Index([0.1], name="true_tox"))
    summarise = MagicMock(return_value=summary_df)
    monkeypatch.setattr(crm_view, "extract_sim_data", summarise)
    sims = [{"recommended_dose": 1}]
    crm_view.render(sims)
    fake_streamlit.warning.assert_called_once()
def test_main_persistence_logic(fake_streamlit, fake_plotly, monkeypatch):
    js_mock = types.SimpleNamespace()
    js_mock.window = types.SimpleNamespace()
    js_mock.window.localStorage = types.SimpleNamespace()
    js_mock.window.localStorage.getItem = MagicMock(return_value="true")
    js_mock.window.localStorage.setItem = MagicMock()
    js_mock.eval = MagicMock()
    
    def mock_create_proxy(f):
        return f

    pyodide_mock = types.SimpleNamespace()
    pyodide_mock.ffi = types.SimpleNamespace()
    pyodide_mock.ffi.create_proxy = mock_create_proxy

    monkeypatch.setitem(sys.modules, "js", js_mock)
    monkeypatch.setitem(sys.modules, "pyodide", pyodide_mock)
    monkeypatch.setitem(sys.modules, "pyodide.ffi", pyodide_mock.ffi)

    main = reload_module("clintrials.visualization.dashboard.main")
    monkeypatch.setattr(main.crm_view, "render", MagicMock())
    monkeypatch.setattr(main.efftox_view, "render", MagicMock())
    monkeypatch.setattr(main.watu_view, "render", MagicMock())
    monkeypatch.setattr(main.winratio_view, "render", MagicMock())
    
    # Simulate a query param override
    fake_streamlit.query_params = {"accessibility_mode": ["false"], "design_type": ["EffTox"]}
    
    class DummyFile:
        def getvalue(self):
            return json.dumps([{"recommended_dose": 1}]).encode("utf-8")
            
    fake_streamlit.sidebar.file_uploader.return_value = DummyFile()
    
    # Simulate Clear Session button press
    fake_streamlit.sidebar.button = MagicMock(return_value=True)

    main.main()
    
    # Check that persistence functions were called
    assert js_mock.window.localStorage.getItem.called
    assert js_mock.eval.called
    
    # Call the IDB load callback
    if hasattr(js_mock.window, "_on_idb_load"):
        js_mock.window._on_idb_load('[{"batch": [{"recommended_dose": 1}]}]')
    
    assert main.st.session_state["idb_loaded"]
    
    # Trigger exception path in idb
    if hasattr(js_mock.window, "_on_idb_load"):
        js_mock.window._on_idb_load('invalid json')


def test_main_persistence_fallback(fake_streamlit, fake_plotly, monkeypatch):
    js_mock = types.SimpleNamespace()
    js_mock.window = types.SimpleNamespace()
    js_mock.window.localStorage = types.SimpleNamespace()
    js_mock.window.localStorage.getItem = MagicMock(return_value="CRM")
    js_mock.window.localStorage.setItem = MagicMock()
    js_mock.eval = MagicMock()
    
    def mock_create_proxy(f):
        return f

    pyodide_mock = types.SimpleNamespace()
    pyodide_mock.ffi = types.SimpleNamespace()
    pyodide_mock.ffi.create_proxy = mock_create_proxy

    monkeypatch.setitem(sys.modules, "js", js_mock)
    monkeypatch.setitem(sys.modules, "pyodide", pyodide_mock)
    monkeypatch.setitem(sys.modules, "pyodide.ffi", pyodide_mock.ffi)

    main = reload_module("clintrials.visualization.dashboard.main")
    monkeypatch.setattr(main.crm_view, "render", MagicMock())
    monkeypatch.setattr(main.efftox_view, "render", MagicMock())
    monkeypatch.setattr(main.watu_view, "render", MagicMock())
    monkeypatch.setattr(main.winratio_view, "render", MagicMock())
    
    # Simulate a query param override with experimental API
    if hasattr(fake_streamlit, "query_params"):
        del fake_streamlit.query_params
    fake_streamlit.experimental_get_query_params = MagicMock(return_value={"accessibility_mode": ["true"], "design_type": ["CRM"]})
    
    class DummyFile:
        def getvalue(self):
            return json.dumps([{"recommended_dose": 1}]).encode("utf-8")
            
    fake_streamlit.sidebar.file_uploader.return_value = DummyFile()
    fake_streamlit.sidebar.button = MagicMock(return_value=False)

    main.main()
    
    # Hit the IDB dict structure branch
    if hasattr(js_mock.window, "_on_idb_load"):
        js_mock.window._on_idb_load('[{"batch": {"Simulations": [{"recommended_dose": 1}]}}]')
        
    # Test JS error handling during clear session
    fake_streamlit.sidebar.button = MagicMock(return_value=True)
    del fake_streamlit.rerun
    main.main()
