import sys
import types
from unittest.mock import MagicMock
import pytest
import importlib

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
    st.cache_data = lambda **kwargs: lambda f: f
    st.spinner = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    st.fragment = lambda func: func
    
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
    sidebar.radio = MagicMock(return_value="Preview Mode")
    sidebar.number_input = MagicMock(return_value=10)
    st.sidebar = sidebar
    monkeypatch.setitem(sys.modules, "streamlit", st)
    return st

def reload_module(name):
    if name in sys.modules:
        importlib.reload(sys.modules[name])
    return importlib.import_module(name)

def test_main_preview_crm(fake_streamlit, monkeypatch):
    fake_streamlit.sidebar.selectbox.return_value = "CRM"
    main = reload_module("clintrials.visualization.dashboard.main")
    
    # Mock simulate to be fast
    def mock_simulate(*args, **kwargs):
        return {"recommended_dose": 1}
    monkeypatch.setattr(main, "get_preview_sims", MagicMock(return_value=[{"recommended_dose": 1}]))
    monkeypatch.setitem(main.PROTOCOL_REGISTRY._designs["CRM"], "render", MagicMock())
    
    main.main()
    main.PROTOCOL_REGISTRY.get_render("CRM").assert_called_once()
    
def test_get_preview_sims_crm():
    main = reload_module("clintrials.visualization.dashboard.main")
    sims = main.get_preview_sims("CRM", target_tox=0.25, cohort_size=3, max_size=10)
    assert len(sims) > 0
    
def test_get_preview_sims_efftox():
    main = reload_module("clintrials.visualization.dashboard.main")
    sims = main.get_preview_sims("EffTox", target_tox=0.25, cohort_size=3, max_size=10)
    assert len(sims) > 0
    
def test_get_preview_sims_watu():
    main = reload_module("clintrials.visualization.dashboard.main")
    sims = main.get_preview_sims("WATU", target_tox=0.25, cohort_size=3, max_size=10)
    assert len(sims) > 0
