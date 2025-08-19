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
    sidebar = types.SimpleNamespace()
    sidebar.header = MagicMock()
    sidebar.write = MagicMock()
    sidebar.json = MagicMock()
    sidebar.selectbox = MagicMock()
    sidebar.file_uploader = MagicMock()
    sidebar.success = MagicMock()
    st.sidebar = sidebar
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

    main = reload_module("clintrials.dashboard.main")
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

    main = reload_module("clintrials.dashboard.main")
    monkeypatch.setattr(main.crm_view, "render", MagicMock())
    monkeypatch.setattr(main.efftox_view, "render", MagicMock())
    main.main()
    main.efftox_view.render.assert_called_once_with(sims)
    main.crm_view.render.assert_not_called()


def test_crm_render_creates_plot(fake_streamlit, fake_plotly, monkeypatch):
    crm_view = reload_module("clintrials.dashboard.views.crm_view")
    monkeypatch.setattr(crm_view, "st", fake_streamlit)
    monkeypatch.setattr(crm_view, "px", fake_plotly)

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
    monkeypatch.setattr(crm_view, "summarise_sims", summarise)

    sims = [{"recommended_dose": 1}, {"recommended_dose": 2}]
    crm_view.render(sims)
    summarise.assert_called_once()
    assert fake_plotly.bar.called
    fake_streamlit.plotly_chart.assert_called_once_with("bar_fig")


def test_efftox_render_creates_plots(fake_streamlit, fake_plotly, monkeypatch):
    efftox_view = reload_module("clintrials.dashboard.views.efftox_view")
    monkeypatch.setattr(efftox_view, "st", fake_streamlit)
    monkeypatch.setattr(efftox_view, "px", fake_plotly)

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
    monkeypatch.setattr(efftox_view, "summarise_sims", summarise)

    sims = [{"recommended_dose": 1, "prob_accept_tox": 0.6, "prob_accept_eff": 0.7}]
    efftox_view.render(sims)
    summarise.assert_called_once()
    assert fake_plotly.bar.called
    assert fake_plotly.line.called
    assert fake_streamlit.plotly_chart.call_count == 2


def test_crm_render_warning_branch(fake_streamlit, fake_plotly, monkeypatch):
    crm_view = reload_module("clintrials.dashboard.views.crm_view")
    monkeypatch.setattr(crm_view, "st", fake_streamlit)

    class DummyPS:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(crm_view, "ParameterSpace", DummyPS)
    summary_df = pd.DataFrame({"N": [2]}, index=pd.Index([0.1], name="true_tox"))
    summarise = MagicMock(return_value=summary_df)
    monkeypatch.setattr(crm_view, "summarise_sims", summarise)
    sims = [{"recommended_dose": 1}]
    crm_view.render(sims)
    fake_streamlit.warning.assert_called_once()
