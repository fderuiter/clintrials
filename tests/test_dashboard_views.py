import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

from clintrials.dashboard import main
from clintrials.dashboard.views import crm_view, efftox_view, winratio_view


def _make_streamlit_mock(selectbox_return="CRM", file_data=None):
    """Create a minimal mock of the streamlit module."""

    class DummyFile:
        def __init__(self, data):
            self._data = data

        def getvalue(self):
            return self._data

    if file_data is None:
        file_data = json.dumps([{"foo": "bar"}]).encode("utf-8")

    sidebar = SimpleNamespace(
        header=MagicMock(),
        selectbox=MagicMock(return_value=selectbox_return),
        file_uploader=MagicMock(return_value=DummyFile(file_data)),
        success=MagicMock(),
        write=MagicMock(),
        json=MagicMock(),
    )

    st = SimpleNamespace(
        title=MagicMock(),
        header=MagicMock(),
        subheader=MagicMock(),
        write=MagicMock(),
        warning=MagicMock(),
        error=MagicMock(),
        plotly_chart=MagicMock(),
        sidebar=sidebar,
    )
    return st


def _make_winratio_streamlit_mock():
    """Create a minimal mock for the Win Ratio view."""

    sidebar = SimpleNamespace(
        header=MagicMock(),
        number_input=MagicMock(
            side_effect=[100, 50, 1000, 0.5, 0.5, 0.75, 0.25, 0.43, 0.27, 0.05]
        ),
        button=MagicMock(return_value=True),
    )

    class DummySpinner:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return None

    st = SimpleNamespace(
        header=MagicMock(),
        sidebar=sidebar,
        spinner=lambda msg: DummySpinner(),
        success=MagicMock(),
        subheader=MagicMock(),
        write=MagicMock(),
    )
    return st


def test_dashboard_main_routes_to_crm(monkeypatch):
    """main() should invoke crm_view.render when CRM is selected."""
    st_mock = _make_streamlit_mock(selectbox_return="CRM")
    monkeypatch.setattr(main, "st", st_mock)

    called = {}

    def fake_render(data):
        called["data"] = data

    monkeypatch.setattr(main.crm_view, "render", fake_render)
    main.main()
    assert called["data"] == [{"foo": "bar"}]


def test_dashboard_main_routes_to_efftox(monkeypatch):
    """main() should invoke efftox_view.render when EffTox is selected."""
    st_mock = _make_streamlit_mock(selectbox_return="EffTox")
    monkeypatch.setattr(main, "st", st_mock)

    called = {}

    def fake_render(data):
        called["data"] = data

    monkeypatch.setattr(main.efftox_view, "render", fake_render)
    main.main()
    assert called["data"] == [{"foo": "bar"}]


def test_dashboard_main_routes_to_winratio(monkeypatch):
    """main() should invoke winratio_view.render when Win Ratio is selected."""
    st_mock = _make_streamlit_mock(selectbox_return="Win Ratio")
    monkeypatch.setattr(main, "st", st_mock)

    called = {}

    def fake_render():
        called["called"] = True

    monkeypatch.setattr(main.winratio_view, "render", fake_render)
    main.main()
    assert called["called"]


def test_crm_view_render_success(monkeypatch):
    """render() should summarise simulations and plot results when data is valid."""
    st_mock = _make_streamlit_mock()
    monkeypatch.setattr(crm_view, "st", st_mock)
    monkeypatch.setattr(crm_view, "ParameterSpace", lambda cfg: "ps")

    summary_df = pd.DataFrame(
        {
            "N": [2],
            "recommended_dose_prob": [{1: 0.5, 2: 0.5}],
        },
        index=pd.Index([0.1], name="true_tox"),
    )
    summarise_mock = MagicMock(return_value=summary_df)
    monkeypatch.setattr(crm_view, "summarise_sims", summarise_mock)

    bar_fig = object()
    monkeypatch.setattr(crm_view.px, "bar", MagicMock(return_value=bar_fig))

    sims = [{"recommended_dose": 1}, {"recommended_dose": 2}]
    crm_view.render(sims)

    summarise_mock.assert_called_once()
    crm_view.px.bar.assert_called_once()
    st_mock.plotly_chart.assert_called_with(bar_fig)


def test_crm_view_warns_without_recommended(monkeypatch):
    """If the summary lacks recommendation information a warning is shown."""
    st_mock = _make_streamlit_mock()
    monkeypatch.setattr(crm_view, "st", st_mock)
    monkeypatch.setattr(crm_view, "ParameterSpace", lambda cfg: "ps")

    summary_df = pd.DataFrame({"N": [1]}, index=pd.Index([0.1], name="true_tox"))
    monkeypatch.setattr(crm_view, "summarise_sims", MagicMock(return_value=summary_df))

    crm_view.render([{}])
    st_mock.warning.assert_called_once()


def test_efftox_view_render_success(monkeypatch):
    """EffTox view should plot recommendation and acceptability probabilities."""
    st_mock = _make_streamlit_mock()
    monkeypatch.setattr(efftox_view, "st", st_mock)
    monkeypatch.setattr(efftox_view, "ParameterSpace", lambda cfg: "ps")

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
        efftox_view, "summarise_sims", MagicMock(return_value=summary_df)
    )

    monkeypatch.setattr(efftox_view.px, "bar", MagicMock(return_value="fig_bar"))
    monkeypatch.setattr(efftox_view.px, "line", MagicMock(return_value="fig_line"))

    efftox_view.render([{}])

    efftox_view.px.bar.assert_called_once()
    efftox_view.px.line.assert_called_once()
    # st.plotly_chart called for both figures
    assert st_mock.plotly_chart.call_count == 2


def test_efftox_view_warns_when_empty(monkeypatch):
    """If the summary dataframe is empty a warning is shown."""
    st_mock = _make_streamlit_mock()
    monkeypatch.setattr(efftox_view, "st", st_mock)
    monkeypatch.setattr(efftox_view, "ParameterSpace", lambda cfg: "ps")
    monkeypatch.setattr(
        efftox_view, "summarise_sims", MagicMock(return_value=pd.DataFrame())
    )

    efftox_view.render([{}])
    st_mock.warning.assert_called_once()


def test_winratio_view_render_success(monkeypatch):
    """Win Ratio view should run the simulation and display results."""
    st_mock = _make_winratio_streamlit_mock()
    monkeypatch.setattr(winratio_view, "st", st_mock)
    run_sim = MagicMock(return_value=(0.8, (0.1, 0.2)))
    monkeypatch.setattr(winratio_view, "run_simulation", run_sim)

    winratio_view.render()

    run_sim.assert_called_once_with(
        100, 50, 1000, 0.5, 0.5, 0.75, 0.25, 0.43, 0.27, 0.05
    )
    st_mock.success.assert_called_once()
    st_mock.write.assert_any_call("Power of the test: 0.8000")
    st_mock.write.assert_any_call("Average 95% Confidence Interval: (0.1000, 0.2000)")
