import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

from clintrials.visualization.dashboard.views.framework import dashboard_view


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


def test_dashboard_view_table_rendering(monkeypatch):  # type: ignore
    st_mock = _make_streamlit_mock()  # type: ignore
    st_mock.session_state = {"accessibility_mode": False}
    monkeypatch.setitem(sys.modules, "streamlit", st_mock)

    summary_df = pd.DataFrame(
        {
            "N": [2.123456],
            "recommended_dose_prob": [{1: 0.5, 2: 0.5}],
        },
        index=pd.Index([0.1], name="true_tox"),
    )

    @dashboard_view(title="Test Title", model_name="CRM", file_prefix="test_prefix")  # type: ignore[misc]
    def dummy_render() -> tuple[pd.DataFrame, list[object]]:
        return summary_df, []

    dummy_render()

    # Verify st_mock.markdown was called
    called_htmls = [args[0] for args, kwargs in st_mock.markdown.call_args_list if args]
    assert any("Simulation Summary" in html for html in called_htmls)
    assert any("True Tox" in html for html in called_htmls)
    assert any("2.1235" in html for html in called_htmls)
    assert any("<details>" not in html for html in called_htmls)


def test_dashboard_view_table_rendering_accessible(monkeypatch):  # type: ignore
    st_mock = _make_streamlit_mock()  # type: ignore
    st_mock.session_state = {"accessibility_mode": True}
    monkeypatch.setitem(sys.modules, "streamlit", st_mock)

    summary_df = pd.DataFrame(
        {
            "N": [2.123456],
            "recommended_dose_prob": [{1: 0.5, 2: 0.5}],
        },
        index=pd.Index([0.1], name="true_tox"),
    )

    @dashboard_view(title="Test Title", model_name="CRM", file_prefix="test_prefix")  # type: ignore[misc]
    def dummy_render() -> tuple[pd.DataFrame, list[object]]:
        return summary_df, []

    dummy_render()

    # Verify st_mock.markdown was called with details/summary disclosure pattern
    called_htmls = [args[0] for args, kwargs in st_mock.markdown.call_args_list if args]
    assert any("Simulation Summary" in html for html in called_htmls)
    assert any("Expand All" in html for html in called_htmls)
    assert any("<details>" in html for html in called_htmls)
