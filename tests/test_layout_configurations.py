import sys
from unittest.mock import MagicMock

from clintrials.visualization.dashboard.views.framework import render_sidebar_config


def test_render_sidebar_config_structure(monkeypatch):
    """
    Verify that render_sidebar_config properly structures the sidebar inputs
    and returns a valid ParameterSpace, without loading browser automation.
    """
    # Mock streamlit
    st_mock = MagicMock()
    st_mock.sidebar.header = MagicMock()
    st_mock.sidebar.json = MagicMock()
    monkeypatch.setitem(sys.modules, "streamlit", st_mock)

    param_space_config = {
        "true_tox": [(0.05, 0.1, 0.2, 0.3, 0.4), (0.1, 0.2, 0.3, 0.4, 0.5)]
    }

    ps = render_sidebar_config(param_space_config)

    # Asserts
    st_mock.sidebar.header.assert_called_once_with("Trial Parameters")
    st_mock.sidebar.json.assert_called_once_with(param_space_config)

    # Check ParameterSpace functionality
    assert ps is not None
    assert "true_tox" in ps.vals_map
