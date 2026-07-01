import pandas as pd
from unittest.mock import MagicMock
from clintrials.visualization.provider import plot_winratio_power_curve
from clintrials.visualization.dashboard.factory import render_accessible_chart
from clintrials.visualization.report import AccessiblePDF, generate_pdf_report
from clintrials.visualization.models import TableSection

def test_winratio_power_curve():
    df = pd.DataFrame({"num_subjects_A": [10, 20], "power": [0.5, 0.8]})
    fig = plot_winratio_power_curve(df)
    assert fig is not None
    assert fig.layout.meta is not None

def test_render_accessible_chart_no_mode():
    st_module = MagicMock()
    st_module.session_state = {"accessibility_mode": False}
    fig = MagicMock()
    fig.layout.meta = "mock meta"
    render_accessible_chart(st_module, fig)
    st_module.plotly_chart.assert_called_with(fig)
    st_module.markdown.assert_called_with("mock meta")

def test_render_accessible_chart_with_mode():
    st_module = MagicMock()
    st_module.session_state = {"accessibility_mode": True}
    fig = MagicMock()
    fig.layout.meta = "mock meta"
    render_accessible_chart(st_module, fig)
    st_module.plotly_chart.assert_not_called()
    st_module.markdown.assert_called_with("mock meta")

def test_generate_pdf_report():
    df = pd.DataFrame({"A": [1, 2], "B": [3.14159, 4.5]})
    section = TableSection("My Table", df)
    pdf_bytes = generate_pdf_report(df, "CRM", text_summaries=[section, "Simple text"])
    assert isinstance(pdf_bytes, bytearray) or isinstance(pdf_bytes, bytes)

def test_format_label():
    from clintrials.visualization.models import _format_label
    assert _format_label("some_cool_label") == "Some Cool Label"
    assert _format_label(42) == 42

def test_text_section():
    from clintrials.visualization.models import TextSection
    ts = TextSection("Hello World")
    assert str(ts) == "Hello World"

def test_table_section_str():
    from clintrials.visualization.models import TableSection
    df = pd.DataFrame({"some_val": [1, 2.12345], "other": ["A", "B"]})
    ts = TableSection("My Test", df)
    res = str(ts)
    assert "**Data Summary: My Test**" in res
    assert "| Some Val | Other |" in res
    assert "| 2.1235 | B |" in res
