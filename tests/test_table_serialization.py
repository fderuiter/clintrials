import pandas as pd

from clintrials.visualization.helpers import (
    build_html_table,
    format_label,
    format_number,
)
from clintrials.visualization.models import MultiFormatSummaryContainer


def test_format_label_standard():
    """Test that format_label standardizes column names to title-cased labels."""
    assert format_label("trial_phase") == "Trial Phase"
    assert format_label("dose_level") == "Dose Level"
    assert format_label("recommended_dose_prob") == "Recommended Dose Prob"
    assert format_label(1234) == 1234


def test_format_number_standard():
    """Test that format_number rounds floats to exactly 4 decimal places."""
    assert format_number(1.234567) == "1.2346"
    assert format_number(0.00001) == "0.0000"
    assert format_number(5) == "5"
    assert format_number("string_val") == "string_val"


def test_build_html_table_flat():
    """Test standard flat HTML table serialization outputs correct labels, rounded floats, and correct structure."""
    df = pd.DataFrame({
        "trial_phase": ["Phase I", "Phase II"],
        "dose_level": [10.12345, 20.987654],
        "n_patients": [15, 30]
    })

    html = build_html_table(df)

    # Check structures
    assert "<table>" in html
    assert "<thead>" in html
    assert "<tbody>" in html
    assert 'th scope="col"' in html

    # Check mapped column labels
    assert "Trial Phase" in html
    assert "Dose Level" in html
    assert "N Patients" in html

    # Check formatted cell values
    assert "10.1235" in html
    assert "20.9877" in html
    assert "<td>15</td>" in html
    assert "<td>30</td>" in html


def test_container_flat_table_rendering():
    """Test that MultiFormatSummaryContainer returns correct flat HTML table in standard mode."""
    df = pd.DataFrame({
        "dose_level": [1.123456, 2.5]
    })
    container = MultiFormatSummaryContainer("Dose Summary", df)
    html = container.html

    assert "<strong>Data Summary: Dose Summary</strong><br><br>\n" in html
    assert "Dose Level" in html
    assert "1.1235" in html
    assert "2.5000" in html


def test_container_hierarchical_rendering(monkeypatch):
    """Test that MultiFormatSummaryContainer renders hierarchical lists with proper markup in accessibility mode."""
    import streamlit as st
    monkeypatch.setattr(st, "session_state", {"accessibility_mode": True})

    df = pd.DataFrame({
        "Trial": ["Trial A", "Trial A", "Trial B"],
        "Cohort": ["Cohort 1", "Cohort 2", "Cohort 1"],
        "Value": [1.23456, 2.34567, 3.45678]
    })

    container = MultiFormatSummaryContainer("Accessible Trial Stats", df)
    html = container.html

    assert "<strong>Data Summary: Accessible Trial Stats</strong>" in html
    assert "<details>" in html
    assert "<summary" in html
    assert 'aria-label="Expand all data sections"' in html
    assert 'aria-label="Collapse all data sections"' in html

    # Verify hierarchical grouping and headings
    assert "Trial: Trial A" in html
    assert "Cohort: Cohort 1" in html

    # Check computed summaries in summaries
    assert "N=2" in html or "N=1" in html
    assert "Mean Value: 1.7901" in html  # (1.23456 + 2.34567) / 2 = 1.790115

    # Check that flat tables are rendered at the leaf level using build_html_table
    assert "<table>" in html
    assert "Value" in html
