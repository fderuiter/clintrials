import pandas as pd
from clintrials.visualization.models import MultiFormatSummaryContainer, TextSection, _format_label

def test_text_section():
    ts = TextSection("Hello")
    assert str(ts) == "Hello"

def test_format_label():
    assert _format_label("some_label") == "Some Label"
    assert _format_label(123) == 123

def test_multi_format_summary_container():
    df = pd.DataFrame({"a_col": [1.123456, 2], "b_col": ["str", "other"]})
    c = MultiFormatSummaryContainer("Title", df)
    
    md = c.markdown
    assert "Title" in md
    assert "A Col" in md
    assert "1.1235" in md
    
    assert str(c) == md
    
    html = c.html
    assert "<strong>Data Summary: Title</strong>" in html
    assert "A Col" in html
    assert "1.1235" in html
    assert "<td>str</td>" in html
    assert "<details>" not in html  # By default, should not have details

def test_multi_format_summary_container_hierarchical(monkeypatch):
    import streamlit as st
    monkeypatch.setattr(st, "session_state", {"accessibility_mode": True})
    
    df = pd.DataFrame({
        "Trial": ["T1", "T1", "T2"],
        "Cohort": ["C1", "C2", "C1"],
        "Dose": [1, 2, 1],
        "Value": [0.5, 0.6, 0.7]
    })
    c = MultiFormatSummaryContainer("Hierarchical", df)
    html = c.html
    
    assert "Data Summary: Hierarchical" in html
    assert "Expand All" in html
    assert "Collapse All" in html
    assert "<details>" in html
    assert "<summary" in html
    # Check that the summary has metrics
    assert "N=2" in html or "N=1" in html
    # Check that levels are correctly rendered
    assert "Trial: T1" in html
    assert "Cohort: C1" in html

def test_multi_format_summary_container_hierarchical_fallback(monkeypatch):
    import streamlit as st
    monkeypatch.setattr(st, "session_state", {"accessibility_mode": True})
    
    df = pd.DataFrame({
        "Cat1": ["A", "A", "B"],
        "Cat2": ["X", "Y", "X"],
        "Value": [1, 2, 3]
    })
    c = MultiFormatSummaryContainer("Fallback", df)
    html = c.html
    
    assert "<details>" in html
    assert "Cat1: A" in html

