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

