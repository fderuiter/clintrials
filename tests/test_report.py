import pandas as pd
import numpy as np
from clintrials.visualization.models import TextSection, TableSection, _format_label
from clintrials.visualization.report import generate_pdf_report

def test_text_section():
    section = TextSection("Hello World")
    assert str(section) == "Hello World"

def test_table_section():
    df = pd.DataFrame({"col_a": [1.23456, 2.0], "col_b": ["a", "b"]})
    section = TableSection(title="My Title", df=df)
    output = str(section)
    assert "**Data Summary: My Title**" in output
    assert "| Col A | Col B |" in output
    assert "| --- | --- |" in output
    assert "| 1.2346 | a |" in output
    assert "| 2.0000 | b |" in output

def test_format_label():
    assert _format_label("hello_world") == "Hello World"
    assert _format_label(123) == 123

def test_generate_pdf_report():
    df = pd.DataFrame({"col_a": [1.1, 2.2]})
    text_summaries = [
        TextSection("This is a paragraph."),
        TableSection(title="Table 1", df=df)
    ]
    pdf_bytes = generate_pdf_report(df=df, design_type="My Design", text_summaries=text_summaries)
    assert isinstance(pdf_bytes, bytearray) or isinstance(pdf_bytes, bytes)
