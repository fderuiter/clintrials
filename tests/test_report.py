import pandas as pd
from clintrials.visualization.models import TextSection, MultiFormatSummaryContainer, _format_label
from clintrials.visualization.report import generate_pdf_report

def test_text_section():
    section = TextSection("Hello World")
    assert str(section) == "Hello World"

def test_table_section():
    df = pd.DataFrame({"col_a": [1.23456, 2.0], "col_b": ["a", "b"]})
    section = MultiFormatSummaryContainer(title="My Title", df=df)
    output = str(section)
    assert "**Data Summary: My Title**" in output
    assert "| Col A | Col B |" in output
    assert "| --- | --- |" in output
    assert "| 1.2346 | a |" in output
    assert "| 2.0000 | b |" in output
    
    html_output = section.html
    assert "<strong>Data Summary: My Title</strong>" in html_output
    assert '<th scope="col">Col A</th>' in html_output
    assert '<td>1.2346</td>' in html_output
    assert '<td>2.0000</td>' in html_output

def test_format_label():
    assert _format_label("hello_world") == "Hello World"
    assert _format_label(123) == 123

def test_generate_pdf_report():
    df = pd.DataFrame({"col_a": [1.1, 2.2]})
    text_summaries = [
        TextSection("This is a paragraph."),
        MultiFormatSummaryContainer(title="Table 1", df=df)
    ]
    pdf_bytes = generate_pdf_report(df=df, design_type="My Design", text_summaries=text_summaries)
    assert isinstance(pdf_bytes, bytearray) or isinstance(pdf_bytes, bytes)

from clintrials.validation import parse_pdf_structure, validate_pdf_ua_structure
from clintrials.visualization.report import AccessiblePDF

def test_pdf_structural_nesting_and_mcid():
    """Validates that Tables, TR, TD are nested properly and MCIDs are assigned correctly."""
    pdf = AccessiblePDF()
    pdf.set_font("helvetica", "", 12)
    # The output stream must not be compressed for our basic parser
    pdf.set_compression(False)
    
    with pdf.accessible_table() as table:
        row = table.row()
        row.cell("Header 1")
        row.cell("Header 2")
        
        row = table.row()
        row.cell("Data 1")
        row.cell("Data 2")
        
    pdf_bytes = pdf.output()
    
    # 1. Use the validation utility
    validate_pdf_ua_structure(pdf_bytes)
    
    # 2. Assert structural nesting is present
    elems = parse_pdf_structure(pdf_bytes)
    
    tables = [e for e in elems.values() if e['type'] == 'Table']
    trs = [e for e in elems.values() if e['type'] == 'TR']
    ths = [e for e in elems.values() if e['type'] == 'TH']
    tds = [e for e in elems.values() if e['type'] == 'TD']
    
    assert len(tables) > 0, "PDF should contain a Table tag"
    assert len(trs) > 0, "PDF should contain TR tags"
    assert len(ths) > 0, "PDF should contain TH tags"
    assert len(tds) > 0, "PDF should contain TD tags"
    
    # Verify parent of TR is Table
    for tr in trs:
        parent = elems[tr['parent']]
        assert parent['type'] == 'Table', "TR parent must be Table"
        assert any(str(x).startswith('MCID_') for x in tr['kids']) is False, "TR should not have MCIDs"
        
    # Verify parent of TH and TD is TR
    for cell in ths + tds:
        parent = elems[cell['parent']]
        assert parent['type'] == 'TR', "Cell parent must be TR"
        assert len(cell['kids']) == 1 and str(cell['kids'][0]).startswith('MCID_'), "Cells should have MCIDs"

def test_artifact_tagging():
    """Validates that decorative elements can be tagged as artifacts."""
    pdf = AccessiblePDF()
    pdf.set_font("helvetica", "", 12)
    pdf.set_compression(False)
    
    with pdf.artifact("Layout"):
        pdf.cell(10, 10, "Decorative Text")
        
    pdf_bytes = pdf.output()
    content = pdf_bytes.decode('latin1')
    
    # Verify that the artifact tag is emitted properly
    assert "/Artifact <</Type /Layout>> BDC" in content
    assert "EMC" in content

