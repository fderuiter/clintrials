from fpdf import FPDF
from fpdf.enums import PageMode
from fpdf.prefs import ViewerPreferences
from contextlib import contextmanager
from fpdf.table import Table
import numpy as np


class AccessibleTable(Table):
    def _render_table_row(self, i, row_layout_info, cell_x_positions, **kwargs):
        is_header = (i < self._num_heading_rows)
        with self._fpdf.mark_text("/TR"):
            self._current_row_is_header = is_header
            super()._render_table_row(i, row_layout_info, cell_x_positions, **kwargs)

    def _render_table_cell(self, i, j, cell, row_height, cell_height_info=None, cell_x_positions=None, **kwargs):
        height_query_only = (cell_height_info is None)
        if height_query_only:
            return super()._render_table_cell(i, j, cell, row_height, cell_height_info, cell_x_positions, **kwargs)
        
        tag = "/TH" if getattr(self, "_current_row_is_header", False) else "/TD"
        with self._fpdf.mark_text(tag):
            return super()._render_table_cell(i, j, cell, row_height, cell_height_info, cell_x_positions, **kwargs)


class AccessiblePDF(FPDF):
    def __init__(self, title="Trial Simulation Report"):
        super().__init__()
        self.pdf_version = "1.7"
        self.set_title(title)
        self.set_lang("en-US")
        self.page_mode = PageMode.USE_OUTLINES
        self.viewer_preferences = ViewerPreferences(display_doc_title=True)
        self.add_page()

    @contextmanager
    def mark_text(self, struct_type="/P"):
        """Context manager to mark text for PDF/UA structural tagging."""
        mcid = self.struct_builder.next_mcid_for_page(self.page)
        struct_elem = self._add_marked_content(struct_type=struct_type, mcid=mcid)
        self._out(f"{struct_type} <</MCID {mcid}>> BDC")
        yield struct_elem
        self._out("EMC")

    @contextmanager
    def accessible_table(self, *args, **kwargs):
        """Context manager for generating an accessible PDF table."""
        kwargs.setdefault("num_heading_rows", 1)
        table = AccessibleTable(self, *args, **kwargs)
        yield table
        with self.mark_text("/Table"):
            table.render()

    def add_h1(self, text):
        """Adds a heading 1 tagged element to the PDF."""
        self.set_font("helvetica", "B", 16)
        with self.mark_text(struct_type="/H1"):
            self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(10)

    def add_p(self, text):
        """Adds a paragraph tagged element to the PDF."""
        self.set_font("helvetica", "", 12)
        with self.mark_text(struct_type="/P"):
            self.multi_cell(0, 10, text)
        self.ln(5)


def generate_pdf_report(df, design_type, text_summaries=None):
    """Generates an accessibility-first PDF report for trial simulations."""
    if text_summaries is None:
        text_summaries = []

    pdf = AccessiblePDF(f"{design_type} Simulation Report")

    pdf.add_h1(f"{design_type} Simulation Report")

    pdf.add_p(
        f"This is an automated accessibility-first report for {design_type} trial simulations."
    )
    pdf.add_p(f"Number of scenarios/simulations summarized: {len(df)}")

    pdf.add_h1("Simulation Data Summary")

    def fmt(v):
        if isinstance(v, (float, np.float64)):
            return f"{v:.4f}"
        return str(v)

    from clintrials.visualization.models import TableSection, _format_label

    for summary in text_summaries:
        if isinstance(summary, TableSection):
            pdf.add_p(f"Data Summary: {summary.title}")
            pdf.set_font("helvetica", "", 10)
            with pdf.accessible_table() as table:
                row = table.row()
                for col in summary.df.columns:
                    row.cell(str(_format_label(col)))
                for _, data_row in summary.df.iterrows():
                    row = table.row()
                    for val in data_row:
                        row.cell(fmt(val))
            pdf.ln(5)
        else:
            pdf.add_p(str(summary))

    return pdf.output()
