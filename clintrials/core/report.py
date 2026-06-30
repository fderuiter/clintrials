from fpdf import FPDF
from fpdf.enums import PageMode
from fpdf.prefs import ViewerPreferences
from contextlib import contextmanager


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

    for summary in text_summaries:
        pdf.add_p(summary.replace("**", ""))

    return pdf.output()
