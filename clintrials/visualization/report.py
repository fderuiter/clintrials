"""Reporting module for generating accessible PDF summaries."""

import re
from contextlib import contextmanager

from fpdf import FPDF
from fpdf.enums import PageMode
from fpdf.prefs import ViewerPreferences
from fpdf.table import Table


class AccessibleTable(Table):
    """An accessible table representation for PDF rendering."""
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
    """An accessible PDF generation wrapper around FPDF."""
    def __init__(self, title="Trial Simulation Report"):
        """Initializes the accessible PDF with metadata and accessibility features."""
        super().__init__()
        self.pdf_version = "1.7"
        self.set_title(title)
        self.set_lang("en-US")
        self.page_mode = PageMode.USE_OUTLINES
        self.viewer_preferences = ViewerPreferences(display_doc_title=True)
        self.struct_stack = []
        self.mcid_counter = {}
        self.add_page()

        # Patch the structure builder's iterator to support nested structure elements
        def recursive_iter(builder):
            yield builder.struct_tree_root
            yield builder.doc_struct_elem
            yield builder.struct_tree_root.parent_tree

            def walk(elem):
                for kid in elem.k:
                    if hasattr(kid, 'k'):  # Check if it's a StructElem
                        yield kid
                        yield from walk(kid)

            yield from walk(builder.doc_struct_elem)

        if hasattr(self, 'struct_builder'):
            self.struct_builder.__class__.__iter__ = recursive_iter

    @contextmanager
    def artifact(self, artifact_type="Layout"):
        """Context manager to mark content as an artifact (ignored by screen readers)."""
        if artifact_type:
            self._out(f"/Artifact <</Type /{artifact_type}>> BDC")
        else:
            self._out("/Artifact BMC")
        yield
        self._out("EMC")

    @contextmanager
    def mark_text(self, struct_type="/P"):
        """Context manager to mark text for PDF/UA structural tagging."""
        is_container = struct_type in ("/Table", "/TR")

        if is_container:
            mcid = None
        else:
            mcid = self.mcid_counter.get(self.page, 0)
            self.mcid_counter[self.page] = mcid + 1

        struct_elem = self._add_marked_content(struct_type=struct_type, mcid=mcid)

        if self.struct_stack:
            parent = self.struct_stack[-1]
            if struct_elem in self.struct_builder.doc_struct_elem.k:
                self.struct_builder.doc_struct_elem.k.remove(struct_elem)
            struct_elem.p = parent
            parent.k.append(struct_elem)

        self.struct_stack.append(struct_elem)

        if mcid is not None:
            self._out(f"{struct_type} <</MCID {mcid}>> BDC")
        else:
            self._out(f"{struct_type} BDC")

        yield struct_elem

        self.struct_stack.pop()
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

    from clintrials.visualization.helpers import format_label as _format_label
    from clintrials.visualization.helpers import format_number as fmt
    from clintrials.visualization.models import MultiFormatSummaryContainer

    for summary in text_summaries:
        if isinstance(summary, MultiFormatSummaryContainer):
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

    return bytes(pdf.output())


def parse_pdf_structure(pdf_bytes: bytes) -> dict:
    """Parses a generated PDF to extract its logical structure tree.

    Returns a dictionary of structure elements.

    Args:
        pdf_bytes (bytes): The byte string of the PDF content.

    Returns:
        dict: A dictionary of extracted logical structure elements.
    """
    content = pdf_bytes.decode('latin1', errors='ignore')
    objects = {}

    for match in re.finditer(r'(\d+)\s+0\s+obj(.*?)endobj', content, re.DOTALL):
        obj_id = int(match.group(1))
        objects[obj_id] = match.group(2)

    struct_elems = {}
    for obj_id, obj_content in objects.items():
        if '/Type /StructElem' in obj_content:
            s_match = re.search(r'/S\s+/(\w+)', obj_content)
            s_type = s_match.group(1) if s_match else None

            p_match = re.search(r'/P\s+(\d+)\s+0\s+R', obj_content)
            parent_id = int(p_match.group(1)) if p_match else None

            k_match = re.search(r'/K\s+\[(.*?)\]', obj_content, re.DOTALL)
            kids = []
            if k_match:
                k_content = k_match.group(1)
                kids_refs = re.findall(r'(\d+)\s+0\s+R', k_content)
                if kids_refs:
                    kids = [int(x) for x in kids_refs]
                else:
                    kids_ints = re.findall(r'\b\d+\b', k_content)
                    if kids_ints:
                        kids = [f"MCID_{x}" for x in kids_ints]

            struct_elems[obj_id] = {
                'id': obj_id,
                'type': s_type,
                'parent': parent_id,
                'kids': kids
            }

    return struct_elems

def validate_pdf_ua_structure(pdf_bytes: bytes):
    """Validates that a PDF's structure tree is correctly nested and MCIDs are only on leaves.

    Args:
        pdf_bytes (bytes): The byte string of the PDF content.

    Returns:
        bool: True if the structure is correctly nested.

    Raises:
        ValueError: If the structural nesting is incorrect or MCIDs are on containers.
    """
    elems = parse_pdf_structure(pdf_bytes)

    # Validation 1: Nesting correctness
    for obj_id, elem in elems.items():
        if elem['type'] == 'TR':
            parent = elems.get(elem['parent'])
            if parent and parent['type'] not in ('Table', 'THead', 'TBody', 'TFoot'):
                raise ValueError(f"TR (id {obj_id}) must be a child of a Table element")

        if elem['type'] in ('TH', 'TD'):
            parent = elems.get(elem['parent'])
            if parent and parent['type'] != 'TR':
                raise ValueError(f"{elem['type']} (id {obj_id}) must be a child of a TR element")

        # Validation 2: MCIDs are not on structural containers
        if elem['type'] in ('Table', 'TR', 'THead', 'TBody', 'TFoot'):
            for kid in elem['kids']:
                if isinstance(kid, str) and kid.startswith('MCID_'):
                    raise ValueError(f"Structural container {elem['type']} (id {obj_id}) cannot have MCID assigned directly")

    return True
