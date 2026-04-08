"""
PDF DFM report generation using fpdf2.
"""

import io
import datetime
from fpdf import FPDF


class DFMReport(FPDF):
    """Custom PDF class for DFM reports."""

    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "DFM Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 6, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_fill_color(41, 128, 185)
        self.set_text_color(255, 255, 255)
        self.cell(0, 9, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def key_value(self, key, value):
        self.set_font("Helvetica", "B", 10)
        self.cell(65, 7, key, new_x="END")
        self.set_font("Helvetica", "", 10)
        self.cell(0, 7, str(value), new_x="LMARGIN", new_y="NEXT")


def generate_report(analysis_result: dict, filename: str = "part") -> bytes:
    """
    Generate a PDF DFM report from analysis results.

    Parameters
    ----------
    analysis_result : dict from inference.predict.run_inference()
    filename : str  – name of the uploaded file

    Returns
    -------
    bytes : the PDF content
    """
    pdf = DFMReport()
    pdf.alias_nb_pages()
    pdf.add_page()

    # --- Part Information ---
    pdf.section_title("Part Information")
    pdf.key_value("File Name:", filename)

    mesh_info = analysis_result.get("mesh_info", {})
    pdf.key_value("Vertices:", f"{mesh_info.get('vertices', 'N/A'):,}")
    pdf.key_value("Faces:", f"{mesh_info.get('faces', 'N/A'):,}")
    bb = mesh_info.get("bounding_box_mm", [0, 0, 0])
    pdf.key_value("Bounding Box (mm):", f"{bb[0]:.1f} x {bb[1]:.1f} x {bb[2]:.1f}")
    pdf.key_value("Watertight:", "Yes" if mesh_info.get("is_watertight") else "No")
    pdf.key_value("Euler Number:", str(mesh_info.get("euler_number", "N/A")))

    sa = mesh_info.get("surface_area_mm2")
    if sa is not None:
        pdf.key_value("Surface Area (mm²):", f"{sa:,.1f}")
    vol = mesh_info.get("volume_mm3")
    if vol is not None:
        pdf.key_value("Mesh Volume (mm³):", f"{vol:,.1f}")

    pdf.ln(5)

    # --- DFM Predictions ---
    pdf.section_title("DFM Model Predictions")
    pdf.key_value("Predicted Log Volume:", str(analysis_result.get("predicted_log_volume", "N/A")))
    pdf.key_value("Predicted Volume (mm³):", f"{analysis_result.get('predicted_volume_mm3', 0):,.2f}")
    pdf.ln(3)

    # Constraints table
    pdf.set_font("Helvetica", "B", 10)
    col_w = [60, 40, 50]
    headers = ["Constraint", "Status", "Confidence"]
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 8, h, border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    constraints = analysis_result.get("constraints", {})
    for name, info in constraints.items():
        status = "PASS" if info["passed"] else "FAIL"
        conf = f"{info['confidence']:.1f}%"

        # Color coding
        if info["passed"]:
            pdf.set_text_color(39, 174, 96)
        else:
            pdf.set_text_color(231, 76, 60)

        pdf.cell(col_w[0], 7, name.replace("_", " ").title(), border=1)
        pdf.cell(col_w[1], 7, status, border=1, align="C")
        pdf.cell(col_w[2], 7, conf, border=1, align="C")
        pdf.ln()

    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # --- Recommendations ---
    pdf.section_title("DFM Recommendations")
    pdf.set_font("Helvetica", "", 10)

    failed = [n for n, i in constraints.items() if not i["passed"]]
    if not failed:
        pdf.multi_cell(0, 6, "All constraint checks passed. The part appears suitable for FDM 3D printing.")
    else:
        pdf.multi_cell(0, 6, f"The following constraints failed: {', '.join(failed)}.\n")
        recommendations = {
            "overhang": "- Reduce overhanging surfaces or add supports. Keep overhang angles below 45 degrees.",
            "area": "- Large flat surfaces may warp. Add ribs, reduce surface area, or use a heated bed.",
            "contour_count": "- High contour count increases print time. Simplify geometry where possible.",
            "contour_length": "- Long contour lengths may cause quality issues. Consider splitting the part.",
            "pass_fail": "- Overall DFM check failed. Review and address the individual constraints above.",
        }
        for f in failed:
            rec = recommendations.get(f, f"- Review the '{f}' constraint.")
            pdf.multi_cell(0, 6, rec)
            pdf.ln(1)

    # Return PDF bytes
    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
