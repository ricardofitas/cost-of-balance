"""Render both risk decompositions from the full-order aggregate measurements."""
import json
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory

from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "risk-components.json").read_text())
INK = HexColor("#172434")
MUTED = HexColor("#46556A")
GRID = HexColor("#DBE1E7")
POSITIVE = HexColor("#286480")
NEGATIVE = HexColor("#A24925")


def draw_figure(path, panels, title):
    width, height = 495, 145
    c = Canvas(str(path), pagesize=(width, height), invariant=True)
    c.setTitle(title)
    labels = [("adjacency","Adjacency",92), ("residual","Residual",74),
              ("interaction","Interaction",56), ("net","Net",34)]
    for key, label, y in labels:
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold" if key == "net" else "Helvetica", 8.8)
        c.drawString(1, y-3, label)

    start = 62
    pitch = (width-start)/len(panels)
    for i,(key, title, subhead) in enumerate(panels):
        panel = DATA[key]
        left = start + i*pitch
        center = left + pitch/2
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 10.3)
        c.drawCentredString(center, 129, title)
        c.setFont("Helvetica", 7.4)
        c.setFillColor(MUTED)
        c.drawCentredString(center, 115, subhead)

        def x(value):
            return center + (pitch-4)*value / 35

        for tick in (-10,-5,0,5,10):
            c.setStrokeColor(MUTED if tick == 0 else GRID)
            c.setLineWidth(0.65 if tick == 0 else 0.35)
            c.line(x(tick), 26, x(tick), 104)
            c.setFillColor(MUTED)
            c.setFont("Helvetica", 7)
            c.drawCentredString(x(tick), 16, str(tick))
        c.setStrokeColor(GRID)
        c.line(left+3, 45, left+pitch-3, 45)

        for component, _, y in labels:
            value = panel[component] * 1000
            color = POSITIVE if value > 0 else NEGATIVE
            c.setFillColor(color)
            c.rect(min(x(value),x(0)), y-3.5, abs(x(value)-x(0)), 7, fill=1, stroke=0)
            c.setFont("Helvetica-Bold" if component == "net" else "Helvetica", 7.4)
            c.setFillColor(INK if component == "net" else color)
            if value >= 0:
                c.drawString(x(value)+2.5, y-2.6, f"{value:+.2f}")
            else:
                c.drawRightString(x(value)-2.5, y-2.6, f"{value:+.2f}")
        change = panel["relative_mse_reduction"]
        direction = "lower" if change > 0 else "higher"
        c.setFont("Helvetica-Bold", 8.3)
        c.setFillColor(INK)
        c.drawCentredString(center, 2, f"Williams MSE {abs(change):.2%} {direction}")
    c.showPage()
    c.save()


def render(filename, panels, title):
    with TemporaryDirectory() as temporary:
        source = Path(temporary) / "unembedded.pdf"
        draw_figure(source, panels, title)
        run(["gs", "-q", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.4",
             "-dEmbedAllFonts=true", "-dSubsetFonts=true", "-dOmitInfoDate=true",
             "-dOmitID=true", "-sOutputFile=" + str(HERE / filename), "-c",
             "<< /NeverEmbed [] >> setdistillerparams", "-f", str(source)], check=True)


if __name__ == "__main__":
    render("matched-components.pdf", [
        ("HellaSwag (3B)", "HellaSwag", "validation"),
        ("MMLU validation (3B)", "MMLU", "validation"),
        ("LogiQA 2.0 (3B)", "LogiQA 2.0", "English test"),
    ], "Four shared orders with Qwen 3B and SmolLM3 3B")
    render("risk-components.pdf", [
        ("HellaSwag", "HellaSwag", "Qwen 1.5B and Granite 2B, 6,144 items"),
        ("MMLU validation (3B)", "MMLU", "Qwen 3B and SmolLM3 3B, 992 items"),
    ], "HellaSwag and MMLU with different model pairs")
