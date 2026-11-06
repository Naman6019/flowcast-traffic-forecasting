"""Export reports/final_report.md to PDF. Run by `python -m src.pipeline report`."""

from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parents[1]
STYLE = """
@page { size: A4; margin: 2cm; }
body { font-family: Helvetica; font-size: 10pt; line-height: 1.35; color: #222; }
h1 { font-size: 20pt; }
h2 { font-size: 14pt; margin-top: 14pt; }
h3 { font-size: 11.5pt; margin-top: 10pt; }
table { width: 100%; margin: 6pt 0; }
th { background-color: #eeeeee; font-weight: bold; }
th, td { border: 0.5pt solid #bbbbbb; padding: 3pt; font-size: 8.5pt; }
code { font-family: Courier; font-size: 9pt; }
"""
# The PDF's built-in fonts lack these characters, so they're spelled out.
REPLACEMENTS = {"≤": "at most", "≥": "at least", "−": "-", "×": "x", "ᵀ": "^T", "²": "^2"}
FIGURES = [
    ("final_volume_residuals.png", "Predicted against actual and residuals, test window, next-window volume."),
    ("deep_loss_curves.png", "Training and validation loss for every deep run."),
    ("eda_daily_profile.png", "Daily profile of volume and speed."),
    ("eda_weather.png", "Volume, speed and accident rate by weather."),
]


def main() -> None:
    text = (ROOT / "reports" / "final_report.md").read_text(encoding="utf-8")
    for character, replacement in REPLACEMENTS.items():
        text = text.replace(character, replacement)
    body = markdown.markdown(text, extensions=["tables"])
    figures = "".join(f'<p><img src="{(ROOT / "reports" / "figures" / name).as_posix()}" width="480"/><br/><i>{caption}</i></p>' for name, caption in FIGURES)
    html = f"<html><head><meta charset='utf-8'><style>{STYLE}</style></head><body>{body}<h2>Figures</h2>{figures}</body></html>"
    with open(ROOT / "reports" / "final_report.pdf", "wb") as output:
        result = pisa.CreatePDF(html, dest=output, encoding="utf-8")
    if result.err:
        raise RuntimeError(f"PDF export failed with {result.err} errors")


if __name__ == "__main__":
    main()
