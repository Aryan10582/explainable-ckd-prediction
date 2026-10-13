"""Export the deck to PDF with Microsoft PowerPoint (Windows) and write slide previews.

    python presentation/export_pdf.py
"""
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PPTX = HERE / "CKD_XAI_presentation.pptx"
PDF = HERE / "CKD_XAI_presentation.pdf"
PREVIEW = HERE.parent / "reports" / "build"


def main():
    ps = ("$p = New-Object -ComObject PowerPoint.Application; "
          f"$d = $p.Presentations.Open('{PPTX}', $true, $false, $false); "
          f"$d.SaveAs('{PDF}', 32); $d.Close(); $p.Quit()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, timeout=300)
    import pymupdf

    PREVIEW.mkdir(parents=True, exist_ok=True)
    for old in PREVIEW.glob("slide_*.png"):
        old.unlink()
    doc = pymupdf.open(str(PDF))
    for i, page in enumerate(doc, 1):
        page.get_pixmap(dpi=90).save(str(PREVIEW / f"slide_{i:02d}.png"))
    print(f"wrote {PDF.name} ({doc.page_count} slides) and previews in reports/build/")


if __name__ == "__main__":
    main()
