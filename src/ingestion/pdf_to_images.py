"""
Pre-ingestion step for Book Intelligence Agent.

Takes a PDF (or a folder of PDFs) -> renders each page to a PNG image ->
saves to data/raw/pages/. Those page images are the input to vision_llm.py.

The source scans arrive as multi-page PDFs (e.g. "Pages_8-18.pdf"), but the
vision extractor works on one page image at a time, so we split first.

Usage:
    python pdf_to_images.py data/raw/images/Pages_8-18.pdf
    python pdf_to_images.py data/raw/images          # every PDF in the folder
"""

import sys
from pathlib import Path

import fitz  # PyMuPDF

# Render resolution. 200 DPI is a good balance for scanned book pages —
# high enough for the vision model to read small print, without producing
# needlessly large base64 payloads.
DPI = 200

OUTPUT_DIR = Path("data/raw/pages")


def pdf_to_images(pdf_path: Path, output_dir: Path) -> list[Path]:
    """Render every page of one PDF to a PNG and return the written paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    with fitz.open(pdf_path) as doc:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            pixmap = page.get_pixmap(dpi=DPI)

            # e.g. "Pages_8-18.pdf" -> "Pages_8-18_p01.png". Zero-pad so the
            # files sort correctly (p02 before p10).
            out_name = f"{pdf_path.stem}_p{page_index + 1:02d}.png"
            out_path = output_dir / out_name
            pixmap.save(out_path)
            written.append(out_path)

    return written


def main():
    if len(sys.argv) != 2:
        print("Usage: python pdf_to_images.py <path_to_pdf_or_folder>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"Path not found: {target}")
        sys.exit(1)

    # Accept either a single PDF or a directory of PDFs.
    if target.is_dir():
        pdfs = sorted(target.glob("*.pdf"))
        if not pdfs:
            print(f"No PDFs found in: {target}")
            sys.exit(1)
    else:
        pdfs = [target]

    total = 0
    for pdf_path in pdfs:
        pages = pdf_to_images(pdf_path, OUTPUT_DIR)
        total += len(pages)
        print(f"{pdf_path.name}: {len(pages)} page(s) -> {OUTPUT_DIR}/")

    print(f"Done. {total} page image(s) written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
