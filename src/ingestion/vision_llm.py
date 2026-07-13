"""
Base-layer ingestion script for Book Intelligence Agent.

Takes one page image -> sends to Claude vision -> gets back structured JSON
(articles, sidebars, tables, captions) -> saves to data/processed/.

Usage:
    python vision_llm.py data/raw/pages/page_001.png    # one page
    python vision_llm.py data/raw/pages                  # every image in the folder
"""

import base64
import json
import os
import sys
from pathlib import Path

from anthropic import Anthropic, APIError
from dotenv import load_dotenv

load_dotenv()  # reads .env from project root into environment

client = Anthropic()  # reads ANTHROPIC_API_KEY from environment

# Override with e.g. MODEL=claude-opus-4-8 to retry pages a different model
# handles differently (some pages are blocked by output content filtering).
MODEL = os.environ.get("MODEL", "claude-sonnet-5")

# The extraction schema. Keep this loose enough to handle pages that don't
# have every section (e.g. no table, no sidebar) — empty lists are fine.
EXTRACTION_PROMPT = """
You are extracting content from a scanned page of a football history book.
The book is organized chronologically. Each page has a small header line
near the top (e.g. "THE 1880s", "1899-1900", "EARLY YEARS", "FOOTBALL
FEATURE") above a horizontal rule — this is the era/section label and is
critical metadata, capture it exactly as printed even if it's just a decade
or date range.

The layout is irregular: pages may have one or more articles (each with its
own headline and multi-column body text), a "FOOTBALL FOCUS" style sidebar
of bullet points, a "FINAL SCORE" box containing one or more separate named
mini-tables (e.g. "FA Cup Finals", "Scottish FA Cup Finals", "International
Firsts"), and photo/sketch captions. Not every page has every section —
some have no table, some have no sidebar. Reading order within a multi-column
article matters: reconstruct the body text in the correct column order, not
left-to-right across the whole page.

Return ONLY valid JSON in this exact structure, with no other text:

{
  "page_number": <int or null if not visible, from page footer>,
  "era_label": "<the section/date header at top of page, e.g. 'THE 1880s', or null if none>",
  "articles": [
    {"headline": "...", "body": "..."}
  ],
  "sidebars": [
    {"title": "...", "bullet_points": ["...", "..."]}
  ],
  "score_tables": [
    {
      "section_title": "<e.g. 'FINAL SCORE', or null>",
      "sub_tables": [
        {
          "title": "<e.g. 'FA Cup Finals'>",
          "rows": [
            {"year": "...", "detail": "..."}
          ]
        }
      ]
    }
  ],
  "captions": [
    {"caption": "...", "near_headline": "<headline of the article this image sits nearest to, or null>"}
  ]
}

Rules:
- If a section type isn't present on this page, return an empty list for it.
- Transcribe text exactly as printed, don't paraphrase or summarize.
- If text is unclear/illegible, use "[unclear]" rather than guessing.
- For score_tables rows, keep "detail" as the raw printed line (teams, score,
  venue etc combined) rather than trying to force it into rigid columns —
  the source formatting varies too much to guarantee clean column splits.
"""


def image_to_base64(image_path: Path) -> tuple[str, str]:
    """Read an image file and return (base64_data, media_type)."""
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    media_type = media_types.get(image_path.suffix.lower(), "image/jpeg")
    data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
    return data, media_type


def extract_page(image_path: Path) -> dict:
    """Send one page image to Claude vision and return parsed JSON."""
    image_data, media_type = image_to_base64(image_path)

    response = client.messages.create(
        model=MODEL,
        # Dense pages (multiple articles + score tables) plus the model's
        # thinking tokens can exceed a small budget and truncate the JSON
        # mid-object, so give it plenty of headroom.
        max_tokens=8192,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            }
        ],
    )

    # The model may return a thinking block before the text block, so grab
    # the first text block rather than assuming it's at index 0.
    raw_text = next(
        block.text for block in response.content if block.type == "text"
    ).strip()

    # Model sometimes wraps JSON in ```json fences — strip those if present.
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        raw_text = raw_text.removeprefix("json").strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Don't crash the batch — save the raw output so you can inspect
        # and fix the prompt for this page later.
        return {"error": "failed_to_parse", "raw_response": raw_text}


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

OUTPUT_DIR = Path("data/processed")


def process_image(image_path: Path) -> None:
    """Extract one page image to JSON, skipping it if already done."""
    output_path = OUTPUT_DIR / f"{image_path.stem}.json"
    if output_path.exists():
        print(f"skip (exists): {image_path.name}")
        return

    # An API error on one page (rate limit, content filter, etc.) shouldn't
    # kill the whole batch — record it and move on so the run completes.
    try:
        result = extract_page(image_path)
    except APIError as exc:
        result = {"error": "api_error", "detail": str(exc)}

    output_path.write_text(json.dumps(result, indent=2))

    # Flag pages that failed (parse or API) so they stand out in a batch run.
    if result.get("error"):
        print(f"FAILED ({result['error']}): {output_path}")
    else:
        print(f"Saved: {output_path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python vision_llm.py <path_to_image_or_folder>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"Path not found: {target}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Accept either a single image or a directory of images.
    if target.is_dir():
        images = sorted(p for p in target.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            print(f"No images found in: {target}")
            sys.exit(1)
    else:
        images = [target]

    for image_path in images:
        process_image(image_path)


if __name__ == "__main__":
    main()