# Turning-Football-Articles-into-RAG-Structured-Data
This repository is to explore different concepts I learned over my Cambridge University module D001 Non Structured Data. I will be using Doc2Vec, Spacy and other NLP techniques.



# Pipeline Ingestion

# 1. Split PDFs into per-page images
python src/ingestion/pdf_to_images.py data/raw/images

# 2. Extract structured JSON from each page
python src/ingestion/vision_llm.py data/raw/pages/Pages_8-18_p01.png

# 3. Extract structured JSON from entire pages folder
python src/ingestion/vision_llm.py data/raw/pages


## Extraction Prompt: Evolution

The extraction prompt started off a little simple but was altered to include all the different sections I can find in the book.

```plaintext
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

```

## Problem 2: Anthropic flags visual content if it doesn't follow their policy
Tried Sonnet first as it is the cheaper model to work on first. The model proved largely successful apart from 6. Then ran Opus to see if Opus understood that this content was not against their policy, but it only recovered 1 more page. Therefore, I have 5 pages that are not correctly identified. I'm not entirely sure what rules are broken through these pages, but this will have to be done manually.

# Checklist
- 12/07/2026: Go through the 5 pages to fill in the content, the era_labels are there now.