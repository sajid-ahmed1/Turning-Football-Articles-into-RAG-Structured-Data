# Turning Football Articles into RAG-Structured Data

Personal learning project built on Cambridge module D001 (Non-Structured Data).
Corpus: a scanned football history book (c. 1880s–1920s), extracted page-by-page
into JSON by a vision LLM.

## How Claude should work on this repo

**This is a learning project. Do not hand over answers.**

- Don't write the implementation when Sajid is learning the concept. Sketch the
  shape (function signatures, docstrings, `TODO`s, the sequence of steps) and let
  him fill in the body.
- When he's stuck, ask a question that moves him one step, don't skip to the fix.
  "What does `.shape` say right before that line?" beats "add `.toarray()`".
- When something breaks, work it out together: read the traceback aloud, form a
  hypothesis, test the hypothesis. The debugging *is* the lesson.
- Do write: boilerplate he's already done once, plumbing, plots, data loading.
- Always be honest about whether a technique will actually work on this corpus
  (see Corpus reality below). Flattering a bad plan wastes his time.
- Exception: if he explicitly says "just write it", write it.

## Layout

```
data/raw/          source PDF + per-page PNGs (gitignored, 154MB)
data/processed/    30 JSON files, one per page — the corpus
src/ingestion/     pdf_to_images.py, vision_llm.py (DONE)
notebooks/         exploration
```

Ingestion is complete. See README.md for the extraction prompt and its history.

## Data shape

Each `data/processed/*.json` is one page:

```
page_number, era_label, articles[{headline, body}],
sidebars[{title, bullet_points[]}],
score_tables[{section_title, sub_tables[{title, rows[{year, detail}]}]}],
captions[{caption, near_headline}]
```

Known facts (as of 2026-07-17):
- 30 pages, 81 articles, ~16,300 body words. Median article ≈183 words, max 783.
- 22 era labels: 18 season labels (`1899-1900` … `1920-21`, one page each), plus
  `EARLY YEARS` (2), `THE 1880s` (2), `THE 1890s` (2), `FOOTBALL FEATURE` (6).
- Non-season era_labels are multi-page spreads; articles continue across the page
  break. Matters for chunking and for treating a page as a document.
- One empty article body remains: `Pages 30-37_p01.json` / "Another Olympic
  triumph for England".
- Two filename conventions coexist: `Pages_8-18_p01.json` and `Pages 30-37_p01.json`
  (space, not underscore). Glob accordingly.

## Corpus reality (read before choosing a method)

81 documents / 16k words is **small**. This constrains what's honest:

- TF-IDF: fine at this size.
- LDA: 81 documents is below where topics are stable. Expect to defend the choice
  or work at a different unit of analysis (paragraphs? sentences?).
- Supervised classification: there are **no labels** in this data. Any supervised
  task needs labels invented and hand-annotated first — that's the real cost.
- Sentiment: no in-domain labelled data, and 1900s sports prose is out of
  distribution for off-the-shelf sentiment models. Validate on hand-read examples
  before trusting any number.

## Roadmap

1. **Preprocess** — load JSON → flat records; tokenize, sentence-segment,
   lemmatize (spaCy); decide unit of analysis (page vs article vs paragraph);
   handle `[unclear]` markers and cross-page continuations.
2. **Features** — TF-IDF, LDA topics, Doc2Vec / embeddings.
3. **Analysis** — sentiment or another downstream task. Open question, see below.
4. **RAG** — chunk + index, per the repo name.

## Open decisions

- What is a "document"? Drives everything downstream.
- What's the actual research question for step 3? "Sentiment" is a method, not a
  question. Something like "does coverage tone shift across the WWI break?" is a
  question a 16k-word corpus can maybe answer.

## Environment

`.venv/`, Python 3.12. Deps in `requirements.txt` (python-dotenv, anthropic,
PyMuPDF). `.env` holds `ANTHROPIC_API_KEY` — gitignored, never print it.
