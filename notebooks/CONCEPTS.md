# Concepts — running notes

Plain-English answers to questions that came up while working through the
notebooks. Append here whenever something clicks, so we don't re-derive it.

---

## What is a "document" in this corpus?

**One document = one article.** `df.shape` is `(81, …)`, so 81 documents.

This was a *choice*, not the only option — see the top "Open decision" in
CLAUDE.md. Alternatives:

- **Article-level (chosen):** ~81 short docs, each tightly about one thing.
  Crisp top-terms. Good for "what is this article about."
- **Page-level:** ~30 longer docs, but a page can mix two unrelated articles,
  so top-terms blur. More natural as a RAG retrieval chunk later.

Everything downstream (TF-IDF rows, LDA docs, similarity) inherits this unit.

---

## TF-IDF — what the weights mean

TF-IDF turns each article into a row of numbers over a shared vocabulary.
A word's weight is **how distinctive it is to THIS document vs the rest of the
corpus** — high when it appears a lot *in this article* but rarely *across the
other 80*.

- High weight = a fingerprint word for that document (`barnsley`, `albion`,
  `utley` for the Barnsley match report).
- Common words (`the`, `and`, `match`, `football`) appear everywhere, so their
  IDF crushes their weight to ~0. That's TF-IDF working, not a bug.

**Top-terms recipe** for document `i`:
```python
row = X[i].toarray().ravel()          # doc i's weights, one per vocab word
top_idx = np.argsort(row)[::-1][:10]  # positions of the 10 biggest weights
for j in top_idx:
    print(f"{terms[j]:20s} {row[j]:.3f}")
```
Two index spaces to keep straight: `i` indexes **documents** (rows of `X`),
`j` indexes the **vocabulary** (`terms[j]`). `terms[i]` is meaningless.

---

## `ngram_range=(1, 2)` — unigrams + bigrams

Counts single words AND adjacent word pairs as separate columns, so `extra`,
`time`, and `extra time` all become their own terms.

- **Upside:** phrases carry meaning singles lose (`extra time`, `fa cup`,
  `aston villa`).
- **Cost on THIS corpus:** bigrams ~multiply the vocabulary. With 81 tiny docs,
  most bigrams appear once and never again → sparser matrix, top-terms lists can
  fill with one-off phrases that look distinctive only because they're rare.
- **Manage it:** `min_df=2` drops any term in <2 documents (kills one-off
  bigrams). Compare `(1,1)` vs `(1,2)` top-terms to see which reads better.

---

## Document similarity — article vs article

With article-level docs, similarity compares two rows of `X`. Standard measure
is **cosine similarity**: the angle between two TF-IDF vectors, 0 (nothing in
common) to 1 (identical direction). Two articles score high when they put weight
on the *same* distinctive words.

```python
from sklearn.metrics.pairwise import cosine_similarity
S = cosine_similarity(X)   # 81×81 matrix; S[a, b] = similarity of doc a and b
```
Caveats on this corpus:
- **Empty-body row** (`Pages 30-37_p01.json`, "Another Olympic triumph") is an
  all-zeros vector. Cosine is undefined (0/0); sklearn returns 0 — that's "no
  information," not "very dissimilar." Know whether it's in `df`.
- 81 short docs + bigrams = very sparse vectors, so most pairs score near 0 and
  only genuinely-overlapping articles light up. Normal here.

---

## LDA uses counts, TF-IDF uses weights — why swap the vectorizer?

Notebook 03 rebuilds the matrix with `CountVectorizer`, not `TfidfVectorizer`.
Same vocabulary and knobs — only the cell *values* change (integer counts vs
rescaled weights). Two reasons:

- **LDA is a generative count model.** Its story is "each document was made by
  repeatedly drawing words from topics," which only makes sense with **integer
  counts** ("this word occurred 3 times"). A TF-IDF weight like `0.524` isn't a
  count — feeding it in breaks the probability math LDA is built on.
- **LDA down-weights common words its own way.** A word in every document gets
  spread across all topics and stops being distinctive *inside the model*, so
  IDF is redundant here — not just unwanted.

One-liner: **LDA needs raw counts because it models word occurrences and handles
commonness internally; TF-IDF pre-rescales, which is both invalid input and
redundant for LDA.**

---

## Why the LDA topics were hard to name (and what the stopword pass showed)

**First run (5 topics, plain English stopwords):** every topic's top words shared
`cup`, `final`, `half`, `league`, `goal`. Those are football-generic — in *every*
article — so LDA can't use them to separate topics; they smear across all five.
TF-IDF's IDF would crush them, but LDA models raw counts, so nothing does.
`max_df=0.9` didn't catch them because no single word is in >90% of docs.

**Domain-stopword pass (dropped the generic words, re-fit):**
- *Interpretability improved* — distinctive words surfaced: club/place names
  (`villa`, `newcastle`, `celtic`, `rangers`), `war`, `england`/`scotland`,
  `division`. So the first-run mush **was** a vocabulary problem. Confirmed.
- *Stability still failed* — re-fitting with `random_state=1` scrambles the
  topics; only a Scottish-football cluster (`celtic/rangers/quinn/referee`)
  survives both seeds. This is the signature of **n=81 being too small for
  stable LDA**. Cleaning vocab bought interpretability, not stability.

**Takeaways:**
- More/fewer topics was never the real lever — vocabulary was (for naming),
  and corpus size is (for stability).
- At this scale LDA mostly captures **proper-noun co-occurrence** (articles about
  the same club), not abstract themes.
- `war` recurring is a genuine lead for the real research question (coverage
  shift across the WWI break) — chase that directly, not via LDA.
- Housekeeping: add `second` to domain stopwords (`second half`/`second
  division` artefact).

---

## Visualising LDA (what's honest at n=81)

- **Article × topic heatmap** (`lda.transform(X)`, 81×n_topics): shows whether
  articles are decisively one topic or smeared. Lots of mid-range = weak topics.
- **Topic × era heatmap** (crosstab of dominant topic vs `era_label`): the useful
  one — does topic track the timeline? Speaks to the WWI-break question.
- **pyLDAvis** deliberately skipped: heavy dependency, and its intertopic-distance
  map is unreliable with 5 topics / 81 docs — looks impressive, means little here.

---

## Open threads to revisit

- **Empty body row** — does the loader skip empty bodies or keep a blank string
  (zero row)? Affects similarity and any per-doc stats.
- **No lemmatization** — sklearn's tokenizer treats `goal`/`goals`/`scored`/
  `scoring` as separate columns. Does it matter here? (Roadmap step 1.)
