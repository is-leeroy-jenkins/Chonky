# Processing

The Processing tab is the second stage in the Chonky workflow. It transforms loaded `raw_text` into
cleaner `processed_text` that can be analyzed, tokenized, embedded, and searched.

Processing is where extraction artifacts, formatting noise, repeated text, markup, punctuation
issues, and other document-quality problems are reduced before downstream work begins.

```text
raw_text
   │
   ▼
Text Processing
   │
   ▼
processed_text
```

## 🧭 Purpose

The Processing tab prepares source text for analysis and embedding.

It supports cleanup and transformation operations such as whitespace normalization, HTML removal,
Markdown removal, XML tag removal, stopword removal, punctuation cleanup, symbol cleanup, number
removal, sentence splitting, tokenization, stemming, lemmatization, and page-header/footer cleanup.

The primary output is:

```text
processed_text
```

This output becomes the main input for semantic analysis, tokenization diagnostics, embedding
generation, and vector search.

## 🧱 Workflow Position

Processing follows Loading and precedes Analysis.

```text
Loading
  → Processing
  → Analysis
  → Tokenization
  → Embeddings
  → Vector Database
```

The Processing tab assumes the Loading tab has already produced usable `raw_text`.

## 📦 Processor Module

The processing layer is implemented in:

```text
processors.py
```

The module contains reusable parser and processor classes for text cleanup, token operations,
document analysis, and file-specific processing.

Primary processing classes include:

| Class        | Purpose                                                                                  |
| ------------ | ---------------------------------------------------------------------------------------- |
| `Processor`  | Base class for shared processing state and common NLP utilities.                         |
| `TextParser` | Text cleanup, normalization, tokenization, chunking, vocabulary, and corpus diagnostics. |
| `NltkParser` | NLTK-backed corpus and NLP operations.                                                   |
| `WordParser` | Word document and paragraph-oriented processing support.                                 |
| `PdfParser`  | PDF extraction cleanup, page reconstruction, and layout-related processing.              |

## 📥 Primary Input

The Processing tab normally consumes:

| Input           | Source      | Purpose                                                                   |
| --------------- | ----------- | ------------------------------------------------------------------------- |
| `raw_text`      | Loading tab | Combined text extracted from the active loader.                           |
| `documents`     | Loading tab | Active LangChain document list when document-aware operations are needed. |
| `active_loader` | Loading tab | Identifies the loader that produced the current source text.              |

Before processing, confirm that `raw_text` is populated and readable.

## 📤 Primary Output

The Processing tab writes:

| Output                | Purpose                                               |
| --------------------- | ----------------------------------------------------- |
| `processed_text`      | Cleaned and transformed text used by downstream tabs. |
| `processed_text_view` | Display-oriented processed text state where used.     |
| `tokens`              | Token output where processing includes tokenization.  |
| `lines`               | Line-level text output where applicable.              |
| `chunks`              | Chunk output where applicable.                        |
| `vocabulary`          | Unique term set where vocabulary operations are run.  |

The most important output is `processed_text`.

## 🧹 Common Processing Operations

| Operation                  | Purpose                                                                        |
| -------------------------- | ------------------------------------------------------------------------------ |
| Collapse whitespace        | Reduces repeated spaces, tabs, and line breaks.                                |
| Normalize text             | Converts text into a consistent case or normalized representation.             |
| Remove punctuation         | Removes or normalizes punctuation while preserving useful sentence delimiters. |
| Reduce repeated symbols    | Converts repeated punctuation or symbols into cleaner single markers.          |
| Remove HTML                | Extracts visible text from HTML content.                                       |
| Remove XML                 | Removes XML tags while preserving inner text.                                  |
| Remove Markdown            | Removes Markdown formatting syntax.                                            |
| Remove stopwords           | Removes common words that may add noise to analysis.                           |
| Remove fragments           | Removes very short fragments that may not be useful.                           |
| Remove numbers or numerals | Removes numeric characters or roman numerals where needed.                     |
| Remove images              | Removes Markdown images, HTML image tags, and image URLs.                      |
| Remove encoding artifacts  | Cleans Unicode, HTML entities, and control characters.                         |
| Remove headers/footers     | Removes repeated page-boundary lines from text documents.                      |

## 🧾 Processing by Source Type

Different sources usually need different cleanup choices.

| Source      | Recommended Focus                                                                                            |
| ----------- | ------------------------------------------------------------------------------------------------------------ |
| PDF         | Header/footer cleanup, whitespace normalization, hyphenation repair, page-break review, punctuation cleanup. |
| HTML        | HTML removal, whitespace cleanup, boilerplate review, link/image cleanup.                                    |
| Markdown    | Markdown removal, image removal, heading review, whitespace cleanup.                                         |
| XML         | XML tag removal or structured XML review before flattening.                                                  |
| CSV / Excel | Whitespace cleanup, token review, column-derived text inspection.                                            |
| Word        | Paragraph cleanup, punctuation review, whitespace normalization.                                             |
| Web pages   | HTML cleanup, boilerplate removal, stopword review, repeated navigation cleanup.                             |
| Notebooks   | Markdown cleanup, traceback cleanup, output-length review.                                                   |

## 📕 PDF Processing Notes

PDFs often require the most careful processing because text extraction can introduce layout
artifacts.

Common PDF issues include:

| Issue            | Description                                                           |
| ---------------- | --------------------------------------------------------------------- |
| Repeated headers | Page titles, document names, or section names repeated on every page. |
| Repeated footers | Page numbers, dates, classification labels, or file references.       |
| Broken spacing   | Words joined together or split unnaturally.                           |
| Hyphenation      | Words split across line endings.                                      |
| Page artifacts   | Captions, marginalia, or layout fragments.                            |
| Table artifacts  | Columns merged into poorly ordered text.                              |

For PDFs, review `raw_text` first, then apply cleanup progressively rather than applying every
option at once.

## 🧬 Markup Processing

Chonky can process markup-heavy sources such as HTML, Markdown, and XML.

| Markup Type | Processing Goal                                             |
| ----------- | ----------------------------------------------------------- |
| HTML        | Remove tags and extract visible readable text.              |
| Markdown    | Remove formatting syntax while retaining content.           |
| XML         | Remove tags or extract text while preserving inner content. |

When XML structure matters, use the XML loader’s structured tree and XPath workflow before
flattening the text.

## 🔤 Token-Oriented Processing

Token-oriented operations prepare text for analysis and embedding.

| Operation             | Purpose                                                   |
| --------------------- | --------------------------------------------------------- |
| Word tokenization     | Splits text into word-level tokens.                       |
| Sentence tokenization | Splits text into sentence-level units.                    |
| Tiktoken tokenization | Uses model-oriented tokenization for embedding readiness. |
| Lemmatization         | Converts words to base forms.                             |
| Stemming              | Reduces words to stems.                                   |
| Vocabulary creation   | Builds a unique term set.                                 |
| Word bag creation     | Builds token-frequency structures.                        |

Token operations are useful when checking corpus shape or preparing text for downstream model
workflows.

## 📊 Corpus Diagnostics

Processing can support corpus-level inspection before embedding.

Useful diagnostics include:

| Diagnostic               | Purpose                                         |
| ------------------------ | ----------------------------------------------- |
| Frequency distribution   | Identifies common tokens and repeated language. |
| Conditional distribution | Compares token usage under selected conditions. |
| Vocabulary size          | Measures unique term count.                     |
| Token counts             | Estimates document size and model-readiness.    |
| Sentence counts          | Helps evaluate segmentation quality.            |
| Paragraph counts         | Helps understand document structure.            |

These diagnostics help determine whether text is clean enough to move forward.

## 🧠 Preparing for Embeddings

Embedding quality depends heavily on text quality.

Before generating embeddings, confirm:

| Check                               | Reason                                                          |
| ----------------------------------- | --------------------------------------------------------------- |
| Text is not empty                   | Empty text cannot produce useful vectors.                       |
| Formatting artifacts are reduced    | Artifacts can distort semantic similarity.                      |
| Repeated headers are removed        | Repetition can dominate vector meaning.                         |
| Chunks are readable                 | Poor chunks create poor retrieval results.                      |
| Token counts are reasonable         | Oversized text may fail provider limits or produce weak chunks. |
| Source-specific noise is controlled | Web, PDF, and notebook outputs often need cleanup.              |

## 🛠 Recommended Processing Workflow

Use this sequence for a typical document:

1. Load the source document.
2. Review `raw_text`.
3. Apply minimal cleanup first.
4. Inspect the processed result.
5. Add source-specific cleanup only if needed.
6. Generate tokens or chunks.
7. Review analysis and token diagnostics.
8. Proceed to embeddings only when text quality is acceptable.

## 🧯 Troubleshooting

| Symptom                                  | Likely Cause                                                 | Action                                                       |
| ---------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `processed_text` is empty                | Cleanup removed too much content                             | Disable aggressive options and rerun processing.             |
| Text still has markup                    | HTML, Markdown, or XML removal was not applied               | Apply the relevant markup cleanup operation.                 |
| Text has repeated page headers           | PDF or scanned document contains repeated page-boundary text | Use header/footer cleanup or adjust extraction settings.     |
| Tokens look fragmented                   | Source extraction created broken words or symbols            | Normalize whitespace and reduce symbols before tokenization. |
| Stopword removal makes text hard to read | Stopword removal is too aggressive for review                | Use stopword removal only when needed for analysis.          |
| Embeddings perform poorly                | Text is noisy, repetitive, or poorly chunked                 | Reprocess text and review chunks before embedding.           |

## ✅ Processing Checklist

Before moving to Analysis, confirm:

| Check                                         | Complete |
| --------------------------------------------- | -------- |
| `raw_text` was reviewed                       |          |
| Cleanup choices match the source type         |          |
| `processed_text` is populated                 |          |
| Processed text is readable                    |          |
| Repeated artifacts are reduced                |          |
| Markup is removed where appropriate           |          |
| Text is suitable for chunking or tokenization |          |

## 🧾 Summary

The Processing tab is where Chonky turns extracted text into analysis-ready text.

Good processing improves every later stage: semantic analysis becomes clearer, token diagnostics
become more meaningful, embeddings become more accurate, and vector search produces better retrieval
results.
