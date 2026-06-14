# Tokenization

The Tokenization tab is the fourth stage in the Chonky workflow. It helps users inspect how
processed text is divided into sentences, words, tokens, and fixed-width token windows before
embedding generation or vector search.

Tokenization is a diagnostic stage. It gives visibility into whether the processed text is
structured well enough for model-oriented workflows.

```text id="a1q7pn"
processed_text
      │
      ▼
Tokenization
      │
      ▼
tokens + sentence rows + token diagnostics
```

## 🧭 Purpose

The Tokenization tab helps evaluate text readiness before embeddings are generated.

It supports inspection of:

| Diagnostic           | Purpose                                            |
| -------------------- | -------------------------------------------------- |
| Sentence rows        | Shows how text is split into sentence-level units. |
| Word tokens          | Shows word-level tokenization output.              |
| Token grids          | Displays fixed-width token windows for review.     |
| Token counts         | Estimates text size and model-readiness.           |
| Sparsity diagnostics | Identifies empty, short, or uneven token regions.  |
| Frequency signals    | Highlights repeated or dominant terms.             |

This stage is useful when source text comes from PDFs, web pages, OCR-like extraction, notebooks, or
other formats that can produce fragmented text.

## 🧱 Workflow Position

Tokenization follows Analysis and precedes Embeddings.

```text id="g0qpcm"
Loading
  → Processing
  → Analysis
  → Tokenization
  → Embeddings
  → Vector Database
```

The Tokenization tab assumes the text has already been loaded, cleaned, and reviewed.

## 📥 Primary Input

The Tokenization tab commonly consumes:

| Input               | Source                              | Purpose                                                   |
| ------------------- | ----------------------------------- | --------------------------------------------------------- |
| `processed_text`    | Processing tab                      | Cleaned text used for sentence and token operations.      |
| `tokens`            | Processing or tokenization workflow | Existing token list where available.                      |
| `chunks`            | Analysis tab                        | Chunked text used for token-window inspection.            |
| `chunked_documents` | Analysis tab                        | Document-aware chunks where metadata should be preserved. |

If these values are missing, return to the earlier workflow stages and regenerate them.

## 📤 Primary Output

The Tokenization tab can produce:

| Output            | Purpose                                                     |
| ----------------- | ----------------------------------------------------------- |
| `tokens`          | Token list produced from active text.                       |
| sentence rows     | Sentence-level representation of processed text.            |
| token grids       | Fixed-width token windows for inspection.                   |
| token counts      | Numeric token diagnostics.                                  |
| readiness metrics | Signals used to decide whether text is ready for embedding. |

These outputs help users decide whether to continue to embeddings or return to processing.

## 🔤 Token Types

Chonky may expose several token views depending on the workflow.

| Token Type      | Description                                                            |
| --------------- | ---------------------------------------------------------------------- |
| Word tokens     | Human-readable word-level tokens.                                      |
| Sentence tokens | Sentence-level units created through sentence segmentation.            |
| Model tokens    | Tokenizer-specific units used to estimate model or embedding workload. |
| Chunk tokens    | Token windows derived from chunks or fixed token ranges.               |

Each token type serves a different purpose. Word and sentence tokens are easier to inspect. Model
tokens are better for estimating provider or model input size.

## 🧾 Sentence Segmentation

Sentence segmentation splits processed text into sentence-level units.

Good sentence segmentation should produce rows that are:

| Quality       | Description                                                                     |
| ------------- | ------------------------------------------------------------------------------- |
| Complete      | Sentences should not be cut off unnecessarily.                                  |
| Readable      | Each row should contain coherent text.                                          |
| Not too long  | Extremely long sentences may indicate missing punctuation or extraction issues. |
| Not too short | Too many fragments may indicate broken source extraction.                       |

Sentence review is especially useful before chunking or embedding large documents.

## 🔢 Token Grids

Token grids display tokens in fixed-width windows.

They are useful for checking:

| Check             | Reason                                                |
| ----------------- | ----------------------------------------------------- |
| Token density     | Sparse rows can indicate empty or fragmented text.    |
| Token continuity  | Broken sequences can show extraction artifacts.       |
| Repetition        | Repeated headers or boilerplate become easier to see. |
| Window boundaries | Helps evaluate chunking and overlap choices.          |
| Readability       | Confirms that token order still makes sense.          |

A clean token grid usually indicates that the text is ready for embedding.

## 📏 Token Counts

Token counts help estimate how much text will be sent into an embedding or model workflow.

Token count diagnostics can help identify:

| Signal                  | Meaning                                                          |
| ----------------------- | ---------------------------------------------------------------- |
| Very low token count    | Text may be empty, over-cleaned, or too short.                   |
| Very high token count   | Text may need chunking before embedding.                         |
| Uneven token windows    | Text may contain extraction artifacts or inconsistent structure. |
| Repeated token patterns | Headers, footers, or boilerplate may still be present.           |

Token count review helps avoid embedding oversized or poor-quality text.

## 🧹 Readiness Diagnostics

Readiness diagnostics help determine whether the current text should move forward.

A text sample is usually ready for embedding when:

| Check                           | Expected Result                                    |
| ------------------------------- | -------------------------------------------------- |
| Text is populated               | There is enough content to represent.              |
| Sentences are coherent          | Sentence splitting produces readable rows.         |
| Token windows are balanced      | Token grids are not mostly empty or fragmented.    |
| Repetition is controlled        | Boilerplate does not dominate the token output.    |
| Chunk boundaries are reasonable | Chunks preserve meaningful context.                |
| Token counts are manageable     | Text can fit into the intended embedding workflow. |

If one or more checks fail, return to Processing or Analysis.

## 🧠 Relationship to Embeddings

Embeddings are numerical representations of text. Poor tokenization can produce poor embeddings.

Common tokenization problems that affect embeddings include:

| Problem                  | Effect                                              |
| ------------------------ | --------------------------------------------------- |
| Fragmented words         | Reduces semantic clarity.                           |
| Repeated boilerplate     | Makes unrelated chunks look similar.                |
| Empty chunks             | Produces unusable or skipped embeddings.            |
| Oversized chunks         | Can exceed limits or dilute meaning.                |
| Poor sentence boundaries | Can split related meaning across unrelated windows. |

Tokenization review helps catch these issues before vectors are generated.

## 🛠 Recommended Tokenization Workflow

Use this sequence:

1. Confirm `processed_text` exists.
2. Generate sentence rows.
3. Review sentence segmentation.
4. Generate word or model tokens.
5. Inspect token counts.
6. Review token grids.
7. Check for sparse, repeated, or fragmented token windows.
8. Return to Processing or Analysis if text quality is poor.
9. Proceed to Embeddings when token diagnostics look reasonable.

## 🧯 Troubleshooting

| Symptom                     | Likely Cause                                                             | Action                                                       |
| --------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------ |
| No tokens are produced      | `processed_text` is empty or missing                                     | Return to Processing and regenerate `processed_text`.        |
| Sentences are too long      | Punctuation was removed too aggressively or source text lacks delimiters | Reprocess with punctuation preservation or sentence cleanup. |
| Sentences are too short     | Source extraction created fragments                                      | Normalize whitespace and reduce symbols before tokenizing.   |
| Token grid is sparse        | Text has many blanks, fragments, or short chunks                         | Reprocess or adjust chunk size.                              |
| Token count is too high     | Text is too large for the intended workflow                              | Use chunking before embedding.                               |
| Repeated tokens dominate    | Headers, footers, or boilerplate remain                                  | Return to Processing and remove repeated artifacts.          |
| Important terms are missing | Cleanup may have removed too much content                                | Reduce aggressive processing options.                        |

## ✅ Tokenization Checklist

Before moving to Embeddings, confirm:

| Check                              | Complete |
| ---------------------------------- | -------- |
| `processed_text` exists            |          |
| Sentence rows were generated       |          |
| Sentences are readable             |          |
| Tokens were generated              |          |
| Token counts are reasonable        |          |
| Token grids are not mostly empty   |          |
| Repeated boilerplate is controlled |          |
| Text is ready for embedding        |          |

## 🧾 Summary

The Tokenization tab helps verify that processed text is structurally ready for embedding and
retrieval.

Use this stage to inspect sentence boundaries, token windows, token counts, and sparsity before
generating vectors. Good tokenization improves embedding quality and strengthens semantic search
results.
