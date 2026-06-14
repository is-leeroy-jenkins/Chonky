# Analysis

The Analysis tab is the third stage in the Chonky workflow. It helps inspect processed text before
tokenization, embedding generation, and vector search.

This stage focuses on chunking, vocabulary creation, frequency distributions, corpus metrics, and
other diagnostics that reveal whether the text is structurally ready for downstream semantic
workflows.

```text id="ma5va5"
processed_text
      │
      ▼
Semantic Analysis
      │
      ▼
chunks + vocabulary + frequency diagnostics
```

## 🧭 Purpose

The Analysis tab gives users visibility into the shape and quality of processed text.

It helps answer practical questions:

| Question                         | Why It Matters                                                             |
| -------------------------------- | -------------------------------------------------------------------------- |
| Is the processed text usable?    | Empty, noisy, or malformed text will produce weak embeddings.              |
| Are chunks the right size?       | Poor chunk boundaries reduce retrieval quality.                            |
| Which terms dominate the corpus? | Repeated terms, headers, or boilerplate may distort analysis.              |
| Is the vocabulary reasonable?    | Vocabulary size helps reveal sparsity, repetition, or extraction problems. |
| Are there obvious artifacts?     | Extraction noise should be fixed before embedding.                         |

## 🧱 Workflow Position

Analysis follows Processing and precedes Tokenization and Embeddings.

```text id="k23zge"
Loading
  → Processing
  → Analysis
  → Tokenization
  → Embeddings
  → Vector Database
```

The Analysis tab assumes that the Processing tab has produced `processed_text`.

## 📥 Primary Input

The Analysis tab primarily consumes:

| Input            | Source                              | Purpose                                                          |
| ---------------- | ----------------------------------- | ---------------------------------------------------------------- |
| `processed_text` | Processing tab                      | Cleaned text used for chunking and diagnostics.                  |
| `documents`      | Loading tab                         | Active document objects where document-aware analysis is needed. |
| `tokens`         | Processing or tokenization workflow | Token list used by frequency and vocabulary diagnostics.         |
| `active_loader`  | Loading tab                         | Identifies the source workflow that produced the current text.   |

If `processed_text` is missing or empty, return to the Processing tab and regenerate it from
`raw_text`.

## 📤 Primary Output

The Analysis tab can produce:

| Output              | Purpose                                                                 |
| ------------------- | ----------------------------------------------------------------------- |
| `chunks`            | Text chunks produced for inspection, embedding, or retrieval workflows. |
| `chunked_documents` | LangChain `Document` chunks where document-aware splitting is used.     |
| `vocabulary`        | Unique token or term set.                                               |
| `token_counts`      | Token frequency information.                                            |
| `df_frequency`      | Frequency distribution table.                                           |
| `df_chunks`         | Chunk preview or chunk diagnostics table.                               |

These outputs help determine whether the text should move forward to tokenization and embeddings.

## 🧩 Chunking

Chunking divides long text into smaller sections.

Chunks are useful because embedding models and retrieval workflows work better with focused text
windows than with large, mixed-topic documents.

| Chunking Method  | Description                                                             |
| ---------------- | ----------------------------------------------------------------------- |
| Character chunks | Splits text into fixed-size character windows with optional overlap.    |
| Token chunks     | Splits token lists into fixed-size token windows with optional overlap. |
| Document chunks  | Splits LangChain `Document` objects while preserving document metadata. |

## 📏 Chunk Size and Overlap

Chunk size and overlap control how text is divided.

| Setting            | Effect                                                                |
| ------------------ | --------------------------------------------------------------------- |
| Larger chunk size  | More context per chunk, but less precise retrieval.                   |
| Smaller chunk size | More precise retrieval, but less surrounding context.                 |
| More overlap       | Preserves continuity between chunks, but increases duplicate content. |
| Less overlap       | Reduces duplication, but may split related ideas.                     |

A practical starting point is to use moderate chunk sizes and enough overlap to avoid cutting
sentences or concepts too sharply.

## 🔍 Chunk Review

After chunking, inspect the chunk output before creating embeddings.

Good chunks should be:

| Quality                | Description                                                            |
| ---------------------- | ---------------------------------------------------------------------- |
| Readable               | The chunk should contain coherent text.                                |
| Focused                | The chunk should not mix unrelated sections when avoidable.            |
| Complete enough        | The chunk should preserve enough context to make sense.                |
| Not mostly boilerplate | Headers, footers, navigation, and repeated labels should not dominate. |
| Not empty or sparse    | Blank or very short chunks should be removed or avoided.               |

Poor chunks usually indicate that the source text needs more cleanup or different chunk settings.

## 📚 Vocabulary

Vocabulary analysis identifies the unique terms present in the active text.

Vocabulary can help reveal:

| Signal                     | Meaning                                                           |
| -------------------------- | ----------------------------------------------------------------- |
| Very small vocabulary      | Text may be too short, repetitive, or over-cleaned.               |
| Very large vocabulary      | Text may contain noise, identifiers, artifacts, or mixed content. |
| Many malformed terms       | Extraction or encoding cleanup may be needed.                     |
| Repeated boilerplate terms | Headers, footers, or navigation may still be present.             |

Vocabulary review is useful before embedding because dominant artifacts can affect semantic
similarity.

## 📊 Frequency Distribution

Frequency distributions show how often tokens appear.

Common uses include:

| Use                             | Description                                                                        |
| ------------------------------- | ---------------------------------------------------------------------------------- |
| Identify dominant terms         | Find words that appear most often.                                                 |
| Detect boilerplate              | Repeated page headers, footers, and navigation can appear as high-frequency terms. |
| Review source-specific language | Confirm that important domain terms appear in the text.                            |
| Check cleanup effects           | Compare frequency results before and after processing choices.                     |

Frequency analysis should be treated as a diagnostic tool, not as a replacement for reading the
processed text.

## 📈 Corpus Diagnostics

Corpus diagnostics help evaluate whether text is ready for tokenization and embedding.

Useful diagnostics include:

| Diagnostic           | Purpose                                            |
| -------------------- | -------------------------------------------------- |
| Character count      | Measures overall text size.                        |
| Token count          | Estimates model and embedding workload.            |
| Unique token count   | Measures vocabulary breadth.                       |
| Type-token ratio     | Compares vocabulary size to total token count.     |
| Average token length | Helps identify artifacts or abnormal tokenization. |
| Stopword ratio       | Shows how much common language remains.            |
| Lexical density      | Estimates information-bearing word concentration.  |
| Readability metrics  | Provides a rough text-complexity signal.           |

These metrics are most useful when compared across processing attempts.

## 🧠 Preparing for Embeddings

The Analysis tab is the best place to catch problems before embeddings are generated.

Before moving to Embeddings, confirm:

| Check                                                | Reason                                                     |
| ---------------------------------------------------- | ---------------------------------------------------------- |
| Chunks are readable                                  | Embeddings represent chunk content.                        |
| Chunk sizes are reasonable                           | Oversized or tiny chunks can weaken retrieval.             |
| Repeated boilerplate is reduced                      | Repetition can dominate vector similarity.                 |
| Vocabulary looks plausible                           | Malformed or noisy vocabulary indicates extraction issues. |
| Frequency distribution is not dominated by artifacts | Dominant artifacts can distort search.                     |
| Processed text still contains meaningful content     | Over-cleaning can remove useful information.               |

## 🛠 Recommended Analysis Workflow

Use this sequence:

1. Confirm `processed_text` exists.
2. Generate chunks using a reasonable size and overlap.
3. Review chunk previews.
4. Generate vocabulary diagnostics.
5. Review frequency distribution.
6. Look for extraction artifacts or repeated boilerplate.
7. Return to Processing if text quality is poor.
8. Proceed to Tokenization or Embeddings when analysis outputs look reasonable.

## 🧯 Troubleshooting

| Symptom                                      | Likely Cause                                  | Action                                                               |
| -------------------------------------------- | --------------------------------------------- | -------------------------------------------------------------------- |
| No chunks are produced                       | `processed_text` is empty or missing          | Return to Processing and regenerate `processed_text`.                |
| Chunks are too short                         | Chunk size is too small or text is fragmented | Increase chunk size or improve cleanup.                              |
| Chunks are too large                         | Chunk size is too large                       | Reduce chunk size before embedding.                                  |
| Chunks repeat the same header                | Source has repeated page headers or footers   | Return to Processing and remove repeated headers/footers.            |
| Vocabulary contains broken words             | Extraction or encoding artifacts remain       | Apply encoding cleanup, whitespace normalization, or symbol cleanup. |
| Frequency table is dominated by common words | Stopwords or boilerplate remain               | Apply stopword cleanup or remove repeated boilerplate.               |
| Important terms are missing                  | Processing may be too aggressive              | Reduce cleanup options and rerun analysis.                           |

## ✅ Analysis Checklist

Before moving to Tokenization or Embeddings, confirm:

| Check                                                | Complete |
| ---------------------------------------------------- | -------- |
| `processed_text` exists                              |          |
| Chunks were generated                                |          |
| Chunks are readable                                  |          |
| Chunk size and overlap are reasonable                |          |
| Vocabulary looks plausible                           |          |
| Frequency distribution is not dominated by artifacts |          |
| Text quality is good enough for embedding            |          |

## 🧾 Summary

The Analysis tab helps verify that Chonky’s processed text is structurally ready for tokenization,
embedding, and retrieval.

This stage is where users should inspect chunks, vocabulary, frequency distributions, and corpus
diagnostics before committing the text to vector generation.
