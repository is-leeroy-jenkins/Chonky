# User Guide

The Chonky user guide explains how to move through the application from document loading to
retrieval-ready outputs.

Chonky is organized around a staged workflow. Each tab produces state that later tabs consume. The
intended sequence is:

```text
Loading
  → Processing
  → Analysis
  → Tokenization
  → Embeddings
  → Vector Database
```

## 🧭 Purpose

This guide helps users understand what each Chonky workflow stage does, what inputs it expects, what
outputs it produces, and how those outputs support downstream analysis, embedding, and semantic
retrieval.

Use this guide when you need to:

| Task                                                    | Where to Start                        |
| ------------------------------------------------------- | ------------------------------------- |
| Load source documents                                   | [Loading](loading.md)                 |
| Clean and normalize text                                | [Processing](processing.md)           |
| Inspect chunks, vocabulary, and frequency distributions | [Analysis](analysis.md)               |
| Review token and sentence diagnostics                   | [Tokenization](tokenization.md)       |
| Generate hosted or local embedding vectors              | [Embeddings](embeddings.md)           |
| Store vectors and run semantic search                   | [Vector Database](vector-database.md) |

## 🧱 Workflow Overview

Chonky follows a document-intelligence pipeline.

| Stage | Tab             | Primary Input                                                           | Primary Output                              |
| ----- | --------------- | ----------------------------------------------------------------------- | ------------------------------------------- |
| 1     | Loading         | Source files, URLs, corpora, cloud objects, notebooks, or email content | `documents`, `raw_documents`, `raw_text`    |
| 2     | Processing      | `raw_text`                                                              | `processed_text`                            |
| 3     | Analysis        | `processed_text`                                                        | chunks, vocabulary, frequency distributions |
| 4     | Tokenization    | `processed_text`, chunks, or tokenized text                             | token rows, sentence rows, token metrics    |
| 5     | Embeddings      | processed text or chunks                                                | embedding vectors                           |
| 6     | Vector Database | embedding vectors and source text                                       | persisted vectors, search results           |

## 📥 Loading First

The Loading tab is the starting point for most workflows.

It converts source material into LangChain `Document` objects and builds the shared raw-text buffer
used by downstream tabs.

Chonky supports local files, corpora, PDFs, Word documents, spreadsheets, notebooks, web pages,
public sources, email files, and cloud-backed documents.

Typical loading outputs are:

```text
documents
raw_documents
raw_text
active_loader
```

Do not proceed to processing until the expected text appears in the application.

## 🧹 Processing Second

The Processing tab converts `raw_text` into cleaner `processed_text`.

Use this stage to remove formatting artifacts, normalize whitespace, remove markup, strip stopwords,
handle punctuation, remove headers and footers, or prepare text for tokenization and embedding.

Processing is especially important for content extracted from:

| Source        | Common Issue                                                  |
| ------------- | ------------------------------------------------------------- |
| PDFs          | Repeated headers, page breaks, hyphenation, spacing artifacts |
| HTML          | Tags, navigation text, scripts, boilerplate                   |
| Markdown      | Formatting syntax and image references                        |
| XML           | Tags and nested markup                                        |
| OCR-like text | Encoding artifacts, fragmented tokens, repeated symbols       |
| Spreadsheets  | Column-derived text that may need normalization               |

## 📊 Analyze Before Embedding

The Analysis tab helps evaluate text quality and structure before creating vectors.

Use it to inspect:

| Diagnostic             | Purpose                                       |
| ---------------------- | --------------------------------------------- |
| Chunks                 | Confirm text is split into usable windows.    |
| Vocabulary             | Understand the unique term set.               |
| Frequency distribution | Identify dominant terms and repeated content. |
| Corpus metrics         | Evaluate density, readability, and structure. |

This stage helps catch poor extraction or excessive noise before embedding vectors are created.

## 🔢 Tokenization Diagnostics

The Tokenization tab provides a closer look at sentence and token structure.

Use it when you need to understand whether text is ready for embedding or model input.

Useful checks include:

| Check                 | Purpose                                                        |
| --------------------- | -------------------------------------------------------------- |
| Sentence segmentation | Confirm sentences are separated correctly.                     |
| Token grids           | Inspect fixed-width token windows.                             |
| Token counts          | Estimate model input size.                                     |
| Sparsity diagnostics  | Identify blank, short, or uneven text regions.                 |
| Readiness metrics     | Determine whether the text should be cleaned or chunked again. |

## 🧠 Embedding Generation

The Embeddings tab converts text or chunks into numerical vectors.

Chonky supports two embedding paths:

| Path             | Module         | Use Case                                                 |
| ---------------- | -------------- | -------------------------------------------------------- |
| Hosted providers | `embedders.py` | OpenAI, Gemini, and Grok-compatible embedding workflows. |
| Local providers  | `loca.py`      | GGUF embedding models through `llama-cpp-python`.        |

Hosted providers require API credentials. Local providers require the expected GGUF model files to
exist at the configured local paths.

## 🗄 Vector Storage and Search

The Vector Database tab stores embedding vectors and supports similarity search.

Chonky uses `sqlite-vec` for local vector persistence. This gives the application a lightweight
retrieval layer without requiring a separate vector database service.

The vector workflow supports:

| Capability         | Description                                          |
| ------------------ | ---------------------------------------------------- |
| Vector persistence | Save embedding vectors and associated chunk text.    |
| Similarity search  | Retrieve chunks close to a query vector.             |
| Search review      | Inspect semantically relevant source content.        |
| Reusable output    | Preserve retrieval-ready chunks for downstream work. |

## 🧩 Typical Workflow

A standard Chonky workflow looks like this:

1. Load a file, corpus, web page, notebook, email, or cloud document.
2. Confirm `raw_text` was extracted.
3. Clean and normalize the text.
4. Review processed text.
5. Generate chunks and frequency diagnostics.
6. Inspect token and sentence structure.
7. Generate embedding vectors.
8. Store vectors in the local vector database.
9. Run semantic search.
10. Review retrieved chunks.

## 🛠 Recommended Practices

| Practice                                      | Reason                                                         |
| --------------------------------------------- | -------------------------------------------------------------- |
| Start with one document                       | Confirms loader behavior before batch workflows.               |
| Inspect raw text before processing            | Helps identify extraction problems early.                      |
| Process text before embedding                 | Reduces noise and improves retrieval quality.                  |
| Review chunks before vectorization            | Prevents poor chunk boundaries from entering the vector store. |
| Use token diagnostics                         | Helps avoid sparse or oversized chunks.                        |
| Keep provider credentials out of source files | Prevents accidental credential exposure.                       |
| Rebuild documentation after source changes    | Confirms docstrings still render correctly.                    |

## 🧯 Troubleshooting

| Symptom                     | Likely Cause                                                        | Action                                              |
| --------------------------- | ------------------------------------------------------------------- | --------------------------------------------------- |
| No raw text appears         | Loader did not extract usable content                               | Try another loader mode or inspect the source file. |
| Processed text is empty     | Cleanup settings removed too much text                              | Reduce processing options and retry.                |
| Chunks look too short       | Chunk size is too small or source text is fragmented                | Increase chunk size or improve text cleanup.        |
| Embeddings fail             | Missing API key, invalid provider model, or missing local GGUF file | Check provider settings and model paths.            |
| Search returns weak matches | Source text may be noisy or chunks may be poor                      | Reprocess text and regenerate embeddings.           |
| MkDocs warns about griffe   | A docstring contains malformed Google-style sections                | Fix the referenced source docstring.                |

## 🧾 Summary

The user guide follows the same sequence as the Chonky application. Start with loading, clean the
extracted text, analyze corpus structure, inspect tokenization, generate embeddings, and then
persist vectors for semantic retrieval.

Each stage is intentionally separated so users can inspect results before committing data to the
next stage.
