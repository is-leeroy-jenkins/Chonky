# Vector Database

The Vector Database tab is the final stage in the Chonky workflow. It stores embedding vectors,
preserves their relationship to source text, and supports semantic similarity search over embedded
document chunks.

This stage connects embedding generation to retrieval.

```text id="b21lxm"
embedding vectors + source chunks
          │
          ▼
Vector Database
          │
          ▼
semantic search results
```

## 🧭 Purpose

The Vector Database tab provides local vector persistence and retrieval for Chonky.

It supports:

| Capability                  | Description                                                                                 |
| --------------------------- | ------------------------------------------------------------------------------------------- |
| Vector persistence          | Stores embedding vectors and associated text records.                                       |
| Text-to-vector traceability | Keeps vectors connected to the source chunks that produced them.                            |
| Similarity search           | Retrieves semantically related chunks from stored vectors.                                  |
| Local retrieval workflow    | Uses local vector storage without requiring an external vector database service.            |
| RAG-ready output            | Returns relevant chunks that can be reused in downstream retrieval or generative workflows. |

The primary output is:

```text id="q6y6vg"
search_results
```

## 🧱 Workflow Position

The Vector Database tab follows Embeddings.

```text id="69t64w"
Loading
  → Processing
  → Analysis
  → Tokenization
  → Embeddings
  → Vector Database
```

This tab assumes embedding vectors have already been generated and that those vectors can be traced
back to source text or chunks.

## 📥 Primary Input

The Vector Database tab commonly consumes:

| Input                 | Source         | Purpose                                                          |
| --------------------- | -------------- | ---------------------------------------------------------------- |
| `embeddings`          | Embeddings tab | Vector values to persist or search.                              |
| `embedding_documents` | Embeddings tab | Document or chunk records associated with vectors.               |
| `df_embedding_output` | Embeddings tab | Tabular embedding output where available.                        |
| `embedding_provider`  | Embeddings tab | Provider that generated the vectors.                             |
| `embedding_model`     | Embeddings tab | Model that generated the vectors.                                |
| `embedding_source`    | Embeddings tab | Source text, chunks, or document set used for vector generation. |

The vector workflow is strongest when each vector maps to one readable chunk.

## 📤 Primary Output

The Vector Database tab can produce:

| Output                | Purpose                                                |
| --------------------- | ------------------------------------------------------ |
| persisted vector rows | Stored vector records and related text metadata.       |
| vector table          | Local table used for similarity search.                |
| `search_results`      | Results returned from semantic search.                 |
| retrieved chunks      | Source text chunks most similar to a query.            |
| similarity scores     | Ranking signals for retrieved matches where available. |

These outputs allow users to inspect semantic matches and reuse relevant chunks.

## 🗄 Local Vector Storage

Chonky uses `sqlite-vec` for local vector persistence.

`sqlite-vec` provides vector-search capability inside SQLite. This keeps the retrieval workflow
portable and local to the project without requiring a separate vector database service.

A typical vector record should preserve:

| Field            | Purpose                                                       |
| ---------------- | ------------------------------------------------------------- |
| vector           | Numerical embedding values.                                   |
| text             | Source chunk or text associated with the vector.              |
| provider         | Embedding provider used to create the vector.                 |
| model            | Embedding model used to create the vector.                    |
| source           | Source document or loader metadata where available.           |
| chunk identifier | Identifier that connects the vector back to the source chunk. |

## 🧩 Chunk-to-Vector Contract

The vector database workflow depends on a stable chunk-to-vector relationship.

```text id="p09jd0"
chunk_001 → embedding_001 → vector row 001
chunk_002 → embedding_002 → vector row 002
chunk_003 → embedding_003 → vector row 003
```

This contract matters because search results are only useful when the returned vector can be traced
back to readable text.

## 🔎 Similarity Search

Similarity search compares a query vector to stored vectors and returns the closest matches.

The workflow is:

```text id="9u38lm"
query text
   │
   ▼
query embedding
   │
   ▼
vector comparison
   │
   ▼
ranked matching chunks
```

Similarity search helps users find document sections related to a concept, question, phrase, or
topic even when the exact words differ.

## 📊 Search Results

Search results should be reviewed as ranked evidence, not as final answers.

Useful result fields include:

| Field            | Description                                                      |
| ---------------- | ---------------------------------------------------------------- |
| matched text     | Source chunk returned by the search.                             |
| source metadata  | Document, loader, page, table, or file metadata where available. |
| similarity score | Ranking value used to compare matches.                           |
| provider/model   | Embedding context used to generate the stored vector.            |
| chunk identifier | Traceability back to the source chunk.                           |

A strong search result should be semantically related to the query and readable enough to use as
source context.

## 🧠 Retrieval and RAG-Ready Outputs

Vector search produces retrieval-ready chunks that can support downstream workflows.

Common uses include:

| Use                 | Description                                               |
| ------------------- | --------------------------------------------------------- |
| Semantic review     | Find relevant sections across loaded documents.           |
| Evidence collection | Gather related chunks for manual analysis.                |
| Question answering  | Retrieve context that can support an answer.              |
| RAG workflows       | Supply retrieved chunks to a generative model as context. |
| Reuse               | Preserve vectorized chunks for later search.              |

Chonky’s retrieval output is designed to be inspectable before it is used elsewhere.

## 🧪 Before Persisting Vectors

Before writing vectors to storage, confirm:

| Check                            | Reason                                                     |
| -------------------------------- | ---------------------------------------------------------- |
| Embeddings exist                 | The database cannot store missing vectors.                 |
| Vector dimensions are consistent | A table should not mix models with different vector sizes. |
| Chunks are readable              | Search results return the associated text.                 |
| Empty chunks are excluded        | Empty text produces poor or unusable retrieval.            |
| Provider and model are known     | Stored vectors should retain generation context.           |
| Source metadata is available     | Results should be traceable to source material.            |

## 🛠 Recommended Vector Workflow

Use this sequence:

1. Load source material.
2. Process and clean extracted text.
3. Generate chunks.
4. Review chunk quality.
5. Generate embeddings.
6. Confirm vector count and dimensions.
7. Persist vectors to the local vector database.
8. Enter a search query.
9. Generate or use a query embedding.
10. Run similarity search.
11. Review retrieved chunks and metadata.

## 🧯 Troubleshooting

| Symptom                        | Likely Cause                                             | Action                                                          |
| ------------------------------ | -------------------------------------------------------- | --------------------------------------------------------------- |
| No vectors are available       | Embeddings were not generated                            | Return to the Embeddings tab and generate vectors.              |
| Vector storage fails           | Vector dimensions are inconsistent or data is malformed  | Regenerate embeddings with one model and validate vector shape. |
| Search returns no results      | Vector table is empty or query embedding failed          | Confirm vectors were persisted and the query is valid.          |
| Search results are irrelevant  | Source chunks are noisy or poorly sized                  | Reprocess, rechunk, and regenerate embeddings.                  |
| Duplicate results dominate     | Chunks contain repeated headers, footers, or boilerplate | Clean repeated text before embedding.                           |
| Results lack source context    | Metadata was not preserved                               | Review loader metadata and chunk-to-vector mapping.             |
| Database file cannot be opened | Path, permission, or lock issue                          | Confirm the database path and close competing processes.        |

## ✅ Vector Database Checklist

Before relying on search results, confirm:

| Check                                             | Complete |
| ------------------------------------------------- | -------- |
| Embeddings were generated                         |          |
| Vector dimensions are consistent                  |          |
| Vectors were persisted successfully               |          |
| Each vector maps to readable text                 |          |
| Provider and model metadata are known             |          |
| Search query is clear                             |          |
| Results are reviewed manually                     |          |
| Retrieved chunks are traceable to source material |          |

## 🧾 Summary

The Vector Database tab completes the Chonky workflow by storing embedding vectors and returning
semantically related chunks.

Good retrieval depends on every earlier stage: reliable loading, careful processing, useful
analysis, sound tokenization, and clean embeddings. When those stages are handled well, the vector
database can return useful, reviewable, retrieval-ready results.
