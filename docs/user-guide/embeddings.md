# Embeddings

The Embeddings tab is the fifth stage in the Chonky workflow. It converts processed text or text
chunks into numerical vectors that can be inspected, persisted, and searched.

Embeddings are the bridge between text processing and semantic retrieval. They allow Chonky to
compare text by meaning rather than by exact keyword overlap.

```text id="w9b3sl"
processed_text or chunks
        │
        ▼
Embedding Provider
        │
        ▼
embedding vectors
        │
        ▼
Vector Database
```

## 🧭 Purpose

The Embeddings tab generates vector representations of document text.

It supports:

| Capability                  | Description                                                        |
| --------------------------- | ------------------------------------------------------------------ |
| Hosted embedding generation | Uses provider APIs through `embedders.py`.                         |
| Local embedding generation  | Uses local GGUF models through `loca.py`.                          |
| Batch embedding             | Creates vectors for multiple chunks or text rows.                  |
| Embedding diagnostics       | Supports inspection of vector shape and reduced-space projections. |
| Vector database preparation | Produces vectors that can be persisted for similarity search.      |

The main output is:

```text id="tfuzjs"
embeddings
```

## 🧱 Workflow Position

Embeddings follows Tokenization and precedes the Vector Database workflow.

```text id="ehw5q4"
Loading
  → Processing
  → Analysis
  → Tokenization
  → Embeddings
  → Vector Database
```

The Embeddings tab assumes the text has been loaded, processed, reviewed, and chunked or tokenized
appropriately.

## 📥 Primary Input

The Embeddings tab can consume several upstream values.

| Input                | Source                   | Purpose                                            |
| -------------------- | ------------------------ | -------------------------------------------------- |
| `processed_text`     | Processing tab           | Cleaned text used when embedding one text body.    |
| `chunks`             | Analysis tab             | Text chunks used for batch embedding.              |
| `chunked_documents`  | Analysis tab             | Document chunks with metadata.                     |
| `df_embedding_input` | Embedding workflow       | Tabular input prepared for vector generation.      |
| provider settings    | Sidebar or configuration | API keys, model selections, and local model paths. |

For retrieval workflows, chunked input is usually better than embedding one large text body.

## 📤 Primary Output

The Embeddings tab can write:

| Output                | Purpose                                                     |
| --------------------- | ----------------------------------------------------------- |
| `embeddings`          | Generated vector values.                                    |
| `embedding_provider`  | Provider used to create the vectors.                        |
| `embedding_model`     | Model used to create the vectors.                           |
| `embedding_source`    | Source text or chunk set used for generation.               |
| `embedding_documents` | Document records associated with vectors.                   |
| `df_embedding_input`  | Input rows used for embedding generation.                   |
| `df_embedding_output` | Dataframe containing vector output and metadata where used. |

These outputs become the input to vector persistence and semantic search.

## 🧠 Hosted Embedding Providers

Hosted embedding providers are implemented in:

```text id="l2sdfc"
embedders.py
```

Supported hosted-provider wrappers include:

| Wrapper  | Purpose                                                              |
| -------- | -------------------------------------------------------------------- |
| `GPT`    | Creates embeddings through OpenAI embedding models.                  |
| `Gemini` | Creates embeddings through Google GenAI embedding models.            |
| `Grok`   | Provides a Grok/Groq-compatible embedding workflow where configured. |

Hosted providers usually require API keys or environment settings.

## 🖥 Local Embedding Providers

Local embedding providers are implemented in:

```text id="po6pyi"
loca.py
```

Supported local-provider wrappers include:

| Wrapper  | Model Role                                      |
| -------- | ----------------------------------------------- |
| `Booger` | Local BGE small English GGUF embedding wrapper. |
| `Nomnom` | Local Nomic GGUF embedding wrapper.             |
| `Bobo`   | Local Mixedbread GGUF embedding wrapper.        |

Local providers use `llama-cpp-python` and require the expected GGUF files to exist in the
configured model directories.

## 🔑 Provider Configuration

Hosted providers require credentials.

| Setting                                     | Purpose                                                                  |
| ------------------------------------------- | ------------------------------------------------------------------------ |
| `OPENAI_API_KEY`                            | Required for OpenAI embedding requests.                                  |
| `GEMINI_API_KEY` or related Google settings | Required for Gemini or Google-backed workflows where configured.         |
| `GROQ_API_KEY`                              | Required for Groq/Grok-compatible workflows where configured.            |
| `GOOGLE_APPLICATION_CREDENTIALS`            | Used by Google-backed services that require service-account credentials. |

Credentials can be supplied through environment variables or the Streamlit sidebar where supported.

Do not store provider secrets directly in source files.

## 📏 Single vs Batch Embeddings

Chonky supports both single-text and batch embedding patterns.

| Pattern          | Use Case                                                              |
| ---------------- | --------------------------------------------------------------------- |
| Single embedding | Useful for testing provider connectivity or embedding one text value. |
| Batch embedding  | Preferred for chunked documents and retrieval workflows.              |

Batch embedding is usually the better path for vector search because each vector maps to a smaller,
more focused text chunk.

## 🧩 Chunk-to-Vector Relationship

A strong retrieval workflow depends on a clean relationship between chunks and vectors.

```text id="n4hzas"
chunk_001 → embedding_001
chunk_002 → embedding_002
chunk_003 → embedding_003
```

Each vector should be traceable back to the text chunk that produced it. This makes search results
reviewable and reusable.

## 📊 Embedding Diagnostics

Embedding diagnostics help verify that vectors were created successfully.

Useful checks include:

| Diagnostic               | Purpose                                                              |
| ------------------------ | -------------------------------------------------------------------- |
| Vector count             | Confirms one vector was created for each expected text row or chunk. |
| Vector dimension         | Confirms the model returned the expected vector length.              |
| Empty-vector check       | Identifies failed or skipped inputs.                                 |
| Provider/model metadata  | Confirms the source of the embedding output.                         |
| Reduced-space projection | Supports visual inspection of vector relationships where available.  |

If vector count does not match the expected number of chunks, review the embedding input and
provider response.

## 🗄 Preparing for Vector Storage

Before moving to the Vector Database tab, confirm:

| Check                            | Reason                                                      |
| -------------------------------- | ----------------------------------------------------------- |
| Embeddings exist                 | No vectors means nothing can be persisted.                  |
| Vectors align to chunks          | Search results need text-to-vector traceability.            |
| Provider and model are known     | Stored vectors should preserve generation context.          |
| Empty inputs were removed        | Empty chunks should not enter the vector database.          |
| Vector dimensions are consistent | Mixed dimensions cannot be stored in the same vector table. |

A clean embedding output improves vector persistence and search quality.

## 🛠 Recommended Embedding Workflow

Use this sequence:

1. Load source material.
2. Process and clean the extracted text.
3. Generate and review chunks.
4. Inspect tokenization and readiness diagnostics.
5. Select an embedding provider.
6. Select the embedding model.
7. Generate embeddings.
8. Confirm vector count and shape.
9. Review provider/model metadata.
10. Proceed to Vector Database.

## 🧯 Troubleshooting

| Symptom                              | Likely Cause                                             | Action                                                       |
| ------------------------------------ | -------------------------------------------------------- | ------------------------------------------------------------ |
| Embedding request fails              | Missing API key or invalid provider settings             | Check credentials and provider configuration.                |
| Local model fails to load            | GGUF file is missing or path is incorrect                | Confirm the model file exists at the configured path.        |
| No embeddings are produced           | Input text or chunks are empty                           | Return to Processing or Analysis and regenerate usable text. |
| Vector count is lower than expected  | Blank chunks may have been skipped                       | Review chunk input and remove empty records.                 |
| Vectors have inconsistent dimensions | Different models were used in one output set             | Regenerate embeddings with one model.                        |
| Search results are weak              | Chunks may be noisy, too large, too small, or repetitive | Reprocess and rechunk before embedding again.                |
| Provider times out                   | Batch size or source volume may be too large             | Reduce batch size or embed fewer chunks.                     |

## ✅ Embedding Checklist

Before moving to Vector Database, confirm:

| Check                                                   | Complete |
| ------------------------------------------------------- | -------- |
| Text or chunks were reviewed                            |          |
| Provider is selected                                    |          |
| Model is selected                                       |          |
| Required credentials or local model files are available |          |
| Embeddings were generated                               |          |
| Vector count is reasonable                              |          |
| Vector dimensions are consistent                        |          |
| Embedding metadata is known                             |          |
| Output is ready for persistence                         |          |

## 🧾 Summary

The Embeddings tab turns Chonky’s processed text and chunks into vector representations.

Good embeddings depend on good upstream loading, processing, analysis, and tokenization. Review text
quality before generating vectors, and confirm vector output before storing embeddings in the vector
database.
