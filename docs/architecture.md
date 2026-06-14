![Chonky Architecture](img/chonky-architecture.png)

___

Chonky is organized as a staged document-processing pipeline. The application moves content from
source documents into normalized text, semantic chunks, token diagnostics, embedding vectors, vector
storage, and retrieval-ready outputs.

The architecture keeps the Streamlit interface separate from the reusable processing modules.
`app.py` coordinates the user workflow, `config.py` centralizes settings and session defaults,
`loaders.py` handles document ingestion, `processors.py` handles text transformation and analysis,
`embedders.py` handles hosted embedding providers, and `loca.py` handles local GGUF embedding
models.

## 🧭 Purpose

The architecture is designed to support a complete document intelligence workflow:

1. Load documents from local, web, cloud, public-data, notebook, and email sources.
2. Normalize source material into consistent LangChain `Document` objects.
3. Clean and transform raw text into analysis-ready text.
4. Generate chunks, vocabulary, frequency distributions, and token diagnostics.
5. Produce hosted or local embedding vectors.
6. Persist vectors with `sqlite-vec`.
7. Run similarity search and return retrieval-ready chunks.

This structure keeps each workflow stage explicit, testable, and replaceable.

## 🧱 Core Workflow

| Stage             | Module                    | Responsibility                                                                                              | Primary Output                               |
| ----------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| Loading           | `loaders.py`              | Loads documents from files, web sources, public repositories, cloud storage, notebooks, email, and corpora. | `documents`, `raw_documents`, `raw_text`     |
| Text Processing   | `processors.py`           | Cleans, normalizes, parses, tokenizes, and prepares text for downstream analysis.                           | `processed_text`                             |
| Semantic Analysis | `processors.py`           | Builds chunks, vocabulary, frequency tables, and corpus diagnostics.                                        | `chunks`, `vocabulary`, `frequencies`        |
| Data Tokenization | `processors.py`           | Produces sentence rows, token grids, token metrics, and readiness diagnostics.                              | `tokens`, sentence rows, token metrics       |
| Tensor Embeddings | `embedders.py`, `loca.py` | Generates hosted or local embedding vectors from processed text and chunks.                                 | embedding vectors, reduced-space diagnostics |
| Vector Database   | `app.py`, `sqlite-vec`    | Persists vectors and supports semantic similarity search.                                                   | search results, retrieved chunks             |
| Retrieval Outputs | application workflow      | Returns semantically relevant chunks for analysis, retrieval, and RAG-ready downstream use.                 | relevant chunks, semantic matches            |

## 📥 Document Sources

Chonky accepts a broad range of document inputs through the loading layer.

| Source Type         | Examples                                                               |
| ------------------- | ---------------------------------------------------------------------- |
| Local files         | Text, CSV, PDF, Word, Excel, PowerPoint, Markdown, HTML, JSON, XML     |
| Corpora             | NLTK Brown, Gutenberg, Reuters, WebText, Inaugural, State of the Union |
| Web sources         | Web pages, recursive crawls, Wikipedia, ArXiv, PubMed, GitHub          |
| Notebook sources    | Jupyter notebooks with optional outputs and tracebacks                 |
| Email sources       | Outlook messages and email files                                       |
| Cloud sources       | Google Cloud Storage, AWS S3, Google Drive, SharePoint, OneDrive       |
| Public-data sources | Open City Data and other supported public loaders                      |

Each loader converts source-specific content into a consistent document contract used by the rest of
the pipeline.

## 🧰 Loading Layer

The loading layer is implemented in `loaders.py`.

Its primary responsibility is to normalize heterogeneous inputs into LangChain `Document` objects.
Each document carries `page_content` and metadata that identifies the loader, source, file name,
parsing mode, or related extraction settings.

The loading layer supports both simple file ingestion and more complex workflows such as PDF
extraction, XML parsing, notebook loading, cloud-object loading, and recursive web crawling.

## 🧹 Text Processing Layer

The processing layer is implemented in `processors.py`.

This layer transforms raw document text into cleaner, more consistent text for analysis and
embedding. Processing operations include whitespace normalization, punctuation handling, symbol
removal, HTML cleanup, Markdown cleanup, XML text extraction, stopword removal, header/footer
cleanup, sentence splitting, tokenization, lemmatization, stemming, and related NLP preparation.

The processing layer is intentionally separated from the UI so parsing and cleanup logic can be
documented, tested, and reused independently.

## 📊 Semantic Analysis Layer

The semantic analysis layer prepares text for measurement and downstream embedding.

Typical outputs include:

| Output                  | Purpose                                                           |
| ----------------------- | ----------------------------------------------------------------- |
| Chunks                  | Break long text into smaller windows for embedding and retrieval. |
| Vocabulary              | Track unique terms and corpus shape.                              |
| Frequency distributions | Identify dominant terms and token patterns.                       |
| Corpus diagnostics      | Support readability, density, and readiness analysis.             |

This layer gives users visibility into the document before vectors are generated.

## 🔢 Data Tokenization Layer

The tokenization layer provides diagnostics between text processing and embedding generation.

It supports sentence segmentation, token grids, token counts, and readiness metrics. These
diagnostics help determine whether text is clean enough, sparse enough, or appropriately chunked for
embedding workflows.

Token diagnostics are especially useful when source material comes from PDFs, OCR-like extraction,
web pages, or highly formatted documents.

## 🧠 Embedding Layer

Chonky supports both hosted and local embedding workflows.

| Provider Type     | Module         | Providers                                 |
| ----------------- | -------------- | ----------------------------------------- |
| Hosted embeddings | `embedders.py` | OpenAI, Gemini, Grok-compatible workflows |
| Local embeddings  | `loca.py`      | Booger, Nomnom, Bobo GGUF wrappers        |

The hosted embedding layer calls external provider APIs. The local embedding layer uses
`llama-cpp-python` with GGUF embedding models. Both paths return vector lists that can be inspected,
reduced for diagnostics, and persisted for search.

## 🗄 Vector Database Layer

The vector database layer stores embedding vectors and associated text chunks.

Chonky uses `sqlite-vec` for local vector persistence. This keeps the retrieval workflow lightweight
and portable while still supporting similarity search over embedded document chunks.

The vector database layer connects embedding outputs to retrieval outputs.

## 🎯 Retrieval Outputs

The final stage returns retrieval-ready results.

These outputs can be used for:

| Output                  | Use                                                         |
| ----------------------- | ----------------------------------------------------------- |
| Relevant chunks         | Review the most semantically related document segments.     |
| Semantic matches        | Compare user queries against embedded content.              |
| Reusable embedding data | Preserve vectorized chunks for later search or analysis.    |
| RAG-ready context       | Prepare retrieved text for downstream generative workflows. |

## ⚙️ Cross-Cutting Modules

Two modules support the full workflow.

| Module      | Role                                                                                                                                             |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `app.py`    | Coordinates the Streamlit interface, tab layout, session state, loader calls, processor calls, embedding workflow, and vector database workflow. |
| `config.py` | Centralizes paths, constants, provider settings, model options, logging paths, and session-state defaults.                                       |

These modules are not a single pipeline stage. They support orchestration, configuration, and
application-level consistency across the entire system.

## 🧪 Architecture Validation

The architecture should be validated in three ways:

```powershell
python -m compileall .
mkdocs build
python -m streamlit run app.py
```

A successful build confirms that the Python modules compile, the documentation can be generated by
MkDocs, and the Streamlit workflow can still launch from the same source structure.

## 🧩 Design Principles

Chonky follows several practical design principles:

| Principle                   | Description                                                                         |
| --------------------------- | ----------------------------------------------------------------------------------- |
| Source-driven documentation | API pages are generated from Python docstrings using `mkdocstrings`.                |
| Explicit workflow stages    | Each major document-processing step is visible in the UI and documentation.         |
| Modular source files        | Loading, processing, embeddings, configuration, and UI orchestration are separated. |
| Local-first vector storage  | `sqlite-vec` supports portable local vector persistence.                            |
| Provider flexibility        | Hosted and local embedding providers share a common workflow role.                  |
| Inspection before embedding | Users can review text, tokens, chunks, and diagnostics before vector generation.    |

## 🧾 Summary

Chonky’s architecture is a modular document-intelligence pipeline. It begins with flexible document
ingestion, moves through cleaning and analysis, produces embedding vectors, stores those vectors
locally, and returns semantic retrieval outputs.

The design keeps the user interface, configuration, loaders, processors, hosted embeddings, and
local embeddings clearly separated while preserving a single end-to-end workflow for document
preparation, analysis, vectorization, and search.
