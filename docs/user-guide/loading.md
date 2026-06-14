# Loading

The Loading tab is the first stage in the Chonky workflow. It converts source material into
LangChain `Document` objects and prepares the shared raw-text state used by processing, analysis,
tokenization, embeddings, and vector search.

The loading workflow is designed to normalize different document types into a consistent application
contract.

```text id="trzv8v"
Source Material
      │
      ▼
Loader Wrapper
      │
      ▼
LangChain Document Objects
      │
      ▼
raw_text + raw_documents + documents
```

## 🧭 Purpose

The Loading tab provides a single place to ingest documents from local files, corpora, web sources,
notebooks, email, cloud storage, and public data services.

It produces the core state used by the rest of the application:

| Output          | Purpose                                                              |
| --------------- | -------------------------------------------------------------------- |
| `documents`     | Active LangChain `Document` objects used by downstream tabs.         |
| `raw_documents` | Original loaded documents before later processing or transformation. |
| `raw_text`      | Combined text buffer extracted from the active document set.         |
| `active_loader` | Name of the loader that produced the current application state.      |

A successful loading step should produce readable `raw_text` before any text processing or embedding
work begins.

## 🧱 Workflow Position

Loading is the first stage in the Chonky pipeline.

```text id="n16ujs"
Loading
  → Processing
  → Analysis
  → Tokenization
  → Embeddings
  → Vector Database
```

Downstream tabs assume that the Loading tab has already produced usable document state.

## 📦 Loader Module

The loading layer is implemented in:

```text id="w6022r"
loaders.py
```

The loader classes wrap LangChain document loaders and related source-specific extraction logic.
Each loader is responsible for validating the source, extracting text, adding metadata, and
returning LangChain `Document` objects.

Common loader behavior includes:

| Behavior             | Description                                                     |
| -------------------- | --------------------------------------------------------------- |
| Path validation      | Confirms local files exist before loading.                      |
| Source normalization | Converts source-specific content into `Document` objects.       |
| Metadata assignment  | Tracks loader name, source, path, mode, or extraction strategy. |
| Chunk support        | Supports splitting loaded documents where available.            |
| Error wrapping       | Raises logged application errors when loading fails.            |

## 📥 Supported Source Categories

Chonky supports a wide range of source types.

| Category         | Supported Sources                                                  |
| ---------------- | ------------------------------------------------------------------ |
| Local documents  | Text, CSV, PDF, Word, Excel, PowerPoint, Markdown, HTML, JSON, XML |
| Corpora          | Brown, Gutenberg, Reuters, WebText, Inaugural, State of the Union  |
| Web sources      | Web pages, recursive crawls, Wikipedia, ArXiv, GitHub, PubMed      |
| Notebook sources | Jupyter notebooks                                                  |
| Email sources    | Outlook and email files                                            |
| Cloud sources    | Google Cloud Storage, AWS S3, Google Drive, SharePoint, OneDrive   |
| Public data      | Open City Data                                                     |

## 📝 Text Loader

Use the Text Loader for plain text files.

Supported examples:

```text id="hiumcg"
.txt
.text
.log
```

The Text Loader reads the file content and returns a single LangChain `Document` with text content
and source metadata.

Typical output:

```text id="rkdxmc"
documents
raw_documents
raw_text
active_loader = TextLoader
```

## 📑 CSV Loader

Use the CSV Loader for delimited records.

The loader supports delimiter and quote-character controls. This is useful when processing exported
tables, flat files, or structured records that need to become document text.

Common controls include:

| Control         | Purpose                                                   |
| --------------- | --------------------------------------------------------- |
| Delimiter       | Defines the separator between fields.                     |
| Quote Character | Defines the quoting character used by the CSV file.       |
| Load            | Loads the uploaded CSV into documents.                    |
| Clear           | Clears the active CSV state when it is the active loader. |
| Save            | Exports the current raw text when available.              |

## 🧬 XML Loader

Use the XML Loader for XML files.

Chonky supports two XML paths:

| Mode                        | Purpose                                                                                   |
| --------------------------- | ----------------------------------------------------------------------------------------- |
| Semantic XML loading        | Uses unstructured loading to create document objects suitable for analysis and embedding. |
| Structured XML tree loading | Parses XML into an `lxml` tree for XPath-style inspection.                                |

The XML workflow can support both document extraction and structured element review.

Common XML outputs include:

```text id="a9rzu8"
xml_documents
xml_split_documents
xml_tree_loaded
xml_namespaces
xml_xpath_results
```

## 📘 Word Loader

Use the Word Loader for `.docx` documents.

The loader extracts text from Word documents and stores the result as LangChain `Document` objects.
It is useful for memos, reports, correspondence, requirements documents, and other narrative source
material.

Typical metadata includes:

| Metadata | Purpose                                             |
| -------- | --------------------------------------------------- |
| `loader` | Identifies the loader as `WordLoader`.              |
| `source` | Stores the uploaded file name or source identifier. |

## 📕 PDF Loader

Use the PDF Loader for `.pdf` files.

PDFs often require special handling because extracted text can include headers, footers, page
breaks, hyphenation, layout artifacts, repeated text, or poor spacing.

Chonky supports geometry-aware extraction and legacy loader paths.

Common PDF controls include:

| Control                 | Purpose                                                       |
| ----------------------- | ------------------------------------------------------------- |
| Mode                    | Selects document-level or page-level extraction behavior.     |
| Extract                 | Controls plain or layout-aware extraction where supported.    |
| Include Images          | Enables image-related extraction where supported.             |
| Use Geometry Extraction | Uses page geometry for layout-aware parsing.                  |
| Header Band             | Defines the top region treated as a candidate header band.    |
| Footer Band             | Defines the bottom region treated as a candidate footer band. |
| Preserve Page Breaks    | Keeps explicit page-break boundaries in extracted text.       |

After loading, inspect `raw_text` before processing. PDF extraction quality can vary significantly
by source document.

## 📽 PowerPoint Loader

Use the PowerPoint Loader for `.pptx` files.

This loader extracts slide text into document objects. It is useful for presentations, briefings,
training material, and slide-based documentation.

The output can then move through the same processing, analysis, embedding, and retrieval workflow as
other document sources.

## 📓 Jupyter Notebook Loader

Use the Jupyter Notebook Loader for `.ipynb` files.

Notebook loading can include or exclude outputs and tracebacks.

Common controls include:

| Control           | Purpose                                       |
| ----------------- | --------------------------------------------- |
| Include Outputs   | Includes notebook cell outputs when selected. |
| Max Output Length | Limits output text length.                    |
| Remove Newline    | Controls newline handling.                    |
| Include Traceback | Includes traceback output when selected.      |

This loader is useful when analyzing notebook code, markdown cells, generated outputs, or experiment
notes.

## 📊 Excel Loader

Use the Excel Loader for `.xlsx` and `.xls` files.

Chonky supports two spreadsheet workflows:

| Mode                  | Purpose                                                                    |
| --------------------- | -------------------------------------------------------------------------- |
| Tabular + SQLite      | Loads sheets into SQLite tables and creates document text from sheet data. |
| Unstructured Document | Uses an unstructured document loader to extract spreadsheet content.       |

The tabular workflow is useful when preserving sheet/table structure is important. The unstructured
workflow is useful when spreadsheet content should be treated as narrative or semi-structured text.

## 🧾 Markdown Loader

Use the Markdown Loader for `.md` and `.markdown` files.

Markdown files are useful sources for repository documentation, notes, project plans, dataset cards,
and technical guides.

After loading, the Processing tab can remove Markdown syntax if the downstream workflow requires
plain text.

## 🌐 HTML Loader

Use the HTML Loader for `.html` and `.htm` files.

HTML loading extracts page content from local HTML files. Use Processing options afterward to remove
tags, formatting artifacts, and boilerplate content.

## 🧩 JSON Loader

Use the JSON Loader for `.json` or `.jsonl` files.

JSON loading is useful for structured text records, exported conversations, event logs,
metadata-rich documents, or datasets with nested fields.

Typical controls include:

| Control     | Purpose                                     |
| ----------- | ------------------------------------------- |
| jq Schema   | Selects the JSON structure to extract.      |
| Content Key | Identifies the field used as document text. |
| JSON Lines  | Treats the file as newline-delimited JSON.  |

## 🌍 Web and Public Loaders

Chonky supports web and public-data loading through several wrappers.

| Loader Type       | Use                                                    |
| ----------------- | ------------------------------------------------------ |
| Web pages         | Load one or more URLs.                                 |
| Recursive crawler | Crawl linked pages from a starting URL.                |
| Wikipedia         | Load article content.                                  |
| ArXiv             | Load research-paper metadata and text where available. |
| GitHub            | Load files from repositories.                          |
| PubMed            | Load biomedical publication records.                   |
| Open City Data    | Load supported public-data resources.                  |

Web and public sources can include boilerplate or navigation text, so review `raw_text` and apply
processing before embedding.

## ☁️ Cloud and Connected Sources

Chonky includes loader wrappers for connected document sources and object storage.

| Source               | Purpose                                            |
| -------------------- | -------------------------------------------------- |
| Google Cloud Storage | Load files or objects from Google Cloud buckets.   |
| AWS S3               | Load files or directories from S3.                 |
| Google Drive         | Load documents from Drive-backed sources.          |
| SharePoint           | Load SharePoint-hosted documents where configured. |
| OneDrive             | Load OneDrive documents where configured.          |

Cloud-backed workflows may require credentials or environment configuration before loading succeeds.

## 📤 Loading Outputs

The Loading tab writes shared state used by later workflow stages.

| State Key           | Description                                                 |
| ------------------- | ----------------------------------------------------------- |
| `documents`         | Current active LangChain document list.                     |
| `raw_documents`     | Original loaded document list.                              |
| `raw_text`          | Combined document text.                                     |
| `active_loader`     | Loader name that produced the current state.                |
| `processed_text`    | Usually reset after new loading so processing can be rerun. |
| `lines`             | Cleared or reset where loader output changes text scope.    |
| `chunked_documents` | Cleared or reset when source material changes.              |
| `df_chunks`         | Cleared or reset when source material changes.              |

## 🧪 Recommended Loading Workflow

Use this sequence when loading new source material:

1. Select the appropriate loader.
2. Upload or enter the source input.
3. Configure loader-specific options.
4. Click **Load**.
5. Confirm that documents were loaded.
6. Inspect `raw_text`.
7. Clear the loader state if the wrong source was loaded.
8. Proceed to Processing only after text extraction looks usable.

## 🧯 Troubleshooting

| Symptom                               | Likely Cause                                                 | Action                                                     |
| ------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| No documents loaded                   | Missing file, wrong file type, empty source, or invalid path | Confirm the file exists and use the correct loader.        |
| Raw text is empty                     | Source content could not be extracted                        | Try a different loading mode or inspect the source file.   |
| PDF text is poorly spaced             | PDF layout or extraction artifacts                           | Try geometry extraction and adjust header/footer controls. |
| XML XPath returns nothing             | XML tree was not loaded or namespace prefixes are missing    | Load the XML tree and inspect namespaces.                  |
| Web source fails                      | URL is unavailable, blocked, or slow                         | Check the URL and retry with a simpler page.               |
| Cloud source fails                    | Missing credentials or incorrect object path                 | Confirm credential settings and storage path.              |
| Existing downstream results disappear | New loading changed the active source state                  | Rerun Processing, Analysis, Tokenization, and Embeddings.  |

## ✅ Loading Checklist

Before moving to the Processing tab, confirm:

| Check                               | Complete |
| ----------------------------------- | -------- |
| Correct loader selected             |          |
| Source loaded without error         |          |
| `raw_text` is populated             |          |
| Text is readable enough for cleanup |          |
| Metadata appears reasonable         |          |
| Wrong or stale loader state cleared |          |

## 🧾 Summary

The Loading tab is the foundation of Chonky’s workflow. It standardizes many source types into
LangChain `Document` objects and produces the raw text used by every later stage.

Good loading results make processing, analysis, tokenization, embedding, and retrieval more
reliable.
