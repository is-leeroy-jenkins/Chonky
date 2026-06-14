# Development

This page describes the development workflow for Chonky. It covers the local environment, source
layout, documentation build process, validation commands, logging pattern, and safe change practices
used to maintain the application.

Chonky combines a Streamlit application shell with reusable Python modules for document loading,
text processing, embedding generation, local vector persistence, and retrieval.

## 🧭 Purpose

The development workflow is designed to keep Chonky stable while source files, documentation, and UI
features evolve.

Development work should preserve:

| Area               | Requirement                                                                                  |
| ------------------ | -------------------------------------------------------------------------------------------- |
| Runtime behavior   | Do not change loader, processor, embedding, or vector behavior unless explicitly intended.   |
| Streamlit workflow | Preserve tab order, expanders, controls, and session-state contracts.                        |
| Documentation      | Keep Python docstrings compatible with MkDocs, mkdocstrings, and griffe.                     |
| Logging            | Preserve wrapped exception handling and write wrapped errors through the application logger. |
| Validation         | Compile source files and build documentation before publishing changes.                      |

## 🧱 Source Layout

The primary source files are located at the repository root.

```text id="o38qz0"
Chonky/
├── app.py
├── config.py
├── embedders.py
├── loca.py
├── loaders.py
├── processors.py
├── requirements.txt
├── mkdocs.yml
└── docs/
```

## 📦 Core Modules

| Module          | Development Role                                                                                                                                     |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app.py`        | Streamlit UI orchestration, tab layout, session-state coordination, loader calls, processor calls, embedding workflow, and vector database workflow. |
| `config.py`     | Runtime constants, paths, logging settings, provider keys, model options, required corpora, and session-state defaults.                              |
| `loaders.py`    | LangChain-backed document loaders and file/source normalization logic.                                                                               |
| `processors.py` | Text cleaning, tokenization, parsing, corpus analysis, PDF processing, and NLP utilities.                                                            |
| `embedders.py`  | Hosted embedding provider wrappers.                                                                                                                  |
| `loca.py`       | Local GGUF embedding wrappers using `llama-cpp-python`.                                                                                              |

## 🧰 Local Environment

Create a virtual environment from the repository root.

### Windows PowerShell

```powershell id="seql9l"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash id="pf3s8s"
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## ▶️ Run the Application

Run Chonky from the repository root.

```powershell id="a6wzk8"
python -m streamlit run app.py
```

The application starts the Streamlit interface and loads the tabbed workflow defined in `app.py`.

## 📚 Build the Documentation

Build the MkDocs site.

```powershell id="qnb1c8"
mkdocs build
```

Serve it locally during development.

```powershell id="qex7tq"
mkdocs serve
```

The documentation is generated from Markdown pages under `docs/` and API docstrings from the Python
source files using `mkdocstrings`.

## 🧾 Documentation Standard

Python source documentation should use Google-style docstrings that are compatible with
`mkdocstrings` and `griffe`.

Use this pattern:

```python id="sdirjy"
def load( self, filepath: str ) -> List[ Document ] | None:
	"""Load a source file into LangChain documents.

	Purpose:
		Validates the file path, initializes the loader, extracts source content, and returns
		LangChain ``Document`` objects for downstream text processing and embedding workflows.

	Args:
		filepath: Path to the source file.

	Returns:
		List[Document] | None: Loaded document objects.

	Raises:
		Error: Raised when validation or document loading fails.
	"""
```

Use these docstring section names only when applicable:

```text id="lbv6i9"
Purpose:
Args:
Attributes:
Returns:
Raises:
Notes:
Examples:
```

Avoid underline-style sections such as:

```text id="zgaqiv"
Parameters:
-----------
Returns:
--------
```

Those can cause griffe parsing warnings.

## 🧯 Logging Pattern

Existing wrapped exception handlers should write to the application logger before re-raising the
wrapped `Error`.

Use this pattern:

```python id="g5gd8l"
except Exception as e:
	exception = Error( e )
	exception.module = 'chonky'
	exception.cause = 'ClassName'
	exception.method = 'method_name( self, *args ) -> ReturnType'
	Logger( ).write( exception )
	raise exception
```

The `exception.method` field should contain a stable method signature. It should not contain live
data, file contents, full file paths, API keys, tokens, user text, dataframe contents, or object
memory addresses.

## 🔐 Configuration and Secrets

Provider keys and service credentials should be supplied through environment variables or Streamlit
UI inputs.

| Setting                          | Purpose                                 |
| -------------------------------- | --------------------------------------- |
| `OPENAI_API_KEY`                 | OpenAI embedding workflows.             |
| `GEMINI_API_KEY`                 | Gemini-related workflows.               |
| `GROQ_API_KEY`                   | Groq/Grok-compatible workflows.         |
| `GOOGLE_API_KEY`                 | Google-backed services where used.      |
| `GOOGLE_APPLICATION_CREDENTIALS` | Google service-account credential path. |
| `PINECONE_API_KEY`               | Pinecone configuration where enabled.   |

Do not commit secrets, API keys, service-account JSON files, local model binaries, vector databases,
or generated runtime logs.

## 🧪 Validation Commands

Run these checks after source changes.

```powershell id="innj2d"
python -m py_compile .\config.py
python -m py_compile .\embedders.py
python -m py_compile .\loca.py
python -m py_compile .\loaders.py
python -m py_compile .\processors.py
python -m py_compile .\app.py
```

Run a project-wide compile check.

```powershell id="dkc7af"
python -m compileall .
```

Build the documentation.

```powershell id="lq6jhi"
mkdocs build
```

Launch the app.

```powershell id="v1t39p"
python -m streamlit run app.py
```

## 🧭 Safe Change Workflow

Use this sequence for source changes:

| Step | Action                                                                       |
| ---- | ---------------------------------------------------------------------------- |
| 1    | Identify the module and workflow stage being changed.                        |
| 2    | Preserve public names, signatures, return contracts, and session-state keys. |
| 3    | Update docstrings when behavior or public parameters change.                 |
| 4    | Preserve existing exception handling and logging patterns.                   |
| 5    | Compile the changed module.                                                  |
| 6    | Run `python -m compileall .`.                                                |
| 7    | Run `mkdocs build`.                                                          |
| 8    | Launch Streamlit and verify the affected tab manually.                       |

## 🧩 Streamlit Session-State Contracts

The Streamlit application uses shared session state to pass data between workflow stages.

Common keys include:

| Key                 | Purpose                                                    |
| ------------------- | ---------------------------------------------------------- |
| `documents`         | Active LangChain document objects.                         |
| `raw_documents`     | Original loaded document objects before later processing.  |
| `raw_text`          | Combined raw text from the active loading workflow.        |
| `processed_text`    | Text after processing and cleanup.                         |
| `tokens`            | Tokenized text output.                                     |
| `vocabulary`        | Unique token or term collection.                           |
| `chunks`            | Text chunks produced for analysis or embedding.            |
| `chunked_documents` | Chunked LangChain document objects.                        |
| `embeddings`        | Generated embedding vectors.                               |
| `search_results`    | Semantic search output.                                    |
| `active_loader`     | Name of the loader that produced the current source state. |

When editing `app.py`, avoid reusing one session key for unrelated data. Each tab depends on
upstream state produced by earlier workflow stages.

## 📥 Loader Development Notes

Loader wrappers should preserve the LangChain `Document` contract.

Each loader should return:

```text id="q73e6q"
List[Document] | None
```

Document metadata should identify the source and loader where possible.

Common metadata keys include:

| Metadata Key | Purpose                                                      |
| ------------ | ------------------------------------------------------------ |
| `loader`     | Loader class or workflow name.                               |
| `source`     | Source file or source identifier.                            |
| `path`       | File path where already used by the existing implementation. |
| `mode`       | Loader mode where applicable.                                |
| `extract`    | Extraction strategy where applicable.                        |

## 🧹 Processor Development Notes

Processor methods should preserve their existing input and output contracts.

Text-processing changes should be evaluated carefully because downstream stages depend on stable
text behavior. Changes to punctuation handling, tokenization, lowercasing, stopword removal, or
chunking can affect analysis results and embedding output.

When adding new processor methods:

1. Add type hints.
2. Add a Google-style docstring.
3. Validate required parameters with `throw_if`.
4. Preserve safe exception wrapping.
5. Return a predictable type.

## 🧠 Embedding Development Notes

Hosted and local embedding wrappers should return list-based vector structures.

| Method Type      | Expected Shape      |
| ---------------- | ------------------- |
| Single embedding | `List[float]`       |
| Batch embeddings | `List[List[float]]` |

Embedding methods should preserve cleaned empty-input behavior. Blank text or a batch with no usable
strings should return an empty list when that is the existing contract.

## 🗄 Documentation Development Notes

The documentation site uses this structure:

```text id="p3p58l"
docs/
├── index.md
├── architecture.md
├── development.md
├── api/
├── assets/
├── img/
└── user-guide/
```

The API reference pages should remain source-driven:

Do not add long manual API descriptions that duplicate the Python docstrings. Keep API pages simple
and let `mkdocstrings` render the source documentation.

## 🚀 GitHub Pages Build

After local validation succeeds, publish the documentation with the project’s selected GitHub Pages
workflow.

Typical local validation before publishing:

```powershell id="ow75xk"
Remove-Item -Recurse -Force .\site -ErrorAction SilentlyContinue
mkdocs build
```

The generated `site/` directory should not be edited manually. Treat it as build output.

## ✅ Pre-Commit Checklist

Before committing source or documentation changes:

| Check                                    | Complete                         |
| ---------------------------------------- | -------------------------------- |
| Source files compile                     | `python -m compileall .`         |
| Documentation builds                     | `mkdocs build`                   |
| Streamlit starts                         | `python -m streamlit run app.py` |
| No griffe docstring warnings             | Review build output              |
| No missing nav pages                     | Review MkDocs output             |
| No broken image paths                    | Review MkDocs output             |
| No secrets committed                     | Review changed files             |
| No generated runtime artifacts committed | Review changed files             |


