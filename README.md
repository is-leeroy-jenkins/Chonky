###### Chonky-Py
![](https://github.com/is-leeroy-jenkins/Chonky/blob/main/resources/images/github/chonky_project.png)
___


A modular text-processing framework baseed in python tailored for analysts, data scientists,
and machine learning practitioners working with unstructured text. Chonky provideds a text-processing 
pipeline from ingestion through preprocessing, tokenization, embedding, semantic inspection, and vector persistence.  It unifies high-performance NLP
utilities and machine learning-ready pipelines to support text ingestion, cleaning, tokenization,
feature extraction, and document analysis.

#### Local-setup
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/is-leeroy-jenkins/Chonky/blob/main/ipynb/pipes.ipynb)

#### Web-based
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://chonky-py.streamlit.app/)

## 🧠 Features

- **Text Preprocessing**: Clean and normalize text by removing HTML, punctuation, special
  characters, and stopwords.
- **Tokenization**: Sentence and word-level tokenization with support for HuggingFace and OpenAI
  tokenizers.
- **Chunking**: Token or word chunking for long document management and model input preparation.
- **Text Analysis**: Frequency distributions, conditional frequency analysis, TF-IDF, Word2Vec
  embeddings.
- **Multi-format Support**: Extracts from `.txt`, `.docx`, and `.pdf` files with high fidelity.
- **Custom Pipelines**: Utilities for JSONL export, batch cleaning, and document segmentation.
- **LLM-Compatible**: Includes OpenAI and HuggingFace tokenizer interfaces for seamless integration.

___
###### Demo
![](https://github.com/is-leeroy-jenkins/Chonky/blob/main/resources/images/Chonky-streamlit.gif)


## 🧰 Setup Instructions

To ensure a clean and isolated environment for running **Chonky**, follow these steps:

## ⚡ Clone the Repository

```

git clone https://github.com/yourusername/Chonky.git
cd Chonky
cd venv/Scripts
./activate.bat
cd ../../
pip install -r requirements.txt

```


## 🧠 `Text` 

- General-purpose text processor.
  - Methods: `load_text`, `normalize_text`, `remove_html`, `remove_punctuation`, `lemmatize_tokens`,
    `tokenize_text`, `chunk_text`, `create_word2vec`, `create_tfidf`, and more.
    | Method | Description |
    |--------|-------------|
    | `load_text(path)` | Loads raw text from a file. |
    | `split_lines(path)` | Splits text into individual lines. |
    | `split_pages(path, delimit)` | Splits text by page delimiters. |
    | `collapse_whitespace(text)` | Collapses multiple whitespaces into single spaces. |
    | `remove_punctuation(text)` | Removes punctuation from text. |
    | `remove_special(text)` | Removes special characters while preserving select symbols. |
    | `remove_html(text)` | Strips HTML tags using BeautifulSoup. |
    | `remove_errors(text)` | Removes non-English or misspelled words. |
    | `correct_errors(text)` | Attempts to autocorrect spelling using TextBlob. |
    | `remove_markdown(text)` | Strips Markdown syntax like `*`, `#`, etc. |
    | `remove_stopwords(text)` | Removes English stopwords. |
    | `remove_headers(pages)` | Removes repetitive headers/footers using frequency. |
    | `normalize_text(text)` | Normalizes text to lowercase ASCII. |
    | `lemmatize_tokens(tokens)` | Lemmatizes tokens using NLTK's WordNet. |
    | `tokenize_text(text)` | Cleans and tokenizes raw text. |
    | `tokenize_words(words)` | Tokenizes a list of words. |
    | `tokenize_sentences(text)` | Sentence tokenization using NLTK. |
    | `split_paragraphs(path)` | Splits text file into paragraphs. |
    | `chunk_text(text, size)` | Splits text into word chunks. |
    | `chunk_words(words, size)` | Chunks a list of tokenized words. |
    | `split_sentences(text)` | Returns a list of sentences. |
    | `compute_frequency_distribution(lines)` | Computes frequency of tokens. |
    | `compute_conditional_distribution(lines)` | Computes frequency grouped by condition. |
    | `create_vocabulary(freq_dist)` | Creates vocabulary list from token frequency. |
    | `create_wordbag(words)` | Constructs Bag-of-Words from token list. |
    | `create_word2vec(words)` | Trains a Word2Vec model from tokenized sentences. |
    | `create_tfidf(words)` | Generates TF-IDF matrix. |
    | `clean_files(src, dest)` | Batch cleans `.txt` files from source to destination. |
    | `convert_jsonl(src, dest)` | Converts text files into JSONL format. |
    | `visualize_embeddings`     |  Visualize word vectors using PCA (2D).  |
    | `filter_tokens`            | Removes stopwords and short tokens.  |
    | `vectorize_corpus`         | Converts all tokenized sentences into vector embeddings.|
    | `semantic_search`          | Find top-K similar sentences to query using cosine similarity.|
    | `encode_sentences`         | Generate contextual sentence embeddings w/ SentenceTransformer.|
 

## 📄 `Word` 

- Parses `.docx` files using Python-docx.
- Sentence segmentation, vocabulary extraction, frequency computation.

  | Method | Description |
  |--------|-------------|
  | `extract_text()` | Extracts text and paragraphs from a `.docx` file. |
  | `split_sentences()` | Splits extracted text into sentences. |
  | `clean_sentences()` | Cleans individual sentences: lowercases and removes punctuation. |
  | `create_vocabulary()` | Builds vocabulary list from cleaned sentences. |
  | `compute_frequency_distribution()` | Computes token frequency from sentences. |
  | `summarize()` | Prints summary stats: paragraphs, sentences, vocab size. |


## 📑 `PDF` 

- Reads `.pdf` files using `PyMuPDF`.
- Extracts structured or unstructured text and exports CSV/Excel.

| Method | Description |
|--------|-------------|
| `extract_lines(path, max)` | Extracts and cleans lines of text from PDF pages. |
| `extract_text(path, max)` | Extracts full concatenated text from PDF. |
| `extract_tables(path, max)` | Extracts tables into pandas DataFrames. |
| `export_csv(tables, filename)` | Exports extracted tables to CSV files. |
| `export_text(lines, path)` | Writes extracted text lines to a `.txt` file. |
| `export_excel(tables, path)` | Saves tables to an Excel workbook. |




## 🧪 Example Usage

```

    python
    from processing import Text
    processor = Text()
    text = processor.load_text("example.txt")
    clean = processor.remove_stopwords(text)
    tokens = processor.tokenize_words(clean)
    chunks = processor.chunk_text(clean, size=100)
    
```

## 🧩 Initialize Processor

```

  processor = Text()

```

## 📂 Load Raw Text

```

  raw_text = processor.load_text("data/sample.txt")
  
```

## 🧼 Clean & Normalize

```

  text = processor.remove_html(raw_text)                     # 🧹 Strip HTML
  text = processor.normalize_text(text)                      # 🔡 Lowercase + ASCII
  text = processor.remove_markdown(text)                     # ✨ Remove markdown (#, *, etc.)
  text = processor.remove_special(text)                      # ❌ Remove special chars
  text = processor.remove_punctuation(text)                  # 🪛 Remove punctuation
  text = processor.collapse_whitespace(text)                 # 📏 Collapse whitespace
  
```

## 🧠 Spelling & Stopwords

```
  cleaned_text = processor.remove_errors(text)               # 🧬 Remove misspellings
  corrected_text = processor.correct_errors(cleaned_text)    # 🔁 Auto-correct spelling
  no_stopwords_text = processor.remove_stopwords(corrected_text)  # 🚫 Remove stopwords
  
```

## ✂️ Tokenization

```
  word_tokens = processor.tokenize_words(no_stopwords_text)       # 🧩 Word tokens
  sentence_tokens = processor.tokenize_sentences(no_stopwords_text)  # 🧾 Sentence tokens
```

## 🌱 Lemmatization

```
  lemmatized_tokens = processor.lemmatize_tokens(word_tokens)
```


## 📦 Chunking

```  
  text_chunks = processor.chunk_text(no_stopwords_text, max=800)   # 🧳 Word chunked text
  word_chunks = processor.chunk_words(word_tokens, max=100, over=50)  # 🎒 Token chunks
```

## 📚 Structural Splitting

```
  line_groups = processor.split_lines("data/sample.txt")           # 📏 Lines
  paragraphs = processor.split_paragraphs("data/sample.txt")       # 📄 Paragraphs
  pages = processor.split_pages("data/sample.txt", delimit="\f")   # 📃 Pages (form-feed)
```

## 📊 Frequency & Vocabulary

```
  freq_dist = processor.compute_frequency_distribution(word_tokens)  # 📈 Frequency dist
  cond_freq = processor.compute_conditional_distribution(word_tokens, condition="POS")  # 🧮
  Conditional
  vocabulary = processor.create_vocabulary(freq_dist, min=2)         # 📖 Vocabulary
```

## 🧠 Vector Representations

```
  bow_vector = processor.create_wordbag(word_tokens)                 # 🧰 Bag-of-Words
  word2vec_model = processor.create_word2vec([word_tokens], vector_size=100, window=5)  # 🧬 Word2Vec
  tfidf_matrix, feature_names = processor.create_tfidf(word_tokens, max_features=500)   # 📐 TF-IDF
```

## 🗃️ Batch Utilities

```
  processor.clean_files("data/input_dir", "data/cleaned_output_dir")     # 🧼 Clean .txt files in bulk
  processor.convert_jsonl("data/cleaned_output_dir", "data/jsonl_output_dir")  # 🔄 .txt ➡️ .jsonl
```
### Application Tabs (High-Level Flow)

1. **Loading**
2. **Processing**
3. **Data Tokenization**
4. **Tensor Embedding**
5. **Vector Database**

Each tab is responsible for a **single stage** in the pipeline and only consumes outputs from upstream stages, ensuring deterministic execution and reproducibility.

### End-to-End Processing Flow

```
    ┌────────────┐
│  Loading   │
│  (Sources) │
└─────┬──────┘
      │ raw_text
      ▼
┌────────────┐
│ Processing │
│ (Cleaning) │
└─────┬──────┘
      │ processed_text
      ▼
┌────────────────┐
│ Data Tokenization │
│ (Sentences / Tokens)
└─────┬──────────┘
      │ token diagnostics
      ▼
┌────────────────┐
│ Tensor Embedding │
│ (Vectorization)  │
└─────┬──────────┘
      │ embeddings
      ▼
┌────────────────┐
│ Vector Database │
│ (sqlite-vec)    │
└─────┬──────────┘
      │ similarity search
      ▼
┌────────────────┐
│ RAG / Retrieval │
│ (Downstream)    │
└────────────────┘

```

## 📥 Loaders

The Loading tab supports **single-source document ingestion** from multiple loaders, including:

* Plain text
* PDF documents
* Word documents
* Web pages
* Wikipedia articles
* NLTK corpora
* Web crawling sources

All loaders normalize extracted content into a single **raw text buffer**, which is then propagated downstream for processing. Only one document or request is active at a time to preserve provenance and reproducibility.


## 🧼 Processing

The Processing tab provides a **repeatable, multi-pass text cleaning pipeline** with fine-grained controls, including:

* HTML, Markdown, XML, and encoding removal
* Symbol, numeral, and punctuation stripping
* Stopword removal
* Lemmatization
* Error and fragment cleanup
* Whitespace normalization

### Key Characteristics

* Processing can be applied **multiple times** to the same text.
* Each run deterministically replaces the previous processed output.
* Timing metrics and before/after statistics are computed automatically.

Outputs from this tab form the **authoritative processed text** for all downstream analysis.



## 🔤 Tokenization

The Data Tokenization tab transforms processed text into **sentence- and token-level representations** suitable for chunking and embedding.

### Core Outputs

* Sentence segmentation
* Token lists
* Token frequency distributions
* Fixed-width token grids (D0–D14)

### Visualizations

This tab includes analytical panels designed to validate embedding readiness:

* **Top-N token frequency histograms**
* **Sentence length distributions**
* **Token grid sparsity / padding analysis**
* **Embedding readiness scorecard** (tokens, vocabulary, hapax ratio)

These diagnostics help identify:

* Boilerplate dominance
* Over-truncation or padding
* Low-information text regions



## 🧠 Embedding Generation

The Tensor Embedding tab generates vector embeddings for tokenized chunks using supported embedding providers and models.

### Supported Capabilities

* SentenceTransformer-based embeddings
* Model-aware dimensional validation
* Batch embedding generation

### Semantic Diagnostics (t-SNE / UMAP)

Chonky includes **interactive dimensionality-reduction diagnostics** to visually inspect embedding quality:

* **t-SNE** for local neighborhood analysis
* **UMAP** for global structure preservation
* Adjustable hyperparameters (perplexity, neighbors, seed)
* Interactive scatter plots with chunk previews
* Optional tabular inspection of reduced coordinates

These diagnostics are **read-only** and are intended solely for validating embedding quality prior to persistence.



## 🗄️  Persistence

The Vector Database SQL CRUD with **vector-aware persistence**, tailored specifically for embedding workflows.

### Storage Stack

* **sqlite-vec** for vector storage and ANN search
* **LangChain Community VectorStores**
* **SentenceTransformer embeddings**

### Vector-Aware CRUD Semantics

| Operation | Vector Meaning                                  |
| --------- | ----------------------------------------------- |
| Create    | Create a vector table with fixed dimensionality |
| Insert    | Persist chunk embeddings with provenance        |
| Read      | Inspect metadata and sample rows                |
| Delete    | Drop entire vector tables                       |

Tables are deterministically named using:

```
<document>__<provider>__<model>__<dimension>
```

This prevents accidental mixing of embeddings from different models or dimensions.



## 🔍 Similarity Search 

Chonky includes an **interactive semantic similarity search interface** built directly on sqlite-vec.

### Features

* Free-text query input
* Top-K result selection
* **Similarity threshold slider** for precision control
* Ranked semantic matches with cosine similarity scores
* Expandable result previews

Similarity search is **read-only** and does not mutate stored vectors, making it safe for exploratory analysis and RAG workflows.

### Similarity Retrieval (sqlite-vec)

```python

import sqlite3
import sqlite_vec
from langchain_community.vectorstores import SQLiteVec
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings
)

# Connect to the same sqlite database used by Chonky
conn = sqlite3.connect("vectors.db")
conn.enable_load_extension(True)
sqlite_vec.load(conn)

# Initialize the embedding function (must match persisted model)
embedding_fn = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load the vector store
vector_store = SQLiteVec(
    connection=conn,
    table_name="epa_budget_2024__sentence_transformer__all-MiniLM-L6-v2__384",
    embedding=embedding_fn
)

# Perform semantic retrieval
query = "How are EPA discretionary funds allocated?"
results = vector_store.similarity_search(
    query=query,
    k=5
)

for doc in results:
    print(doc.page_content)

```

## 🧠 Retreival Augmentation 

With persistent vector storage and similarity search, Chonky is ready for:

* Retrieval-Augmented Generation (RAG)
* Semantic document exploration
* Chunk-level knowledge retrieval
* Integration with LangChain retrievers and chains



Below is an **updated, drop-in replacement** for your **📦 Dependencies** table that reflects **all functionality present in the current `app.py`**, including:

* Streamlit UI
* Embedding generation
* t-SNE / UMAP diagnostics
* sqlite-vec persistence
* LangChain vector stores and RAG usage


### LLM Retreival Augmentation

```python

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

response = qa_chain(
    {"query": "Summarize EPA funding priorities for FY2024."}
)

print(response["result"])

```

## 📦 Dependencies

| Package                 | Description                                                                                     |
| ----------------------- | ----------------------------------------------------------------------------------------------- |
| `streamlit`             | Interactive web application framework used to build the Chonky UI.                              |
| `nltk`                  | Natural Language Toolkit for tokenization, stopwords, lemmatization, and sentence segmentation. |
| `gensim`                | Library for Word2Vec and traditional topic modeling utilities.                                  |
| `spacy`                 | Industrial-strength NLP for linguistic parsing and tagging (optional advanced processing).      |
| `scikit-learn`          | Machine learning utilities, including TF-IDF, t-SNE, clustering, and supporting metrics.        |
| `umap-learn`            | UMAP dimensionality-reduction algorithm used for embedding diagnostics and visualization.       |
| `pandas`                | Data analysis and tabular manipulation for metrics, diagnostics, and previews.                  |
| `numpy`                 | Core numerical computing library for vector and tensor operations.                              |
| `tiktoken`              | OpenAI tokenizer for GPT-family models (token counting and diagnostics).                        |
| `transformers`          | Hugging Face model and tokenizer interfaces used by some embedding pipelines.                   |
| `sentence-transformers` | Sentence-level embedding models used for vector generation and semantic search.                 |
| `langchain-community`   | LangChain vector stores, embedding wrappers, and RAG utilities (including `SQLiteVec`).         |
| `sqlite-vec`            | SQLite extension providing vector storage and approximate nearest-neighbor (ANN) search.        |
| `pymupdf`               | PDF text extraction via PyMuPDF (a.k.a. `fitz`).                                                |
| `python-docx`           | Extracts text from Microsoft Word `.docx` documents.                                            |
| `beautifulsoup4`        | Parses and cleans HTML/XML content from web sources.                                            |
| `pydantic`              | Data validation and parsing using Python type hints.                                            |




## 📝 License

Chonky is published under the [MIT General Public License v3](https://github.com/is-leeroy-jenkins/Chonky/blob/main/LICENSE.txt)


