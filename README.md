###### Chonky-Py
![](https://github.com/is-leeroy-jenkins/Chonky/blob/main/resources/images/github/chonky_project.png)

A powerful, modular text-processing framework baseed in python tailored for analysts, data scientists,
and machine learning practitioners working with unstructured text. It unifies high-performance NLP
utilities and machine learning-ready pipelines to support text ingestion, cleaning, tokenization,
feature extraction, and document analysis.

---

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

---


## 🧰 Setup Instructions

To ensure a clean and isolated environment for running **Chonky**, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Chonky.git
cd Chonky
cd venv/Scripts
./activate.bat
cd ../../
pip install -r requirements.txt
```

##  Module Overview

### 🧠 `Text` Class

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
  | `compute_conditional_distribution(lines)` | Computes conditional frequency grouped by
  condition. |
  | `create_vocabulary(freq_dist)` | Creates vocabulary list from token frequency. |
  | `create_wordbag(words)` | Constructs Bag-of-Words from token list. |
  | `create_word2vec(words)` | Trains a Word2Vec model from tokenized sentences. |
  | `create_tfidf(words)` | Generates TF-IDF matrix. |
  | `clean_files(src, dest)` | Batch cleans `.txt` files from source to destination. |
  | `convert_jsonl(src, dest)` | Converts text files into JSONL format. |


### 📄 `Word` Class

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


### 📑 `PDF` Class

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


### 🧬 `Token` Class

- Interface to HuggingFace tokenizers and models.
- Supports token conversion, encoding/decoding, and vocabulary extraction.
  | Method | Description |
  |--------|-------------|
  | `encode(text)` | Encodes text using HuggingFace tokenizer. |
  | `batch_encode(texts)` | Batch encodes multiple texts. |
  | `decode(ids)` | Converts token IDs back to readable text. |
  | `convert_tokens(tokens)` | Converts tokens to string representations. |
  | `convert_ids(ids)` | Converts token IDs to strings. |
  | `create_vocabulary()` | Builds vocabulary from tokenizer model. |
  | `save_tokenizer(path)` | Saves tokenizer to disk. |
  | `load_tokenizer(path)` | Loads tokenizer from saved path. |

---

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


## 📦 Dependencies

| Package         | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `nltk`          | Natural language toolkit for tokenization, stopwords, tagging, etc.         |
| `gensim`        | Library for Word2Vec and topic modeling.                                    |
| `spacy`         | Industrial-strength NLP for tagging and parsing.                            |
| `scikit-learn`  | Machine learning library used for TF-IDF and dimensionality reduction.      |
| `pandas`        | Data analysis and manipulation tool.                                        |
| `numpy`         | Fundamental package for scientific computing.                               |
| `tiktoken`      | OpenAI’s tokenizer for GPT models.                                          |
| `transformers`  | HuggingFace’s model and tokenizer interface.                               |
| `pymupdf`       | PDF extraction with PyMuPDF (a.k.a. `fitz`).                               |
| `python-docx`   | Extracts text from Microsoft Word `.docx` documents.                        |
| `beautifulsoup4`| Parses and cleans HTML/XML content.                                         |
| `pydantic`      | Data validation and parsing with Python type hints.                         |


---

## 📝 License

Chonky is published under
the [MIT General Public License v3](https://github.com/is-leeroy-jenkins/Chonky/blob/main/LICENSE).

___