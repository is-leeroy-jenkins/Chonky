###### Chonky-Py
![](https://github.com/is-leeroy-jenkins/Chonky/blob/main/resources/images/github/chonky_project.png)

A modular text-processing framework baseed in python tailored for analysts, data scientists,
and machine learning practitioners working with unstructured text. It unifies high-performance NLP
utilities and machine learning-ready pipelines to support text ingestion, cleaning, tokenization,
feature extraction, and document analysis.



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

## 🧩 0. Initialize Processor

```
  processor = Text()

```

## 📂 1. Load Raw Text

```
  raw_text = processor.load_text("data/sample.txt")
```

## 🧼 2. Clean & Normalize

```
  text = processor.remove_html(raw_text)                     # 🧹 Strip HTML
  text = processor.normalize_text(text)                      # 🔡 Lowercase + ASCII
  text = processor.remove_markdown(text)                     # ✨ Remove markdown (#, *, etc.)
  text = processor.remove_special(text)                      # ❌ Remove special chars
  text = processor.remove_punctuation(text)                  # 🪛 Remove punctuation
  text = processor.collapse_whitespace(text)                 # 📏 Collapse whitespace
```

## 🧠 3. Spelling & Stopwords

```
  cleaned_text = processor.remove_errors(text)               # 🧬 Remove misspellings
  corrected_text = processor.correct_errors(cleaned_text)    # 🔁 Auto-correct spelling
  no_stopwords_text = processor.remove_stopwords(corrected_text)  # 🚫 Remove stopwords
```

## ✂️ 4. Tokenization

```
  word_tokens = processor.tokenize_words(no_stopwords_text)       # 🧩 Word tokens
  sentence_tokens = processor.tokenize_sentences(no_stopwords_text)  # 🧾 Sentence tokens
```

## 🌱 5. Lemmatization

```
  lemmatized_tokens = processor.lemmatize_tokens(word_tokens)
```


## 📦 6. Chunking

```  
  text_chunks = processor.chunk_text(no_stopwords_text, max=800)   # 🧳 Word chunked text
  word_chunks = processor.chunk_words(word_tokens, max=100, over=50)  # 🎒 Token chunks
```

## 📚 7. Structural Splitting

```
  line_groups = processor.split_lines("data/sample.txt")           # 📏 Lines
  paragraphs = processor.split_paragraphs("data/sample.txt")       # 📄 Paragraphs
  pages = processor.split_pages("data/sample.txt", delimit="\f")   # 📃 Pages (form-feed)
```

## 📊 8. Frequency & Vocabulary

```
  freq_dist = processor.compute_frequency_distribution(word_tokens)  # 📈 Frequency dist
  cond_freq = processor.compute_conditional_distribution(word_tokens, condition="POS")  # 🧮
  Conditional
  vocabulary = processor.create_vocabulary(freq_dist, min=2)         # 📖 Vocabulary
```

## 🧠 9. Vector Representations

```
  bow_vector = processor.create_wordbag(word_tokens)                 # 🧰 Bag-of-Words
  word2vec_model = processor.create_word2vec([word_tokens], vector_size=100, window=5)  # 🧬 Word2Vec
  tfidf_matrix, feature_names = processor.create_tfidf(word_tokens, max_features=500)   # 📐 TF-IDF
```

## 🗃️ 10. Batch Utilities

```
  processor.clean_files("data/input_dir", "data/cleaned_output_dir")     # 🧼 Clean .txt files in bulk
  processor.convert_jsonl("data/cleaned_output_dir", "data/jsonl_output_dir")  # 🔄 .txt ➡️ .jsonl
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




## 📝 License

Chonky is published under
the [MIT General Public License v3](https://github.com/is-leeroy-jenkins/Chonky/blob/main/LICENSE.txt)


