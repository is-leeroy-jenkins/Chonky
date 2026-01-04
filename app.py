'''
  ******************************************************************************************
      Assembly:                Name
      Filename:                name.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="guro.py" company="Terry D. Eppler">

	     name.py
	     Copyright ©  2022  Terry Eppler

     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the “Software”),
     to deal in the Software without restriction,
     including without limitation the rights to use,
     copy, modify, merge, publish, distribute, sublicense,
     and/or sell copies of the Software,
     and to permit persons to whom the Software is furnished to do so,
     subject to the following conditions:

     The above copyright notice and this permission notice shall be included in all
     copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
     ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
     DEALINGS IN THE SOFTWARE.

     You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov

  </copyright>
  <summary>
    name.py
  </summary>
  ******************************************************************************************
'''
# ******************************************************************************************
# Assembly:                Chonky
# Filename:                app.py
# Author:                  Terry D. Eppler (integration)
# Created:                 01-04-2026
# ******************************************************************************************

from __future__ import annotations

import streamlit as st
import tempfile
import os
from typing import List

from processing import (
    TXT, DOCX, PDF, Markdown, HTML, CSV, Web,
    Processor
)

from langchain_core.documents import Document
import pandas as pd

# ==========================================================================================
# Session State Initialization
# ==========================================================================================

STATE_KEYS = [
    "uploaded_files",
    "documents",
    "raw_text",
    "processed_text",
    "lines",
    "paragraphs",
    "pages",
    "sentences",
    "tokens",
    "vocabulary",
    "frequency_df",
    "chunks"
]

for key in STATE_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None

# ==========================================================================================
# Page Config
# ==========================================================================================

st.set_page_config(
    page_title="Chonky – Document Processing Workbench",
    layout="wide"
)

st.title("📦 Chonky – Document Processing Workbench")

# ==========================================================================================
# Sidebar — Ingestion & Configuration
# ==========================================================================================

st.sidebar.header("📥 Ingestion")

source_type = st.sidebar.radio(
    "Data Source",
    ["Local Files", "Web URLs"]
)

loader_type = st.sidebar.selectbox(
    "Loader Type",
    ["TXT", "PDF", "DOCX", "Markdown", "HTML", "CSV", "Web"]
)

chunk_size = st.sidebar.number_input(
    "Chunk Size",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100
)

overlap = st.sidebar.number_input(
    "Chunk Overlap",
    min_value=0,
    max_value=1000,
    value=200,
    step=50
)

# ==========================================================================================
# Loader Resolution
# ==========================================================================================

LOADER_MAP = {
    "TXT": TXT,
    "PDF": PDF,
    "DOCX": DOCX,
    "Markdown": Markdown,
    "HTML": HTML,
    "CSV": CSV,
    "Web": Web
}

# ==========================================================================================
# File Upload / URL Input
# ==========================================================================================

if source_type == "Local Files":
    uploaded = st.sidebar.file_uploader(
        "Upload Documents",
        type=["txt", "pdf", "docx", "md", "html", "csv"],
        accept_multiple_files=True
    )
else:
    urls = st.sidebar.text_area(
        "Enter URLs (one per line)"
    )

load_button = st.sidebar.button("Load Documents")

# ==========================================================================================
# Load Documents
# ==========================================================================================

if load_button:
    documents: List[Document] = []

    if source_type == "Local Files" and uploaded:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []

            for f in uploaded:
                path = os.path.join(tmpdir, f.name)
                with open(path, "wb") as out:
                    out.write(f.read())
                paths.append(path)

            loader = LOADER_MAP[loader_type]()

            for path in paths:
                docs = loader.load(path)
                documents.extend(docs)

    elif source_type == "Web URLs" and urls.strip():
        url_list = [u.strip() for u in urls.splitlines() if u.strip()]
        loader = Web()
        documents = loader.load(url_list)

    st.session_state.documents = documents
    st.session_state.raw_text = "\n\n".join(
        d.page_content for d in documents
    )

    st.success(f"Loaded {len(documents)} document segments")

# ==========================================================================================
# Main Tabs
# ==========================================================================================

tabs = st.tabs([
    "📄 Documents",
    "🧹 Preprocessing Pipeline",
    "📐 Structural Views",
    "🔤 Tokens & Vocabulary",
    "📊 Analysis & Statistics",
    "🧩 Vectorization & Chunking",
    "📤 Export"
])

# ==========================================================================================
# Tab 1 — Documents
# ==========================================================================================

with tabs[0]:
    st.header("Documents")

    if st.session_state.documents:
        st.write(f"Total Documents: {len(st.session_state.documents)}")
        st.text_area(
            "Raw Text Preview",
            st.session_state.raw_text,
            height=400
        )
    else:
        st.info("No documents loaded")

# ==========================================================================================
# Tab 2 — Preprocessing Pipeline
# ==========================================================================================

with tabs[1]:
    st.header("Preprocessing Pipeline")

    if st.session_state.raw_text:
        processor = Processor()

        normalize = st.checkbox("Normalize Text")
        remove_punct = st.checkbox("Remove Punctuation")
        remove_stop = st.checkbox("Remove Stopwords")
        lemmatize = st.checkbox("Lemmatize Tokens")

        if st.button("Apply Pipeline"):
            text = st.session_state.raw_text

            if normalize:
                text = processor.normalize_text(text)

            if remove_punct:
                text = processor.remove_punctuation(text)

            if remove_stop:
                text = processor.remove_stopwords(text)

            if lemmatize:
                tokens = processor.tokenize_text(text)
                tokens = processor.lemmatize_tokens(tokens)
                text = " ".join(tokens)

            st.session_state.processed_text = text

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Before")
            st.text_area("", st.session_state.raw_text, height=300)

        with col2:
            st.subheader("After")
            st.text_area("", st.session_state.processed_text or "", height=300)

    else:
        st.info("Load documents first")

# ==========================================================================================
# Tab 3 — Structural Views
# ==========================================================================================

with tabs[2]:
    st.header("Structural Views")

    if st.session_state.processed_text:
        processor = Processor()

        view = st.selectbox(
            "View Type",
            ["Lines", "Paragraphs", "Sentences"]
        )

        if view == "Lines":
            st.session_state.lines = st.session_state.processed_text.splitlines()
            st.write(len(st.session_state.lines))
            st.dataframe(pd.DataFrame(st.session_state.lines, columns=["Line"]))

        elif view == "Paragraphs":
            st.session_state.paragraphs = st.session_state.processed_text.split("\n\n")
            st.write(len(st.session_state.paragraphs))
            st.dataframe(pd.DataFrame(st.session_state.paragraphs, columns=["Paragraph"]))

        elif view == "Sentences":
            st.session_state.sentences = processor.tokenize_sentences(
                st.session_state.processed_text
            )
            st.write(len(st.session_state.sentences))
            st.dataframe(pd.DataFrame(st.session_state.sentences, columns=["Sentence"]))

    else:
        st.info("Run preprocessing first")

# ==========================================================================================
# Tab 4 — Tokens & Vocabulary
# ==========================================================================================

with tabs[3]:
    st.header("Tokens & Vocabulary")

    if st.session_state.processed_text:
        processor = Processor()
        tokens = processor.tokenize_text(st.session_state.processed_text)
        st.session_state.tokens = tokens

        st.write(f"Token Count: {len(tokens)}")
        st.dataframe(pd.DataFrame(tokens, columns=["Token"]))

        vocab = processor.create_vocabulary(tokens)
        st.session_state.vocabulary = vocab

        st.write(f"Vocabulary Size: {len(vocab)}")
        st.dataframe(pd.DataFrame(vocab, columns=["Word"]))

    else:
        st.info("Run preprocessing first")

# ==========================================================================================
# Tab 5 — Analysis & Statistics
# ==========================================================================================

with tabs[4]:
    st.header("Analysis & Statistics")

    if st.session_state.tokens:
        processor = Processor()
        freq_df = processor.create_frequency_distribution(
            st.session_state.tokens
        )
        st.session_state.frequency_df = freq_df
        st.dataframe(freq_df)

    else:
        st.info("Generate tokens first")

# ==========================================================================================
# Tab 6 — Vectorization & Chunking
# ==========================================================================================

with tabs[5]:
    st.header("Vectorization & Chunking")

    if st.session_state.documents:
        loader = LOADER_MAP[loader_type]()
        loader.documents = st.session_state.documents

        if st.button("Chunk Documents"):
            chunks = loader.split(chunk_size, overlap)
            st.session_state.chunks = chunks
            st.success(f"Created {len(chunks)} chunks")

        if st.session_state.chunks:
            st.write("Chunk Preview")
            for i, c in enumerate(st.session_state.chunks[:5]):
                st.text_area(f"Chunk {i}", c.page_content, height=150)

    else:
        st.info("Load documents first")

# ==========================================================================================
# Tab 7 — Export
# ==========================================================================================

with tabs[6]:
    st.header("Export")

    if st.session_state.processed_text:
        st.download_button(
            "Download Cleaned Text",
            st.session_state.processed_text,
            file_name="cleaned_text.txt"
        )

    if st.session_state.frequency_df is not None:
        csv = st.session_state.frequency_df.to_csv(index=False)
        st.download_button(
            "Download Frequency Table",
            csv,
            file_name="frequency.csv"
        )

    if st.session_state.chunks:
        chunk_text = "\n\n".join(c.page_content for c in st.session_state.chunks)
        st.download_button(
            "Download Chunks",
            chunk_text,
            file_name="chunks.txt" )
