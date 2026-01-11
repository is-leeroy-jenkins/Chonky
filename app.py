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

	     app.py
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
    app.py
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations

import os
import tempfile
from typing import List

import pandas as pd
import streamlit as st
from PIL import Image
from langchain_core.documents import Document

import config as cfg
from processing import Processor
from loaders import (
    TextLoader,
    CsvLoader,
    PdfLoader,
    ExcelLoader,
    WordLoader,
    MarkdownLoader,
    HtmlLoader,
    JsonLoader,
    PowerPointLoader,
    WikiLoader,
    GithubLoader,
    WebLoader,
    ArXivLoader
)

CHUNKABLE_LOADERS = {
    "TextLoader": ["chars", "tokens"],
    "CsvLoader": ["chars"],
    "PdfLoader": ["chars"],
    "ExcelLoader": ["chars"],
    "WordLoader": ["chars"],
    "MarkdownLoader": ["chars"],
    "HtmlLoader": ["chars"],
    "JsonLoader": ["chars"],
    "PowerPointLoader": ["chars"],
}

# ======================================================================================
# Page Configuration
# ======================================================================================

st.set_page_config(
    page_title="Chonky",
    layout="wide",
    page_icon=cfg.ICON,
)
SESSION_STATE_DEFAULTS = {
    # Ingestion
    "documents": None,
    "raw_documents": None,
    "raw_text": None,
    "active_loader": None,

    # Processing
    "processed_text": None,

    # Tokenization
    "tokens": None,
    "vocabulary": None,
    "token_counts": None,

    # Chunking
    "chunks": None,

    # Vectorization
    "embeddings": None,
    "embedding_model": None,

    # Retrieval / Search
    "search_results": None,
}

for key, default in SESSION_STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown("#### NLPlumbing")

# ======================================================================================
# Session State Initialization
# ======================================================================================

if "documents" not in st.session_state:
    st.session_state.documents = None

if "active_loader" not in st.session_state:
    st.session_state.active_loader = None

# Baseline snapshot of the initially ingested corpus (pre-processing, pre-chunking).
if "raw_text" not in st.session_state:
    st.session_state.raw_text = None

# Optional: keep a copy of the original Documents as loaded (pre-chunking).
if "raw_documents" not in st.session_state:
    st.session_state.raw_documents = None
    
if "vocabulary" not in st.session_state:
    st.session_state.vocabulary = None

if "token_counts" not in st.session_state:
    st.session_state.token_counts = None

# ======================================================================================
# Sidebar (Branding / Global Only)
# ======================================================================================

with st.sidebar:
	st.header( "Chonky" )
	st.caption( "Document ingestion & NLP plumbing" )
	st.divider( )
	st.subheader( "" )

# ======================================================================================
# Tabs
# ======================================================================================

tabs = st.tabs(
    [
        "📄 Loading",
        "🧹 Processing",
        "📐 Scaffolding",
        "🔤 Tokens & Vocabulary",
        "📊 Analysis & Statistics",
        "🧩 Vectorization & Chunking",
        "📤 Export",
    ]
)

# ======================================================================================
# Tab 1 — Loading (PER-EXPANDER CLEAR / RESET, NO SIDEBAR)
# ======================================================================================

with tabs[0]:

    # ------------------------------------------------------------------
    # Defensive session_state initialization
    # ------------------------------------------------------------------
    for key, default in {
        "documents": None,
        "raw_documents": None,
        "raw_text": None,
        "processed_text": None,
        "active_loader": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    left, right = st.columns([1, 1.5])

    # ------------------------------------------------------------------
    # Helper: clear only if this loader is active
    # ------------------------------------------------------------------
    def clear_if_active(loader_name: str) -> None:
        if st.session_state.active_loader == loader_name:
            st.session_state.documents = None
            st.session_state.raw_documents = None
            st.session_state.raw_text = None
            st.session_state.processed_text = None
            st.session_state.active_loader = None

    # ------------------------------------------------------------------
    # LEFT COLUMN — Loader Expanders
    # ------------------------------------------------------------------
    with left:
        st.subheader("Load Documents")

        # --------------------------- Text Loader
        with st.expander("📄 Text", expanded=False):
            files = st.file_uploader(
                "Upload TXT files",
                type=["txt"],
                accept_multiple_files=True,
                key="txt_upload",
            )

            col_load, col_clear = st.columns(2)
            load_txt = col_load.button("Load", key="txt_load")
            clear_txt = col_clear.button("Clear", key="txt_clear")

            if clear_txt:
                clear_if_active("TextLoader")
                st.info("Text Loader state cleared.")

            if load_txt and files:
                docs = []
                for f in files:
                    text = f.read().decode("utf-8", errors="ignore")
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={"source": f.name, "loader": "TextLoader"},
                        )
                    )

                st.session_state.documents = docs
                st.session_state.raw_documents = list(docs)
                st.session_state.raw_text = "\n\n".join(d.page_content for d in docs)
                st.session_state.processed_text = None
                st.session_state.active_loader = "TextLoader"

                st.success(f"Loaded {len(docs)} text document(s).")

        # --------------------------- CSV Loader
        with st.expander("📑 CSV", expanded=False):
            csv_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload")
            delimiter = st.text_input("Delimiter", value="\n\n", key="csv_delim")
            quotechar = st.text_input("Quote Character", value='"', key="csv_quote")

            col_load, col_clear = st.columns(2)
            load_csv = col_load.button("Load", key="csv_load")
            clear_csv = col_clear.button("Clear", key="csv_clear")

            if clear_csv:
                clear_if_active("CsvLoader")
                st.info("CSV Loader state cleared.")

            if load_csv and csv_file:
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, csv_file.name)
                    with open(path, "wb") as f:
                        f.write(csv_file.read())

                    loader = CsvLoader()
                    docs = loader.load(
                        path,
                        columns=None,
                        delimiter=delimiter,
                        quotechar=quotechar,
                    )

                st.session_state.documents = docs
                st.session_state.raw_documents = list(docs)
                st.session_state.raw_text = "\n\n".join(d.page_content for d in docs)
                st.session_state.processed_text = None
                st.session_state.active_loader = "CsvLoader"

                st.success(f"Loaded {len(docs)} CSV document(s).")

        # --------------------------- PDF Loader
        with st.expander("📕 PDF", expanded=False):
            pdf = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_upload")
            mode = st.selectbox("Mode", ["single", "elements"], key="pdf_mode")
            extract = st.selectbox("Extract", ["plain", "ocr"], key="pdf_extract")
            include = st.checkbox("Include Images", value=True, key="pdf_include")
            fmt = st.selectbox("Format", ["markdown-img", "text"], key="pdf_fmt")

            col_load, col_clear = st.columns(2)
            load_pdf = col_load.button("Load", key="pdf_load")
            clear_pdf = col_clear.button("Clear", key="pdf_clear")

            if clear_pdf:
                clear_if_active("PdfLoader")
                st.info("PDF Loader state cleared.")

            if load_pdf and pdf:
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, pdf.name)
                    with open(path, "wb") as f:
                        f.write(pdf.read())

                    loader = PdfLoader()
                    docs = loader.load(
                        path,
                        mode=mode,
                        extract=extract,
                        include=include,
                        format=fmt,
                    )

                st.session_state.documents = docs
                st.session_state.raw_documents = list(docs)
                st.session_state.raw_text = "\n\n".join(d.page_content for d in docs)
                st.session_state.processed_text = None
                st.session_state.active_loader = "PdfLoader"

                st.success(f"Loaded {len(docs)} PDF document(s).")

        # --------------------------- Markdown Loader
        with st.expander("🧾 Markdown", expanded=False):
            md = st.file_uploader("Upload Markdown", type=["md", "markdown"], key="md_upload")

            col_load, col_clear = st.columns(2)
            load_md = col_load.button("Load", key="md_load")
            clear_md = col_clear.button("Clear", key="md_clear")

            if clear_md:
                clear_if_active("MarkdownLoader")
                st.info("Markdown Loader state cleared.")

            if load_md and md:
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, md.name)
                    with open(path, "wb") as f:
                        f.write(md.read())

                    loader = MarkdownLoader()
                    docs = loader.load(path)

                st.session_state.documents = docs
                st.session_state.raw_documents = list(docs)
                st.session_state.raw_text = "\n\n".join(d.page_content for d in docs)
                st.session_state.processed_text = None
                st.session_state.active_loader = "MarkdownLoader"

                st.success(f"Loaded {len(docs)} Markdown document(s).")

        # --------------------------- HTML Loader
        with st.expander("🌐 HTML", expanded=False):
            html = st.file_uploader("Upload HTML", type=["html", "htm"], key="html_upload")

            col_load, col_clear = st.columns(2)
            load_html = col_load.button("Load", key="html_load")
            clear_html = col_clear.button("Clear", key="html_clear")

            if clear_html:
                clear_if_active("HtmlLoader")
                st.info("HTML Loader state cleared.")

            if load_html and html:
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, html.name)
                    with open(path, "wb") as f:
                        f.write(html.read())

                    loader = HtmlLoader()
                    docs = loader.load(path)

                st.session_state.documents = docs
                st.session_state.raw_documents = list(docs)
                st.session_state.raw_text = "\n\n".join(d.page_content for d in docs)
                st.session_state.processed_text = None
                st.session_state.active_loader = "HtmlLoader"

                st.success(f"Loaded {len(docs)} HTML document(s).")

        # --------------------------- JSON Loader
        with st.expander("🧩 JSON", expanded=False):
            js = st.file_uploader("Upload JSON", type=["json"], key="json_upload")
            is_lines = st.checkbox("JSON Lines", value=False, key="json_lines")

            col_load, col_clear = st.columns(2)
            load_json = col_load.button("Load", key="json_load")
            clear_json = col_clear.button("Clear", key="json_clear")

            if clear_json:
                clear_if_active("JsonLoader")
                st.info("JSON Loader state cleared.")

            if load_json and js:
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, js.name)
                    with open(path, "wb") as f:
                        f.write(js.read())

                    loader = JsonLoader()
                    docs = loader.load(
                        path,
                        is_text=True,
                        is_lines=is_lines,
                    )

                st.session_state.documents = docs
                st.session_state.raw_documents = list(docs)
                st.session_state.raw_text = "\n\n".join(d.page_content for d in docs)
                st.session_state.processed_text = None
                st.session_state.active_loader = "JsonLoader"

                st.success(f"Loaded {len(docs)} JSON document(s).")

        # --------------------------- PowerPoint Loader
        with st.expander("📽 Power Point", expanded=False):
            pptx = st.file_uploader("Upload PPTX", type=["pptx"], key="pptx_upload")
            mode = st.selectbox("Mode", ["single", "multiple"], key="pptx_mode")

            col_load, col_clear = st.columns(2)
            load_pptx = col_load.button("Load", key="pptx_load")
            clear_pptx = col_clear.button("Clear", key="pptx_clear")

            if clear_pptx:
                clear_if_active("PowerPointLoader")
                st.info("PowerPoint Loader state cleared.")

            if load_pptx and pptx:
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, pptx.name)
                    with open(path, "wb") as f:
                        f.write(pptx.read())

                    loader = PowerPointLoader()
                    docs = (
                        loader.load(path)
                        if mode == "single"
                        else loader.load_multiple(path)
                    )

                st.session_state.documents = docs
                st.session_state.raw_documents = list(docs)
                st.session_state.raw_text = "\n\n".join(d.page_content for d in docs)
                st.session_state.processed_text = None
                st.session_state.active_loader = "PowerPointLoader"

                st.success(f"Loaded {len(docs)} PowerPoint document(s).")
                
        # --------------------------- arXiv Loader (APPEND, PARAMETER-COMPLETE)
        with st.expander( "🧠 ArXiv", expanded=False ):
            arxiv_query = st.text_input(
                "Query",
                placeholder="e.g., transformer OR llm",
                key="arxiv_query",
            )
            
            arxiv_max_chars = st.number_input(
                "Max characters per document",
                min_value=250,
                max_value=100000,
                value=1000,
                step=250,
                key="arxiv_max_chars",
                help="Passed as max_chars to ArXivLoader.load(query, max_chars=...).",
            )
            
            col_load, col_clear = st.columns( 2 )
            arxiv_fetch = col_load.button( "Fetch", key="arxiv_fetch" )
            arxiv_clear = col_clear.button( "Clear", key="arxiv_clear" )
            
            if arxiv_clear and st.session_state.get( "documents" ):
                st.session_state.documents = [
		                d for d in st.session_state.documents
		                if d.metadata.get( "loader" ) != "ArXivLoader"
                ]
                st.info( "ArXivLoader documents removed." )
            
            if arxiv_fetch and arxiv_query:
                loader = ArXivLoader( )
                docs = loader.load( arxiv_query, max_chars=int( arxiv_max_chars ) ) or [ ]
                
                for d in docs:
	                d.metadata[ "loader" ] = "ArXivLoader"
	                d.metadata[ "source" ] = arxiv_query
                
                if docs:
	                if st.session_state.documents:
		                st.session_state.documents.extend( docs )
	                else:
		                st.session_state.documents = docs
		                # Baseline snapshot only on first corpus load
		                st.session_state.raw_documents = list( docs )
		                st.session_state.raw_text = "\n\n".join(
			                x.page_content for x in docs )
	                
	                st.session_state.active_loader = "ArXivLoader"
	                st.success( f"Fetched {len( docs )} arXiv document(s)." )
        
        # --------------------------- Wikipedia Loader (APPEND, PARAMETER-COMPLETE)
        with st.expander( "📚 Wikipedia", expanded=False ):
            wiki_query = st.text_input(
                "Query",
                placeholder="e.g., Natural language processing",
                key="wiki_query",
            )
            
            wiki_max_docs = st.number_input(
                "Max documents",
                min_value=1,
                max_value=250,
                value=25,
                step=1,
                key="wiki_max_docs",
                help="Passed as max_docs to WikiLoader.load(query, max_docs=..., max_chars=...).",
            )
            
            wiki_max_chars = st.number_input(
                "Max characters per document",
                min_value=250,
                max_value=100000,
                value=4000,
                step=250,
                key="wiki_max_chars",
                help="Passed as max_chars to WikiLoader.load(query, max_docs=..., max_chars=...).",
            )
            
            col_load, col_clear = st.columns( 2 )
            wiki_fetch = col_load.button( "Fetch", key="wiki_fetch" )
            wiki_clear = col_clear.button( "Clear", key="wiki_clear" )
            
            if wiki_clear and st.session_state.get( "documents" ):
                st.session_state.documents = [
		                d for d in st.session_state.documents
		                if d.metadata.get( "loader" ) != "WikiLoader"
                ]
                st.info( "WikiLoader documents removed." )
            
            if wiki_fetch and wiki_query:
                loader = WikiLoader( )
                docs = loader.load(
	                wiki_query,
	                max_docs=int( wiki_max_docs ),
	                max_chars=int( wiki_max_chars ),
                ) or [ ]
                
                for d in docs:
	                d.metadata[ "loader" ] = "WikiLoader"
	                d.metadata[ "source" ] = wiki_query
                
                if docs:
	                if st.session_state.documents:
		                st.session_state.documents.extend( docs )
	                else:
		                st.session_state.documents = docs
		                # Baseline snapshot only on first corpus load
		                st.session_state.raw_documents = list( docs )
		                st.session_state.raw_text = "\n\n".join(
			                x.page_content for x in docs )
	                
	                st.session_state.active_loader = "WikiLoader"
	                st.success( f"Fetched {len( docs )} Wikipedia document(s)." )
        
        # --------------------------- GitHub Loader
        with st.expander( "🐙 GitHub", expanded=False ):
            gh_url = st.text_input(
                "GitHub API URL",
                placeholder="https://api.github.com",
                value="https://api.github.com",
                key="gh_url",
                help="Passed as the 'url' argument to GithubLoader.load(...).",
            )
            
            gh_repo = st.text_input(
                "Repo (owner/name)",
                placeholder="openai/openai-python",
                key="gh_repo",
                help="Passed as the 'repo' argument to GithubLoader.load(...).",
            )
            
            gh_branch = st.text_input(
                "Branch",
                placeholder="main",
                value="main",
                key="gh_branch",
                help="Passed as the 'branch' argument to GithubLoader.load(...).",
            )
            
            gh_filetype = st.text_input(
                "File type filter",
                value=".md",
                key="gh_filetype",
                help="Passed as the 'filetype' argument (default .md). Example: .py, .md, .txt",
            )
            
            col_load, col_clear = st.columns( 2 )
            gh_fetch = col_load.button( "Fetch", key="gh_fetch" )
            gh_clear = col_clear.button( "Clear", key="gh_clear" )
            
            if gh_clear and st.session_state.get( "documents" ):
                st.session_state.documents = [
		                d for d in st.session_state.documents
		                if d.metadata.get( "loader" ) != "GithubLoader"
                ]
                st.info( "GithubLoader documents removed." )
            
            if gh_fetch and gh_repo and gh_branch:
                loader = GithubLoader( )
                docs = loader.load(
	                gh_url,
	                gh_repo,
	                gh_branch,
	                gh_filetype,
                ) or [ ]
                
                for d in docs:
	                d.metadata[ "loader" ] = "GithubLoader"
	                d.metadata[ "source" ] = f"{gh_repo}@{gh_branch}"
                
                if docs:
	                if st.session_state.documents:
		                st.session_state.documents.extend( docs )
	                else:
		                st.session_state.documents = docs
		                # Baseline snapshot only on first corpus load
		                st.session_state.raw_documents = list( docs )
		                st.session_state.raw_text = "\n\n".join(
			                x.page_content for x in docs )
	                
	                st.session_state.active_loader = "GithubLoader"
	                st.success( f"Fetched {len( docs )} GitHub document(s)." )
                
        # --------------------------- Web Loader
        with st.expander( "🔗 Web Loader", expanded=False ):
            urls = st.text_area(
                "Enter one URL per line",
                placeholder="https://example.com\nhttps://another.com",
                key="web_urls",
            )
            
            col_load, col_clear = st.columns( 2 )
            load_web = col_load.button( "Fetch", key="web_fetch" )
            clear_web = col_clear.button( "Clear", key="web_clear" )
            
            if clear_web and st.session_state.get( "documents" ):
                st.session_state.documents = [
		                d for d in st.session_state.documents
		                if d.metadata.get( "loader" ) != "WebLoader"
                ]
                st.info( "WebLoader documents removed." )
            
            if load_web and urls.strip( ):
                loader = WebLoader( recursive=False )
                new_docs = [ ]
                
                for url in [ u.strip( ) for u in urls.splitlines( ) if u.strip( ) ]:
	                docs = loader.load( url )
	                for d in docs:
		                d.metadata[ "loader" ] = "WebLoader"
		                d.metadata[ "source" ] = url
	                new_docs.extend( docs )
                
                if new_docs:
	                if st.session_state.documents:
		                st.session_state.documents.extend( new_docs )
	                else:
		                st.session_state.documents = new_docs
		                st.session_state.raw_documents = list( new_docs )
		                st.session_state.raw_text = "\n\n".join(
			                d.page_content for d in new_docs
		                )
	                
	                st.session_state.active_loader = "WebLoader"
	                st.success( f"Fetched {len( new_docs )} web document(s)." )
	            
        # --------------------------- Web Crawler (RECURSIVE)
        with st.expander( "🕷️ Web Crawler", expanded=False ):
            start_url = st.text_input(
                "Start URL",
                placeholder="https://example.com",
                key="crawl_start_url",
            )
            
            max_depth = st.number_input(
                "Max crawl depth",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
                key="crawl_depth",
            )
            
            stay_on_domain = st.checkbox(
                "Stay on starting domain",
                value=True,
                key="crawl_domain_lock",
            )
            
            col_run, col_clear = st.columns( 2 )
            run_crawl = col_run.button( "Crawl", key="crawl_run" )
            clear_crawl = col_clear.button( "Clear", key="crawl_clear" )
            
            if clear_crawl and st.session_state.get( "documents" ):
                st.session_state.documents = [
		                d for d in st.session_state.documents
		                if d.metadata.get( "loader" ) != "WebCrawler"
                ]
                st.info( "Web crawler documents removed." )
            
            if run_crawl and start_url:
                loader = WebLoader(
	                recursive=True,
	                max_depth=max_depth,
	                prevent_outside=stay_on_domain,
                )
                
                docs = loader.load( start_url )
                
                for d in docs:
	                d.metadata[ "loader" ] = "WebCrawler"
	                d.metadata[ "source" ] = start_url
                
                if st.session_state.documents:
	                st.session_state.documents.extend( docs )
                else:
	                st.session_state.documents = docs
	                st.session_state.raw_documents = list( docs )
	                st.session_state.raw_text = "\n\n".join(
		                d.page_content for d in docs
	                )
                
                st.session_state.active_loader = "WebCrawler"
                st.success( f"Crawled {len( docs )} document(s)." )
    
    # ------------------------------------------------------------------
    # RIGHT COLUMN — Document Preview
    # ------------------------------------------------------------------
    with right:
        st.subheader("Document Preview")

        docs = st.session_state.documents
        if not docs:
            st.info("No documents loaded.")
        else:
            st.caption(f"Active Loader: {st.session_state.active_loader}")
            st.write(f"Documents: {len(docs)}")

            for i, d in enumerate(docs[:5]):
                with st.expander(f"Document {i + 1}", expanded=False):
                    st.json(d.metadata)
                    st.text_area(
                        "Content",
                        d.page_content[:5000],
                        height=200,
                        key=f"preview_doc_{i}",
                    )
       

# ==========================================================================================
# Tab 2 — Preprocessing Pipeline
# ==========================================================================================

with tabs[ 1 ]:
	st.header( "" )
	
	if st.session_state.raw_text:
		processor = Processor( )
		
		normalize = st.checkbox( "Normalize Text" )
		remove_punct = st.checkbox( "Remove Punctuation" )
		remove_stop = st.checkbox( "Remove Stopwords" )
		lemmatize = st.checkbox( "Lemmatize Tokens" )
		
		if st.button( "Apply Pipeline" ):
			text = st.session_state.raw_text
			
			if normalize:
				text = processor.normalize_text( text )
			
			if remove_punct:
				text = processor.remove_punctuation( text )
			
			if remove_stop:
				text = processor.remove_stopwords( text )
			
			if lemmatize:
				tokens = processor.tokenize_text( text )
				tokens = processor.lemmatize_tokens( tokens )
				text = " ".join( tokens )
			
			st.session_state.processed_text = text
		
		col1, col2 = st.columns( 2 )
		
		with col1:
			st.subheader( "Before" )
			st.text_area( "", st.session_state.raw_text, height=300 )
		
		with col2:
			st.subheader( "After" )
			st.text_area( "", st.session_state.processed_text or "", height=300 )
	
	else:
		st.info( "Load documents first" )

# ==========================================================================================
# Tab 3 — Structural Views
# ==========================================================================================

with tabs[ 2 ]:
	st.header( "" )
	
	if st.session_state.processed_text:
		processor = Processor( )
		
		view = st.selectbox(
			"View Type",
			[ "Lines",
			  "Paragraphs",
			  "Sentences" ]
		)
		
		if view == "Lines":
			lines = st.session_state.processed_text.splitlines( )
			st.dataframe( pd.DataFrame( lines, columns=[ "Line" ] ) )
		
		elif view == "Paragraphs":
			paragraphs = st.session_state.processed_text.split( "\n\n" )
			st.dataframe( pd.DataFrame( paragraphs, columns=[ "Paragraph" ] ) )
		
		elif view == "Sentences":
			sentences = processor.tokenize_sentences(
				st.session_state.processed_text
			)
			st.dataframe( pd.DataFrame( sentences, columns=[ "Sentence" ] ) )
	
	else:
		st.info( "Run preprocessing first" )

# ==========================================================================================
# Tab 4 — Tokens & Vocabulary
# ==========================================================================================

with tabs[ 3 ]:
	st.header( "" )
	
	if st.session_state.processed_text:
		processor = Processor( )
		tokens = processor.tokenize_text( st.session_state.processed_text )
		vocab = processor.create_vocabulary( tokens )
		
		st.write( f"Token Count: {len( tokens )}" )
		st.dataframe( pd.DataFrame( tokens, columns=[ "Token" ] ) )
		
		st.write( f"Vocabulary Size: {len( vocab )}" )
		st.dataframe( pd.DataFrame( vocab, columns=[ "Word" ] ) )
		
		st.session_state.tokens = tokens
		st.session_state.vocabulary = vocab
	
	else:
		st.info( "Run preprocessing first" )

# ==========================================================================================
# Tab 5 — Analysis & Statistics
# ==========================================================================================

with tabs[ 4 ]:
	st.header( "" )
	
	if st.session_state.tokens:
		processor = Processor( )
		freq_df = processor.create_frequency_distribution(
			st.session_state.tokens
		)
		st.session_state.frequency_df = freq_df
		st.dataframe( freq_df )
	
	else:
		st.info( "Generate tokens first" )

with tabs[5]:  # 🧩 Vectorization & Chunking (or Chunking tab index)

    st.subheader("Chunking")

    docs = st.session_state.get("documents")
    loader_name = st.session_state.get("active_loader")

    if not docs:
        st.warning("No documents loaded. Please load documents first.")
        st.stop()

    if not loader_name:
        st.warning("No active loader found.")
        st.stop()

    chunk_modes = CHUNKABLE_LOADERS.get(loader_name)

    if not chunk_modes:
        st.info(f"Chunking is not supported for loader: {loader_name}")
        st.stop()

    st.caption(f"Source Loader: {loader_name}")

    # ---------------------------
    # Chunking Controls
    # ---------------------------

    mode = st.selectbox(
        "Chunking Mode",
        options=chunk_modes,
        help="Select how documents should be chunked",
    )

    col_a, col_b = st.columns(2)

    with col_a:
        chunk_size = st.number_input(
            "Chunk Size",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
        )

    with col_b:
        overlap = st.number_input(
            "Overlap",
            min_value=0,
            max_value=2000,
            value=200,
            step=50,
        )

    col_run, col_reset = st.columns(2)

    run_chunking = col_run.button("Apply Chunking")
    reset_chunking = col_reset.button("Reset")

    # ---------------------------
    # Actions
    # ---------------------------

    if reset_chunking:
        # No mutation of source docs here unless you already snapshot originals
        st.info("Chunking controls reset.")

    if run_chunking:
        processor = Processor(docs)

        if mode == "chars":
            chunked_docs = processor.chunk_chars(
                size=chunk_size,
                overlap=overlap,
            )

        elif mode == "tokens":
            chunked_docs = processor.chunk_tokens(
                size=chunk_size,
                overlap=overlap,
            )

        else:
            st.error(f"Unsupported chunking mode: {mode}")
            st.stop()

        st.session_state.documents = chunked_docs

        st.success(
            f"Chunking complete: {len(chunked_docs)} chunks generated "
            f"(mode={mode}, size={chunk_size}, overlap={overlap})"
        )

# ==========================================================================================
# Tab 7 — Export
# ==========================================================================================

with tabs[ 6 ]:
	st.header( "" )
	
	if st.session_state.processed_text:
		st.download_button(
			"Download Cleaned Text",
			st.session_state.processed_text,
			file_name="cleaned_text.txt"
		)
	
	if st.session_state.frequency_df is not None:
		csv = st.session_state.frequency_df.to_csv( index=False )
		st.download_button(
			"Download Frequency Table",
			csv,
			file_name="frequency.csv"
		)
	
	if st.session_state.chunks:
		chunk_text = "\n\n".join( c.page_content for c in st.session_state.chunks )
		st.download_button(
			"Download Chunks",
			chunk_text,
			file_name="chunks.txt"
		)
