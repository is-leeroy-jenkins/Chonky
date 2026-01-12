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
import sqlite3
import tempfile
from typing import List

import pandas as pd
import streamlit as st
from PIL import Image
from langchain_core.documents import Document

import config as cfg
from processing import Processor, TextParser, WordParser, PdfParser
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

st.markdown("")

# ======================================================================================
# Session State Initialization
# ======================================================================================

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

# ======================================================================================
# Sidebar (Branding / Global Only)
# ======================================================================================

with st.sidebar:
	st.header( "Chonky" )
	st.caption( "Pipe" )
	st.divider( )
	st.subheader( "" )

# ======================================================================================
# Tabs
# ======================================================================================

tabs = st.tabs(
    [
        "Loading",
        "Processing",
        "Scaffolding",
        "Tokens & Vocabulary",
        "Analysis & Statistics",
        "Vectorization & Chunking",
        "Export",
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
        st.subheader("")

        # --------------------------- Text Loader
        with st.expander("📄 Text Loader", expanded=False):
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
        with st.expander("📑 CSV Loader", expanded=False):
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
        with st.expander("📕 PDF Loader", expanded=False):
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
        with st.expander("🧾 Markdown Loader", expanded=False):
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
        with st.expander("🌐 HTML Loader", expanded=False):
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
        with st.expander("🧩 JSON Loader", expanded=False):
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
        with st.expander("📽 Power Point Loader", expanded=False):
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
        # --------------------------- Excel Loader (FILE + SQLITE)
        with st.expander("📊 Excel Loader", expanded=False):

            excel_file = st.file_uploader(
                "Upload Excel file",
                type=["xlsx", "xls"],
                key="excel_upload",
            )

            sheet_name = st.text_input(
                "Sheet name (leave blank for all sheets)",
                key="excel_sheet",
            )

            table_prefix = st.text_input(
                "SQLite table prefix",
                value="excel",
                help="Each sheet will be written as <prefix>_<sheetname>",
                key="excel_table_prefix",
            )

            col_load, col_clear = st.columns(2)
            load_excel = col_load.button("Load", key="excel_load")
            clear_excel = col_clear.button("Clear", key="excel_clear")

            # ------------------------------------------------------
            # Clear logic (remove only ExcelLoader documents)
            # ------------------------------------------------------
            if clear_excel and st.session_state.get("documents"):
                st.session_state.documents = [
                    d for d in st.session_state.documents
                    if d.metadata.get("loader") != "ExcelLoader"
                ]
                st.info("ExcelLoader documents removed.")

            # ------------------------------------------------------
            # Load + SQLite ingestion
            # ------------------------------------------------------
            if load_excel and excel_file:

                # Ensure SQLite directory exists
                sqlite_path = os.path.join("stores", "sqlite", "data.db")
                os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)

                with tempfile.TemporaryDirectory() as tmp:
                    excel_path = os.path.join(tmp, excel_file.name)
                    with open(excel_path, "wb") as f:
                        f.write(excel_file.read())

                    # Read Excel into DataFrames
                    if sheet_name.strip():
                        dfs = {
                            sheet_name: pd.read_excel(excel_path, sheet_name=sheet_name)
                        }
                    else:
                        dfs = pd.read_excel(excel_path, sheet_name=None)

                # Open SQLite connection
                conn = sqlite3.connect(sqlite_path)

                docs = []

                for sheet, df in dfs.items():
                    if df.empty:
                        continue

                    # Normalize table name
                    table_name = f"{table_prefix}_{sheet}".replace(" ", "_").lower()

                    # Write DataFrame to SQLite
                    df.to_sql(
                        table_name,
                        conn,
                        if_exists="replace",
                        index=False,
                    )

                    # Convert DataFrame to text for NLP pipeline
                    text = df.to_csv(index=False)

                    docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "loader": "ExcelLoader",
                                "source": excel_file.name,
                                "sheet": sheet,
                                "table": table_name,
                                "sqlite_db": sqlite_path,
                            },
                        )
                    )

                conn.close()

                if docs:
                    if st.session_state.documents:
                        st.session_state.documents.extend(docs)
                    else:
                        st.session_state.documents = docs
                        st.session_state.raw_documents = list(docs)
                        st.session_state.raw_text = "\n\n".join(
                            d.page_content for d in docs
                        )

                    st.session_state.active_loader = "ExcelLoader"
                    st.success(
                        f"Loaded {len(docs)} sheet(s) and stored in SQLite."
                    )
                else:
                    st.warning("No data loaded (empty sheets or invalid selection).")

        # --------------------------- arXiv Loader (APPEND, PARAMETER-COMPLETE)
        with st.expander( "🧠 ArXiv Loader", expanded=False ):
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
                help="Maximum characters read",
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
        with st.expander( "📚 Wikipedia Loader", expanded=False ):
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
                help="Maximum number of documents loaded",
            )
            
            wiki_max_chars = st.number_input(
                "Max characters per document",
                min_value=250,
                max_value=100000,
                value=4000,
                step=250,
                key="wiki_max_chars",
                help="Upper limit on the number of characters",
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
        with st.expander( "🐙 GitHub Loader", expanded=False ):
            gh_url = st.text_input(
                "GitHub API URL",
                placeholder="https://api.github.com",
                value="https://api.github.com",
                key="gh_url",
                help="web url to a github repository",
            )
            
            gh_repo = st.text_input(
                "Repo (owner/name)",
                placeholder="openai/openai-python",
                key="gh_repo",
                help="Name of the repository",
            )
            
            gh_branch = st.text_input(
                "Branch",
                placeholder="main",
                value="main",
                key="gh_branch",
                help="The branch of the repository",
            )
            
            gh_filetype = st.text_input(
                "File type filter",
                value=".md",
                key="gh_filetype",
                help="Filering by file type. Example: .py, .md, .txt",
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
        st.subheader("")

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
       
# ======================================================================================
# Tab — Processing / Preprocessing (Grouped Expanders)
# ======================================================================================

with tabs[1]:

    st.subheader("")

    # ------------------------------------------------------------------
    # Defensive session_state initialization (local safety)
    # ------------------------------------------------------------------
    for key, default in {
        "raw_text": None,
        "processed_text": None,
        "active_loader": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    has_text = bool(st.session_state.raw_text)

    if not has_text:
        st.info("No raw text available yet. Load documents to enable processing.")

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    left, right = st.columns([1, 1.5])

    # ------------------------------------------------------------------
    # RIGHT COLUMN — Text Views
    # ------------------------------------------------------------------
    with right:
        st.markdown("##### Raw Text")
        st.text_area(
            "Raw Text",
            st.session_state.raw_text or "No text loaded yet.",
            height=300,
            disabled=True,
            key="raw_text_view",
        )

        st.markdown("##### Processed Text")
        st.text_area(
            "Processed Text",
            st.session_state.processed_text or "",
            height=300,
            key="processed_text_view",
        )

    # ------------------------------------------------------------------
    # LEFT COLUMN — Controls (Grouped Expanders)
    # ------------------------------------------------------------------
    with left:

        active = st.session_state.active_loader

        # ==============================================================
        # Common Text Processing (TextParser)
        # ==============================================================
        with st.expander("🧠 Text Processing", expanded=True):

            remove_html = st.checkbox("Remove HTML")
            remove_markdown = st.checkbox("Remove Markdown")
            remove_special = st.checkbox("Remove Special Characters")
            remove_numbers = st.checkbox("Remove Numbers")
            remove_punctuation = st.checkbox("Remove Punctuation")
            remove_stopwords = st.checkbox("Remove Stopwords")
            normalize_text = st.checkbox("Normalize (lowercase)")
            lemmatize_text = st.checkbox("Lemmatize")
            remove_fragments = st.checkbox("Remove Fragments")
            collapse_whitespace = st.checkbox("Collapse Whitespace")

        # ==============================================================
        # Word-Specific Processing (WordParser)
        # ==============================================================
        extract_tables = extract_paragraphs = False
        with st.expander("📄 Word Processing", expanded=False):
            if active == "WordLoader":
                extract_tables = st.checkbox("Extract Tables")
                extract_paragraphs = st.checkbox("Extract Paragraphs")
            else:
                st.caption("Available when Word documents are loaded.")

        # ==============================================================
        # PDF-Specific Processing (PdfParser)
        # ==============================================================
        remove_headers = join_hyphenated = False
        with st.expander("📕 PDF Processing", expanded=False):
            if active == "PdfLoader":
                remove_headers = st.checkbox("Remove Headers / Footers")
                join_hyphenated = st.checkbox("Join Hyphenated Lines")
            else:
                st.caption("Available when PDF documents are loaded.")

        # ==============================================================
        # HTML-Specific Processing (Structural)
        # ==============================================================
        strip_scripts = keep_headings = keep_paragraphs = keep_tables = False
        with st.expander("🌐 HTML Processing", expanded=False):
            if active == "HtmlLoader":
                strip_scripts = st.checkbox("Strip <script> / <style>")
                keep_headings = st.checkbox("Keep Headings")
                keep_paragraphs = st.checkbox("Keep Paragraphs")
                keep_tables = st.checkbox("Keep Tables")
            else:
                st.caption("Available when HTML documents are loaded.")

        st.divider()

        # ==============================================================
        # Actions
        # ==============================================================
        col_apply, col_reset, col_clear = st.columns(3)

        apply_processing = col_apply.button(
            "Apply", disabled=not has_text
        )
        reset_processing = col_reset.button(
            "Reset to Raw", disabled=not has_text
        )
        clear_processing = col_clear.button(
            "Clear", disabled=not has_text
        )

        # ==============================================================
        # Reset / Clear
        # ==============================================================
        if reset_processing:
            st.session_state.processed_text = st.session_state.raw_text
            st.success("Processed text reset to raw text.")

        if clear_processing:
            st.session_state.processed_text = None
            st.success("Processed text cleared.")

        # ==============================================================
        # Apply Processing (Execution Order Matters)
        # ==============================================================
        if apply_processing:
            text = st.session_state.raw_text

            # ----------------------------------------------------------
            # Format-specific FIRST
            # ----------------------------------------------------------
            if active == "WordLoader":
                parser = WordParser()
                if extract_tables and hasattr(parser, "extract_tables"):
                    text = parser.extract_tables(text) or text
                if extract_paragraphs and hasattr(parser, "extract_paragraphs"):
                    text = parser.extract_paragraphs(text) or text

            if active == "PdfLoader":
                parser = PdfParser()
                if remove_headers and hasattr(parser, "remove_headers"):
                    text = parser.remove_headers(text) or text
                if join_hyphenated and hasattr(parser, "join_hyphenated"):
                    text = parser.join_hyphenated(text) or text

            if active == "HtmlLoader":
                if strip_scripts:
                    text = TextParser().remove_html(text) or text
                # Structural selectors can be refined later

            # ----------------------------------------------------------
            # Common TextParser pipeline (string → string)
            # ----------------------------------------------------------
            tp = TextParser()

            if remove_html:
                text = tp.remove_html(text) or text
            if remove_markdown:
                text = tp.remove_markdown(text) or text
            if remove_special:
                text = tp.remove_special(text) or text
            if remove_numbers:
                text = tp.remove_numbers(text) or text
            if remove_punctuation:
                text = tp.remove_punctuation(text) or text
            if remove_stopwords:
                text = tp.remove_stopwords(text) or text
            if normalize_text:
                text = tp.normalize_text(text) or text
            if lemmatize_text:
                text = tp.lemmatize_text(text) or text
            if remove_fragments:
                text = tp.remove_fragments(text) or text
            if collapse_whitespace:
                text = tp.collapse_whitespace(text) or text

            st.session_state.processed_text = text
            st.success("Text processing applied.")


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

    st.subheader("")

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
