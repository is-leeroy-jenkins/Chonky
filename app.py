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

from collections import Counter
import base64
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import List
import math
import nltk
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

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
	import textstat
	TEXTSTAT_AVAILABLE = True
except ImportError:
	TEXTSTAT_AVAILABLE = False
	
# ================================================================================
# Contants / Helpers / Utilities
# ============================================================================
BASE_DIR = os.path.dirname( os.path.abspath( __file__ ) )

CHUNKABLE_LOADERS = {
		'TextLoader': [ 'chars', 'tokens' ],
		'CsvLoader': [ 'chars' ],
		'PdfLoader': [ 'chars' ],
		'ExcelLoader': [ 'chars' ],
		'WordLoader': [ 'chars' ],
		'MarkdownLoader': [ 'chars' ],
		'HtmlLoader': [ 'chars' ],
		'JsonLoader': [ 'chars' ],
		'PowerPointLoader': [ 'chars' ],
}

BLUE_DIVIDER = "<div style='height:2px;align:left;background:#0078FC;margin:6px 0 10px 0;'></div>"

def encode_image_base64( path: str ) -> str:
	data = Path( path ).read_bytes( )
	return base64.b64encode( data ).decode( "utf-8" )

SESSION_STATE_DEFAULTS = {
		# Ingestion
		'documents': None,
		'raw_documents': None,
		'active_loader': None,
		# Input
		'raw_text': None,
		'raw_text_view': None,
		# Processing
		'processed_text': None,
		'processed_text_view': None,
		# Tokenization
		'tokens': None,
		'vocabulary': None,
		'token_counts': None,
		# SQLite / Excel
		'sqlite_tables': [ ],
		'active_table': None,
		# Chunking
		'chunks': None,
		# Vectorization
		'embeddings': None,
		'embedding_model': None,
		# Retrieval / Search
		'search_results': None,
		# DataFrames
		'df_frequency': None,
		'df_tables': None,
		'df_schema': None,
		'df_preview': None,
		'df_count': None
}

def clear_if_active( loader_name: str ) -> None:
	if st.session_state.active_loader == loader_name:
		st.session_state.documents = None
		st.session_state.raw_documents = None
		st.session_state.raw_text = None
		st.session_state.raw_text_view = None
		st.session_state.processed_text = None
		st.session_state.processed_text_view = None
		st.session_state.active_loader = None
		st.session_state.tokens = None
		st.session_state.vocabulary = None
		st.session_state.token_counts = None
		st.session_state.chunks = NOne
		st.session_state.embeddings = None
		st.session_state.embedding_model = None
		st.session_state.sqlite_tables = None
		st.session_state.active_table = None
		st.session_state.df_frequency = None
		st.session_state.df_tables = None
		st.session_state.df_schema = None
		st.session_state.df_preview = None
		st.session_state.df_count = None

# ======================================================================================
# Page Configuration
# ======================================================================================

st.set_page_config(
	page_title='Chonky',
	layout='wide',
	page_icon=cfg.ICON,
)

st.markdown( "" )

# ======================================================================================
# Session State Initialization
# ======================================================================================

for key, default in SESSION_STATE_DEFAULTS.items( ):
	if key not in st.session_state:
		st.session_state[ key ] = default

# ======================================================================================
# Sidebar (Branding / Global Only)
# ======================================================================================
with st.sidebar:
	try:
		logo_b64 = encode_image_base64( cfg.LOGO )
		st.markdown(
			f"""
            <div style="
                display: flex;
                justify-content: center;
                align-items: center;
                margin-bottom: 0.95rem;
            ">
                <img src="data:image/png;base64,{logo_b64}"
                     style="max-height: 30px;" />
            </div>
            """,
			unsafe_allow_html=True
		)
	except Exception:
		st.write( "Bro" )
	
	st.header( 'Chonky' )
	st.caption( 'Pipelines & Plumbling' )
	st.markdown( BLUE_DIVIDER, unsafe_allow_html=True )
	st.subheader( "" )

# ======================================================================================
# Tabs
# ======================================================================================

tabs = st.tabs(
	[
			'Loading',
			'Processing',
			'Scaffolding',
			'Tokens & Vocabulary',
			'Analysis & Statistics',
			'Vectorization & Chunking',
			'Export',
			'Data'
	]
)

# ======================================================================================
# Tab 1 — Loading (PER-EXPANDER CLEAR / RESET, NO SIDEBAR)
# ======================================================================================

with tabs[ 0 ]:
	metrics_container = st.container( )
	def render_metrics_panel( ):
		raw_text = st.session_state.get( 'raw_text' )
		if not isinstance( raw_text, str ) or not raw_text.strip( ):
			st.info( 'Load data to view corpus metrics.' )
			return
		
		# -------------------------------
		# Tokenization
		# -------------------------------
		try:
			tokens = [
					t.lower( )
					for t in word_tokenize( raw_text )
					if t.isalpha( )
			]
		except LookupError:
			st.error(
				'NLTK resources missing.\n\n'
				'Run:\n'
				'`python -m nltk.downloader punkt stopwords`'
			)
			return
		
		if not tokens:
			st.warning( 'No valid alphabetic tokens found.' )
			return
		
		# -------------------------------
		# Core counts
		# -------------------------------
		char_count = len( raw_text )
		token_count = len( tokens )
		vocab = set( tokens )
		vocab_size = len( vocab )
		counts = Counter( tokens )
		
		# -------------------------------
		# Lexical statistics
		# -------------------------------
		hapax_count = sum( 1 for c in counts.values( ) if c == 1 )
		hapax_ratio = hapax_count / vocab_size if vocab_size else 0.0
		avg_word_len = sum( len( t ) for t in tokens ) / token_count
		ttr = vocab_size / token_count
		
		# Defaults (CRITICAL)
		stopword_ratio = 0.0
		lexical_density = 0.0
		
		try:
			stop_words = set( stopwords.words( 'english' ) )
			stopword_ratio = sum( 1 for t in tokens if t in stop_words ) / token_count
			lexical_density = 1.0 - stopword_ratio
		except LookupError:
			# Keep defaults — do NOT error
			pass
		
		# -------------------------------
		# Corpus Metrics
		# -------------------------------
		with st.expander( '📊 Corpus Metrics', expanded=False ):
			c1, c2, c3, c4 = st.columns( 4 )
			c1.metric( 'Characters', f'{char_count:,}' )
			c2.metric( 'Tokens', f'{token_count:,}' )
			c3.metric( 'Unique Tokens', f'{vocab_size:,}' )
			c4.metric( 'TTR', f'{ttr:.3f}' )
			
			c5, c6, c7, c8 = st.columns( 4 )
			c5.metric( 'Hapax Ratio', f'{hapax_ratio:.3f}' )
			c6.metric( 'Avg Word Length', f'{avg_word_len:.2f}' )
			c7.metric( 'Stopword Ratio', f'{stopword_ratio:.2%}' )
			c8.metric( 'Lexical Density', f'{lexical_density:.2%}' )
		
		# -------------------------------
		# Readability
		# -------------------------------
		with st.expander( '📖 Readability', expanded=False ):
			if TEXTSTAT_AVAILABLE:
				r1, r2, r3 = st.columns( 3 )
				r1.metric( 'Flesch Reading Ease', f'{textstat.flesch_reading_ease( raw_text ):.1f}' )
				r2.metric(
					'Flesch–Kincaid Grade',
					f'{textstat.flesch_kincaid_grade( raw_text ):.1f}',
				)
				r3.metric(
					'Gunning Fog',
					f'{textstat.gunning_fog( raw_text ):.1f}',
				)
			else:
				st.caption( 'Install `textstat` to enable readability metrics.' )
		
		# -------------------------------
		# Top Tokens
		# -------------------------------
		with st.expander( "🔤 Top Tokens", expanded=False ):
			top_tokens = counts.most_common( 10 )
			st.table( [ { "token": tok, "count": cnt } for tok, cnt in top_tokens ] )
	
	# ------------------------------------------------------------------
	# Defensive session_state initialization
	# ------------------------------------------------------------------
	for key, default in SESSION_STATE_DEFAULTS.items( ):
		if key not in st.session_state:
			st.session_state[ key ] = default
	
	# ------------------------------------------------------------------
	# Layout
	# ------------------------------------------------------------------
	left, right = st.columns( [ 1, 1.5 ] )
	
	
	# ------------------------------------------------------------------
	# LEFT COLUMN — Loader Expanders
	# ------------------------------------------------------------------
	with left:
		st.subheader( "" )
		
		# --------------------------- Text Loader
		with st.expander( '📄 Text Loader', expanded=False ):
			files = st.file_uploader(
				'Upload TXT files',
				type=[ 'txt' ],
				accept_multiple_files=True,
				key='txt_upload',
			)
			
			col_load, col_clear = st.columns( 2 )
			load_txt = col_load.button( 'Load', key='txt_load' )
			clear_txt = col_clear.button( 'Clear', key='txt_clear' )
			
			if clear_txt:
				clear_if_active( 'TextLoader' )
				st.info( 'Text Loader state cleared.' )
			
			if load_txt and files:
				docs = [ ]
				for f in files:
					text = f.read( ).decode( 'utf-8', errors='ignore' )
					docs.append(
						Document(
							page_content=text,
							metadata={
									'source': f.name,
									'loader': 'TextLoader' },
						)
					)
				
				st.session_state.documents = docs
				st.session_state.raw_documents = list( docs )
				st.session_state.raw_text = "\n\n".join( d.page_content for d in docs )
				st.session_state.processed_text = None
				st.session_state.active_loader = "TextLoader"
				
				st.success( f'Loaded {len( docs )} text document(s).' )
		
		# --------------------------- NLTK Loader (BUILT-IN + LOCAL)
		with st.expander( '📚 Corpora Loader', expanded=False ):
			import nltk
			from nltk.corpus import (
				brown,
				gutenberg,
				reuters,
				webtext,
				inaugural,
				state_union,
			)
			
			st.markdown( '#### NLTK Corpora' )
			
			corpus_name = st.selectbox(
				'Select corpus',
				[
						'Brown',
						'Gutenberg',
						'Reuters',
						'WebText',
						'Inaugural',
						'State of the Union',
				],
				key='nltk_corpus_name',
			)
			
			file_ids = [ ]
			try:
				if corpus_name == 'Brown':
					file_ids = brown.fileids( )
				elif corpus_name == 'Gutenberg':
					file_ids = gutenberg.fileids( )
				elif corpus_name == 'Reuters':
					file_ids = reuters.fileids( )
				elif corpus_name == 'WebText':
					file_ids = webtext.fileids( )
				elif corpus_name == 'Inaugural':
					file_ids = inaugural.fileids( )
				elif corpus_name == 'State of the Union':
					file_ids = state_union.fileids( )
			except LookupError:
				st.error(
					'NLTK corpus not found. Run:\n\n'
					'python -m nltk.downloader all\n\n'
					'or download individual corpora.'
				)
			
			selected_files = st.multiselect(
				"Select files (leave empty to load all)",
				options=file_ids,
				key="nltk_file_ids",
			)
			
			st.divider( )
			
			st.markdown( '#### Local Corpus' )
			
			local_corpus_dir = st.text_input(
				'Local directory',
				placeholder="path/to/text/files",
				key="nltk_local_dir",
			)
			
			col_load, col_clear = st.columns( 2 )
			load_nltk = col_load.button( 'Load', key='nltk_load' )
			clear_nltk = col_clear.button( 'Clear', key='nltk_clear' )
			
			# ------------------------------------------------------
			# Clear logic
			# ------------------------------------------------------
			if clear_nltk and st.session_state.get( "documents" ):
				st.session_state.documents = [
						d for d in st.session_state.documents
						if d.metadata.get( "loader" ) != "NLTKLoader"
				]
				st.info( "NLTKLoader documents removed." )
			
			# ------------------------------------------------------
			# Load logic
			# ------------------------------------------------------
			if load_nltk:
				docs = [ ]
				
				# Built-in corpora
				if file_ids:
					files_to_load = selected_files or file_ids
					
					for fid in files_to_load:
						try:
							if corpus_name == 'Brown':
								text = ' '.join( brown.words( fid ) )
							elif corpus_name == 'Gutenberg':
								text = gutenberg.raw( fid )
							elif corpus_name == 'Reuters':
								text = reuters.raw( fid )
							elif corpus_name == 'WebText':
								text = webtext.raw( fid )
							elif corpus_name == 'Inaugural':
								text = inaugural.raw( fid )
							elif corpus_name == 'State of the Union':
								text = state_union.raw( fid )
							
							docs.append(
								Document(
									page_content=text,
									metadata={
											'loader': 'NLTKLoader',
											'corpus': corpus_name,
											'file_id': fid,
									},
								)
							)
						except Exception:
							continue
				
				# Local corpus
				if local_corpus_dir and os.path.isdir( local_corpus_dir ):
					for fname in os.listdir( local_corpus_dir ):
						path = os.path.join( local_corpus_dir, fname )
						if os.path.isfile( path ) and fname.lower( ).endswith( ".txt" ):
							with open( path, "r", encoding="utf-8", errors="ignore" ) as f:
								text = f.read( )
							
							docs.append(
								Document(
									page_content=text,
									metadata={
											"loader": "NLTKLoader",
											"source": path,
									},
								)
							)
				
				if docs:
					if st.session_state.documents:
						st.session_state.documents.extend( docs )
					else:
						st.session_state.documents = docs
						st.session_state.raw_documents = list( docs )
						st.session_state.raw_text = "\n\n".join(
							d.page_content for d in docs
						)
					
					st.session_state.active_loader = "NLTKLoader"
					st.success( f"Loaded {len( docs )} document(s) from NLTK." )
				else:
					st.warning( "No documents loaded." )
		
		# --------------------------- CSV Loader
		with st.expander( "📑 CSV Loader", expanded=False ):
			csv_file = st.file_uploader( "Upload CSV", type=[ "csv" ], key="csv_upload" )
			delimiter = st.text_input( "Delimiter", value="\n\n", key="csv_delim" )
			quotechar = st.text_input( "Quote Character", value='"', key="csv_quote" )
			
			col_load, col_clear = st.columns( 2 )
			load_csv = col_load.button( "Load", key="csv_load" )
			clear_csv = col_clear.button( "Clear", key="csv_clear" )
			
			if clear_csv:
				clear_if_active( "CsvLoader" )
				st.info( "CSV Loader state cleared." )
			
			if load_csv and csv_file:
				with tempfile.TemporaryDirectory( ) as tmp:
					path = os.path.join( tmp, csv_file.name )
					with open( path, "wb" ) as f:
						f.write( csv_file.read( ) )
					
					loader = CsvLoader( )
					docs = loader.load(
						path,
						columns=None,
						delimiter=delimiter,
						quotechar=quotechar,
					)
				
				st.session_state.documents = docs
				st.session_state.raw_documents = list( docs )
				st.session_state.raw_text = "\n\n".join( d.page_content for d in docs )
				st.session_state.processed_text = None
				st.session_state.active_loader = "CsvLoader"
				
				st.success( f"Loaded {len( docs )} CSV document(s)." )
		
		# --------------------------- PDF Loader
		with st.expander( '📕 PDF Loader', expanded=False ):
			pdf = st.file_uploader( 'Upload PDF', type=[ 'pdf' ], key='pdf_upload' )
			mode = st.selectbox( 'Mode', [ 'single',
			                               'elements' ], key='pdf_mode' )
			extract = st.selectbox( 'Extract', [ 'plain',
			                                     'ocr' ], key='pdf_extract' )
			include = st.checkbox( 'Include Images', value=True, key='pdf_include' )
			fmt = st.selectbox( 'Format', [ 'markdown-img',
			                                'text' ], key='pdf_fmt' )
			
			col_load, col_clear = st.columns( 2 )
			load_pdf = col_load.button( 'Load', key='pdf_load' )
			clear_pdf = col_clear.button( 'Clear', key='pdf_clear' )
			
			if clear_pdf:
				clear_if_active( "PdfLoader" )
				st.info( "PDF Loader state cleared." )
			
			if load_pdf and pdf:
				with tempfile.TemporaryDirectory( ) as tmp:
					path = os.path.join( tmp, pdf.name )
					with open( path, "wb" ) as f:
						f.write( pdf.read( ) )
					
					loader = PdfLoader( )
					docs = loader.load(
						path,
						mode=mode,
						extract=extract,
						include=include,
						format=fmt,
					)
				
				st.session_state.documents = docs
				st.session_state.raw_documents = list( docs )
				st.session_state.raw_text = '\n\n'.join( d.page_content for d in docs )
				st.session_state.processed_text = None
				st.session_state.active_loader = 'PdfLoader'
				
				st.success( f'Loaded {len( docs )} PDF document(s).' )
		
		# --------------------------- Markdown Loader
		with st.expander( '🧾 Markdown Loader', expanded=False ):
			md = st.file_uploader( 'Upload Markdown', type=[ 'md',
			                                                 'markdown' ], key='md_upload' )
			
			col_load, col_clear = st.columns( 2 )
			load_md = col_load.button( 'Load', key='md_load' )
			clear_md = col_clear.button( 'Clear', key='md_clear' )
			
			if clear_md:
				clear_if_active( 'MarkdownLoader' )
				st.info( "Markdown Loader state cleared." )
			
			if load_md and md:
				with tempfile.TemporaryDirectory( ) as tmp:
					path = os.path.join( tmp, md.name )
					with open( path, "wb" ) as f:
						f.write( md.read( ) )
					
					loader = MarkdownLoader( )
					docs = loader.load( path )
				
				st.session_state.documents = docs
				st.session_state.raw_documents = list( docs )
				st.session_state.raw_text = "\n\n".join( d.page_content for d in docs )
				st.session_state.processed_text = None
				st.session_state.active_loader = "MarkdownLoader"
				
				st.success( f"Loaded {len( docs )} Markdown document(s)." )
		
		# --------------------------- HTML Loader
		with st.expander( '🌐 HTML Loader', expanded=False ):
			html = st.file_uploader( 'Upload HTML', type=[ 'html', 'htm' ], key='html_upload' )
			
			col_load, col_clear = st.columns( 2 )
			load_html = col_load.button( 'Load', key='html_load' )
			clear_html = col_clear.button( 'Clear', key='html_clear' )
			
			if clear_html:
				clear_if_active( "HtmlLoader" )
				st.info( "HTML Loader state cleared." )
			
			if load_html and html:
				with tempfile.TemporaryDirectory( ) as tmp:
					path = os.path.join( tmp, html.name )
					with open( path, "wb" ) as f:
						f.write( html.read( ) )
					
					loader = HtmlLoader( )
					docs = loader.load( path )
				
				st.session_state.documents = docs
				st.session_state.raw_documents = list( docs )
				st.session_state.raw_text = "\n\n".join( d.page_content for d in docs )
				st.session_state.processed_text = None
				st.session_state.active_loader = "HtmlLoader"
				
				st.success( f"Loaded {len( docs )} HTML document(s)." )
		
		# --------------------------- JSON Loader
		with st.expander( '🧩 JSON Loader', expanded=False ):
			js = st.file_uploader( 'Upload JSON', type=[ 'json' ], key='json_upload' )
			is_lines = st.checkbox( 'JSON Lines', value=False, key='json_lines' )
			
			col_load, col_clear = st.columns( 2 )
			load_json = col_load.button( 'Load', key='json_load' )
			clear_json = col_clear.button( 'Clear', key='json_clear' )
			
			if clear_json:
				clear_if_active( 'JsonLoader' )
				st.info( 'JSON Loader state cleared.' )
			
			if load_json and js:
				with tempfile.TemporaryDirectory( ) as tmp:
					path = os.path.join( tmp, js.name )
					with open( path, 'wb' ) as f:
						f.write( js.read( ) )
					
					loader = JsonLoader( )
					docs = loader.load(
						path,
						is_text=True,
						is_lines=is_lines,
					)
				
				st.session_state.documents = docs
				st.session_state.raw_documents = list( docs )
				st.session_state.raw_text = "\n\n".join( d.page_content for d in docs )
				st.session_state.processed_text = None
				st.session_state.active_loader = "JsonLoader"
				
				st.success( f"Loaded {len( docs )} JSON document(s)." )
		
		# --------------------------- PowerPoint Loader
		with st.expander( '📽 Power Point Loader', expanded=False ):
			pptx = st.file_uploader( 'Upload PPTX', type=[ 'pptx' ], key='pptx_upload' )
			mode = st.selectbox( 'Mode', [ 'single', 'multiple' ], key='pptx_mode' )
			col_load, col_clear = st.columns( 2 )
			load_pptx = col_load.button( 'Load', key='pptx_load' )
			clear_pptx = col_clear.button( 'Clear', key='pptx_clear' )
			if clear_pptx:
				clear_if_active( 'PowerPointLoader' )
				st.info( 'PowerPoint Loader state cleared.' )
			
			if load_pptx and pptx:
				with tempfile.TemporaryDirectory( ) as tmp:
					path = os.path.join( tmp, pptx.name )
					with open( path, "wb" ) as f:
						f.write( pptx.read( ) )
					
					loader = PowerPointLoader( )
					docs = (
							loader.load( path )
							if mode == "single"
							else loader.load_multiple( path )
					)
				
				st.session_state.documents = docs
				st.session_state.raw_documents = list( docs )
				st.session_state.raw_text = "\n\n".join( d.page_content for d in docs )
				st.session_state.processed_text = None
				st.session_state.active_loader = "PowerPointLoader"
				
				st.success( f"Loaded {len( docs )} PowerPoint document(s)." )
		
		# --------------------------- Excel Loader (FILE + SQLITE)
		with st.expander( '📊 Excel Loader', expanded=False ):
			excel_file = st.file_uploader(
				'Upload Excel file',
				type=[ 'xlsx', 'xls' ],
				key='excel_upload',
			)
			
			sheet_name = st.text_input( 'Sheet name (leave blank for all sheets)', key='excel_sheet', )
			
			table_prefix = st.text_input(
				'SQLite table prefix',
				value='excel',
				help='Each sheet will be written as <prefix>_<sheetname>',
				key='excel_table_prefix',
			)
			
			col_load, col_clear = st.columns( 2 )
			load_excel = col_load.button( 'Load', key='excel_load' )
			clear_excel = col_clear.button( 'Clear', key='excel_clear' )
			
			# ------------------------------------------------------
			# Clear logic (remove only ExcelLoader documents)
			# ------------------------------------------------------
			if clear_excel and st.session_state.get( "documents" ):
				st.session_state.documents = [
						d for d in st.session_state.documents
						if d.metadata.get( "loader" ) != "ExcelLoader"
				]
				st.info( "ExcelLoader documents removed." )
			
			# ------------------------------------------------------
			# Load + SQLite ingestion
			# ------------------------------------------------------
			if load_excel and excel_file:
				# Ensure SQLite directory exists
				sqlite_path = os.path.join( "stores", "sqlite", "data.db" )
				os.makedirs( os.path.dirname( sqlite_path ), exist_ok=True )
				
				with tempfile.TemporaryDirectory( ) as tmp:
					excel_path = os.path.join( tmp, excel_file.name )
					with open( excel_path, "wb" ) as f:
						f.write( excel_file.read( ) )
					
					# Read Excel into DataFrames
					if sheet_name.strip( ):
						dfs = {
								sheet_name: pd.read_excel( excel_path, sheet_name=sheet_name )
						}
					else:
						dfs = pd.read_excel( excel_path, sheet_name=None )
				
				# Open SQLite connection
				conn = sqlite3.connect( sqlite_path )
				
				docs = [ ]
				
				for sheet, df in dfs.items( ):
					if df.empty:
						continue
					
					# Normalize table name
					table_name = f"{table_prefix}_{sheet}".replace( " ", "_" ).lower( )
					
					# Write DataFrame to SQLite
					df.to_sql(
						table_name,
						conn,
						if_exists="replace",
						index=False,
					)
					
					# Convert DataFrame to text for NLP pipeline
					text = df.to_csv( index=False )
					
					docs.append(
						Document(
							page_content=text,
							metadata={
									'loader': 'ExcelLoader',
									'source': excel_file.name,
									'sheet': sheet,
									'table': table_name,
									'sqlite_db': sqlite_path,
							},
						)
					)
				
				conn.close( )
				
				if docs:
					if st.session_state.documents:
						st.session_state.documents.extend( docs )
					else:
						st.session_state.documents = docs
						st.session_state.raw_documents = list( docs )
						st.session_state.raw_text = "\n\n".join(
							d.page_content for d in docs
						)
					
					st.session_state.active_loader = "ExcelLoader"
					st.success(
						f"Loaded {len( docs )} sheet(s) and stored in SQLite."
					)
				else:
					st.warning( "No data loaded (empty sheets or invalid selection)." )
		
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
		st.subheader( "" )
		
		docs = st.session_state.documents
		if not docs:
			st.info( 'No documents loaded.' )
		else:
			st.caption( f'Active Loader: {st.session_state.active_loader}' )
			st.write( f'Documents: {len( docs )}' )
			
			for i, d in enumerate( docs[ :5 ] ):
				with st.expander( f'Document {i + 1}', expanded=False ):
					st.json( d.metadata )
					st.text_area(
						'Content',
						d.page_content[ :5000 ],
						height=200,
						key=f'preview_doc_{i}',
					)

	with metrics_container:
		render_metrics_panel( )
		
# ======================================================================================
# Tab — Processing / Preprocessing (Grouped Expanders)
# ======================================================================================
with tabs[ 1 ]:
	for key, default in SESSION_STATE_DEFAULTS.items( ):
		if key not in st.session_state:
			st.session_state[ key ] = default
			
	raw_text = st.session_state.get( 'raw_text' )
	processed_text = st.session_state.get( 'processed_text' )
	has_text = isinstance( raw_text, str ) and bool( raw_text.strip( ) )
	
	# ------------------------------------------------------------------
	# Cascade: whenever raw_text changes, re-seed BOTH processed_text and the widget view
	# ------------------------------------------------------------------
	if has_text:
		seed_hash = hash( raw_text )
		
		if st.session_state.get( "_processing_seed_hash" ) != seed_hash:
			# New load detected → seed processed text AND the widget-backed view
			st.session_state.processed_text = processed_text
			st.session_state.processed_text_view = processed_text
			
			# Keep raw view in sync as well (disabled widget still has state)
			st.session_state.raw_text_view = raw_text
			
			st.session_state._processing_seed_hash = processed_text
	else:
		st.session_state.raw_text_view = ""

	if not has_text:
		st.info( "No raw text available yet. Load documents to enable processing." )

	# ------------------------------------------------------------------
	# Layout
	# ------------------------------------------------------------------
	left, right = st.columns( [ 1, 1.5 ] )
	
	# ------------------------------------------------------------------
	# LEFT COLUMN — Controls (Grouped Expanders)
	# ------------------------------------------------------------------
	with left:
		active = st.session_state.get( "active_loader" )

		# ==============================================================
		# Common Text Processing (TextParser)
		# ==============================================================
		with st.expander( "🧠 Text Processing", expanded=True ):
			remove_html = st.checkbox( "Remove HTML" )
			remove_markdown = st.checkbox( "Remove Markdown" )
			remove_special = st.checkbox( "Remove Special Characters" )
			remove_numbers = st.checkbox( "Remove Numbers" )
			remove_punctuation = st.checkbox( "Remove Punctuation" )
			remove_formatting = st.checkbox( 'Remove Formatting' )
			remove_stopwords = st.checkbox( "Remove Stopwords" )
			remove_numerals = st.checkbox( 'Remove Numerals' )
			remove_encodings = st.checkbox( 'Remove Encoding' )
			normalize_text = st.checkbox( "Normalize (lowercase)" )
			lemmatize_text = st.checkbox( "Lemmatize" )
			remove_fragments = st.checkbox( "Remove Fragments" )
			collapse_whitespace = st.checkbox( "Collapse Whitespace" )
			compress_whitespace = st.checkbox( "Compress Whitespace" )

		# ==============================================================
		# Word-Specific Processing (WordParser)
		# ==============================================================
		extract_tables = extract_paragraphs = False
		with st.expander( "📄 Word Processing", expanded=False ):
			if active == "WordLoader":
				extract_tables = st.checkbox( "Extract Tables" )
				extract_paragraphs = st.checkbox( "Extract Paragraphs" )
			else:
				st.caption( "Available when Word documents are loaded." )

		# ==============================================================
		# PDF-Specific Processing (PdfParser)
		# ==============================================================
		remove_headers = join_hyphenated = False
		with st.expander( "📕 PDF Processing", expanded=False ):
			if active == "PdfLoader":
				remove_headers = st.checkbox( "Remove Headers / Footers" )
				join_hyphenated = st.checkbox( "Join Hyphenated Lines" )
			else:
				st.caption( "Available when PDF documents are loaded." )

		# ==============================================================
		# HTML-Specific Processing (Structural)
		# ==============================================================
		strip_scripts = keep_headings = keep_paragraphs = keep_tables = False
		with st.expander( "🌐 HTML Processing", expanded=False ):
			if active == "HtmlLoader":
				strip_scripts = st.checkbox( "Strip <script> / <style>" )
				keep_headings = st.checkbox( "Keep Headings" )
				keep_paragraphs = st.checkbox( "Keep Paragraphs" )
				keep_tables = st.checkbox( "Keep Tables" )
			else:
				st.caption( "Available when HTML documents are loaded." )
		
		st.markdown( BLUE_DIVIDER, unsafe_allow_html=True )

		# ==============================================================
		# Actions
		# ==============================================================
		col_apply, col_reset, col_clear = st.columns( 3 )

		apply_processing = col_apply.button( "Apply", disabled=not has_text )
		reset_processing = col_reset.button( "Reset", disabled=not has_text )
		clear_processing = col_clear.button( "Clear", disabled=not has_text )

		# ==============================================================
		# Reset / Clear
		# ==============================================================
		if reset_processing:
			st.session_state.processed_text = ''
			st.session_state.processed_text_view = ''
			st.success( "Processed text reset to raw text." )
		
		if clear_processing:
			st.session_state.processed_text = ""
			st.session_state.processed_text_view = ""
			st.success( "Processed text cleared." )

		# ==============================================================
		# Apply Processing (Execution Order Matters)
		# ==============================================================
		if apply_processing:
			processed_text = raw_text
			tp = TextParser( )

			if remove_html:
				processed_text = tp.remove_html( processed_text )
			if remove_markdown:
				processed_text = tp.remove_markdown( processed_text )
			if remove_special:
				processed_text = tp.remove_special( processed_text )
			if remove_numbers:
				processed_text = tp.remove_numbers( processed_text )
			if remove_formatting:
				processed_text = tp.remove_formatting( processed_text )
			if remove_punctuation:
				processed_text = tp.remove_punctuation( processed_text )
			if remove_stopwords:
				processed_text = tp.remove_stopwords( processed_text )
			if remove_numerals:
				processed_text = tp.remove_numerals( processed_text )
			if remove_encodings:
				processed_text = tp.remove_encodings( processed_text )
			if normalize_text:
				processed_text = tp.normalize_text( processed_text )
			if lemmatize_text:
				processed_text = tp.lemmatize_text( processed_text )
			if remove_fragments:
				processed_text = tp.remove_fragments( processed_text )
			if collapse_whitespace:
				processed_text = tp.collapse_whitespace( processed_text )
			if compress_whitespace:
				processed_text = tp.compress_whitespace( processed_text )

			# ----------------------------------------------------------
			# Format-specific FIRST
			# ----------------------------------------------------------
			if active == 'WordLoader':
				parser = WordParser( )
				if extract_tables and hasattr( parser, 'extract_tables' ):
					processed_text = parser.extract_tables( processed_text )
				if extract_paragraphs and hasattr( parser, 'extract_paragraphs' ):
					processed_text = parser.extract_paragraphs( processed_text )
	
			if active == 'PdfLoader':
				parser = PdfParser( )
				if remove_headers and hasattr( parser, 'remove_headers' ):
					processed_text = parser.remove_headers( processed_text )
				if join_hyphenated and hasattr( parser, 'join_hyphenated' ):
					processed_text = parser.join_hyphenated( processed_text )
	
			if active == 'HtmlLoader':
				if strip_scripts:
					processed_text = TextParser( ).remove_html( processed_text )
				
			# Structural selectors can be refined later
			st.session_state.processed_text = processed_text
			st.session_state.processed_text_view = processed_text
			st.success( 'Text processing applied.' )
	
	# ------------------------------------------------------------------
	# RIGHT COLUMN — Text Views
	# ------------------------------------------------------------------
	with right:
		st.text_area(
			'Raw Text',
			value=st.session_state.raw_text_view or 'No text loaded yet.',
			height=150,
			disabled=False,
			key='raw_text_view',
		)
		
		st.text_area(
			'Processed Text',
			value=st.session_state.processed_text_view or "",
			height=300,
			key='processed_text_view',
		)
		

# ==========================================================================================
# Tab 3 — Structural Views
# ==========================================================================================
with tabs[ 2 ]:
	st.header( "" )
	for key, default in SESSION_STATE_DEFAULTS.items( ):
		if key not in st.session_state:
			st.session_state[ key ] = default

	df_frequency = st.session_state.get( 'df_frequency' )
	dr_tables = st.session_state.get( 'df_tables' )
	df_count = st.session_state.get( 'df_count' )
	df_schema = st.session_state.get( 'df_schema' )
	df_preview = st.session_state.get( 'df_preview' )
	chunks = st.session_state.get( 'chunks' )
	embedding_model = st.session_state.get( 'embedding_model' )
	embeddings = st.session_state.get( 'embeddings' )
	sqlite_tables = st.session_state.get( 'sqlite_tables' )
	active_table = st.session_state.get( 'active_table' )
	
	if st.session_state.processed_text:
		processor = TextParser( )
		
		view = st.selectbox(
			"View Type",
			[ 'Lines',
			  'Paragraphs',
			  'Sentences' ]
		)
		
		if view == 'Lines':
			lines = processor.split_sentences( processed_text )
			st.dataframe( pd.DataFrame( lines, columns=[ 'Line' ] ))
		
		elif view == 'Paragraphs':
			paragraphs = processed_text.split( '\n\n' )
			st.dataframe( pd.DataFrame( paragraphs, columns=[ 'Paragraph' ] ))
		
		elif view == 'Sentences':
			sentences = processor.split_sentences( processed_text )
			st.dataframe( pd.DataFrame( sentences, columns=[ 'Sentence' ] ))
	
	else:
		st.info( 'Run preprocessing first' )

# ==========================================================================================
# Tab 4 — Tokens & Vocabulary
# ==========================================================================================

with tabs[ 3 ]:
	st.header( "" )
	for key, default in SESSION_STATE_DEFAULTS.items( ):
		if key not in st.session_state:
			st.session_state[ key ] = default
	
	if st.session_state.processed_text:
		processor = TextParser( )
		tokens = word_tokenize( processed_text )
		vocab = processor.create_vocabulary( tokens )
		
		st.write( f'Token Count: {len( tokens )}' )
		st.dataframe( pd.DataFrame( tokens, columns=[ 'Token' ] )  )
		
		st.write( f'Vocabulary Size: {len( vocab )}' )
		st.dataframe( pd.DataFrame( vocab, columns=[ 'Word' ] ) )
		st.session_state.tokens = tokens
		st.session_state.vocabulary = vocab
	
	else:
		st.info( 'Run preprocessing first' )

# ==========================================================================================
# Tab 5 — Analysis & Statistics
# ==========================================================================================

with tabs[ 4 ]:
	st.header( "" )
	for key, default in SESSION_STATE_DEFAULTS.items( ):
		if key not in st.session_state:
			st.session_state[ key ] = default
			
	df_frequency = st.session_state.get( 'df_frequency' )
	df_table = st.session_state.get( 'df_table' )
	
	if st.session_state.tokens:
		processor = TextParser( )
		df_frequency = processor.create_frequency_distribution( st.session_state.tokens )
		st.session_state.df_frequency = df_frequency
		st.dataframe( df_frequency )
	
	else:
		st.info( 'Generate tokens first' )

# ==========================================================================================
# Tab 6 — 🧩 Vectorization & Chunking (or Chunking tab index)
# ==========================================================================================
with tabs[ 5 ]:
	st.subheader( "" )
	for key, default in SESSION_STATE_DEFAULTS.items( ):
		if key not in st.session_state:
			st.session_state[ key ] = default
	
	docs = st.session_state.get( 'documents' )
	loader_name = st.session_state.get( 'active_loader' )
	
	if not docs:
		st.warning( 'No documents loaded. Please load documents first.' )
		st.stop( )
	
	if not loader_name:
		st.warning( 'No active loader found.' )
		st.stop( )
	
	chunk_modes = CHUNKABLE_LOADERS.get( loader_name )
	
	if not chunk_modes:
		st.info( f'Chunking is not supported for loader: {loader_name}' )
		st.stop( )
	
	st.caption( f'Source Loader: {loader_name}' )
	
	# ---------------------------
	# Chunking Controls
	# ---------------------------
	mode = st.selectbox(
		'Chunking Mode',
		options=chunk_modes,
		help='Select how documents should be chunked',
	)
	
	col_a, col_b = st.columns( 2 )
	
	with col_a:
		chunk_size = st.number_input(
			'Chunk Size',
			min_value=100,
			max_value=5000,
			value=1000,
			step=100,
		)
	
	with col_b:
		overlap = st.number_input(
			'Overlap',
			min_value=0,
			max_value=2000,
			value=200,
			step=50,
		)
	
	col_run, col_reset = st.columns( 2 )
	
	run_chunking = col_run.button( "Chunk" )
	reset_chunking = col_reset.button( "Reset" )
	
	# ---------------------------
	# Actions
	# ---------------------------
	if reset_chunking:
		# No mutation of source docs here unless you already snapshot originals
		st.info( 'Chunking controls reset.' )
	
	if run_chunking:
		processor = TextParser( )
		
		if mode == 'chars':
			chunked_docs = processor.chunk_text( text=processed_text, size=chunk_size )
		
		elif mode == 'tokens':
			chunked_docs = word_tokenize( processed_text )
		else:
			st.error( f'Unsupported chunking mode: {mode}' )
			st.stop( )
		
		st.session_state.documents = chunked_docs
		
		st.success(
			f"Chunking complete: {len( chunked_docs )} chunks generated "
			f"(mode={mode}, size={chunk_size}, overlap={overlap})"
		)

# ==========================================================================================
# Tab 5 — Analysis & Statistics
# ==========================================================================================
with tabs[ 4 ]:
	st.header( "" )
	for key, default in SESSION_STATE_DEFAULTS.items( ):
		if key not in st.session_state:
			st.session_state[ key ] = default
	tokens = st.session_state.get( 'tokens' )

	if isinstance( tokens, list ) and tokens:
		processor = TextParser( )

		try:
			df_frequency = processor.create_frequency_distribution( tokens )
		except Exception as ex:
			st.error( f'Failed to compute frequency distribution: {ex}' )
			st.stop( )

		st.session_state.df_frequency = df_frequency
		st.dataframe( df_frequency, use_container_width=True )

	else:
		st.info( "Generate tokens first to view frequency statistics." )


# ==========================================================================================
# Tab 6 — Chunking
# ==========================================================================================
with tabs[ 5 ]:
	st.subheader( "" )
	for key, default in SESSION_STATE_DEFAULTS.items( ):
		if key not in st.session_state:
			st.session_state[ key ] = default

	docs = st.session_state.get( 'documents' )
	loader_name = st.session_state.get( 'active_loader' )

	if not docs:
		st.warning( 'No documents loaded. Please load documents first.' )
		st.stop( )

	if not loader_name:
		st.warning( 'No active loader found.' )
		st.stop( )

	# Defensive: CHUNKABLE_LOADERS may not exist
	try:
		chunk_modes = CHUNKABLE_LOADERS.get( loader_name )
	except Exception:
		st.error( 'Chunking configuration is missing (CHUNKABLE_LOADERS).' )
		st.stop( )

	if not chunk_modes:
		st.info( f'Chunking is not supported for loader: {loader_name}' )
		st.stop( )

	st.caption( f'Source Loader: {loader_name}' )

	# ---------------------------
	# Chunking Controls
	# ---------------------------
	mode = st.selectbox(
		'Chunking Mode',
		options=chunk_modes,
		help='Select how documents should be chunked',
	)

	col_a, col_b = st.columns( 2 )

	with col_a:
		chunk_size = st.number_input(
			'Chunk Size',
			min_value=100,
			max_value=5000,
			value=1000,
			step=100,
		)

	with col_b:
		overlap = st.number_input(
			'Overlap',
			min_value=0,
			max_value=2000,
			value=200,
			step=50,
		)

	col_run, col_reset = st.columns( 2 )

	run_chunking = col_run.button( 'Chunk' )
	reset_chunking = col_reset.button( 'Reset' )

	# ---------------------------
	# Actions
	# ---------------------------
	if reset_chunking:
		st.info( 'Chunking controls reset.' )

	if run_chunking:
		processor = TextParser( )

		try:
			if mode == 'chars':
				chunked_docs = processor.chunk_sentences( docs )

			elif mode == 'tokens':
				chunked_docs = processor.chunk_text( docs )

			else:
				st.error( f'Unsupported chunking mode: {mode}' )
				st.stop( )

		except Exception as ex:
			st.error( f'Chunking failed: {ex}' )
			st.stop( )

		if not isinstance( chunked_docs, list ) or not chunked_docs:
			st.error( 'Chunking produced no output.' )
			st.stop( )

		st.session_state.documents = chunked_docs

		st.success(
			f"Chunking complete: {len( chunked_docs )} chunks generated "
			f"(mode={mode}, size={chunk_size}, overlap={overlap})"
		)

# ======================================================================================
# Tab — SQLite Preview
# ======================================================================================
with tabs[ 7 ]:
	st.subheader( "" )
	for key, default in SESSION_STATE_DEFAULTS.items( ):
		if key not in st.session_state:
			st.session_state[ key ] = default
	
	sqlite_path = os.path.join( 'stores', 'sqlite', 'data.db' )
	
	if not os.path.exists( sqlite_path ):
		st.info( 'No SQLite database found. Load Excel data first.' )
	else:
		conn = sqlite3.connect( sqlite_path )
		
		# --------------------------------------------------
		# Tables
		# --------------------------------------------------
		df_tables = pd.read_sql(
			"SELECT name FROM sqlite_master WHERE type='table';",
			conn,
		)
		
		if df_tables.empty:
			st.info( 'Database exists but contains no tables.' )
			conn.close( )
		else:
			st.markdown( '### Tables' )
			st.dataframe( df_tables, use_container_width=True )
			
			table_name = st.selectbox(
				'Select table',
				df_tables[ 'name' ].tolist( ),
			)
			
			# --------------------------------------------------
			# Row Count
			# --------------------------------------------------
			df_count = pd.read_sql(
				f"SELECT COUNT(*) AS rows FROM {table_name};",
				conn,
			)
			st.metric( "Row Count", int( df_count.iloc[ 0 ][ "rows" ] ) )
			
			# --------------------------------------------------
			# Schema
			# --------------------------------------------------
			st.markdown( '### Schema' )
			df_schema = pd.read_sql(
				f'PRAGMA table_info({table_name});',
				conn,
			)
			st.dataframe(
				df_schema[ [ 'cid', 'name', 'type', 'notnull' ] ],
				use_container_width=True,
			)
			
			# --------------------------------------------------
			# Preview Data
			# --------------------------------------------------
			st.markdown( '### Preview (First 100 Rows)' )
			df_preview = pd.read_sql( f"SELECT * FROM {table_name} LIMIT 100;", conn )
			st.dataframe( df_preview, use_container_width=True )
			
			conn.close( )
