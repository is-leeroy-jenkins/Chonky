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
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_extras.chart_container import chart_container
from streamlit_extras.grid import grid
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
	ArXivLoader,
	XmlLoader
)

try:
	import textstat
	TEXTSTAT_AVAILABLE = True
except ImportError:
	TEXTSTAT_AVAILABLE = False
	
	
from nltk import sent_tokenize
from nltk.corpus import stopwords, wordnet, words
from nltk.tokenize import word_tokenize

REQUIRED_CORPORA = [
    'brown',
    'gutenberg',
    'reuters',
    'webtext',
    'inaugural',
    'state_union',
    'punkt',
    'stopwords',
]

for corpus in REQUIRED_CORPORA:
    try:
        nltk.data.find(f"corpora/{corpus}")
    except LookupError:
        nltk.download(corpus)
	    
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
		'active_table': None,
		# Chunking
		'chunks': None,
		'chunk_modes': None,
		"chunked_documents": None,
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
		'df_count': None,
		'df_chunks': None,
		# Data
		'data_connection': None
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
		st.session_state.chunks = None
		st.session_state.chunk_modes = None
		st.session_state.chunked_documents = None
		st.session_state.embeddings = None
		st.session_state.embedding_model = None
		st.session_state.active_table = None
		st.session_state.df_frequency = None
		st.session_state.df_tables = None
		st.session_state.df_schema = None
		st.session_state.df_preview = None
		st.session_state.df_count = None
		st.session_state.df_chunks = None

def metric_with_tooltip( label: str, value: str, tooltip: str ):
	"""
		Renders a metric with a hover tooltip using a two-column layout.
		Left column = the metric itself
		Right column = hoverable ℹ️ icon
	"""
	col_metric, col_info = st.columns( [ 0.5, 0.5 ] )
	
	with col_metric:
		st.metric( label, value )
	
	with col_info:
		if label not in [ 'Characters', 'Tokens', 'Unique Tokens', 'Avg Length' ]:
			st.markdown(
				f"""
	            <span style="
	                cursor: help;
	                font-size: 0.85rem;
	                color:#888;
	                vertical-align: super;
	            " title="{tooltip}">ℹ️ </span>
	            """,
				unsafe_allow_html=True,
			)

# ======================================================================================
# Page Configuration
# ======================================================================================
st.set_page_config( page_title='Chonky', layout='wide', page_icon=cfg.ICON )

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
			'Data View',
			'Tokens & Vocabulary',
			'Vectorization & Chunking',
			'Database'
	]
)

# ======================================================================================
# Tab 1 — Loaders
# ======================================================================================
with tabs[ 0 ]:
	# ------------------------------------------------------------------
	# Metrics container (single owner)
	# ------------------------------------------------------------------
	metrics_container = st.container( )
	def render_metrics_panel( ):
		raw_text = st.session_state.get( 'raw_text' )
		if not isinstance( raw_text, str ) or not raw_text.strip( ):
			st.info( 'Load data to view corpus metrics.' )
			return
		
		try:
			tokens = [ t.lower( ) for t in word_tokenize( raw_text ) if t.isalpha( ) ]
		except LookupError:
			st.error( 'NLTK resources missing.\n\n'
				'Run:\n'
				'`python -m nltk.downloader punkt stopwords`'
			)
			return
		
		if not tokens:
			st.warning( 'No valid alphabetic tokens found.' )
			return
		
		char_count = len( raw_text )
		token_count = len( tokens )
		vocab = set( tokens )
		vocab_size = len( vocab )
		counts = Counter( tokens )		
		hapax_count = sum( 1 for c in counts.values( ) if c == 1 )
		hapax_ratio = hapax_count / vocab_size if vocab_size else 0.0
		avg_word_len = sum( len( t ) for t in tokens ) / token_count
		ttr = vocab_size / token_count
		stopword_ratio = 0.0
		lexical_density = 0.0
		
		try:
			stop_words = set( stopwords.words( 'english' ) )
			stopword_ratio = sum( 1 for t in tokens if t in stop_words ) / token_count
			lexical_density = 1.0 - stopword_ratio
		except LookupError:
			pass
		
		# -------------------------------
		# Top Tokens
		# -------------------------------
		with st.expander( '🔤 Top Tokens', expanded=False ):
			top_tokens = counts.most_common( 10 )
			df = pd.DataFrame( top_tokens, columns=[ 'token', 'count' ] ).set_index( 'token' )
			st.area_chart( df, color='#01438A' )
			
		# -------------------------------
		# Corpus Metrics
		# -------------------------------
		with st.expander( '📊 Corpus Metrics', expanded=False ):
			# -----------------------------
			# Absolute Metrics (with tooltips)
			# -----------------------------
			col1, col2, col3, col4 = st.columns( 4, border=True )
			with col1:
				metric_with_tooltip( 'Characters', f'{char_count:,}',
					'Total number of characters in the raw text.', )
			with col2:
				metric_with_tooltip( 'Tokens', f'{token_count:,}',
					'Token Count: total number of tokenized words after cleanup.', )
			with col3:
				metric_with_tooltip( 'Unique Tokens', f'{vocab_size:,}',
					'Vocabulary Size: number of distinct word types in the text.', )
			with col4:
				metric_with_tooltip( 'TTR', f'{ttr:.3f}',
					'Type–Token Ratio: unique words ÷ total words', )
			col5, col6, col7, col8 = st.columns( 4, border=True )
			with col5:
				metric_with_tooltip( 'Hapax Ratio', f'{hapax_ratio:.3f}',
					'Hapax Ratio: proportion of words that occur only once (lexical rarity).'  )
			with col6:
				metric_with_tooltip( 'Avg Length', f'{avg_word_len:.2f}',
					'Average number of characters per token (after cleanup).', )
			with col7:
				metric_with_tooltip( 'Stopword Ratio', f'{stopword_ratio:.2%}',
					'Stopword Ratio: Percentage of words that provide little  semantic context', )
			with col8:
				metric_with_tooltip( 'Lexical Density', f'{lexical_density:.2%}',
					'Lexical Density: proportion of nouns, verbs, adjectives, adverbs', )
			
		# -------------------------------
		# Readability
		# -------------------------------
		with st.expander( '📖 Readability', expanded=False ):
			if TEXTSTAT_AVAILABLE:
				r1, r2, r3, r4 = st.columns( 4, border=True )
				with r1:
					metric_with_tooltip( 'Flesch Reading Ease',
						f'{textstat.flesch_reading_ease( raw_text ):.1f}',
						'Higher scores = easier to read. Based on sentence length and syllable count.', )
				with r2:
					metric_with_tooltip( 'Flesch–Kincaid Grade',
						f'{textstat.flesch_kincaid_grade( raw_text ):.1f}',
						'Estimated U.S. grade level needed to comprehend the text.', )
				with r3:
					metric_with_tooltip( 'Gunning Fog',
						f'{textstat.gunning_fog( raw_text ):.1f}',
						'Weighted average of words per sentence, and the number of long words per word',
					)
					
				with r4:
					metric_with_tooltip( 'Coleman-Liau Index',
						f'{textstat.coleman_liau_index( raw_text ):.1f}',
						'Average characters per 100 words and sentences per 100 words',
					)
			else:
				st.caption( 'Install `textstat` to enable readability metrics.' )
	
	# ------------------------------------------------------------------
	# SINGLE metrics 
	# ------------------------------------------------------------------
	with metrics_container:
		render_metrics_panel( )

	for key, default in SESSION_STATE_DEFAULTS.items( ):
		if key not in st.session_state:
			st.session_state[ key ] = default
	
	# ------------------------------------------------------------------
	# Left Layout 
	# ------------------------------------------------------------------
	left, right = st.columns( [ 1, 1.5 ] )
	with left:
		# --------------------------- Text Loader
		with st.expander( '📄 Text Loader', expanded=False ):
			files = st.file_uploader( 'Upload TXT files', type=[ 'txt' ],
				accept_multiple_files=True, key='txt_upload' )
			
			# ------------------------------------------------------------------
			# Buttons: Load / Clear / Save (same placement + interaction model)
			# ------------------------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			load_txt = col_load.button( 'Load', key='txt_load' )
			clear_txt = col_clear.button( 'Clear', key='txt_clear' )
			
			# Save is enabled only when THIS loader is active and raw_text exists
			can_save = ( st.session_state.get( 'active_loader' ) == 'TextLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( ) )
			
			if can_save:
				col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
					file_name='text_loader_output.txt', mime='text/plain', key='txt_save' )
			else:
				col_save.button( 'Save', key='txt_save_disabled', disabled=True )
			
			# ------------------------------------------------------------------
			# Clear (unchanged behavior)
			# ------------------------------------------------------------------
			if clear_txt:
				clear_if_active( 'TextLoader' )
				st.info( 'Text Loader state cleared.' )
			
			# ------------------------------------------------------------------
			# Load (unchanged behavior)
			# ------------------------------------------------------------------
			if load_txt and files:
				documents = [ ]
				for f in files:
					text = f.read( ).decode( 'utf-8', errors='ignore' )
					documents.append( Document( page_content=text,
							metadata={ 'source': f.name, 'loader': 'TextLoader' }, ) )
				
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = "\n\n".join( d.page_content for d in documents )
				st.session_state.processed_text = None
				st.session_state.active_loader = "TextLoader"
				
				st.success( f'Loaded {len( documents )} text document(s).' )
		
		# --------------------------- NLTK Loader
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
			
			st.markdown( '###### NLTK Corpora' )
			
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
					"NLTK corpus not found. Run:\n\n"
					"python -m nltk.downloader all\n\n"
					"or download individual corpora."
				)
			
			selected_files = st.multiselect(
				'Select files (leave empty to load all)',
				options=file_ids,
				key='nltk_file_ids',
			)
			
			st.divider( )
			
			st.markdown( '###### Local Corpus' )
			
			local_corpus_dir = st.text_input(
				'Local directory',
				placeholder='path/to/text/files',
				key='nltk_local_dir',
			)
			
			# ------------------------------------------------------------------
			# Load / Clear / Save controls
			# ------------------------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			load_nltk = col_load.button( 'Load', key='nltk_load' )
			clear_nltk = col_clear.button( 'Clear', key='nltk_clear' )
			_docs = st.session_state.get( 'documents' ) or [ ]
			_nltk_docs = [
					d for d in _docs
					if getattr( d, 'metadata', { } ).get( 'loader' ) == 'NLTKLoader'
			]
			_nltk_text = "\n\n".join( d.page_content for d in _nltk_docs ) if _nltk_docs else ""
			_export_name = f"nltk_{corpus_name.lower( ).replace( ' ', '_' )}.txt"
			
			col_save.download_button(
				'Save',
				data=_nltk_text or "",
				file_name=_export_name,
				mime='text/plain',
				disabled=not bool( _nltk_text.strip( ) ) )
			
			if clear_nltk and st.session_state.get( 'documents' ):
				st.session_state.documents = [
						d for d in st.session_state.documents
						if d.metadata.get( 'loader' ) != 'NLTKLoader' ]
				
				# 🔑 rebuild raw_text after clear
				st.session_state.raw_text = (
						"\n\n".join( d.page_content for d in st.session_state.documents )
						if st.session_state.documents else None
				)
				
				st.info( 'NLTKLoader documents removed.' )
			
			if load_nltk:
				documents = [ ]
				
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
							
							documents.append(
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
						if os.path.isfile( path ) and fname.lower( ).endswith( '.txt' ):
							with open( path, 'r', encoding='utf-8', errors='ignore' ) as f:
								text = f.read( )
							
							documents.append(
								Document(
									page_content=text,
									metadata={
											'loader': 'NLTKLoader',
											'source': path,
									},
								)
							)
				
				if documents:
					if st.session_state.documents:
						st.session_state.documents.extend( documents )
					else:
						st.session_state.documents = documents
						st.session_state.raw_documents = list( documents )
					
					# 🔑 rebuild raw_text after load / append
					st.session_state.raw_text = "\n\n".join(
						d.page_content for d in st.session_state.documents )
					
					st.session_state.active_loader = 'NLTKLoader'
					st.success( f'Loaded {len( documents )} document(s) from NLTK.' )
					st.rerun( )
		
		# --------------------------- CSV Loader
		with st.expander( "📑 CSV Loader", expanded=False ):
			csv_file = st.file_uploader(
				"Upload CSV",
				type=[ "csv" ],
				key="csv_upload",
			)
			
			delimiter = st.text_input(
				"Delimiter",
				value="\n\n",
				key="csv_delim",
			)
			
			quotechar = st.text_input(
				"Quote Character",
				value='"',
				key="csv_quote",
			)
			
			# --------------------------------------------------
			# Buttons: Load / Clear / Save (same row, same style)
			# --------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			
			load_csv = col_load.button(
				"Load",
				key="csv_load",
			)
			
			clear_csv = col_clear.button(
				"Clear",
				key="csv_clear",
			)
			
			# Save enabled only when CsvLoader is active and raw_text exists
			can_save = (
					st.session_state.get( "active_loader" ) == "CsvLoader"
					and isinstance( st.session_state.get( "raw_text" ), str )
					and st.session_state.get( "raw_text" ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					"Save",
					data=st.session_state.get( "raw_text" ),
					file_name="csv_loader_output.txt",
					mime="text/plain",
					key="csv_save",
				)
			else:
				col_save.button(
					"Save",
					key="csv_save_disabled",
					disabled=True,
				)
			
			# --------------------------------------------------
			# Clear
			# --------------------------------------------------
			if clear_csv:
				clear_if_active( "CsvLoader" )
				st.info( "CSV Loader state cleared." )
			
			# --------------------------------------------------
			# Load
			# --------------------------------------------------
			if load_csv and csv_file:
				with tempfile.TemporaryDirectory( ) as tmp:
					path = os.path.join( tmp, csv_file.name )
					with open( path, "wb" ) as f:
						f.write( csv_file.read( ) )
					
					loader = CsvLoader( )
					documents = loader.load( path, columns=None, delimiter=delimiter, quotechar=quotechar )
				
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = "\n\n".join( d.page_content for d in documents )
				st.session_state.processed_text = None
				st.session_state.active_loader = "CsvLoader"
				
				st.success( f"Loaded {len( documents )} CSV document(s)." )
		
		# -------------------------- XML Loader Expander
		with st.expander( '🧬 XML Loader', expanded=False ):
			
			# ------------------------------------------------------------------
			# Session-backed loader instance
			# ------------------------------------------------------------------
			if 'xml_loader' not in st.session_state:
				st.session_state.xml_loader = XmlLoader( )
			
			loader  = st.session_state.xml_loader
			xml_file = st.file_uploader( label='Select XML file', type=[ 'xml' ],
				accept_multiple_files=False, key='xml_file_uploader' )
			st.subheader( 'Semantic XML Loading (Unstructured)' )
			
			col1, col2 = st.columns( 2 )
			
			with col1:
				chunk_size = st.number_input( 'Chunk Size', min_value=100,
					max_value=5000, value=1000, step=100 )
			
			with col2:
				overlap_amount = st.number_input( 'Chunk Overlap', min_value=0, max_value=1000,
					value=200, step=50 )
			
			if st.button( 'Load XML (Semantic)', use_container_width=True ):
				if xml_file is None:
					st.warning( 'Please select an XML file.' )
				else:
					with st.spinner( 'Loading XML via UnstructuredXMLLoader...' ):
						documents = loader.load( xml_file.name )
						if documents:
							st.success( f'Loaded {len( documents )} semantic document elements.' )
							st.session_state[ 'xml_documents' ] = documents
			
			if st.button( 'Split Semantic Documents', use_container_width=True ):
				with st.spinner( 'Splitting documents...' ):
					split_docs = loader.split(
						size=int( chunk_size ),
						amount=int( overlap_amount )
					)
					if split_docs:
						st.success( f'Produced {len( split_docs )} document chunks.' )
						st.session_state[ 'xml_split_documents' ] = split_docs
			
			# ------------------------------------------------------------------
			# Structured XML Tree Loading
			# ------------------------------------------------------------------
			st.divider( )
			st.subheader( "Structured XML Tree Loading (XPath)" )
			
			if st.button( "Load XML Tree", use_container_width=True ):
				if xml_file is None:
					st.warning( "Please select an XML file." )
				else:
					with st.spinner( "Parsing XML into ElementTree..." ):
						tree = loader.load_tree( xml_file.name )
						if tree is not None:
							st.success( "XML tree loaded successfully." )
							st.session_state[ "xml_tree_loaded" ] = True
							st.session_state[ "xml_namespaces" ] = loader.xml_namespaces
			
			# ------------------------------------------------------------------
			# XPath Query Interface
			# ------------------------------------------------------------------
			if loader.xml_root is not None:
				st.markdown( "**XPath Query**" )
				
				xpath_expr = st.text_input(
					"XPath Expression",
					value="//*",
					help="Use namespace prefixes if applicable."
				)
				
				if st.button( "Run XPath Query", use_container_width=True ):
					with st.spinner( "Executing XPath..." ):
						elements = loader.get_elements( xpath_expr )
						if elements is not None:
							st.success( f"Returned {len( elements )} elements." )
							st.session_state[ "xml_xpath_results" ] = elements
				
				# Preview results
				if "xml_xpath_results" in st.session_state:
					preview_count = min( 10, len( st.session_state[ "xml_xpath_results" ] ) )
					st.caption( f"Previewing first {preview_count} elements" )
					
					for el in st.session_state[ "xml_xpath_results" ][ :preview_count ]:
						st.code(
							etree.tostring( el, pretty_print=True, encoding="unicode" ),
							language="xml"
						)
			
			# ------------------------------------------------------------------
			# Debug / Introspection
			# ------------------------------------------------------------------
			with st.expander( "ℹ Loader State" ):
				st.json( {
						"file_path": loader.file_path,
						"documents_loaded": loader.documents is not None,
						"xml_tree_loaded": loader.xml_tree is not None,
						"namespaces": loader.xml_namespaces,
						"chunk_size": loader.chunk_size,
						"overlap_amount": loader.overlap_amount,
				} )
		
		# --------------------------- PDF Loader
		with st.expander( '📕 PDF Loader', expanded=False ):
			pdf = st.file_uploader(
				'Upload PDF',
				type=[ 'pdf' ],
				key='pdf_upload',
			)
			
			mode = st.selectbox(
				'Mode',
				[ 'single',
				  'elements' ],
				key='pdf_mode',
			)
			
			extract = st.selectbox(
				'Extract',
				[ 'plain',
				  'ocr' ],
				key='pdf_extract',
			)
			
			include = st.checkbox(
				'Include Images',
				value=False,
				key='pdf_include',
			)
			
			fmt = st.selectbox(
				'Format',
				[ 'markdown-img',
				  'text' ],
				key='pdf_fmt',
			)
			
			# --------------------------------------------------
			# Buttons: Load / Clear / Save (same row, same style)
			# --------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			load_pdf = col_load.button( 'Load', key='pdf_load', )
			clear_pdf = col_clear.button( 'Clear', key='pdf_clear', )
			
			# Save enabled only when PdfLoader is active and raw_text exists
			can_save = ( st.session_state.get( 'active_loader' ) == 'PdfLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( ) )
			
			if can_save:
				col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
					file_name='pdf_loader_output.txt', mime='text/plain', key='pdf_save', )
			else:
				col_save.button( 'Save', key='pdf_save_disabled', disabled=True, )
			
			# --------------------------------------------------
			# Clear
			# --------------------------------------------------
			if clear_pdf:
				clear_if_active( 'PdfLoader' )
				st.info( 'PDF Loader state cleared.' )
			
			# --------------------------------------------------
			# Load
			# --------------------------------------------------
			if load_pdf and pdf:
				with tempfile.TemporaryDirectory( ) as tmp:
					path = os.path.join( tmp, pdf.name )
					with open( path, 'wb' ) as f:
						f.write( pdf.read( ) )
					loader = PdfLoader( )
					documents = loader.load( path, mode=mode, extract=extract, format=fmt, )
				
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents )
				st.session_state.processed_text = None
				st.session_state.active_loader = 'PdfLoader'
				
				st.success( f'Loaded {len( documents )} PDF document(s).' )
		
		# --------------------------- Markdown Loader
		with st.expander( '🧾 Markdown Loader', expanded=False ):
			md = st.file_uploader(
				'Upload Markdown',
				type=[ 'md',
				       'markdown' ],
				key='md_upload',
			)
			
			# --------------------------------------------------
			# Buttons: Load / Clear / Save (same row, same style)
			# --------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			
			load_md = col_load.button(
				'Load',
				key='md_load',
			)
			
			clear_md = col_clear.button(
				'Clear',
				key='md_clear',
			)
			
			# Save enabled only when MarkdownLoader is active and raw_text exists
			can_save = (
					st.session_state.get( 'active_loader' ) == 'MarkdownLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					'Save',
					data=st.session_state.get( 'raw_text' ),
					file_name='markdown_loader_output.txt',
					mime='text/plain',
					key='md_save',
				)
			else:
				col_save.button(
					'Save',
					key='md_save_disabled',
					disabled=True,
				)
			
			# --------------------------------------------------
			# Clear (UNCHANGED behavior)
			# --------------------------------------------------
			if clear_md:
				clear_if_active( 'MarkdownLoader' )
				st.info( "Markdown Loader state cleared." )
			
			# --------------------------------------------------
			# Load (UNCHANGED behavior)
			# --------------------------------------------------
			if load_md and md:
				with tempfile.TemporaryDirectory( ) as tmp:
					path = os.path.join( tmp, md.name )
					with open( path, "wb" ) as f:
						f.write( md.read( ) )
					
					loader = MarkdownLoader( )
					documents = loader.load( path )
				
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = "\n\n".join(
					d.page_content for d in documents
				)
				st.session_state.processed_text = None
				st.session_state.active_loader = "MarkdownLoader"
				
				st.success(
					f"Loaded {len( documents )} Markdown document(s)."
				)
		
		# --------------------------- HTML Loader
		with st.expander( '🌐 HTML Loader', expanded=False ):
			html = st.file_uploader(
				'Upload HTML',
				type=[ 'html',
				       'htm' ],
				key='html_upload',
			)
			
			# --------------------------------------------------
			# Buttons: Load / Clear / Save (same row, same style)
			# --------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			
			load_html = col_load.button(
				'Load',
				key='html_load',
			)
			
			clear_html = col_clear.button(
				'Clear',
				key='html_clear',
			)
			
			# Save enabled only when HtmlLoader is active and raw_text exists
			can_save = (
					st.session_state.get( 'active_loader' ) == 'HtmlLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					'Save',
					data=st.session_state.get( 'raw_text' ),
					file_name='html_loader_output.txt',
					mime='text/plain',
					key='html_save',
				)
			else:
				col_save.button(
					'Save',
					key='html_save_disabled',
					disabled=True,
				)
			
			# --------------------------------------------------
			# Clear (UNCHANGED behavior)
			# --------------------------------------------------
			if clear_html:
				clear_if_active( "HtmlLoader" )
				st.info( "HTML Loader state cleared." )
			
			# --------------------------------------------------
			# Load (UNCHANGED behavior)
			# --------------------------------------------------
			if load_html and html:
				with tempfile.TemporaryDirectory( ) as tmp:
					path = os.path.join( tmp, html.name )
					with open( path, "wb" ) as f:
						f.write( html.read( ) )
					
					loader = HtmlLoader( )
					documents = loader.load( path )
				
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = "\n\n".join(
					d.page_content for d in documents
				)
				st.session_state.processed_text = None
				st.session_state.active_loader = "HtmlLoader"
				
				st.success(
					f"Loaded {len( documents )} HTML document(s)."
				)
		
		# --------------------------- JSON Loader
		with st.expander( '🧩 JSON Loader', expanded=False ):
			js = st.file_uploader(
				'Upload JSON',
				type=[ 'json' ],
				key='json_upload',
			)
			
			is_lines = st.checkbox(
				'JSON Lines',
				value=False,
				key='json_lines',
			)
			
			# --------------------------------------------------
			# Buttons: Load / Clear / Save (same row, same style)
			# --------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			
			load_json = col_load.button(
				'Load',
				key='json_load',
			)
			
			clear_json = col_clear.button(
				'Clear',
				key='json_clear',
			)
			
			# Save enabled only when JsonLoader is active and raw_text exists
			can_save = (
					st.session_state.get( 'active_loader' ) == 'JsonLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					'Save',
					data=st.session_state.get( 'raw_text' ),
					file_name='json_loader_output.txt',
					mime='text/plain',
					key='json_save',
				)
			else:
				col_save.button(
					'Save',
					key='json_save_disabled',
					disabled=True,
				)
			
			# --------------------------------------------------
			# Clear (UNCHANGED behavior)
			# --------------------------------------------------
			if clear_json:
				clear_if_active( 'JsonLoader' )
				st.info( 'JSON Loader state cleared.' )
			
			# --------------------------------------------------
			# Load (UNCHANGED behavior)
			# --------------------------------------------------
			if load_json and js:
				with tempfile.TemporaryDirectory( ) as tmp:
					path = os.path.join( tmp, js.name )
					with open( path, 'wb' ) as f:
						f.write( js.read( ) )
					
					loader = JsonLoader( )
					documents = loader.load(
						path,
						is_text=True,
						is_lines=is_lines,
					)
				
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = "\n\n".join(
					d.page_content for d in documents
				)
				st.session_state.processed_text = None
				st.session_state.active_loader = "JsonLoader"
				
				st.success(
					f"Loaded {len( documents )} JSON document(s)."
				)
		
		# --------------------------- PowerPoint Loader
		with st.expander( '📽 Power Point Loader', expanded=False ):
			pptx = st.file_uploader(
				'Upload PPTX',
				type=[ 'pptx' ],
				key='pptx_upload',
			)
			
			mode = st.selectbox(
				'Mode',
				[ 'single',
				  'multiple' ],
				key='pptx_mode',
			)
			
			# --------------------------------------------------
			# Buttons: Load / Clear / Save (same row, same style)
			# --------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			
			load_pptx = col_load.button(
				'Load',
				key='pptx_load',
			)
			
			clear_pptx = col_clear.button(
				'Clear',
				key='pptx_clear',
			)
			
			# Save enabled only when PowerPointLoader is active and raw_text exists
			can_save = (
					st.session_state.get( 'active_loader' ) == 'PowerPointLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					'Save',
					data=st.session_state.get( 'raw_text' ),
					file_name='powerpoint_loader_output.txt',
					mime='text/plain',
					key='pptx_save',
				)
			else:
				col_save.button(
					'Save',
					key='pptx_save_disabled',
					disabled=True,
				)
			
			# --------------------------------------------------
			# Clear (UNCHANGED behavior)
			# --------------------------------------------------
			if clear_pptx:
				clear_if_active( 'PowerPointLoader' )
				st.info( 'PowerPoint Loader state cleared.' )
			
			# --------------------------------------------------
			# Load (UNCHANGED behavior)
			# --------------------------------------------------
			if load_pptx and pptx:
				with tempfile.TemporaryDirectory( ) as tmp:
					path = os.path.join( tmp, pptx.name )
					with open( path, "wb" ) as f:
						f.write( pptx.read( ) )
					
					loader = PowerPointLoader( )
					documents = (
							loader.load( path )
							if mode == "single"
							else loader.load_multiple( path )
					)
				
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = "\n\n".join(
					d.page_content for d in documents
				)
				st.session_state.processed_text = None
				st.session_state.active_loader = "PowerPointLoader"
				
				st.success(
					f"Loaded {len( documents )} PowerPoint document(s)."
				)
		
		# --------------------------- Excel Loader
		with st.expander( '📊 Excel Loader', expanded=False ):
			excel_file = st.file_uploader(
				'Upload Excel file',
				type=[ 'xlsx',
				       'xls' ],
				key='excel_upload',
			)
			
			sheet_name = st.text_input(
				'Sheet name (leave blank for all sheets)',
				key='excel_sheet',
			)
			
			table_prefix = st.text_input(
				'SQLite table prefix',
				value='excel',
				help='Each sheet will be written as <prefix>_<sheetname>',
				key='excel_table_prefix',
			)
			
			# --------------------------------------------------
			# Buttons: Load / Clear / Save (same row, same style)
			# --------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			
			load_excel = col_load.button(
				'Load',
				key='excel_load',
			)
			
			clear_excel = col_clear.button(
				'Clear',
				key='excel_clear',
			)
			
			# Save enabled only when ExcelLoader is active and raw_text exists
			can_save = (
					st.session_state.get( 'active_loader' ) == 'ExcelLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					'Save',
					data=st.session_state.get( 'raw_text' ),
					file_name='excel_loader_output.txt',
					mime='text/plain',
					key='excel_save',
				)
			else:
				col_save.button(
					'Save',
					key='excel_save_disabled',
					disabled=True,
				)
			
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
			# Load + SQLite ingestion (UNCHANGED behavior)
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
								sheet_name: pd.read_excel(
									excel_path,
									sheet_name=sheet_name,
								)
						}
					else:
						dfs = pd.read_excel(
							excel_path,
							sheet_name=None,
						)
				
				# Open SQLite connection
				conn = sqlite3.connect( sqlite_path )
				
				documents = [ ]
				
				for sheet, df in dfs.items( ):
					if df.empty:
						continue
					
					# Normalize table name
					table_name = f"{table_prefix}_{sheet}".replace(
						" ", "_"
					).lower( )
					
					# Write DataFrame to SQLite
					df.to_sql(
						table_name,
						conn,
						if_exists="replace",
						index=False,
					)
					
					# Convert DataFrame to text for NLP pipeline
					text = df.to_csv( index=False )
					
					documents.append(
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
				
				if documents:
					if st.session_state.documents:
						st.session_state.documents.extend( documents )
					else:
						st.session_state.documents = documents
						st.session_state.raw_documents = list( documents )
						st.session_state.raw_text = "\n\n".join(
							d.page_content for d in documents
						)
					
					st.session_state.processed_text = None
					st.session_state.active_loader = "ExcelLoader"
					
					st.success(
						f"Loaded {len( documents )} sheet(s) and stored in SQLite."
					)
				else:
					st.warning(
						"No data loaded (empty sheets or invalid selection)."
					)
		
		# --------------------------- arXiv Loader
		with st.expander( '🧠 ArXiv Loader', expanded=False ):
			arxiv_query = st.text_input(
				'Query',
				placeholder='e.g., transformer OR llm',
				key='arxiv_query',
			)
			
			arxiv_max_chars = st.number_input(
				'Max characters per document',
				min_value=250,
				max_value=100000,
				value=1000,
				step=250,
				key='arxiv_max_chars',
				help='Maximum characters read',
			)
			
			# --------------------------------------------------
			# Buttons: Fetch / Clear / Save (same row, same style)
			# --------------------------------------------------
			col_fetch, col_clear, col_save = st.columns( 3 )
			
			arxiv_fetch = col_fetch.button(
				'Fetch',
				key='arxiv_fetch',
			)
			
			arxiv_clear = col_clear.button(
				'Clear',
				key='arxiv_clear',
			)
			
			# Save enabled only when ArXivLoader is active and raw_text exists
			can_save = (
					st.session_state.get( 'active_loader' ) == 'ArXivLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					'Save',
					data=st.session_state.get( 'raw_text' ),
					file_name='arxiv_loader_output.txt',
					mime='text/plain',
					key='arxiv_save',
				)
			else:
				col_save.button(
					"Save",
					key="arxiv_save_disabled",
					disabled=True,
				)
			
			# --------------------------------------------------
			# Clear logic (remove only ArXivLoader documents)
			# --------------------------------------------------
			if arxiv_clear and st.session_state.get( 'documents' ):
				st.session_state.documents = [
						d for d in st.session_state.documents
						if d.metadata.get( 'loader' ) != 'ArXivLoader'
				]
				st.info( 'ArXivLoader documents removed.' )
			
			# --------------------------------------------------
			# Fetch (APPEND semantics preserved)
			# --------------------------------------------------
			if arxiv_fetch and arxiv_query:
				loader = ArXivLoader( )
				documents = loader.load(
					arxiv_query,
					max_chars=int( arxiv_max_chars ),
				) or [ ]
				
				for d in documents:
					d.metadata[ 'loader' ] = 'ArXivLoader'
					d.metadata[ 'source' ] = arxiv_query
				
				if documents:
					if st.session_state.documents:
						st.session_state.documents.extend( documents )
					else:
						st.session_state.documents = documents
						# Baseline snapshot only on first corpus load
						st.session_state.raw_documents = list( documents )
						st.session_state.raw_text = '\n\n'.join(
							x.page_content for x in documents
						)
					
					st.session_state.processed_text = None
					st.session_state.active_loader = 'ArXivLoader'
					
					st.success(
						f'Fetched {len( documents )} arXiv document(s).'
					)
		
		# --------------------------- Wikipedia Loader
		with st.expander( '📚 Wikipedia Loader', expanded=False ):
			wiki_query = st.text_input(
				'Query',
				placeholder='e.g., Natural language processing',
				key='wiki_query',
			)
			
			wiki_max_docs = st.number_input(
				'Max documents',
				min_value=1,
				max_value=250,
				value=25,
				step=1,
				key='wiki_max_docs',
				help='Maximum number of documents loaded',
			)
			
			wiki_max_chars = st.number_input(
				'Max characters per document',
				min_value=250,
				max_value=100000,
				value=4000,
				step=250,
				key='wiki_max_chars',
				help='Upper limit on the number of characters',
			)
			
			# --------------------------------------------------
			# Buttons: Fetch / Clear / Save (same row, same style)
			# --------------------------------------------------
			col_fetch, col_clear, col_save = st.columns( 3 )
			
			wiki_fetch = col_fetch.button(
				'Fetch',
				key='wiki_fetch',
			)
			
			wiki_clear = col_clear.button(
				'Clear',
				key='wiki_clear',
			)
			
			# Save enabled only when WikiLoader is active and raw_text exists
			can_save = (
					st.session_state.get( 'active_loader' ) == 'WikiLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					'Save',
					data=st.session_state.get( 'raw_text' ),
					file_name='wiki_loader_output.txt',
					mime='text/plain',
					key='wiki_save',
				)
			else:
				col_save.button(
					'Save',
					key='wiki_save_disabled',
					disabled=True,
				)
			
			# --------------------------------------------------
			# Clear logic (remove only WikiLoader documents)
			# --------------------------------------------------
			if wiki_clear and st.session_state.get( "documents" ):
				st.session_state.documents = [
						d for d in st.session_state.documents
						if d.metadata.get( "loader" ) != "WikiLoader"
				]
				st.info( "WikiLoader documents removed." )
			
			# --------------------------------------------------
			# Fetch (APPEND semantics preserved)
			# --------------------------------------------------
			if wiki_fetch and wiki_query:
				loader = WikiLoader( )
				documents = loader.load(
					wiki_query,
					max_docs=int( wiki_max_docs ),
					max_chars=int( wiki_max_chars ),
				) or [ ]
				
				for d in documents:
					d.metadata[ "loader" ] = "WikiLoader"
					d.metadata[ "source" ] = wiki_query
				
				if documents:
					if st.session_state.documents:
						st.session_state.documents.extend( documents )
					else:
						st.session_state.documents = documents
						# Baseline snapshot only on first corpus load
						st.session_state.raw_documents = list( documents )
						st.session_state.raw_text = "\n\n".join(
							x.page_content for x in documents
						)
					
					st.session_state.processed_text = None
					st.session_state.active_loader = "WikiLoader"
					
					st.success(
						f"Fetched {len( documents )} Wikipedia document(s)."
					)
		
		# --------------------------- GitHub Loader
		with st.expander( '🐙 GitHub Loader', expanded=False ):
			gh_url = st.text_input(
				'GitHub API URL',
				placeholder="https://api.github.com",
				value="https://api.github.com",
				key='gh_url',
				help='web url to a github repository',
			)
			
			gh_repo = st.text_input(
				'Repo (owner/name)',
				placeholder='openai/openai-python',
				key='gh_repo',
				help='Name of the repository',
			)
			
			gh_branch = st.text_input(
				'Branch',
				placeholder='main',
				value='main',
				key='gh_branch',
				help='The branch of the repository',
			)
			
			gh_filetype = st.text_input(
				'File type filter',
				value='.md',
				key='gh_filetype',
				help='Filtering by file type. Example: .py, .md, .txt',
			)
			
			# --------------------------------------------------
			# Buttons: Fetch / Clear / Save (same row, same style)
			# --------------------------------------------------
			col_fetch, col_clear, col_save = st.columns( 3 )
			
			gh_fetch = col_fetch.button(
				'Fetch',
				key='gh_fetch',
			)
			
			gh_clear = col_clear.button(
				'Clear',
				key='gh_clear',
			)
			
			# Save enabled only when GithubLoader is active and raw_text exists
			can_save = (
					st.session_state.get( 'active_loader' ) == 'GithubLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					'Save',
					data=st.session_state.get( 'raw_text' ),
					file_name='github_loader_output.txt',
					mime='text/plain',
					key='gh_save',
				)
			else:
				col_save.button(
					'Save',
					key='gh_save_disabled',
					disabled=True,
				)
			
			# --------------------------------------------------
			# Clear logic (remove only GithubLoader documents)
			# --------------------------------------------------
			if gh_clear and st.session_state.get( 'documents' ):
				st.session_state.documents = [
						d for d in st.session_state.documents
						if d.metadata.get( 'loader' ) != 'GithubLoader'
				]
				st.info( 'GithubLoader documents removed.' )
			
			# --------------------------------------------------
			# Fetch (APPEND semantics preserved)
			# --------------------------------------------------
			if gh_fetch and gh_repo and gh_branch:
				loader = GithubLoader( )
				documents = loader.load(
					gh_url,
					gh_repo,
					gh_branch,
					gh_filetype,
				) or [ ]
				
				for d in documents:
					d.metadata[ 'loader' ] = 'GithubLoader'
					d.metadata[ 'source' ] = f'{gh_repo}@{gh_branch}'
				
				if documents:
					if st.session_state.documents:
						st.session_state.documents.extend( documents )
					else:
						st.session_state.documents = documents
						# Baseline snapshot only on first corpus load
						st.session_state.raw_documents = list( documents )
						st.session_state.raw_text = '\n\n'.join(
							x.page_content for x in documents
						)
					
					st.session_state.processed_text = None
					st.session_state.active_loader = 'GithubLoader'
					
					st.success(
						f'Fetched {len( documents )} GitHub document(s).'
					)
		
		# --------------------------- Web Loader
		with st.expander( '🔗 Web Loader', expanded=False ):
			urls = st.text_area(
				'Enter one URL per line',
				placeholder="https://example.com\nhttps://another.com",
				key='web_urls',
			)
			
			# --------------------------------------------------
			# Buttons: Fetch / Clear / Save (same row, same style)
			# --------------------------------------------------
			col_fetch, col_clear, col_save = st.columns( 3 )
			
			load_web = col_fetch.button(
				'Fetch',
				key='web_fetch',
			)
			
			clear_web = col_clear.button(
				'Clear',
				key='web_clear',
			)
			
			# Save enabled only when WebLoader is active and raw_text exists
			can_save = (
					st.session_state.get( 'active_loader' ) == 'WebLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					'Save',
					data=st.session_state.get( 'raw_text' ),
					file_name='web_loader_output.txt',
					mime='text/plain',
					key='web_save',
				)
			else:
				col_save.button(
					'Save',
					key='web_save_disabled',
					disabled=True,
				)
			
			# --------------------------------------------------
			# Clear logic (remove only WebLoader documents)
			# --------------------------------------------------
			if clear_web and st.session_state.get( 'documents' ):
				st.session_state.documents = [
						d for d in st.session_state.documents
						if d.metadata.get( 'loader' ) != 'WebLoader'
				]
				st.info( 'WebLoader documents removed.' )
			
			# --------------------------------------------------
			# Fetch (APPEND semantics preserved)
			# --------------------------------------------------
			if load_web and urls.strip( ):
				loader = WebLoader( recursive=False )
				new_docs = [ ]
				
				for url in [ u.strip( ) for u in urls.splitlines( )
						if u.strip( ) ]:
					documents = loader.load( url )
					for d in documents:
						d.metadata[ 'loader' ] = 'WebLoader'
						d.metadata[ 'source' ] = url
					new_docs.extend( documents )
				
				if new_docs:
					if st.session_state.documents:
						st.session_state.documents.extend( new_docs )
					else:
						st.session_state.documents = new_docs
						st.session_state.raw_documents = list( new_docs )
						st.session_state.raw_text = '\n\n'.join(
							d.page_content for d in new_docs
						)
					
					st.session_state.processed_text = None
					st.session_state.active_loader = 'WebLoader'
					
					st.success( f'Fetched {len( new_docs )} web document(s).' )
		
		# --------------------------- Web Crawler
		with st.expander( '🕷️ Web Crawler', expanded=False ):
			start_url = st.text_input(
				'Start URL',
				placeholder="https://example.com",
				key="crawl_start_url",
			)
			
			max_depth = st.number_input(
				'Max crawl depth',
				min_value=1,
				max_value=5,
				value=2,
				step=1,
				key='crawl_depth',
			)
			
			stay_on_domain = st.checkbox(
				'Stay on starting domain',
				value=True,
				key='crawl_domain_lock',
			)
			
			# --------------------------------------------------
			# Buttons: Crawl / Clear / Save (same row, same style)
			# --------------------------------------------------
			col_run, col_clear, col_save = st.columns( 3 )
			
			run_crawl = col_run.button(
				'Crawl',
				key='crawl_run',
			)
			
			clear_crawl = col_clear.button(
				'Clear',
				key='crawl_clear',
			)
			
			# Save enabled only when WebCrawler is active and raw_text exists
			can_save = (
					st.session_state.get( 'active_loader' ) == 'WebCrawler'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					'Save',
					data=st.session_state.get( 'raw_text' ),
					file_name='web_crawler_output.txt',
					mime="text/plain",
					key="crawl_save",
				)
			else:
				col_save.button(
					'Save',
					key='crawl_save_disabled',
					disabled=True,
				)
			
			# --------------------------------------------------
			# Clear logic (remove only WebCrawler documents)
			# --------------------------------------------------
			if clear_crawl and st.session_state.get( 'documents' ):
				st.session_state.documents = [
						d for d in st.session_state.documents
						if d.metadata.get( 'loader' ) != 'WebCrawler'
				]
				st.info( 'Web crawler documents removed.' )
			
			# --------------------------------------------------
			# Crawl (APPEND semantics preserved)
			# --------------------------------------------------
			if run_crawl and start_url:
				loader = WebLoader(
					recursive=True,
					max_depth=max_depth,
					prevent_outside=stay_on_domain,
				)
				
				documents = loader.load( start_url )
				
				for d in documents:
					d.metadata[ 'loader' ] = 'WebCrawler'
					d.metadata[ 'source' ] = start_url
				
				if st.session_state.documents:
					st.session_state.documents.extend( documents )
				else:
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents )
				
				st.session_state.processed_text = None
				st.session_state.active_loader = 'WebCrawler'
				
				st.success( f'Crawled {len( documents )} document(s).' )
	
	# ------------------------------------------------------------------
	# RIGHT COLUMN — Document Preview
	# ------------------------------------------------------------------
	with right:
		documents = st.session_state.documents
		if not documents:
			st.info( 'No documents loaded.' )
		else:
			st.caption( f'Active Loader: {st.session_state.active_loader}' )
			st.write( f'Documents: {len( documents )}' )
			for i, d in enumerate( documents[ :5 ] ):
				with st.expander( f'Document {i + 1}', expanded=True ):
					st.json( d.metadata )
					st.text_area( 'Content', d.page_content[ :5000 ],
						height=500, key=f'preview_doc_{i}' )

# ======================================================================================
# Tab — Processing / Parsing
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
	left, right = st.columns( [ 1, 1.5 ], border=True )
	with left:
		active = st.session_state.get( 'active_loader' )

		# ==============================================================
		# Common Text Processing (TextParser)
		# ==============================================================
		with st.expander( '🧠 Text Processing', expanded=True ):
			remove_html = st.checkbox( 'Remove HTML',
				help='Removes Hypertext Markup Tags, eg. <, \>, etc' )
			remove_markdown = st.checkbox( 'Remove Markdown',
				help=r'Removes symobls used in .md files #, ##, ###, -, etc' )
			remove_symbols = st.checkbox( 'Remove Symbols',
				help=r'Removes @, #, $, ^, *, =, |, \, <, >, ~' )
			remove_numbers = st.checkbox( 'Remove Numbers',
				help='Removes numeric digits 0 thour 9' )
			remove_xml = st.checkbox( 'Remove XML',
				help=r'Removes xml tags ( ex. <xml> & <\xml> )' )
			remove_punctuation = st.checkbox( 'Remove Punctuation',
				help=r'Removes @, #, $, ^, *, =, |, \, <, >, ~ but preserves sentence delimiters' )
			remove_images = st.checkbox( 'Remove Images',
				help=r'Remove image from text, including Markdown, HTML <img> tags, and  image URLs' )
			remove_stopwords = st.checkbox( 'Remove Stopwords',
				help=r'Removes common words (e.g., "the", "is", "and", etc.)' )
			remove_numerals = st.checkbox( 'Remove Numerals',
				help='Removes roman numbers I, II, IV, XI, etc' )
			remove_encodings = st.checkbox( 'Remove Encoding',
				help=r'Removes encoding artifacts and over-encoded byte strings' )
			normalize_text = st.checkbox( 'Normalize (lowercase)' )
			lemmatize_text = st.checkbox( 'Lemmatize',
				help='Reduces words to their base or dictionary form' )
			remove_fragments = st.checkbox( 'Remove Fragments',
				help='Removes words less than 3 characters in length' )
			remove_errors = st.checkbox( 'Remove Errors',
				help='Removes misspelled words' )
			collapse_whitespace = st.checkbox( 'Collapse Whitespace',
				help='Removes extra lines' )
			compress_whitespace = st.checkbox( 'Compress Whitespace',
				help='Removes extra spaces' )

		# ==============================================================
		# Word-Specific Processing (WordParser)
		# ==============================================================
		extract_tables = extract_paragraphs = False
		with st.expander( '📄 Word Processing', expanded=False ):
			if active == 'WordLoader':
				extract_tables = st.checkbox( 'Extract Tables' )
				extract_paragraphs = st.checkbox( 'Extract Paragraphs' )
			else:
				st.caption( 'Available when Word documents are loaded.' )

		# ==============================================================
		# PDF-Specific Processing (PdfParser)
		# ==============================================================
		remove_headers = join_hyphenated = False
		with st.expander( '📕 PDF Processing', expanded=False ):
			if active == 'PdfLoader':
				remove_headers = st.checkbox( 'Remove Headers/Footers' )
				join_hyphenated = st.checkbox( 'Join Hyphenated Lines' )
			else:
				st.caption( 'Available when PDF documents are loaded.' )

		# ==============================================================
		# HTML-Specific Processing (Structural)
		# ==============================================================
		strip_scripts = keep_headings = keep_paragraphs = keep_tables = False
		with st.expander( '🌐 HTML Processing', expanded=False ):
			if active == 'HtmlLoader':
				strip_scripts = st.checkbox( 'Strip <script> / <style>' )
				keep_headings = st.checkbox( 'Keep Headings' )
				keep_paragraphs = st.checkbox( 'Keep Paragraphs' )
				keep_tables = st.checkbox( 'Keep Tables' )
			else:
				st.caption( 'Available when HTML documents are loaded.' )
		
		st.divider( )
		
		# ==============================================================
		# Actions (Apply / Reset / Clear / Save)
		# ==============================================================
		col_apply, col_reset, col_clear, col_save = st.columns( 4 )
		apply_processing = col_apply.button( 'Apply', disabled=not has_text, )
		reset_processing = col_reset.button( 'Reset', disabled=not has_text, )
		clear_processing = col_clear.button( 'Clear', disabled=not has_text, )
		can_save_processed = ( isinstance( st.session_state.get( 'processed_text' ), str )
				and st.session_state.get( 'processed_text' ).strip( ) )
		
		if can_save_processed:
			col_save.download_button( 'Save', data=st.session_state.processed_text,
				file_name='processed_text.txt', mime='text/plain', key='processed_text_save' )
		else:
			col_save.button( 'Save', key='processed_text_save_disabled', disabled=True )

		# ==============================================================
		# Buttons Events
		# ==============================================================
		if reset_processing:
			st.session_state.processed_text = ''
			st.session_state.processed_text_view = ''
			st.success( 'Processed text reset to raw text.' )
		
		if clear_processing:
			st.session_state.processed_text = ""
			st.session_state.processed_text_view = ""
			st.success( 'Processed text cleared.' )

		if apply_processing:
			processed_text = raw_text
			tp = TextParser( )
			# 1 — Structural cleanup
			if remove_html:
				processed_text = tp.remove_html( processed_text )
			if remove_markdown:
				processed_text = tp.remove_markdown( processed_text )
			if remove_images:
				processed_text = tp.remove_images( processed_text )
			if remove_encodings:
				processed_text = tp.remove_encodings( processed_text )
			if remove_xml:
				processed_text = tp.remove_xml( processed_text )
			# 2 — Noise / non-lexical characters
			if remove_symbols:
				processed_text = tp.remove_symbols( processed_text )
			if remove_numbers:
				processed_text = tp.remove_numbers( processed_text )
			if remove_numerals:
				processed_text = tp.remove_numerals( processed_text )
			# 3 — Meaning-critical punctuation shaping
			if remove_punctuation:
				processed_text = tp.remove_punctuation( processed_text )
			# 4 — Word normalization
			if normalize_text:
				processed_text = tp.normalize_text( processed_text )
			# 5 — Lexical refinement
			if remove_stopwords:
				processed_text = tp.remove_stopwords( processed_text )
			if remove_fragments:
				processed_text = tp.remove_fragments( processed_text )
			if remove_errors:
				processed_text = tp.remove_errors( processed_text )
			# 6 — Lemmatization
			if lemmatize_text:
				processed_text = tp.lemmatize_text( processed_text )
			# 7 — Whitespace cleanup
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
		st.text_area( 'Raw Text', st.session_state.raw_text or 'No text loaded yet.',
			height=200, disabled=True, key='raw_text_view' )
		
		with st.expander( '📊 Processing Statistics:', expanded=False ):
			raw = st.session_state.get( 'raw_text' )
			processed = st.session_state.get( 'processed_text' )
			if ( isinstance( raw, str ) and raw.strip( ) 
					and isinstance( processed, str ) and processed.strip( )):
				raw_tokens = raw.split( )
				proc_tokens = processed.split( )
				raw_chars = len( raw )
				proc_chars = len( processed )
				raw_vocab = len( set( raw_tokens ) )
				proc_vocab = len( set( proc_tokens ) )
				
				# ----------------------------
				# Absolute Metrics
				# ----------------------------
				st.text( 'Measures:' )
				ttr = (proc_vocab / len( proc_tokens ) if proc_tokens else 0.0 )
				a1, a2, a3, a4 = st.columns( 4, border=True )
				a1.metric( 'Characters', f'{proc_chars:,}' )
				a2.metric( 'Tokens', f'{len( proc_tokens ):,}' )
				a3.metric( 'Unique Tokens', f'{proc_vocab:,}' )
				a4.metric( 'TTR', f'{ttr:.3f}' )
				
				st.divider( )
				
				# ----------------------------
				# Delta Metrics
				# ----------------------------
				st.text( 'Deltas:' )
				d1, d2, d3, d4 = st.columns( 4, border=True )
				char_delta = proc_chars - raw_chars
				token_delta = len( proc_tokens ) - len( raw_tokens )
				vocab_delta = proc_vocab - raw_vocab
				compression = ( proc_chars / raw_chars if raw_chars > 0 else 0.0 )
				d1.metric( 'Δ Characters', f'{char_delta:+,}' )
				d2.metric( 'Δ Tokens', f'{token_delta:+,}' )
				d3.metric( 'Δ Vocabulary', f'{vocab_delta:+,}')
				d4.metric( 'Compression Ratio', f'{compression:.2%}' )
			else:
				st.caption( 'Load and process text to view absolute and delta statistics.' )
				
		# ----------------------------
		# Processed Text (output)
		# ----------------------------
		st.text_area( "Processed Text", st.session_state.processed_text or "", height=700,
			key="processed_text_view" )

# ==========================================================================================
# Tab  — Data View
# ==========================================================================================
with tabs[ 2 ]:
	st.header( "" )
	for key, default in SESSION_STATE_DEFAULTS.items( ):
		if key not in st.session_state:
			st.session_state[ key ] = default
	line_col, chunk_col = st.columns( [ 0.5, 0.5 ], border=True, vertical_alignment='center' )
	df_frequency = st.session_state.get( 'df_frequency' )
	dr_tables = st.session_state.get( 'df_tables' )
	df_count = st.session_state.get( 'df_count' )
	df_schema = st.session_state.get( 'df_schema' )
	df_preview = st.session_state.get( 'df_preview' )
	df_chunks = st.session_state.get( 'df_chunks' )
	embedding_model = st.session_state.get( 'embedding_model' )
	embeddings = st.session_state.get( 'embeddings' )
	active_table = st.session_state.get( 'active_table' )
	
	with line_col:
		st.caption( '' )
		if st.session_state.processed_text:
			processor = TextParser( )
			view = st.selectbox( label='', options=[ 'Lines', 'Paragraphs', 'Pages' ] )
			if view == 'Lines':
				lines = processor.split_sentences( text=processed_text, size=15 )
				st.dataframe( pd.DataFrame( lines, columns=[ 'Line' ] ) )
			elif view == 'Paragraphs':
				paragraphs = processor.split_sentences( text=processed_text, size=40 )
				st.dataframe( pd.DataFrame( paragraphs, columns=[ 'Paragraph' ] ) )
			elif view == 'Pages':
				sentences = processor.split_sentences( text=processed_text, size=250 )
				st.dataframe( pd.DataFrame( sentences, columns=[ 'Sentence' ] ) )
		else:
			st.info( 'Run preprocessing first' )
			
	with chunk_col:
		st.markdown( '##### Vector Space' )
		st.text( f'Token Vectors: {len( lines ) }')
		if st.session_state.processed_text:
			processor = TextParser( )
			if view == 'Lines':
				dimensions = [ 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7',
				               'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15' ]
				
				lines = processor.split_sentences( text=processed_text, size=15 )
				_chunks = [ l.split( ' ' )  for l in lines ]
				df_chunks = pd.DataFrame( _chunks, columns=dimensions )
				st.dataframe( df_chunks )
		else:
			st.info( 'Run preprocessing first' )

# ==========================================================================================
# Tab - Tokens, Vocabulary
# ==========================================================================================
with tabs[ 3 ]:
    st.header("")
    for key, default in SESSION_STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[ key ] = default

    if st.session_state.processed_text:
        processor = TextParser( )

        # --------------------------------------------------
        # Tokenization & Vocabulary
        # --------------------------------------------------
        tokens = word_tokenize( processed_text )
        vocab = processor.create_vocabulary( tokens )

        # --------------------------------------------------
        # Frequency Distribution
        # --------------------------------------------------
        df_frequency = processor.create_frequency_distribution( tokens )
        st.session_state.df_frequency = df_frequency

        # --------------------------------------------------
        # Three-column layout
        # --------------------------------------------------
        col_tokens, col_vocab, col_freq = st.columns( [ 1, 1, 2 ], border=True,
	        vertical_alignment='center'  )

        # -----------------------
        # Column 1 — Tokens
        # -----------------------
        with col_tokens:
            st.write(f"Tokens: {len(tokens)}")
            st.dataframe( pd.DataFrame(tokens, columns=["Token"]), use_container_width=True )

        # -----------------------
        # Column 2 — Vocabulary
        # -----------------------
        with col_vocab:
            st.write(f"Vocabulary: {len(vocab)}")
            st.dataframe(
                pd.DataFrame(vocab, columns=["Word"]),
                use_container_width=True,
            )
        
        # -----------------------
        # Column 3 — Frequency Histogram
        # -----------------------
        with col_freq:
	        st.markdown( "#### Frequency Distribution" )
	        st.caption( 'Top 100 most frequent tokens')
	        if df_frequency is not None and not df_frequency.empty:
		        # Identify numeric frequency column
		        numeric_cols = df_frequency.select_dtypes( include="number" )
		        
		        if not numeric_cols.empty:
			        freq_col = numeric_cols.columns[ 0 ]
			        
			        # Use top-N most frequent tokens for readability
			        top_n = 100
			        df_top = ( df_frequency.sort_values( freq_col, ascending=False ).head( top_n ) )
			        st.bar_chart(  df_top.set_index( df_top.columns[ 0 ] )[ freq_col ],
				        use_container_width=True, color='#01438A' )
		        else:
			        st.info( 'No numeric frequency column available for charting.' )
	        else:
		        st.info( 'Frequency distribution unavailable.' )
        
        # --------------------------------------------------
        # Persist state
        # --------------------------------------------------
        st.session_state.tokens = tokens
        st.session_state.vocabulary = vocab

    else:
        st.info('Run preprocessing first')

# ==========================================================================================
# Tab  — 🧩 Vectorization & Chunking
# ==========================================================================================
with tabs[ 4 ]:
	st.subheader( "" )
	for key, default in SESSION_STATE_DEFAULTS.items( ):
		if key not in st.session_state:
			st.session_state[ key ] = default
	
	documents = st.session_state.get( 'documents' )
	chunks = st.session_state.get( 'chunks' )
	chunk_modes = st.session_state.get( 'chunk_modes' )
	chunked_documents = st.session_state.get( 'chunked_documents' )
	data_connection = st.session_state.get( 'data_connection' )
	loader_name = st.session_state.get( 'active_loader' )
	
	if  documents is None:
		st.warning( 'No documents loaded. Please load documents first.' )
	elif loader_name is None:
		st.warning( 'No active loader found.' )
	else:
		chunk_modes = CHUNKABLE_LOADERS.get( loader_name )
		
	if chunk_modes is None:
		st.info( f'Chunking is not supported for loader: {loader_name}' )
  
	st.caption( f'Source Loader: {loader_name}' )
	
	# ---------------------------
	# Chunking Controls
	# ---------------------------
	mode = st.selectbox( 'Chunking Mode', options=chunk_modes, help='Select how documents are chunked' )
	
	col_a, col_b = st.columns( 2 )
	
	with col_a:
		chunk_size = st.number_input( 'Chunk Size', min_value=100, max_value=5000, value=1000,
			step=100, key='chunk_size' )
	
	with col_b:
		overlap = st.number_input( 'Overlap', min_value=0, max_value=2000,
			value=200, step=50, key='overlap' )
	
	col_run, col_reset = st.columns( 2 )
	run_chunking = col_run.button( 'Chunk', key='chunk_button' )
	reset_chunking = col_reset.button( 'Reset', key='reset_button' )
	
	# ---------------------------
	# Actions
	# ---------------------------
	if reset_chunking:
		# No mutation of source documents here unless you already snapshot originals
		st.info( 'Chunking controls reset.' )
	
	if run_chunking:
		processor = TextParser( )
		
		if mode == 'chars':
			chunked_documents = processor.chunk_text( text=processed_text, size=chunk_size )
		
		elif mode == 'tokens':
			chunked_documents = word_tokenize( processed_text )
		else:
			st.error( f'Unsupported chunking mode: {mode}' )
		
		st.session_state.chunked_documents = chunked_documents
		
		st.success(
			f'Chunking complete: {len( chunked_documents )} chunks generated '
			f'(mode={mode}, size={chunk_size}, overlap={overlap})' )

# ======================================================================================
# Tab — Database
# ======================================================================================
with tabs[ 5 ]:

    import os
    import sqlite3
    import pandas as pd
    from streamlit_extras.grid import grid  # <-- REQUIRED

    st.subheader("Data")

    sqlite_path = os.path.join("stores", "sqlite", "data.db")

    def q(identifier: str) -> str:
        escaped = identifier.replace('"', '""')
        return f'"{escaped}"'

    if not os.path.exists(sqlite_path):
        st.info("No SQLite database found. Load data first.")
    else:
        conn = sqlite3.connect(sqlite_path)
        try:
            df_tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;",
                conn,
            )

            if df_tables.empty:
                st.info("Database exists but contains no tables.")
            else:
                table_names = df_tables["name"].tolist()

                if "active_table" not in st.session_state:
                    st.session_state.active_table = table_names[0]

                if st.session_state.active_table not in table_names:
                    st.session_state.active_table = table_names[0]

                data_grid = grid(
                    2,
                    [2, 3, 1],
                    1,
                    1,
                    vertical_align="top",
                )

                table_name = data_grid.selectbox(
                    "Select Table",
                    table_names,
                    index=table_names.index(st.session_state.active_table),
                )
                st.session_state.active_table = table_name

                df_count = pd.read_sql(
                    f"SELECT COUNT(*) AS rows FROM {q(table_name)};",
                    conn,
                )
                data_grid.metric("Row Count", int(df_count.iloc[0]["rows"]))

                df_preview = pd.read_sql(
                    f"SELECT * FROM {q(table_name)} LIMIT 250;",
                    conn,
                )
                data_grid.dataframe(df_preview, use_container_width=True)

                numeric_cols = df_preview.select_dtypes(include="number")
                if not numeric_cols.empty:
                    data_grid.line_chart(numeric_cols, use_container_width=True)
                else:
                    data_grid.write("No numeric columns available for charting.")

                df_schema = pd.read_sql(
                    f"PRAGMA table_info({q(table_name)});",
                    conn,
                )
                data_grid.dataframe(
                    df_schema[["cid", "name", "type", "notnull", "pk"]],
                    use_container_width=True,
                )

                data_grid.button("Refresh", use_container_width=True)
                data_grid.button("Export CSV", use_container_width=True)

                with data_grid.expander("Show Filters", expanded=False):
                    st.multiselect(
                        "Columns",
                        df_preview.columns.tolist(),
                        default=df_preview.columns.tolist(),
                    )
                    st.slider(
                        "Max Rows",
                        min_value=10,
                        max_value=1000,
                        value=250,
                        step=10,
                    )
        finally:
            conn.close()

