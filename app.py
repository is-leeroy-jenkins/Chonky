'''
  ******************************************************************************************
      Assembly:                Chonky
      Filename:                app.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="app.py" company="Terry D. Eppler">

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
import sys
import types
import altair
import base64
from collections import Counter
from lxml import etree

import config as cfg
import collections
import functools
import itertools
import json
import sqlite3
import math
import nltk
import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import re
import sqlite3
import sqlite_vec
import sys
import statistics
import streamlit as st
import time
import tempfile
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings.sentence_transformer import (
	SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import SQLiteVec
from processors import Processor, TextParser, WordParser, PdfParser

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

from embedders import GPT, Grok, Gemini

try:
	import textstat
	
	TEXTSTAT_AVAILABLE = True
except Exception:
	TEXTSTAT_AVAILABLE = False
	textstat = None

from nltk import sent_tokenize
from nltk.corpus import stopwords, wordnet, words
from nltk.tokenize import word_tokenize


# ======================================================================================
# Session State Initialization
# ======================================================================================

if 'openai_api_key' not in st.session_state:
	st.session_state[ 'openai_api_key' ] = ''

if 'gemini_api_key' not in st.session_state:
	st.session_state[ 'gemini_api_key' ] = ''

if 'groq_api_key' not in st.session_state:
	st.session_state[ 'groq_api_key' ] = ''

if 'google_api_key' not in st.session_state:
	st.session_state[ 'google_api_key' ] = ''

if 'pinecone_api_key' not in st.session_state:
	st.session_state[ 'pinecone_api_key' ] = ''

if 'google_application_credentials' not in st.session_state:
	st.session_state[ 'google_application_credentials' ] = ''

for key, default in cfg.SESSION_STATE_DEFAULTS.items( ):
	if key not in st.session_state:
		st.session_state[ key ] = default

for corpus in cfg.REQUIRED_CORPORA:
	try:
		nltk.data.find( f'corpora/{corpus}' )
	except LookupError:
		nltk.download( corpus )

def style_subheaders( ) -> None:
	"""
	
		Purpose:
		_________
		Sets the style of subheaders in the main UI
		
	"""
	st.markdown(
		"""
		<style>
		div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stMarkdownContainer"] h3,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h3 {
			color: rgb(0, 120, 252) !important;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)

def encode_image_base64( path: str ) -> str:
	data = Path( path ).read_bytes( )
	return base64.b64encode( data ).decode( "utf-8" )

def clear_if_active( loader_name: str ) -> None:
	if st.session_state.active_loader == loader_name:
		st.session_state.documents = None
		st.session_state.active_loader = None
		st.session_state.tokens = None
		st.session_state.vocabulary = None
		st.session_state.token_counts = None
		st.session_state.chunks = None
		st.session_state.chunk_modes = None
		st.session_state.chunked_documents = None
		st.session_state.embeddings = None
		st.session_state.active_table = None
		st.session_state.df_frequency = None
		st.session_state.df_tables = None
		st.session_state.df_schema = None
		st.session_state.df_preview = None
		st.session_state.df_count = None
		st.session_state.df_chunks = None
		st.session_state.lines = None

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

def normalize_embeddings( emb_array: np.ndarray ) -> np.ndarray:
	if isinstance( emb_array, np.ndarray ) and emb_array.ndim == 1:
		return emb_array.reshape( 1, -1 )
	return emb_array
	
def rebuild_raw_text_from_documents( ) -> str | None:
		docs = st.session_state.get( "documents" ) or [ ]
		if not docs:
			return None
		text = "\n\n".join( d.page_content for d in docs
		                    if hasattr( d, "page_content" )
		                    and isinstance( d.page_content, str )
		                    and d.page_content.strip( ) )
		return text if text.strip( ) else None

# ======================================================================================
# Page Configuration
# ======================================================================================
st.set_page_config( page_title='Chonky', layout='wide',
	page_icon=cfg.ICON, initial_sidebar_state='collapsed' )

# ======================================================================================
# Headers/Title
# ======================================================================================
st.logo( cfg.LOGO, size='large' )

# ======================================================================================
# Sidebar — API Key Configuration
# ======================================================================================
style_subheaders( )
with st.sidebar:
	st.text( 'Settings' )
	st.divider( )
	
	if st.session_state.openai_api_key == '':
		default = cfg.OPENAI_API_KEY
	
	if default:
		st.session_state.openai_api_key = default
	
	if st.session_state.gemini_api_key == '':
		default = cfg.GEMINI_API_KEY
	
	if default:
		st.session_state.gemini_api_key = default
	
	if st.session_state.groq_api_key == '':
		default = cfg.GROQ_API_KEY
	
	if default:
		st.session_state.groq_api_key = default
	
	if st.session_state.google_api_key == '':
		default = cfg.GOOGLE_API_KEY
	
	if default:
		st.session_state.google_api_key = default
	
	if st.session_state.pinecone_api_key == '':
		default = cfg.PINECONE_API_KEY
	
	if default:
		st.session_state.pinecone_api_key = default
	
	if st.session_state.google_application_credentials == '':
		default = cfg.GOOGLE_APPLICATION_CREDENTIALS
	
	if default:
		st.session_state.google_application_credentials = default
	
	with st.expander( "🔐 API Keys", expanded=False ):
		# --- OpenAI ---
		openai_key = st.text_input( 'OpenAI API Key',
			value=st.session_state.openai_api_key, type='password' )
		
		# --- Groq ---
		groq_key = st.text_input( 'Groq API Key',
			value=st.session_state.groq_api_key, type='password' )
		
		# --- Google API ---
		google_key = st.text_input( 'Google API Key',
			value=st.session_state.google_api_key, type='password' )
		
		# --- Google Application Credentials ---
		google_creds_path = st.text_input( 'Google Application Credentials (JSON Path)',
			value=st.session_state.google_application_credentials, type='password' )
		
		# --- Pinecone ---
		pinecone_key = st.text_input( 'Pinecone API Key (future)',
			value=st.session_state.pinecone_api_key, type='password' )

# ======================================================================================
# Tabs
# ======================================================================================
tabs = st.tabs( cfg.TABS )

# ======================================================================================
# Tab - Document Loading
# ======================================================================================
with tabs[ 0 ]:
	tokens = st.session_state[ 'tokens' ]
	documents = st.session_state[ 'documents' ]
	raw_text = st.session_state[ 'raw_text' ]
	
	# ------------------------------------------------------------------
	# LEFT COLUMN - LOADERS
	# ------------------------------------------------------------------
	left, right = st.columns( [ 1, 1.5 ] )
	with left:
		_loader_msg = st.session_state.pop( '_loader_status', None )
		if isinstance( _loader_msg, str ) and _loader_msg.strip( ):
			st.success( _loader_msg )
		
		# --------------------------- Text Loader
		with st.expander( label='Text Loader', icon='📝', expanded=False ):
			files = st.file_uploader( 'Upload Text File(s)', type=[ 'txt', 'text', 'log' ],
				accept_multiple_files=True, key='txt_upload' )
			
			# ------------------------------------------------------------------
			# Buttons: Load / Clear / Save
			# ------------------------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			load_txt = col_load.button( 'Load', key='txt_load' )
			clear_txt = col_clear.button( 'Clear', key='txt_clear' )
			can_save = ( st.session_state.get( 'active_loader' ) == 'TextLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( ) )
			
			if can_save:
				col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
					file_name='text_loader_output.txt', mime='text/plain', key='txt_save' )
			else:
				col_save.button( 'Save', key='txt_save_disabled', disabled=True )
			
			# ------------------------------------------------------------------
			# Clear
			# ------------------------------------------------------------------
			if clear_txt:
				clear_if_active( 'TextLoader' )
				st.info( 'Text Loader state cleared.' )
				st.rerun( )
			
			# ------------------------------------------------------------------
			# Load
			# ------------------------------------------------------------------
			if load_txt and files:
				documents: list[ Document ] = [ ]
				
				with tempfile.TemporaryDirectory( ) as tmp:
					for uploaded_file in files:
						path = os.path.join( tmp, uploaded_file.name )
						
						with open( path, 'wb' ) as handle:
							handle.write( uploaded_file.read( ) )
						
						loader = TextLoader( )
						loaded = loader.load( path ) or [ ]
						for document in loaded:
							if not isinstance( getattr( document, 'metadata', None ), dict ):
								document.metadata = { }
							
							document.metadata[ 'loader' ] = 'TextLoader'
							document.metadata.setdefault( 'source', uploaded_file.name )
						
						documents.extend( loaded )
				
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = "\n\n".join( d.page_content for d in documents
					if hasattr( d, 'page_content' ) and isinstance( d.page_content, str )
					                                     and d.page_content.strip( ) )
				st.session_state.active_loader = 'TextLoader'
				st.success( f'Loaded {len( documents )} text document(s).' )
		
		# --------------------------- NLTK Loader Expander
		with st.expander( label='Corpora Loader', icon='📚', expanded=False ):
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
			
			corpus_name = st.selectbox( 'Select corpus',
				[ 'Brown', 'Gutenberg', 'Reuters', 'WebText', 'Inaugural', 'State of the Union', ],
				key='nltk_corpus_name', )
			
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
			
			selected_files = st.multiselect( 'Select files (leave empty to load all)',
				options=file_ids, key='nltk_file_ids', )
			
			st.divider( )
			
			st.markdown( '###### Local Corpus' )
			
			local_corpus_dir = st.text_input( 'Local directory', placeholder='path/to/text/files',
				key='nltk_local_dir', )
			
			# ------------------------------------------------------------------
			# Load / Clear / Save controls
			# ------------------------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			load_nltk = col_load.button( 'Load', key='nltk_load' )
			clear_nltk = col_clear.button( 'Clear', key='nltk_clear' )
			
			_docs = st.session_state.get( 'documents' ) or [ ]
			_nltk_docs = [ d for d in _docs if d.metadata.get( 'loader' ) == 'NLTKLoader' ]
			_nltk_text = "\n\n".join( d.page_content for d in _nltk_docs )
			_export_name = f"nltk_{corpus_name.lower( ).replace( ' ', '_' )}.txt"
			
			col_save.download_button(
				'Save',
				data=_nltk_text,
				file_name=_export_name,
				mime='text/plain',
				disabled=not bool( _nltk_text.strip( ) ),
			)
			
			# ------------------------------------------------------------------
			# Clear
			# ------------------------------------------------------------------
			if clear_nltk and st.session_state.get( 'documents' ):
				st.session_state.documents = [ d for d in st.session_state.documents
				                               if d.metadata.get( 'loader' ) != 'NLTKLoader'
				                               ]
				
				st.session_state.raw_text = (
						"\n\n".join( d.page_content for d in st.session_state.documents )
						if st.session_state.documents else None)
				
				st.session_state.active_loader = None
				
				st.info( 'NLTKLoader documents removed.' )
			
			# ------------------------------------------------------------------
			# Load
			# ------------------------------------------------------------------
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
							
							if text.strip( ):
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
							
							if text.strip( ):
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
					if st.session_state.get( 'documents' ):
						st.session_state.documents.extend( documents )
					else:
						st.session_state.documents = documents
						st.session_state.raw_documents = list( documents )
					
					st.session_state.raw_text = "\n\n".join(
						d.page_content for d in st.session_state.documents
					)
					
					st.session_state.processed_text = None
					st.session_state.active_loader = 'NLTKLoader'
					
					st.success( f'Loaded {len( documents )} document(s) from NLTK.' )
				else:
					st.warning( 'No documents were loaded.' )
		
		# --------------------------- CSV Loader Expander
		with st.expander( label="CSV Loader", icon='📑', expanded=False ):
			csv_file = st.file_uploader(
				"Upload CSV",
				type=[ "csv" ],
				key="csv_upload",
			)
			
			delimiter = st.text_input(
				"Delimiter",
				value=",",
				key="csv_delim",
			)
			
			quotechar = st.text_input(
				"Quote Character",
				value='"',
				key="csv_quote",
			)
			
			# --------------------------------------------------
			# Buttons: Load / Clear / Save
			# --------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			load_csv = col_load.button( 'Load', key='csv_load' )
			clear_csv = col_clear.button( 'Clear', key='csv_clear' )
			
			can_save = (
					st.session_state.get( 'active_loader' ) == 'CsvLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					'Save',
					data=st.session_state.get( 'raw_text' ),
					file_name='csv_loader_output.txt',
					mime='text/plain',
					key='csv_save',
				)
			else:
				col_save.button( 'Save', key='csv_save_disabled', disabled=True )
			
			# --------------------------------------------------
			# Clear
			# --------------------------------------------------
			if clear_csv:
				clear_if_active( "CsvLoader" )
				st.session_state.raw_text = rebuild_raw_text_from_documents( )
				st.session_state[ "_loader_status" ] = "CSV Loader state cleared."
			
			# --------------------------------------------------
			# Load
			# --------------------------------------------------
			if load_csv and csv_file:
				with tempfile.TemporaryDirectory( ) as tmp:
					path = os.path.join( tmp, csv_file.name )
					with open( path, "wb" ) as f:
						f.write( csv_file.read( ) )
					
					loader = CsvLoader( )
					documents = loader.load(
						path,
						columns=None,
						delimiter=delimiter,
						quotechar=quotechar,
					) or [ ]
				
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = "\n\n".join(
					d.page_content for d in documents
					if hasattr( d, "page_content" )
					and isinstance( d.page_content, str )
					and d.page_content.strip( )
				)
				st.session_state.processed_text = None
				st.session_state.active_loader = "CsvLoader"
				
				st.session_state[ "_loader_status" ] = f"Loaded {len( documents )} CSV document(s)."
		
		# -------------------------- XML Loader Expander
		with st.expander( label='XML Loader', icon='🧬' ,expanded=False ):
			# ------------------------------------------------------------------
			# Session-backed loader instance
			# ------------------------------------------------------------------
			if 'xml_loader' not in st.session_state or st.session_state.xml_loader is None:
				st.session_state.xml_loader = XmlLoader( )
			
			loader = st.session_state.xml_loader
			
			xml_file = st.file_uploader( label='Select XML file', type=[ 'xml' ],
				accept_multiple_files=False, key='xml_file_uploader' )
			
			st.subheader( 'Semantic XML Loading (Unstructured)' )
			
			col1, col2 = st.columns( 2 )
			
			with col1:
				chunk_size = st.number_input( 'Chunk Size', min_value=100, max_value=5000,
					value=1000, step=100 )
			
			with col2:
				overlap_amount = st.number_input( 'Chunk Overlap', min_value=0, max_value=1000,
					value=200, step=50 )
			
			# --------------------------------------------------
			# Semantic Load
			# --------------------------------------------------
			if st.button( 'Load XML (Semantic)', use_container_width=True ):
				if xml_file is None:
					st.warning( 'Please select an XML file.' )
				else:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, xml_file.name )
						with open( path, 'wb' ) as f:
							f.write( xml_file.read( ) )
						
						with st.spinner( 'Loading XML via UnstructuredXMLLoader...' ):
							documents = loader.load( path )
					
					if documents:
						raw_text = '\n\n'.join(
							d.page_content
							for d in documents
							if hasattr( d, 'page_content' )
							and isinstance( d.page_content, str )
							and d.page_content.strip( )
						)
						
						st.session_state.documents = documents
						st.session_state.raw_documents = list( documents )
						st.session_state.raw_text = raw_text
						st.session_state.processed_text = None
						st.session_state.active_loader = 'XmlLoader'
						st.session_state[ 'xml_documents' ] = documents
						st.session_state[ 'xml_tree_loaded' ] = False
						st.session_state[ 'xml_xpath_results' ] = None
						st.session_state[ 'xml_namespaces' ] = None
					else:
						st.warning( 'No extractable text found in XML.' )
			
			# --------------------------------------------------
			# Split Semantic Documents
			# --------------------------------------------------
			if st.button( 'Split Semantic Documents', use_container_width=True ):
				with st.spinner( 'Splitting documents...' ):
					split_docs = loader.split( size=int( chunk_size ), amount=int( overlap_amount ))
				
				if split_docs:
					st.session_state[ 'xml_split_documents' ] = split_docs
					st.success( f'Produced {len( split_docs )} document chunks.' )
			
			# ------------------------------------------------------------------
			# Structured XML Tree Loading
			# ------------------------------------------------------------------
			st.divider( )
			st.subheader( 'Structured XML Tree Loading (XPath)' )
			
			if st.button( 'Load XML Tree', use_container_width=True ):
				if xml_file is None:
					st.warning( 'Please select an XML file.' )
				else:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, xml_file.name )
						with open( path, 'wb' ) as f:
							f.write( xml_file.read( ) )
						
						with st.spinner( 'Parsing XML into ElementTree...' ):
							tree = loader.load_tree( path )
					
					if tree is not None:
						xml_text = etree.tostring( tree, pretty_print=True, encoding='unicode' )
						
						st.session_state.raw_text = xml_text
						st.session_state.processed_text = None
						st.session_state.active_loader = 'XmlLoader'
						st.session_state[ 'xml_tree_loaded' ] = True
						st.session_state[ 'xml_namespaces' ] = loader.xml_namespaces
						st.session_state[ 'xml_xpath_results' ] = None
						
						st.success( 'XML tree loaded successfully.' )
					else:
						st.warning( 'Failed to parse XML tree.' )
			
			# ------------------------------------------------------------------
			# XPath Query Interface
			# ------------------------------------------------------------------
			xml_loader = st.session_state.get( 'xml_loader' )
			
			if xml_loader is None:
				st.info( 'No loader initialized.' )
			elif not hasattr( xml_loader, 'xml_root' ):
				st.info( 'XML loader does not support XML tree operations.' )
			elif xml_loader.xml_root is None:
				st.info( 'XML loader initialized but no XML tree loaded.' )
			else:
				st.markdown( '**XPath Query**' )
				
				xpath_expr = st.text_input( 'XPath Expression', value='//*',
					help='Use namespace prefixes if applicable.' )
				
				if st.button( 'Run XPath Query', use_container_width=True ):
					with st.spinner( 'Executing XPath...' ):
						elements = xml_loader.get_elements( xpath_expr )
					
					if elements is not None:
						st.session_state[ 'xml_xpath_results' ] = elements
						st.success( f'Returned {len( elements )} elements.' )
				
				if 'xml_xpath_results' in st.session_state and \
						st.session_state[ 'xml_xpath_results' ] is not None:
					preview_count = min( 10, len( st.session_state[ 'xml_xpath_results' ] ) )
					
					st.caption( f'Previewing first {preview_count} elements' )
					
					for el in st.session_state[ 'xml_xpath_results' ][ :preview_count ]:
						st.code( etree.tostring( el, pretty_print=True, encoding='unicode' ),
							language='xml' )
			
			# ------------------------------------------------------------------
			# Debug / Introspection
			# ------------------------------------------------------------------
			with st.expander( "ℹ Loader State" ):
				xml_loader = st.session_state.get( 'xml_loader' )
				
				if xml_loader is None:
					st.info( "No loader initialized." )
				else:
					st.json(
						{
								"file_path": getattr( xml_loader, 'file_path', None ),
								"documents_loaded": getattr( xml_loader, 'documents', None ) is not None,
								"xml_tree_loaded": getattr( xml_loader, 'xml_tree', None ) is not None,
								"namespaces": getattr( xml_loader, 'xml_namespaces', None ),
								"chunk_size": getattr( xml_loader, 'chunk_size', None ),
								"overlap_amount": getattr( xml_loader, 'overlap_amount', None ),
						}
					)
		
		# --------------------------- PDF Loader Expander
		with st.expander( label='PDF Loader', icon='📕', expanded=False ):
			pdf = st.file_uploader( 'Upload PDF', type=[ 'pdf' ], key='pdf_upload' )
			
			mode = st.selectbox( 'Mode', [ 'single', 'page' ], key='pdf_mode',
				help='Use "single" for one combined document or "page" for page-wise loading.' )
			
			extract = st.selectbox( 'Extract', [ 'plain', 'layout' ], key='pdf_extract',
				help='Use "plain" for standard extraction or "layout" for layout-aware extraction.')
			
			include = st.checkbox( 'Include Images', value=False, key='pdf_include' )
			
			fmt = st.selectbox( 'Format', [ 'markdown-img', 'html-img', 'text-img' ], key='pdf_fmt',
				help='Inner representation to use when image extraction is enabled.' )
			
			# --------------------------------------------------
			# Buttons: Load / Clear / Save
			# --------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			
			load_pdf = col_load.button( 'Load', key='pdf_load' )
			clear_pdf = col_clear.button( 'Clear', key='pdf_clear' )
			
			can_save = ( st.session_state.get( 'active_loader' ) == 'PdfLoader' 
			             and isinstance( st.session_state.get( 'raw_text' ), str ) 
			             and st.session_state.get( 'raw_text' ).strip( ) )
			
			if can_save:
				col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
					file_name='pdf_loader_output.txt', mime='text/plain', key='pdf_save', )
			else:
				col_save.button( 'Save', key='pdf_save_disabled', disabled=True )
			
			# --------------------------------------------------
			# Clear
			# --------------------------------------------------
			if clear_pdf:
				clear_if_active( 'PdfLoader' )
				st.session_state[ '_loader_status' ] = 'Loader state cleared.'
				st.rerun( )
			
			# --------------------------------------------------
			# Load
			# --------------------------------------------------
			if load_pdf and pdf:
				with tempfile.TemporaryDirectory( ) as tmp:
					path = os.path.join( tmp, pdf.name )
					with open( path, 'wb' ) as f:
						f.write( pdf.read( ) )
					
					loader = PdfLoader( )
					documents = loader.load( path, mode=mode, extract=extract, include=include,
						format=fmt, ) or [ ]
				
				raw_text = '\n\n'.join( d.page_content for d in documents if hasattr( d, 'page_content' ) 
				                        and isinstance( d.page_content, str ) 
				                        and d.page_content.strip( ) )
				
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = raw_text
				st.session_state.processed_text = None
				st.session_state.lines = None
				st.session_state.chunked_documents = None
				st.session_state.df_chunks = None
				st.session_state.active_loader = 'PdfLoader'
				
				st.session_state[ '_loader_status' ] = \
					f'Loaded {len( documents )} PDF document(s).'
		
		# --------------------------- Markdown Loader
		with st.expander( label='Markdown Loader', icon='🧾', expanded=False ):
			md = st.file_uploader(
				'Upload Markdown',
				type=[ 'md',
				       'markdown' ],
				key='md_upload',
			)
			
			mode = st.selectbox(
				'Mode',
				[ 'single', 'elements' ],
				index=0,
				key='md_mode',
				help='Use "single" for one combined document or "elements" for element-wise parsing.'
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
			# Load (same behavior, now with explicit mode)
			# --------------------------------------------------
			if load_md and md:
				with tempfile.TemporaryDirectory( ) as tmp:
					path = os.path.join( tmp, md.name )
					with open( path, "wb" ) as f:
						f.write( md.read( ) )
					
					loader = MarkdownLoader( )
					documents = loader.load( path, mode=mode ) or [ ]
				
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = "\n\n".join(
					d.page_content for d in documents
					if hasattr( d, 'page_content' )
					and isinstance( d.page_content, str )
					and d.page_content.strip( )
				)
				st.session_state.active_loader = "MarkdownLoader"
				
				st.success( f"Loaded {len( documents )} Markdown document(s)." )
		
		# --------------------------- HTML Loader
		with st.expander( label='HTML Loader', icon='🌐', expanded=False ):
			html = st.file_uploader( 'Upload HTML', type=[ 'html', 'htm' ], key='html_upload' )
			
			# --------------------------------------------------
			# Buttons: Load / Clear / Save (same row, same style)
			# --------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			load_html = col_load.button( 'Load', key='html_load' )
			clear_html = col_clear.button( 'Clear', key='html_clear' )
			
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
				st.session_state.raw_text = "\n\n".join( d.page_content for d in documents )
				st.session_state.active_loader = "HtmlLoader"
				st.success( f"Loaded {len( documents )} HTML document(s)." )
		
		# --------------------------- JSON Loader
		with st.expander( label='JSON Loader', icon='🧩', expanded=False ):
			js = st.file_uploader(
				'Upload JSON',
				type=[ 'json', 'jsonl' ],
				key='json_upload',
			)
			
			jq_schema = st.text_input(
				'jq Schema',
				value='.',
				key='json_jq_schema',
				help='Examples: ., .[], .messages[], .content'
			)
			
			content_key = st.text_input(
				'Content Key (optional)',
				value='',
				key='json_content_key',
				help='Use when jq_schema returns objects and you want one field as page_content.'
			)
			
			is_lines = st.checkbox(
				'JSON Lines',
				value=False,
				key='json_lines',
			)
			
			is_text = st.checkbox(
				'Extracted content is already text',
				value=True,
				key='json_text_content',
				help='Turn this off when jq_schema/content_key selects structured values instead of plain text.'
			)
			
			# --------------------------------------------------
			# Buttons: Load / Clear / Save
			# --------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			load_json = col_load.button( 'Load', key='json_load' )
			clear_json = col_clear.button( 'Clear', key='json_clear' )
			
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
			# Clear
			# --------------------------------------------------
			if clear_json:
				clear_if_active( 'JsonLoader' )
				st.info( 'JSON Loader state cleared.' )
			
			# --------------------------------------------------
			# Load
			# --------------------------------------------------
			if load_json and js:
				with tempfile.TemporaryDirectory( ) as tmp:
					path = os.path.join( tmp, js.name )
					with open( path, 'wb' ) as f:
						f.write( js.read( ) )
					
					loader = JsonLoader( )
					documents = loader.load(
						path,
						jq_schema=jq_schema,
						content_key=content_key,
						is_text=is_text,
						is_lines=is_lines,
					) or [ ]
				
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = "\n\n".join(
					d.page_content for d in documents
					if hasattr( d, 'page_content' )
					and isinstance( d.page_content, str )
					and d.page_content.strip( )
				)
				st.session_state.active_loader = "JsonLoader"
				st.success( f"Loaded {len( documents )} JSON document(s)." )
		
		# --------------------------- PowerPoint Loader
		with st.expander( label='Power Point Loader', icon='📽', expanded=False ):
			pptx = st.file_uploader( 'Upload PPTX', type=[ 'pptx' ], key='pptx_upload', )
			mode = st.selectbox( 'Mode', [ 'single', 'elements' ], key='pptx_mode', )
			
			# --------------------------------------------------
			# Buttons: Load / Clear / Save (same row, same style)
			# --------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			load_pptx = col_load.button( 'Load', key='pptx_load', )
			
			clear_pptx = col_clear.button( 'Clear', key='pptx_clear', )
			
			# ---------- Save
			can_save = ( st.session_state.get( 'active_loader' ) == 'PowerPointLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( ) )
			
			if can_save:
				col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
					file_name='powerpoint_loader_output.txt', mime='text/plain', key='pptx_save', )
			else:
				col_save.button( 'Save', key='pptx_save_disabled', disabled=True, )
			
			# ---------- Clear
			if clear_pptx:
				clear_if_active( 'PowerPointLoader' )
				st.info( 'PowerPoint Loader state cleared.' )
			
			# ---------- Load
			if load_pptx and pptx:
				with tempfile.TemporaryDirectory( ) as tmp:
					path = os.path.join( tmp, pptx.name )
					with open( path, "wb" ) as f:
						f.write( pptx.read( ) )
					
					loader = PowerPointLoader( )
					documents = loader.load( path, mode=mode ) or [ ]
				
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = "\n\n".join( d.page_content for d in documents
					if hasattr( d, 'page_content' ) and isinstance( d.page_content, str )
					                                     and d.page_content.strip( ) )
				st.session_state.active_loader = "PowerPointLoader"
				st.success( f"Loaded {len( documents )} PowerPoint document(s)." )
		
		# --------------------------- Excel Loader
		with st.expander( label='Excel Loader', icon='📊', expanded=False ):
			excel_file = st.file_uploader( 'Upload Excel file', type=[ 'xlsx', 'xls' ],
				key='excel_upload', )
			
			load_mode = st.selectbox( 'Load Mode', [ 'Tabular + SQLite', 'Unstructured Document' ],
				index=0, key='excel_load_mode',
				help=( 'Use "Tabular + SQLite" to preserve the current sheet-to-SQLite workflow. '
						'Use "Unstructured Document" to route through ExcelLoader.' ), )
			
			sheet_name = st.text_input( 'Sheet name (leave blank for all sheets)', key='excel_sheet' )
			
			table_prefix = st.text_input( 'table prefix', value='excel',
				help='Each sheet will be written as <prefix>_<sheetname>', key='excel_table_prefix')
			
			unstructured_mode = st.selectbox( 'Document Mode', [ 'single', 'elements' ], index=0,
				key='excel_unstructured_mode', help='Used only with "Unstructured Documents".' )
			
			# --------------------------------------------------
			# Buttons: Load / Clear / Save
			# --------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			load_excel = col_load.button( 'Load', key='excel_load' )
			clear_excel = col_clear.button( 'Clear', key='excel_clear' )
			
			can_save = ( st.session_state.get( 'active_loader' ) == 'ExcelLoader'
			             and isinstance( st.session_state.get( 'raw_text' ), str )
			             and st.session_state.get( 'raw_text' ).strip( ) )
			
			if can_save:
				col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
					file_name='excel_loader_output.txt', mime='text/plain', key='excel_save', )
			else:
				col_save.button( 'Save', key='excel_save_disabled', disabled=True, )
			
			# --------------------------------------------------
			# Clear (remove only ExcelLoader documents)
			# --------------------------------------------------
			if clear_excel and st.session_state.get( 'documents' ):
				st.session_state.documents = [ d for d in st.session_state.documents
				                               if d.metadata.get( 'loader' ) != 'ExcelLoader' ]
				
				st.session_state.raw_documents = [ d for d in st.session_state.documents
                   if isinstance( getattr( d, 'metadata', None ), dict ) ] if st.session_state.documents else [ ]
				
				st.session_state.raw_text = ( '\n\n'.join( d.page_content for d in st.session_state.documents
				                                           if isinstance( d.page_content, str )
				                                           and d.page_content.strip( ) )
				                              if st.session_state.documents else None )
				
				st.session_state.processed_text = None
				st.session_state.active_loader = None
				
				st.info( "ExcelLoader documents removed." )
			
			# --------------------------------------------------
			# Load
			# --------------------------------------------------
			if load_excel and excel_file:
				with tempfile.TemporaryDirectory( ) as tmp:
					excel_path = os.path.join( tmp, excel_file.name )
					with open( excel_path, "wb" ) as f:
						f.write( excel_file.read( ) )
					
					documents = [ ]
					if load_mode == 'Tabular + SQLite':
						sqlite_path = os.path.join( "stores", "sqlite", "data.db" )
						os.makedirs( os.path.dirname( sqlite_path ), exist_ok=True )
						
						if sheet_name.strip( ):
							dfs = {sheet_name: pd.read_excel( excel_path, sheet_name=sheet_name,)}
						else:
							dfs = pd.read_excel( excel_path, sheet_name=None, )
						
						conn = sqlite3.connect( sqlite_path )
						try:
							for sheet, df in dfs.items( ):
								if df.empty:
									continue
								table_name = f"{table_prefix}_{sheet}".replace( " ", "_" ).lower( )
								df.to_sql( table_name, conn, if_exists="replace", index=False, )
								text = df.to_csv( index=False )
								documents.append(
									Document( page_content=text,
										metadata={ 'loader': 'ExcelLoader',
												'source': excel_file.name,
												'sheet': sheet,
												'table': table_name,
												'sqlite_db': sqlite_path,
												'load_mode': 'Tabular + SQLite', }, ) )
						finally:
							conn.close( )
					
					else:
						loader = ExcelLoader( )
						documents = loader.load( excel_path, mode=unstructured_mode,
							has_headers=True ) or [ ]
						
						for document in documents:
							if not isinstance( getattr( document, 'metadata', None ), dict ):
								document.metadata = { }
							
							document.metadata[ 'loader' ] = 'ExcelLoader'
							document.metadata.setdefault( 'source', excel_file.name )
							document.metadata[ 'load_mode' ] = 'Unstructured Document'
							document.metadata[ 'document_mode' ] = unstructured_mode
				
				if documents:
					existing_documents = st.session_state.get( 'documents' )
					if isinstance( existing_documents, list ) and existing_documents:
						st.session_state.documents.extend( documents )
					else:
						st.session_state.documents = list( documents )
					
					st.session_state.raw_documents = list( st.session_state.documents )
					st.session_state.raw_text = "\n\n".join( d.page_content for d in st.session_state.documents
					                                         if isinstance( d.page_content, str )
					                                         and d.page_content.strip( ) )
					
					st.session_state.processed_text = None
					st.session_state.active_loader = 'ExcelLoader'
					
					if load_mode == 'Tabular + SQLite':
						st.success( f"Loaded {len( documents )} sheet(s) and stored in SQLite." )
					else:
						st.success( f"Loaded {len( documents )} Excel {unstructured_mode!r} mode." )
				else:
					if load_mode == 'Tabular + SQLite':
						st.warning( "No data loaded (empty sheets or invalid selection)." )
					else:
						st.warning( "No Excel document content was loaded." )
		
		# --------------------------- ArXiv Loader
		with st.expander( label='ArXiv Loader', icon='🧠', expanded=False ):
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
			
			col_fetch, col_clear, col_save = st.columns( 3 )
			arxiv_fetch = col_fetch.button( 'Load', key='arxiv_fetch' )  # label kept as Load button row convention
			arxiv_clear = col_clear.button( 'Clear', key='arxiv_clear' )
			
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
				col_save.button( 'Save', key='arxiv_save_disabled', disabled=True )
			
			if arxiv_clear and st.session_state.get( 'documents' ):
				st.session_state.documents = [
						d for d in st.session_state.documents
						if d.metadata.get( 'loader' ) != 'ArXivLoader'
				]
				st.session_state.raw_text = rebuild_raw_text_from_documents( )
				st.session_state[ '_loader_status' ] = 'ArXivLoader documents removed.'
			
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
					if st.session_state.get( 'documents' ):
						st.session_state.documents.extend( documents )
					else:
						st.session_state.documents = documents
						st.session_state.raw_documents = list( documents )
					
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state.active_loader = 'ArXivLoader'
					
					st.session_state['_loader_status' ] = f'Fetched {len( documents )} document(s).'
		
		# --------------------------- Wikipedia Loader
		with st.expander( label='Wikipedia Loader', icon='📚', expanded=False ):
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
			
			col_fetch, col_clear, col_save = st.columns( 3 )
			wiki_fetch = col_fetch.button( 'Load', key='wiki_fetch' )
			wiki_clear = col_clear.button( 'Clear', key='wiki_clear' )
			
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
				col_save.button( 'Save', key='wiki_save_disabled', disabled=True )
			
			if wiki_clear and st.session_state.get( 'documents' ):
				st.session_state.documents = [
						d for d in st.session_state.documents
						if d.metadata.get( 'loader' ) != 'WikiLoader'
				]
				st.session_state.raw_text = rebuild_raw_text_from_documents( )
				st.session_state[ '_loader_status' ] = 'WikiLoader documents removed.'
			
			if wiki_fetch and wiki_query:
				loader = WikiLoader( )
				documents = loader.load(
					wiki_query,
					max_docs=int( wiki_max_docs ),
					max_chars=int( wiki_max_chars ),
				) or [ ]
				
				for d in documents:
					d.metadata[ 'loader' ] = 'WikiLoader'
					d.metadata[ 'source' ] = wiki_query
				
				if documents:
					if st.session_state.get( 'documents' ):
						st.session_state.documents.extend( documents )
					else:
						st.session_state.documents = documents
						st.session_state.raw_documents = list( documents )
					
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state.active_loader = 'WikiLoader'
					
					st.session_state[
						'_loader_status' ] = f'Fetched {len( documents )} Wikipedia document(s).'
		
		# --------------------------- GitHub Loader
		with st.expander( label='GitHub Loader', icon='🐙', expanded=False ):
			gh_url = st.text_input(
				"GitHub API URL",
				placeholder="https://api.github.com",
				value="https://api.github.com",
				key="gh_url",
				help="GitHub REST API base URL.",
			)
			
			gh_repo = st.text_input(
				"Repo (owner/name)",
				placeholder="openai/openai-python",
				key="gh_repo",
				help="Name of the repository.",
			)
			
			gh_branch = st.text_input(
				"Branch",
				placeholder="main",
				value="main",
				key="gh_branch",
				help="The branch of the repository.",
			)
			
			gh_filetype = st.text_input(
				"File type filter",
				value=".md",
				key="gh_filetype",
				help="Filtering by file type. Example: .py, .md, .txt",
			)
			
			gh_access_token = st.text_input(
				"GitHub Access Token (optional)",
				value="",
				type="password",
				key="gh_access_token",
				help="Optional personal access token. Useful for private repos or higher rate limits.",
			)
			
			col_fetch, col_clear, col_save = st.columns( 3 )
			gh_fetch = col_fetch.button( "Load", key="gh_fetch" )
			gh_clear = col_clear.button( "Clear", key="gh_clear" )
			
			can_save = (
					st.session_state.get( "active_loader" ) == "GithubLoader"
					and isinstance( st.session_state.get( "raw_text" ), str )
					and st.session_state.get( "raw_text" ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					"Save",
					data=st.session_state.get( "raw_text" ),
					file_name="github_loader_output.txt",
					mime="text/plain",
					key="gh_save",
				)
			else:
				col_save.button( "Save", key="gh_save_disabled", disabled=True )
			
			if gh_clear and st.session_state.get( "documents" ):
				st.session_state.documents = [
						d for d in st.session_state.documents
						if d.metadata.get( "loader" ) != "GithubLoader"
				]
				st.session_state.raw_text = rebuild_raw_text_from_documents( )
				st.session_state[ "_loader_status" ] = "GithubLoader documents removed."
			
			if gh_fetch and gh_repo and gh_branch:
				loader = GithubLoader( )
				documents = loader.load(
					gh_url,
					gh_repo,
					gh_branch,
					gh_filetype,
					gh_access_token,
				) or [ ]
				
				for d in documents:
					if not isinstance( getattr( d, "metadata", None ), dict ):
						d.metadata = { }
					d.metadata[ "loader" ] = "GithubLoader"
					d.metadata[ "source" ] = f"{gh_repo}@{gh_branch}"
				
				if documents:
					if st.session_state.get( "documents" ):
						st.session_state.documents.extend( documents )
					else:
						st.session_state.documents = documents
						st.session_state.raw_documents = list( documents )
					
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state.active_loader = "GithubLoader"
					
					st.session_state[
						"_loader_status"
					] = f"Fetched {len( documents )} GitHub document(s)."
		
		# --------------------------- Web Loader
		with st.expander( label="Web Loader", icon='🔗', expanded=False ):
			urls = st.text_area(
				"Enter one URL per line",
				placeholder="https://example.com\nhttps://another.com",
				key="web_urls",
			)
			
			web_timeout = st.number_input(
				"Timeout (seconds)",
				min_value=1,
				max_value=120,
				value=10,
				step=1,
				key="web_timeout",
			)
			
			web_ignore = st.checkbox(
				"Continue On Failure",
				value=True,
				key="web_ignore",
				help="Keep loading remaining URLs if one page fails."
			)
			
			col_fetch, col_clear, col_save = st.columns( 3 )
			load_web = col_fetch.button( "Load", key="web_fetch" )
			clear_web = col_clear.button( "Clear", key="web_clear" )
			
			can_save = (
					st.session_state.get( "active_loader" ) == "WebLoader"
					and isinstance( st.session_state.get( "raw_text" ), str )
					and st.session_state.get( "raw_text" ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					"Save",
					data=st.session_state.get( "raw_text" ),
					file_name="web_loader_output.txt",
					mime="text/plain",
					key="web_save",
				)
			else:
				col_save.button( "Save", key="web_save_disabled", disabled=True )
			
			if clear_web and st.session_state.get( "documents" ):
				st.session_state.documents = [
						d for d in st.session_state.documents
						if d.metadata.get( "loader" ) != "WebLoader"
				]
				st.session_state.raw_text = rebuild_raw_text_from_documents( )
				st.session_state[ "_loader_status" ] = "WebLoader documents removed."
			
			if load_web and urls.strip( ):
				loader = WebLoader( recursive=False )
				new_docs = [ ]
				
				for url in [ u.strip( ) for u in urls.splitlines( ) if u.strip( ) ]:
					documents = loader.load(
						urls=url,
						timeout=int( web_timeout ),
						ignore=bool( web_ignore ),
						progress=True
					) or [ ]
					
					for d in documents:
						if not isinstance( getattr( d, "metadata", None ), dict ):
							d.metadata = { }
						d.metadata[ "loader" ] = "WebLoader"
						d.metadata[ "source" ] = url
					
					new_docs.extend( documents )
				
				if new_docs:
					if st.session_state.get( "documents" ):
						st.session_state.documents.extend( new_docs )
					else:
						st.session_state.documents = new_docs
						st.session_state.raw_documents = list( new_docs )
					
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state.active_loader = "WebLoader"
					
					st.session_state[
						"_loader_status"
					] = f"Fetched {len( new_docs )} web document(s)."
		
		# --------------------------- Web Crawler
		with st.expander( label="Web Crawler", icon='🕷️', expanded=False ):
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
			
			crawl_timeout = st.number_input(
				"Timeout (seconds)",
				min_value=1,
				max_value=120,
				value=10,
				step=1,
				key="crawl_timeout",
			)
			
			stay_on_domain = st.checkbox(
				"Stay on starting domain",
				value=True,
				key="crawl_domain_lock",
			)
			
			col_run, col_clear, col_save = st.columns( 3 )
			run_crawl = col_run.button( "Load", key="crawl_run" )
			clear_crawl = col_clear.button( "Clear", key="crawl_clear" )
			
			can_save = (
					st.session_state.get( "active_loader" ) == "WebCrawler"
					and isinstance( st.session_state.get( "raw_text" ), str )
					and st.session_state.get( "raw_text" ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					"Save",
					data=st.session_state.get( "raw_text" ),
					file_name="web_crawler_output.txt",
					mime="text/plain",
					key="crawl_save",
				)
			else:
				col_save.button( "Save", key="crawl_save_disabled", disabled=True )
			
			if clear_crawl and st.session_state.get( "documents" ):
				st.session_state.documents = [
						d for d in st.session_state.documents
						if d.metadata.get( "loader" ) != "WebCrawler"
				]
				st.session_state.raw_text = rebuild_raw_text_from_documents( )
				st.session_state[ "_loader_status" ] = "WebCrawler documents removed."
			
			if run_crawl and start_url.strip( ):
				loader = WebLoader(
					recursive=True,
					max_depth=int( max_depth ),
					prevent_outside=bool( stay_on_domain ),
					timeout=int( crawl_timeout ),
					ignore=True,
					progress=True
				)
				
				documents = loader.load(
					urls=start_url.strip( ),
					depth=int( max_depth ),
					timeout=int( crawl_timeout ),
					ignore=True,
					progress=True,
					prevent_outside=bool( stay_on_domain )
				) or [ ]
				
				for d in documents:
					if not isinstance( getattr( d, "metadata", None ), dict ):
						d.metadata = { }
					d.metadata[ "loader" ] = "WebCrawler"
					d.metadata.setdefault( "source", start_url.strip( ) )
				
				if documents:
					if st.session_state.get( "documents" ):
						st.session_state.documents.extend( documents )
					else:
						st.session_state.documents = documents
						st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state.active_loader = "WebCrawler"
					st.session_state[ "_loader_status" ] = f"Crawled {len( documents )} document(s)."
				
	# ------------------------------------------------------------------
	# RIGHT COLUMN — DOCUMENT RENDERING
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
						height=800, key=f'preview_doc_{i}' )
			
	st.divider( )
	
	# ------------------------------------------------------------------
	# NLP METRIC CALCULATIONS
	# ------------------------------------------------------------------
	metrics_container = st.container( )
	with metrics_container:
		if documents is not None:
			# ----------------------------------------------
			# Tokenization (session-cached)
			# ----------------------------------------------
			if st.session_state.tokens is None:
				try:
					raw_text = rebuild_raw_text_from_documents( )
					tokens = [ t.lower( ) for t in word_tokenize( raw_text ) if t.isalpha( ) ]
				except LookupError:
					st.error( 'NLTK resources missing' )
				
				if not tokens:
					st.warning( 'No valid alphabetic tokens found.' )
				
				st.session_state.tokens = tokens
				st.session_state.vocabulary = set( tokens )
				st.session_state.token_counts = Counter( tokens )
				
			tokens = st.session_state.tokens
			vocabulary = st.session_state.vocabulary
			counts = st.session_state.token_counts
			char_count = len( raw_text )
			token_count = len( tokens )
			vocab_size = len( vocabulary )
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
			
			st.subheader( 'Top Tokens' )
			
			# ------------ Top Tokens
			with st.expander( label='Tokens', icon='🉑', expanded=True ):
				top_tokens = counts.most_common( 10 )
				df_top = pd.DataFrame( top_tokens, columns=[ 'token', 'count' ] ).set_index( 'token' )
				st.bar_chart( df_top, color='#01438A' )
			
			st.divider( )
			st.subheader( 'Corpus Metrics')
			
			# ------------ Corpus Metrics
			with st.expander( label='Text', icon='📖', expanded=True ):
				
				col1, col2, col3, col4 = st.columns( 4, border=True )
				with col1:
					metric_with_tooltip( 'Characters', f'{char_count:,}',
						'Total number of characters in the selected text.' )
					
				with col2:
					metric_with_tooltip( 'Tokens', f'{token_count:,}',
						'Token Count: total number of tokenized words after cleanup.' )
					
				with col3:
					metric_with_tooltip( 'Unique Tokens', f'{vocab_size:,}',
						'Vocabulary Size: number of distinct word types in the text.' )
					
				with col4:
					metric_with_tooltip( 'TTR', f'{ttr:.3f}',
						'Type–Token Ratio: unique words ÷ total words.' )
				
				col5, col6, col7, col8 = st.columns( 4, border=True )
				with col5:
					metric_with_tooltip( 'Hapax Ratio', f'{hapax_ratio:.3f}',
						'Hapax Ratio: proportion of words that occur only once.' )
					
				with col6:
					metric_with_tooltip( 'Avg Length', f'{avg_word_len:.2f}',
						'Average number of characters per token.' )
					
				with col7:
					metric_with_tooltip( 'Stopword Ratio', f'{stopword_ratio:.2%}',
						'Percentage of stopwords in the text.' )
					
				with col8:
					metric_with_tooltip(
						'Lexical Density',
						f'{lexical_density:.2%}',
						'Proportion of content-bearing words.' )
			
			st.divider( )
			st.subheader( 'Comprehension Metrics' )
			
			# ------------ Readability
			with st.expander( label='Words', icon='👀', expanded=True ):
				if TEXTSTAT_AVAILABLE:
					r1, r2, r3, r4 = st.columns( 4, border=True )
					with r1:
						metric_with_tooltip( 'Flesch Reading Ease', f'{textstat.flesch_reading_ease( raw_text ):.1f}',
							'Higher scores indicate easier readability.' )
					
					with r2:
						metric_with_tooltip( 'Flesch–Kincaid Grade',
							f'{textstat.flesch_kincaid_grade( raw_text ):.1f}',
							'Estimated U.S. grade level required.' )
					
					with r3:
						metric_with_tooltip( 'Gunning Fog', f'{textstat.gunning_fog( raw_text ):.1f}',
							'Readability based on sentence length and complex words.' )
					
					with r4:
						metric_with_tooltip( 'Coleman–Liau Index',
							f'{textstat.coleman_liau_index( raw_text ):.1f}',
							'Readability based on characters and sentences.' )
				else:
					st.caption( 'Install `textstat` to enable readability metrics.' )

# ======================================================================================
# Tab — Text Processing
# ======================================================================================
with tabs[ 1 ]:
	raw_text = st.session_state.get( 'raw_text' )
	processed_text = st.session_state.get( 'processed_text' )
	active_loader = st.session_state.get( 'active_loader' )
	
	if not isinstance( raw_text, str ) or not raw_text.strip( ):
		st.info( 'Load a document before running text processing.' )
		st.stop( )
	
	if not active_loader:
		st.warning( 'No active loader detected. Load documents first.' )
		st.stop( )
	
	if isinstance( raw_text, str ):
		st.session_state.raw_text_view = raw_text
		processed_text = st.session_state.get( 'processed_text' )
		start_time = st.session_state.get( 'start_time', 0.0 )
		end_time = st.session_state.get( 'end_time', 0.0 )
		total_time = st.session_state.get( 'total_time', 0.0 )
		has_text = isinstance( raw_text, str ) and bool( raw_text.strip( ) )
		
		# ------------------------------------------------------------------
		# Layout
		# ------------------------------------------------------------------
		left, right = st.columns( [ 1, 1.5 ], border=True )
		with left:
			active = st.session_state.get( 'active_loader' )
			
			# ==============================================================
			# Common Text Processing (TextParser)
			# ==============================================================
			with st.expander( label='Text Processing', icon='🧠', expanded=True ):
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
					help=r'Removes @, #, $, ^, *, =, |, \, <, >, ~ but preserves sentence '
					     r'delimiters' )
				remove_images = st.checkbox( 'Remove Images',
					help=r'Remove image from text, including Markdown, HTML <img> tags, '
					     r'and  image URLs' )
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
			with st.expander( label='Word Processing', icon='📄', expanded=False ):
				if active == 'WordLoader':
					extract_tables = st.checkbox( 'Extract Tables' )
					extract_paragraphs = st.checkbox( 'Extract Paragraphs' )
				else:
					st.caption( 'Available when Word documents are loaded.' )
			
			# ==============================================================
			# PDF-Specific Processing (PdfParser)
			# ==============================================================
			remove_headers = join_hyphenated = False
			with st.expander( 'PDF Processing', icon='📕', expanded=False ):
				if active == 'PdfLoader':
					remove_headers = st.checkbox( 'Remove Headers/Footers' )
					join_hyphenated = st.checkbox( 'Join Hyphenated Lines' )
				else:
					st.caption( 'Available when PDF documents are loaded.' )
			
			# ==============================================================
			# HTML-Specific Processing (Structural)
			# ==============================================================
			strip_scripts = keep_headings = keep_paragraphs = keep_tables = False
			with st.expander( 'HTML Processing', icon='🌐', expanded=False ):
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
			apply_processing = col_apply.button( 'Apply', disabled=not has_text,
				key='processing_apply_button' )
			reset_processing = col_reset.button( 'Reset', disabled=not has_text,
				key='processing_reset_button' )
			clear_processing = col_clear.button( 'Clear', disabled=not has_text,
				key='processing_clear_button' )
			can_save_processed = (isinstance( st.session_state.get( 'processed_text' ), str )
			                      and st.session_state.get( 'processed_text' ).strip( ))
			
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
				st.session_state.start_time = 0.0
				st.session_state.end_time = 0.0
				st.session_state.total_time = 0.0
				st.success( 'Processed text reset.' )
			
			if clear_processing:
				st.session_state.processed_text = ''
				st.session_state.processed_text_view = ''
				st.session_state.start_time = 0.0
				st.session_state.end_time = 0.0
				st.session_state.total_time = 0.0
				st.success( 'Processed text cleared.' )
			
			if apply_processing:
				start_time = time.perf_counter( )
				
				# ----------------------------------------------------------
				# Initialize from raw text (authoritative source)
				# ----------------------------------------------------------
				processed_text = raw_text if isinstance( raw_text, str ) else ''
				
				tp = TextParser( )
				
				# ----------------------------------------------------------
				# 1 — Structural cleanup
				# ----------------------------------------------------------
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
				
				# ----------------------------------------------------------
				# 2 — Noise / non-lexical characters
				# ----------------------------------------------------------
				if remove_symbols:
					processed_text = tp.remove_symbols( processed_text )
				if remove_numbers:
					processed_text = tp.remove_numbers( processed_text )
				if remove_numerals:
					processed_text = tp.remove_numerals( processed_text )
				
				# ----------------------------------------------------------
				# 3 — Meaning-critical punctuation shaping
				# ----------------------------------------------------------
				if remove_punctuation:
					processed_text = tp.remove_punctuation( processed_text )
				
				# ----------------------------------------------------------
				# 4 — Word normalization
				# ----------------------------------------------------------
				if normalize_text:
					processed_text = tp.normalize_text( processed_text )
				
				# ----------------------------------------------------------
				# 5 — Lexical refinement
				# ----------------------------------------------------------
				if remove_stopwords:
					processed_text = tp.remove_stopwords( processed_text )
				if remove_fragments:
					processed_text = tp.remove_fragments( processed_text )
				if remove_errors:
					processed_text = tp.remove_errors( processed_text )
				
				# ----------------------------------------------------------
				# 6 — Lemmatization
				# ----------------------------------------------------------
				if lemmatize_text:
					processed_text = tp.lemmatize_text( processed_text )
				
				# ----------------------------------------------------------
				# 7 — Whitespace cleanup
				# ----------------------------------------------------------
				if collapse_whitespace:
					processed_text = tp.collapse_whitespace( processed_text )
				if compress_whitespace:
					processed_text = tp.compress_whitespace( processed_text )
				
				# ----------------------------------------------------------
				# Format-specific FIRST
				# ----------------------------------------------------------
				parser = st.session_state.get( 'parser' )
				
				if active == 'WordLoader':
					if extract_tables and hasattr( parser, 'extract_tables' ):
						parser = WordParser( )
						processed_text = parser.extract_tables( processed_text )
					if extract_paragraphs and hasattr( parser, 'extract_paragraphs' ):
						parser = WordParser( )
						processed_text = parser.extract_paragraphs( processed_text )
				
				if active == 'PdfLoader':
					if remove_headers and hasattr( parser, 'remove_headers' ):
						parser = PdfParser( )
						processed_text = parser.remove_headers( processed_text )
					if join_hyphenated and hasattr( parser, 'join_hyphenated' ):
						parser = PdfParser( )
						processed_text = parser.join_hyphenated( processed_text )
				
				if active == 'HtmlLoader':
					if strip_scripts:
						processed_text = tp.remove_html( processed_text )
				
				# ----------------------------------------------------------
				# Finalize timing
				# ----------------------------------------------------------
				end_time = time.perf_counter( )
				st.session_state.total_time = end_time - start_time
				
				# ----------------------------------------------------------
				# Commit processed text
				# ----------------------------------------------------------
				st.session_state.processed_text = ( processed_text if isinstance( processed_text, str ) else str(
							processed_text ))
				st.session_state.processed_text_view = st.session_state.processed_text
				st.success( f'Text processing applied ({st.session_state.total_time:.1f} s)' )
			
			# ------------------------------------------------------------------
			# RIGHT COLUMN — Text Views
			# ------------------------------------------------------------------
			with right:
				st.text_area( label='Raw Text', height=200, disabled=True, key='raw_text_view' )
				raw_text = st.session_state.get( 'raw_text' )
				with st.expander( '📊 Processing Statistics:', expanded=False ):
					processed_text = st.session_state.get( 'processed_text' )
					if (isinstance( raw_text, str ) and raw_text.strip( ) and
							isinstance( processed_text, str ) and processed_text.strip( )):
						raw_tokens = raw_text.split( )
						proc_tokens = processed_text.split( )
						raw_chars = len( raw_text )
						proc_chars = len( processed_text )
						raw_vocab = len( set( raw_tokens ) )
						proc_vocab = len( set( proc_tokens ) )
						
						# ----------------------------
						# Absolute Metrics
						# ----------------------------
						st.text( 'Measures:' )
						ttr = (proc_vocab / len( proc_tokens ) if proc_tokens else 0.0)
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
						compression = (proc_chars / raw_chars if raw_chars > 0 else 0.0)
						d1.metric( 'Δ Characters', f'{char_delta:+,}' )
						d2.metric( 'Δ Tokens', f'{token_delta:+,}' )
						d3.metric( 'Δ Vocabulary', f'{vocab_delta:+,}' )
						d4.metric( 'Compression Ratio', f'{compression:.2%}' )
					else:
						st.caption( 'Load and process text to view absolute and delta statistics.' )
				
				# ----------------------------
				# Processed Text (output)
				# ----------------------------
				st.text_area( 'Processed Text', st.session_state.processed_text or '', height=700 )

# ======================================================================================
# Tab - Semantic Analysis
# ======================================================================================
with tabs[ 2 ]:
	import pandas as pd
	from nltk.tokenize import word_tokenize
	
	processed_text = st.session_state.get( 'processed_text' )
	
	# ------------------------------------------------------------------
	# Guard: must have processed text to proceed
	# ------------------------------------------------------------------
	if not isinstance( processed_text, str ) or not processed_text.strip( ):
		st.info( 'Run text processing before semantic analysis.' )
		st.stop( )
	
	# ------------------------------------------------------------------
	# Local helpers (avoid relying on unknown TextParser signatures)
	# ------------------------------------------------------------------
	def _chunk_chars_with_overlap( text: str, size: int, overlap: int ) -> list[ str ]:
		"""
			Purpose:
			--------
			Chunk text by characters with overlap.

			Parameters:
			-----------
			text : str
				Input text.
			size : int
				Chunk size in characters.
			overlap : int
				Overlap in characters.

			Returns:
			--------
			list[str]
				List of chunk strings.
		"""
		if not isinstance( text, str ) or not text.strip( ):
			return [ ]
		
		size = int( size )
		overlap = int( overlap )
		size = max( 1, size )
		overlap = max( 0, min( overlap, size - 1 ) )
		
		step = size - overlap
		chunks: list[ str ] = [ ]
		
		i = 0
		n = len( text )
		while i < n:
			part = text[ i: i + size ].strip( )
			if part:
				chunks.append( part )
			i += step
		
		return chunks
	
	def _chunk_tokens_with_overlap( tokens: list[ str ], size: int, overlap: int ) -> list[ str ]:
		"""
			Purpose:
			--------
			Chunk token list into overlapped token windows and return joined strings.

			Parameters:
			-----------
			tokens : list[str]
				Token list.
			size : int
				Tokens per chunk.
			overlap : int
				Overlap in tokens.

			Returns:
			--------
			list[str]
				Chunk strings.
		"""
		if not isinstance( tokens, list ) or not tokens:
			return [ ]
		
		size = int( size )
		overlap = int( overlap )
		size = max( 1, size )
		overlap = max( 0, min( overlap, size - 1 ) )
		
		step = size - overlap
		out: list[ str ] = [ ]
		
		i = 0
		n = len( tokens )
		while i < n:
			window = tokens[ i: i + size ]
			s = " ".join( t for t in window if isinstance( t, str ) and t.strip( ) ).strip( )
			if s:
				out.append( s )
			i += step
		
		return out
	
	# ------------------------------------------------------------------
	# Resolve / defaults
	# ------------------------------------------------------------------
	chunk_modes = st.session_state.get( 'chunk_modes' )
	if not isinstance( chunk_modes, (list, tuple) ) or not chunk_modes:
		chunk_modes = [ 'chars',
		                'tokens' ]
		st.session_state.chunk_modes = list( chunk_modes )
	
	# ------------------------------------------------------------------
	# Controls
	# ------------------------------------------------------------------
	st.subheader( "Semantic Analysis" )
	
	mode = st.selectbox(
		'Chunking Mode',
		options=list( chunk_modes ),
		key='chunk_mode',
		help='Select how documents are chunked',
	)
	
	col_a, col_b = st.columns( 2 )
	
	with col_a:
		chunk_size = st.number_input(
			'Chunk Size',
			min_value=100,
			max_value=5000,
			value=1000,
			step=100,
			key='chunk_count',
		)
	
	with col_b:
		overlap = st.number_input(
			'Overlap',
			min_value=0,
			max_value=2000,
			value=200,
			step=50,
			key='overlap_input',
		)
	
	col_run, col_reset = st.columns( 2 )
	run_chunking = col_run.button( 'Chunk', key='run_button' )
	reset_chunking = col_reset.button( 'Reset', key='reset_control' )
	
	# ------------------------------------------------------------------
	# Actions
	# ------------------------------------------------------------------
	if reset_chunking:
		st.session_state.chunked_documents = None
		st.info( 'Chunking controls reset.' )
	
	if run_chunking:
		if mode == 'chars':
			chunked_documents = _chunk_chars_with_overlap(
				text=processed_text,
				size=int( chunk_size ),
				overlap=int( overlap ),
			)
		elif mode == 'tokens':
			toks = word_tokenize( processed_text )
			chunked_documents = _chunk_tokens_with_overlap(
				tokens=toks,
				size=int( max( 1, chunk_size // 10 ) ),
				# practical default if user picked 1000 chars
				overlap=int( max( 0, overlap // 10 ) ),
			)
		else:
			chunked_documents = [ ]
			st.error( f'Unsupported chunking mode: {mode}' )
		
		st.session_state.chunked_documents = chunked_documents
		st.success(
			f'Chunking complete: {len( chunked_documents )} chunks generated '
			f'(mode={mode}, size={chunk_size}, overlap={overlap})'
		)
	
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	
	# ------------------------------------------------------------------
	# Tokenization & Vocabulary
	# ------------------------------------------------------------------
	processor = TextParser( )
	tokens = word_tokenize( processed_text )
	vocabulary = processor.create_vocabulary( tokens )
	
	st.session_state.tokens = tokens
	st.session_state.vocabulary = vocabulary
	
	# ------------------------------------------------------------------
	# Frequency Distribution
	# ------------------------------------------------------------------
	df_frequency = processor.create_frequency_distribution( tokens )
	st.session_state.df_frequency = df_frequency
	
	if isinstance( df_frequency, pd.DataFrame ) and not df_frequency.empty:
		st.session_state.df_token_frequency = df_frequency.rename(
			columns={ 'Word': 'Token' }
		).copy( )
	else:
		st.session_state.df_token_frequency = None
	
	# ------------------------------------------------------------------
	# Three-column layout
	# ------------------------------------------------------------------
	col_tokens, col_vocab, col_freq = st.columns( [ 1, 1, 2 ], border=True, vertical_alignment='top' )
	
	with col_tokens:
		st.write( f"Tokens: {len( tokens )}" )
		st.data_editor(
			pd.DataFrame( tokens, columns=[ "Token" ] ),
			num_rows='dynamic',
			use_container_width=True,
			height='stretch',
			disabled=True, )
	
	with col_vocab:
		st.write( f"Vocabulary: {len( vocabulary )}" )
		st.data_editor(
			pd.DataFrame( vocabulary, columns=[ "Word" ] ),
			num_rows='dynamic',
			use_container_width=True,
			height='stretch',
			disabled=True, )
	
	with col_freq:
		st.markdown( "#### Frequency Distribution" )
		st.caption( 'Top 100 most frequent tokens' )
		
		if isinstance( df_frequency, pd.DataFrame ) and not df_frequency.empty:
			numeric_cols = df_frequency.select_dtypes( include="number" )
			if not numeric_cols.empty:
				freq_col = numeric_cols.columns[ 0 ]
				label_col = df_frequency.columns[ 0 ]
				df_top = df_frequency.sort_values( freq_col, ascending=False ).head( 100 )
				
				st.bar_chart(
					df_top.set_index( label_col )[ freq_col ],
					use_container_width=True,
				)
			else:
				st.info( 'No numeric frequency column available for charting.' )
		else:
			st.info( 'Frequency distribution unavailable.' )
	
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )

# ======================================================================================
# Tab - Data Tokenization
# ======================================================================================
with tabs[ 3 ]:
	line_col, chunk_col = st.columns( [ 0.5, 0.5 ], border=True, vertical_alignment='top' )
	
	# ------------------------------------------------------------------
	# Session-state
	# ------------------------------------------------------------------
	df_frequency = st.session_state.get( 'df_frequency' )
	df_tables = st.session_state.get( 'df_tables' )
	df_count = st.session_state.get( 'df_count' )
	df_schema = st.session_state.get( 'df_schema' )
	df_preview = st.session_state.get( 'df_preview' )
	df_chunks = st.session_state.get( 'df_chunks' )
	embedding_model = st.session_state.get( 'embedding_model' )
	embeddings = st.session_state.get( 'embeddings' )
	lines = st.session_state.get( 'lines' )
	chunks = st.session_state.get( 'chunks' )
	chunked_documents = st.session_state.get( 'chunked_documents' )
	active_table = st.session_state.get( 'active_table' )
	chunk_modes = st.session_state.get( 'chunk_modes' )
	raw_text = st.session_state.get( 'raw_text' )
	processed_text = st.session_state.get( 'processed_text' )
	tokens = st.session_state.get( 'tokens' )
	
	def pad_or_trim_row( row: list, size: int ) -> list:
		"""
			Purpose:
			--------
			Normalizes a token row to a fixed width by trimming or right-padding with blanks.
			
			Parameters:
			-----------
			row : list
				The token row to normalize.
			
			size : int
				The required fixed width.
			
			Returns:
			--------
			list
				A list of length == size.
		"""
		if not isinstance( row, list ):
			row = [ ]
		
		if len( row ) >= size:
			return row[ :size ]
		
		return row + ([ '' ] * (size - len( row )))
	
	def _safe_sent_tokenize( text: str ) -> list[ str ]:
		"""
			Purpose:
			--------
			Segments text into natural-language sentences for diagnostics using a
			boundary-preserving source.
			
			Parameters:
			-----------
			text : str
				The input text to segment.
			
			Returns:
			--------
			list[str]
				A list of non-empty sentence strings.
		"""
		if not isinstance( text, str ) or not text.strip( ):
			return [ ]
		
		try:
			segments = sent_tokenize( text )
		except LookupError:
			segments = re.split( r'(?<=[.!?;])\s+', text )
		
		return [ s.strip( ) for s in segments if isinstance( s, str ) and s.strip( ) ]
	
	def _safe_word_tokenize( text: str ) -> list[ str ]:
		"""
			Purpose:
			--------
			Tokenizes text into words for diagnostics.
			
			Parameters:
			-----------
			text : str
				The input text to tokenize.
			
			Returns:
			--------
			list[str]
				A list of non-empty tokens.
		"""
		if not isinstance( text, str ) or not text.strip( ):
			return [ ]
		
		try:
			parts = word_tokenize( text )
		except LookupError:
			parts = re.findall( r"\b\w+\b", text, flags=re.UNICODE )
		
		return [ p for p in parts if isinstance( p, str ) and p.strip( ) ]
	
	# ------------------------------------------------------------------
	# Fixed vector-space schema
	# ------------------------------------------------------------------
	dimensions = [
			'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7',
			'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14'
	]
	
	# ------------------------------------------------------------------
	# Canonical diagnostics state
	# ------------------------------------------------------------------
	if isinstance( df_frequency, pd.DataFrame ) and not df_frequency.empty:
		if 'Word' in df_frequency.columns and 'Frequency' in df_frequency.columns:
			st.session_state.df_token_frequency = df_frequency.rename(
				columns={ 'Word': 'Token' }
			).copy( )
		else:
			st.session_state.df_token_frequency = df_frequency.copy( )
	else:
		st.session_state.df_token_frequency = None
	
	# Sentence diagnostics must come from a boundary-preserving source.
	diagnostic_text = None
	
	if isinstance( raw_text, str ) and raw_text.strip( ):
		diagnostic_text = raw_text
	elif isinstance( processed_text, str ) and processed_text.strip( ):
		diagnostic_text = processed_text
	
	sentences = _safe_sent_tokenize( diagnostic_text ) if diagnostic_text else [ ]
	st.session_state.sentences = sentences if sentences else None
	
	if sentences:
		sentence_rows = [ _safe_word_tokenize( s ) for s in sentences ]
		sentence_rows = [ pad_or_trim_row( row, size=len( dimensions ) ) for row in sentence_rows ]
		
		if sentence_rows:
			st.session_state.df_sentence_tokens = pd.DataFrame(
				sentence_rows,
				columns=dimensions
			)
		else:
			st.session_state.df_sentence_tokens = None
	else:
		st.session_state.df_sentence_tokens = None
	
	# ------------------------------------------------------------------
	# LEFT COLUMN — Chunked Data
	# ------------------------------------------------------------------
	with line_col:
		st.text( 'Chunked Data' )
		
		if isinstance( processed_text, str ) and processed_text.strip( ):
			processor = TextParser( )
			lines = processor.split_sentences( text=processed_text, size=15 )
			st.session_state.lines = lines
			
			st.data_editor(
				pd.DataFrame( lines, columns=[ 'Processed Text' ] ),
				num_rows='dynamic',
				width='stretch',
				height='stretch'
			)
		else:
			st.info( 'Run preprocessing first' )
	
	# ------------------------------------------------------------------
	# RIGHT COLUMN — Vector Space View
	# ------------------------------------------------------------------
	with chunk_col:
		if isinstance( lines, (list, tuple) ) and isinstance( dimensions, (list, tuple) ):
			st.text( f"Vector Space: {len( lines ) * len( dimensions ):,}" )
		else:
			st.caption( "Vector space not available yet." )
		
		if isinstance( processed_text, str ) and processed_text.strip( ):
			if not isinstance( lines, list ) or not lines:
				processor = TextParser( )
				lines = processor.split_sentences( text=processed_text, size=15 )
				st.session_state.lines = lines
			
			_chunks = [ (l.split( ) if isinstance( l, str ) else [ ]) for l in lines ]
			_chunks = [ pad_or_trim_row( r, size=len( dimensions ) ) for r in _chunks ]
			st.session_state.chunked_documents = _chunks
			df_chunks = pd.DataFrame( _chunks, columns=dimensions )
			st.session_state.df_chunks = df_chunks
			
			st.data_editor(
				df_chunks,
				num_rows='dynamic',
				width='stretch',
				height='stretch'
			)
		else:
			st.info( 'Run preprocessing first' )
	
	# ------------------------------------------------------------------
	# Existing trailing block
	# ------------------------------------------------------------------
	documents = st.session_state.get( 'documents' )
	data_connection = st.session_state.get( 'data_connection' )
	loader_name = st.session_state.get( 'active_loader' )
	
	if st.session_state.documents is None:
		st.warning( 'No documents loaded. Please load documents first.' )
	elif loader_name is None:
		st.warning( 'No active loader found.' )
	else:
		chunk_modes = cfg.CHUNKABLE_LOADERS.get( loader_name )
	
	if chunk_modes is None:
		st.info( f'Chunking is not supported for loader: {loader_name}' )
	
	# ======================================================================================
	# Diagnostics — Token & Sentence Distributions
	# ======================================================================================
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	st.subheader( 'Tokenization Diagnostics' )
	
	row1_col1, row1_col2 = st.columns(
		[ 0.5, 0.5 ],
		border=True
	)
	
	# ------------------------------------------------------------------
	# Top-N Token Frequency Histogram
	# ------------------------------------------------------------------
	with row1_col1:
		st.caption( 'Top-N Token Frequency Distribution' )
		
		df_token_frequency = st.session_state.get( 'df_token_frequency' )
		
		if isinstance( df_token_frequency, pd.DataFrame ) and not df_token_frequency.empty:
			top_n = st.slider(
				'Top-N Tokens',
				min_value=10,
				max_value=100,
				value=30,
				step=10,
				key='token_freq_top_n'
			)
			
			df_top = df_token_frequency.sort_values(
				by='Frequency',
				ascending=False
			).head( top_n )
			
			st.bar_chart(
				df_top.set_index( 'Token' )[ 'Frequency' ],
				use_container_width=True
			)
		else:
			st.info( 'Token frequency data not available.' )
	
	# ------------------------------------------------------------------
	# Sentence Length Distribution
	# ------------------------------------------------------------------
	with row1_col2:
		st.caption( 'Sentence Length Distribution (Tokens per Sentence)' )
		
		sentences = st.session_state.get( 'sentences' )
		
		if isinstance( sentences, list ) and sentences:
			sentence_lengths = [ len( _safe_word_tokenize( s ) ) for s in sentences ]
			sentence_lengths = [ n for n in sentence_lengths if isinstance( n, int ) and n > 0 ]
			
			if sentence_lengths:
				df_sentence_lengths = pd.Series( sentence_lengths ).value_counts( ).sort_index( )
				df_sentence_lengths = df_sentence_lengths.rename_axis(
					'Tokens per Sentence'
				).to_frame( 'Sentence Count' )
				
				st.bar_chart(
					df_sentence_lengths,
					use_container_width=True
				)
			else:
				st.info( 'No valid sentence lengths computed.' )
		else:
			st.info( 'Sentence data not available.' )
	
	# ======================================================================================
	# Diagnostics — Sparsity & Embedding Readiness
	# ======================================================================================
	row2_col1, row2_col2 = st.columns(
		[ 0.5, 0.5 ],
		border=True
	)
	
	# ------------------------------------------------------------------
	# Padding / Sparsity Analysis (D0–D14)
	# ------------------------------------------------------------------
	with row2_col1:
		st.caption( 'Token Grid Sparsity (Padding Analysis)' )
		
		df_sentence_tokens = st.session_state.get( 'df_sentence_tokens' )
		
		if isinstance( df_sentence_tokens, pd.DataFrame ) and not df_sentence_tokens.empty:
			total_cells = df_sentence_tokens.shape[ 0 ] * df_sentence_tokens.shape[ 1 ]
			empty_cells = (df_sentence_tokens == '').sum( ).sum( )
			filled_cells = total_cells - empty_cells
			
			padding_ratio = empty_cells / total_cells if total_cells > 0 else 0.0
			fill_ratio = filled_cells / total_cells if total_cells > 0 else 0.0
			
			m1, m2, m3 = st.columns( 3 )
			m1.metric( 'Total Cells', f'{total_cells:,}' )
			m2.metric( 'Filled Cells', f'{filled_cells:,}' )
			m3.metric( 'Padding %', f'{padding_ratio:.1%}' )
			
			st.progress( fill_ratio )
		else:
			st.info( 'Sentence token grid not available.' )
	
	# ------------------------------------------------------------------
	# Embedding Readiness Scorecard
	# ------------------------------------------------------------------
	with row2_col2:
		st.caption( 'Embedding Readiness Scorecard' )
		
		tokens = st.session_state.get( 'tokens' )
		sentences = st.session_state.get( 'sentences' )
		token_counts = (
				Counter( tokens )
				if isinstance( tokens, list ) and tokens
				else None
		)
		
		if token_counts:
			total_tokens = len( tokens )
			unique_tokens = len( token_counts )
			hapax_count = sum( 1 for c in token_counts.values( ) if c == 1 )
			hapax_ratio = hapax_count / unique_tokens if unique_tokens > 0 else 0.0
			
			sentence_lengths = (
					[ len( _safe_word_tokenize( s ) ) for s in sentences ]
					if isinstance( sentences, list ) and sentences
					else [ ]
			)
			sentence_lengths = [ n for n in sentence_lengths if isinstance( n, int ) and n > 0 ]
			avg_sentence_len = (
					sum( sentence_lengths ) / len( sentence_lengths )
					if sentence_lengths
					else 0.0
			)
			
			r1, r2 = st.columns( 2 )
			r1.metric( 'Total Tokens', f'{total_tokens:,}' )
			r2.metric( 'Unique Tokens', f'{unique_tokens:,}' )
			
			r3, r4 = st.columns( 2 )
			r3.metric( 'Avg Tokens / Sentence', f'{avg_sentence_len:.1f}' )
			r4.metric( 'Hapax Ratio', f'{hapax_ratio:.1%}' )
			
			st.caption(
				'Lower padding and moderate hapax ratios generally yield more stable embeddings.'
			)
		else:
			st.info( 'Token readiness metrics unavailable.' )

# ======================================================================================
# Tab — Tensor Embeddings
# ======================================================================================
with tabs[ 4 ]:
	import pandas as pd
	
	def project_chunks_for_embedding( chunks: list ) -> list[ str ]:
		"""
		
			Purpose:
			--------
			Project chunked inputs into a clean list of embedding-ready strings.
		
			Guarantees:
			-----------
			- Output is strictly List[str]
			- No None, type, NaN, or non-string objects
			- No empty or whitespace-only strings
		
			Parameters:
			-----------
			chunks : list
				Input chunks (token grids, strings, or mixed structures)
		
			Returns:
			--------
			list[str]
				Clean embedding documents
			
		"""
		texts = [ ]
		if not isinstance( chunks, list ):
			return texts
		for c in chunks:
			if isinstance( c, list ):
				tokens = [ t for t in c if isinstance( t, str ) and t.strip( ) ]
				if tokens is not None:
					texts.append( " ".join( tokens ) )
			elif isinstance( c, str ):
				if c.strip( ):
					texts.append( c.strip( ) )
		return texts
	
	processed_text = st.session_state.get( "processed_text" )
	embeddings = st.session_state.get( "embeddings" )
	chunks = st.session_state.get( "chunks" )
	chunked_documents = st.session_state.get( "chunked_documents" )
	
	# ------------------------------------------------------------------
	# Normalize embedding output state
	# ------------------------------------------------------------------
	if not isinstance( st.session_state.get( "df_embedding_output" ), pd.DataFrame ):
		st.session_state.df_embedding_output = pd.DataFrame( )
	
	if "embedding_documents" not in st.session_state:
		st.session_state.embedding_documents = None
	
	def k( name: str ) -> str:
		return f"emb__{name}"
	
	def resolve_texts( source: str ) -> list[ str ]:
		if source == "Processed Text":
			if isinstance( processed_text, str ) and processed_text.strip( ):
				return [ processed_text.strip( ) ]
			return [ ]
		elif source == "Chunked Documents":
			if isinstance( chunked_documents, list ) and chunked_documents:
				return project_chunks_for_embedding( chunked_documents )
			return [ ]
		else:
			return [ ]
	
	# ------------------------------------------------------------------
	# Layout
	# ------------------------------------------------------------------
	left, right = st.columns( [ 1,
	                            1.5 ], border=True )
	
	# ==================================================
	# LEFT COLUMN — Providers + source selection
	# ==================================================
	with left:
		st.markdown( "##### Embedding Providers" )
		embedding_source = st.radio( 'Text Source', options=[ 'Processed Text',
		                                                      'Chunked Documents' ],
			horizontal=True, key=k( 'text_source' ), )
		
		st.session_state.embedding_source = embedding_source
		texts = resolve_texts( embedding_source )
		has_texts = bool( texts )
		
		st.caption( f"Texts to embed: {len( texts ) if texts else 0:,}" )
		
		# --------------------------------------------------
		# Derived embedding input dataframe (for right column display)
		# --------------------------------------------------
		df_embedding_input = (pd.DataFrame( {
				"text": texts } )
		                      if has_texts
		                      else pd.DataFrame( columns=[ "text" ] ))
		
		if not has_texts:
			st.info( "No text available. Run processing or chunking first." )
		
		# --------------------------------------------------
		# Shared save helpers
		# --------------------------------------------------
		def can_save_output( ) -> bool:
			return (isinstance( st.session_state.get( "df_embedding_output" ), pd.DataFrame )
			        and not st.session_state.df_embedding_output.empty)
		
		# ==================================================
		# 🧠 OpenAI
		# ==================================================
		with st.expander( "🧠 OpenAI Embeddings", expanded=False ):
			model = st.selectbox( "Model", options=cfg.GPT_MODELS, key=k( "openai_model" ), )
			col_run, col_clear, col_save = st.columns( 3 )
			run = col_run.button( "Embed", key=k( "openai_embed" ),
				use_container_width=True, disabled=not has_texts, )
			
			clear = col_clear.button( "Clear", key=k( "openai_clear" ),
				use_container_width=True, disabled=st.session_state.df_embedding_output.empty, )
			
			if can_save_output( ):
				col_save.download_button( "Save CSV",
					data=st.session_state.df_embedding_output.to_csv( index=False ),
					file_name="openai_embeddings.csv", mime="text/csv",
					use_container_width=True, key=k( "openai_save" ), )
			else:
				col_save.button( "Save CSV", disabled=True,
					use_container_width=True, key=k( "openai_save_disabled" ), )
			
			if clear:
				st.session_state.embeddings = None
				st.session_state.embedding_documents = None
				st.session_state.df_embedding_output = pd.DataFrame( )
				st.session_state.embedding_provider = None
				st.session_state.embedding_model = None
				st.success( "Embeddings cleared." )
			
			if run and has_texts:
				with st.spinner( "Embedding with OpenAI..." ):
					embedder = GPT( )
					vectors = embedder.embed( texts, model=model )
					
					# Store output separately (do NOT overwrite chunked_documents)
					df_out = pd.DataFrame(
						{
								"provider": "OpenAI",
								"model": model,
								"row_index": range( len( texts ) ),
								"text": texts,
								"embedding": vectors,
						}
					)
					
					st.session_state.df_embedding_output = df_out
					st.session_state.embedding_documents = df_out.to_dict( "records" )
					st.session_state.embeddings = vectors
					st.session_state.embedding_provider = "OpenAI"
					st.session_state.embedding_model = model
					
					st.success( f"Generated {len( vectors )} embedding(s)." )
		
		# ==================================================
		# ✨ Gemini
		# ==================================================
		with st.expander( "✨ Gemini Embeddings", expanded=False ):
			model = st.selectbox(
				"Model",
				options=cfg.GEMINI_MODELS,
				key=k( "gemini_model" ),
				disabled=not has_texts,
			)
			
			task = st.selectbox(
				"Task Type",
				options=Gemini( ).task_options,
				key=k( "gemini_task" ),
				disabled=not has_texts,
				help="Required to determine embedding intent.",
			)
			
			dimensions = st.number_input(
				"Dimensions",
				min_value=128,
				max_value=2048,
				step=128,
				value=768,
				key=k( "gemini_dimensions" ),
				disabled=not has_texts,
				help="Optional. Must be supported by the selected model.",
			)
			
			col_run, col_clear, col_save = st.columns( 3 )
			can_embed = bool( has_texts and model and task )
			
			run = col_run.button(
				"Embed",
				key=k( "gemini_embed" ),
				use_container_width=True,
				disabled=not can_embed,
			)
			
			clear = col_clear.button(
				"Clear",
				key=k( "gemini_clear" ),
				use_container_width=True,
				disabled=st.session_state.df_embedding_output.empty,
			)
			
			if can_save_output( ):
				col_save.download_button(
					"Save CSV",
					data=st.session_state.df_embedding_output.to_csv( index=False ),
					file_name="gemini_embeddings.csv",
					mime="text/csv",
					use_container_width=True,
					key=k( "gemini_save" ),
				)
			else:
				col_save.button(
					"Save CSV",
					disabled=True,
					use_container_width=True,
					key=k( "gemini_save_disabled" ),
				)
			
			if clear:
				st.session_state.embeddings = None
				st.session_state.embedding_documents = None
				st.session_state.df_embedding_output = pd.DataFrame( )
				st.session_state.embedding_provider = None
				st.session_state.embedding_model = None
				st.success( "Embeddings cleared." )
			
			if run and can_embed:
				with st.spinner( "Embedding with Gemini..." ):
					embedder = Gemini( )
					vectors = embedder.embed(
						texts,
						task=task,
						model=model,
						dimensions=dimensions,
					)
					
					df_out = pd.DataFrame(
						{
								"provider": "Gemini",
								"model": model,
								"task": task,
								"dimensions": dimensions,
								"row_index": range( len( texts ) ),
								"text": texts,
								"embedding": vectors,
						}
					)
					
					st.session_state.df_embedding_output = df_out
					st.session_state.embedding_documents = df_out.to_dict( "records" )
					st.session_state.embeddings = vectors
					st.session_state.embedding_provider = "Gemini"
					st.session_state.embedding_model = model
					
					st.success( f"Generated {len( vectors )} embedding(s)." )
		
		# ==================================================
		# ⚡ Groq
		# ==================================================
		with st.expander( "⚡ Groq Embeddings", expanded=False ):
			model = st.selectbox(
				"Model",
				options=cfg.GROK_MODELS,
				key=k( "groq_model" ),
				disabled=not has_texts,
			)
			
			st.caption(
				"Groq embeddings use provider-defined geometry. "
				"No task type or dimensionality parameters are exposed."
			)
			
			col_run, col_clear, col_save = st.columns( 3 )
			can_embed = bool( has_texts and model )
			
			run = col_run.button(
				"Embed",
				key=k( "groq_embed" ),
				use_container_width=True,
				disabled=not can_embed,
			)
			
			clear = col_clear.button(
				"Clear",
				key=k( "groq_clear" ),
				use_container_width=True,
				disabled=st.session_state.df_embedding_output.empty,
			)
			
			if can_save_output( ):
				col_save.download_button(
					"Save CSV",
					data=st.session_state.df_embedding_output.to_csv( index=False ),
					file_name="groq_embeddings.csv",
					mime="text/csv",
					use_container_width=True,
					key=k( "groq_save" ),
				)
			else:
				col_save.button(
					"Save CSV",
					disabled=True,
					use_container_width=True,
					key=k( "groq_save_disabled" ),
				)
			
			if clear:
				st.session_state.embeddings = None
				st.session_state.embedding_documents = None
				st.session_state.df_embedding_output = pd.DataFrame( )
				st.session_state.embedding_provider = None
				st.session_state.embedding_model = None
				st.success( "Embeddings cleared." )
			
			if run and can_embed:
				with st.spinner( "Embedding with Groq..." ):
					embedder = Grok( )
					vectors = embedder.embed( texts, model=model )
					
					df_out = pd.DataFrame(
						{
								"provider": "Groq",
								"model": model,
								"row_index": range( len( texts ) ),
								"text": texts,
								"embedding": vectors,
						}
					)
					
					st.session_state.df_embedding_output = df_out
					st.session_state.embedding_documents = df_out.to_dict( "records" )
					st.session_state.embeddings = vectors
					st.session_state.embedding_provider = "Groq"
					st.session_state.embedding_model = model
					
					st.success( f"Generated {len( vectors )} embedding(s)." )
	
	# ==================================================
	# RIGHT COLUMN — Embedding input (read-only) + results below
	# ==================================================
	with right:
		st.markdown( "##### Embedding Documents" )
		
		# texts already resolved above; do not recompute
		df_embedding_input = (
				pd.DataFrame( {
						"text": texts } )
				if texts
				else pd.DataFrame( columns=[ "text" ] )
		)
		
		st.data_editor(
			df_embedding_input,
			use_container_width=True,
			hide_index=True,
			disabled=True,
			key=k( "embedding_input_view" ),
		)
	
	# --------------------------------------------------
	# BELOW BOTH COLUMNS — Embedding Results (read-only)
	# --------------------------------------------------
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	
	st.markdown( "##### Embedding Results" )
	if st.session_state.df_embedding_output.empty:
		st.info( "No embeddings generated yet." )
	else:
		st.data_editor(
			st.session_state.df_embedding_output,
			use_container_width=True,
			hide_index=True,
			disabled=True,
			key=k( "embedding_output_view" ),
		)
	
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	
	# ======================================================================================
	# Tensor Embedding — Dimensionality Reduction Diagnostics (t-SNE / UMAP)
	# ======================================================================================
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	st.subheader( 'Embedding Diagnostics (t-SNE / UMAP)' )
	
	embeddings = st.session_state.get( 'embeddings' )
	chunked_documents = st.session_state.get( 'chunked_documents' )
	
	# ------------------------------------------------------------------
	# Guard: embeddings availability
	# ------------------------------------------------------------------
	if not isinstance( embeddings, (list, np.ndarray) ):
		st.info( 'Generate embeddings to enable dimensionality reduction diagnostics.' )
	else:
		emb_array = np.asarray( embeddings )
		if emb_array.ndim != 2 or emb_array.shape[ 0 ] < 5:
			st.warning( 'At least 5 valid embedding vectors are required for meaningful diagnostics.' )
		else:
			ctrl_col1, ctrl_col2, ctrl_col3 = st.columns( 3, border=True )
			with ctrl_col1:
				reduction_method = st.selectbox( 'Reduction Method', options=[ 't-SNE',
				                                                               'UMAP' ],
					key='embedding_reduction_method' )
			
			with ctrl_col2:
				if reduction_method == 't-SNE':
					perplexity = st.slider(
						't-SNE Perplexity',
						min_value=5,
						max_value=min( 50, emb_array.shape[ 0 ] - 1 ),
						value=30,
						step=5,
						key='tsne_perplexity'
					)
				else:
					n_neighbors = st.slider(
						'UMAP Neighbors',
						min_value=5,
						max_value=min( 50, emb_array.shape[ 0 ] - 1 ),
						value=15,
						step=5,
						key='umap_neighbors'
					)
			
			with ctrl_col3:
				random_state = st.number_input(
					'Random Seed',
					min_value=0,
					value=42,
					step=1,
					key='embedding_reduction_seed'
				)
			
			# ------------------------------------------------------------------
			# Dimensionality reduction (diagnostic only)
			# ------------------------------------------------------------------
			reduced = None
			error_message = None
			
			try:
				if reduction_method == 't-SNE':
					from sklearn.manifold import TSNE
					
					reducer = TSNE(
						n_components=2,
						perplexity=perplexity,
						random_state=random_state,
						init='pca',
						learning_rate='auto'
					)
					
					reduced = reducer.fit_transform( emb_array )
				
				else:
					import umap
					
					reducer = umap.UMAP(
						n_components=2,
						n_neighbors=n_neighbors,
						random_state=random_state,
						min_dist=0.1
					)
					
					reduced = reducer.fit_transform( emb_array )
			
			except Exception as ex:
				error_message = str( ex )
			
			# ------------------------------------------------------------------
			# Visualization
			# ------------------------------------------------------------------
			if error_message:
				st.error( f'Dimensionality reduction failed: {error_message}' )
			
			elif isinstance( reduced, np.ndarray ) and reduced.shape[ 1 ] == 2:
				df_reduced = pd.DataFrame(
					reduced,
					columns=[ 'X',
					          'Y' ]
				)
				
				df_reduced[ 'Chunk Index' ] = range( len( df_reduced ) )
				
				if isinstance( chunked_documents, list ):
					df_reduced[ 'Preview' ] = [
							(d[ :120 ] + '…')
							if isinstance( d, str ) and len( d ) > 120
							else str( d )
							for d in chunked_documents
					]
				else:
					df_reduced[ 'Preview' ] = ''
				
				# ----------------------------
				# Scatter plot container
				# ----------------------------
				chart_container = st.container( )
				with chart_container:
					st.caption(
						'Each point represents one embedded chunk. '
						'Proximity reflects semantic similarity.'
					)
					st.scatter_chart(
						df_reduced,
						x='X',
						y='Y',
						size=60,
						use_container_width=True
					)
				
				# ----------------------------
				# Tabular inspection (optional)
				# ----------------------------
				with st.expander( 'View Reduced Coordinates (Table)', expanded=False ):
					st.data_editor(
						df_reduced,
						use_container_width=True,
						num_rows='dynamic'
					)
				
				# ----------------------------
				# Interpretation notes
				# ----------------------------
				with st.expander( 'ℹ️ Interpretation Notes', expanded=False ):
					st.markdown(
						"""
	- **t-SNE** emphasizes local neighborhoods; global distances are not meaningful.
	- **UMAP** preserves more global structure and is typically more stable.
	- Heavy overlap may indicate:
	  - chunk sizes are too small
	  - boilerplate text dominance
	  - insufficient preprocessing
	- These diagnostics are **read-only** and are not persisted.
						"""
					)

# ======================================================================================
# Tab — Vector Database (sqlite-vec)
# ======================================================================================
with tabs[ 5 ]:
	st.subheader( 'Vector Database (sqlite-vec)' )
	
	# ------------------------------------------------------------------
	# Required upstream state
	# ------------------------------------------------------------------
	embeddings = st.session_state.get( 'embeddings' )
	chunked_documents = st.session_state.get( 'chunked_documents' )
	embedding_model = st.session_state.get( 'embedding_model' )
	embedding_provider = st.session_state.get( 'embedding_provider' )
	
	# ------------------------------------------------------------------
	# Guard: embeddings must exist before continuing
	# ------------------------------------------------------------------
	if not (isinstance( embeddings, list ) and isinstance( chunked_documents, list ) and
	        embedding_model and embedding_provider):
		st.info( 'Generate embeddings before persisting to the vector database.' )
		st.stop( )
	
	# ------------------------------------------------------------------
	# Derive vector metadata (SAFE)
	# ------------------------------------------------------------------
	emb_array = np.asarray( embeddings )
	
	if emb_array.ndim == 1:
		emb_array = emb_array.reshape( 1, -1 )
	
	if emb_array.ndim != 2 or emb_array.shape[ 0 ] < 1:
		st.error( 'Invalid embeddings array.' )
		st.stop( )
	
	dim = emb_array.shape[ 1 ]
	
	document_name = st.text_input(
		'Document / Collection Name',
		value='default_document'
	)
	
	table_name = (
			f"{document_name}__"
			f"{embedding_provider}__"
			f"{embedding_model}__"
			f"{dim}"
	)
	
	st.caption( f'Vector Table: `{table_name}`' )
	
	# ------------------------------------------------------------------
	# Database connection
	# ------------------------------------------------------------------
	db_path = st.text_input(
		'SQLite Database Path',
		value='vectors.db'
	)
	
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	
	# ------------------------------------------------------------------
	# Actions
	# ------------------------------------------------------------------
	col_create, col_insert, col_delete = st.columns( 3 )
	
	with col_create:
		if st.button( 'Create Vector Table' ):
			conn = sqlite3.connect( db_path )
			conn.enable_load_extension( True )
			sqlite_vec.load( conn )
			
			SQLiteVec.create_table(
				conn,
				table_name=table_name,
				dimension=dim
			)
			
			conn.close( )
			st.success( f'Created vector table `{table_name}`.' )
	
	with col_insert:
		if st.button( 'Insert Embeddings' ):
			conn = sqlite3.connect( db_path )
			conn.enable_load_extension( True )
			sqlite_vec.load( conn )
			
			vector_store = SQLiteVec(
				connection=conn,
				table_name=table_name,
				embedding=SentenceTransformerEmbeddings(
					model_name=embedding_model
				)
			)
			
			vector_store.add_texts(
				texts=chunked_documents,
				embeddings=embeddings
			)
			
			conn.close( )
			st.success(
				f'Inserted {len( embeddings )} embeddings into `{table_name}`.'
			)
	
	with col_delete:
		if st.button( 'Drop Vector Table', type='secondary' ):
			conn = sqlite3.connect( db_path )
			cur = conn.cursor( )
			cur.execute( f'DROP TABLE IF EXISTS {table_name}' )
			conn.commit( )
			conn.close( )
			st.warning( f'Dropped vector table `{table_name}`.' )
	
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	
	# ------------------------------------------------------------------
	# Verification panel
	# ------------------------------------------------------------------
	if st.checkbox( 'Inspect Vector Table' ):
		conn = sqlite3.connect( db_path )
		df_preview = pd.read_sql_query(
			f'SELECT * FROM {table_name} LIMIT 5',
			conn
		)
		conn.close( )
		
		st.data_editor(
			df_preview,
			use_container_width=True,
			num_rows='dynamic'
		)
	
	# ======================================================================================
	# Similarity Search (sqlite-vec)
	# ======================================================================================
	st.subheader( 'Similarity Search' )
	
	query_text = st.text_area(
		'Query Text',
		placeholder='Enter text to search for semantically similar chunks…',
		height=100
	)
	
	top_k = st.slider(
		'Top-K Results',
		min_value=1,
		max_value=20,
		value=5,
		step=1
	)
	
	similarity_threshold = st.slider(
		'Minimum Similarity Threshold',
		min_value=0.0,
		max_value=1.0,
		value=0.0,
		step=0.01,
		help='Only results with similarity ≥ threshold will be shown.'
	)
	
	if not query_text.strip( ):
		st.info( 'Enter a query to run similarity search.' )
		results = None
	else:
		try:
			conn = sqlite3.connect( db_path )
			conn.enable_load_extension( True )
			sqlite_vec.load( conn )
			
			embedding_fn = SentenceTransformerEmbeddings(
				model_name=embedding_model
			)
			
			vector_store = SQLiteVec(
				connection=conn,
				table_name=table_name,
				embedding=embedding_fn
			)
			
			results = vector_store.similarity_search_with_score(
				query=query_text,
				k=top_k
			)
			
			conn.close( )
		
		except Exception as ex:
			st.error( f'Similarity search failed: {ex}' )
			results = None
	
	# ------------------------------------------------------------------
	# Results Rendering (with similarity threshold)
	# ------------------------------------------------------------------
	if results:
		filtered_results = [
				(doc, score)
				for (doc, score) in results
				if score >= similarity_threshold
		]
		
		st.caption(
			f'Results shown with similarity ≥ {similarity_threshold:.2f}. '
			f'{len( filtered_results )} of {len( results )} results retained.'
		)
		
		if not filtered_results:
			st.warning(
				'No results met the selected similarity threshold. '
				'Try lowering the threshold or increasing Top-K.'
			)
		
		for rank, (doc, score) in enumerate( filtered_results, start=1 ):
			with st.expander(
					f'#{rank} — Similarity Score: {score:.4f}',
					expanded=(rank == 1)
			):
				st.text_area(
					'Chunk Text',
					doc.page_content,
					height=200,
					disabled=True
				)
	else:
		st.info( 'No results to display.' )
	
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )

	