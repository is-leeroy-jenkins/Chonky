'''
  ******************************************************************************************
	  Assembly:                Chonky
	  Filename:                processing.py
	  Author:                  Terry D. Eppler
	  Created:                 05-31-2022

	  Last Modified By:        Terry D. Eppler
	  Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="processing.py" company="Terry D. Eppler">

		 Boo is a df analysis tool that integrates various Generative AI, Text-Processing, and
		 Machine-Learning algorithms for federal analysts.
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
	processing.py
  </summary>
  ******************************************************************************************
  '''
from __future__ import annotations
from boogr import Error, ErrorDialog
from bs4 import BeautifulSoup
import chromadb
from chromadb.config import Settings
from collections import Counter
import html
import glob
import json
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools.base import Tool
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools.base import Tool
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader

try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    UnstructuredExcelLoader,
    WebBaseLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk import FreqDist, ConditionalFreqDist
from nltk.corpus import stopwords, wordnet, words
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import os
import pandas as pd
from pandas import DataFrame
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
import re
import spacy
import sqlite3
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from typing import List, Optional, Dict
import tiktoken
from tiktoken.core import Encoding
import unicodedata


try:
	nltk.data.find( 'tokenizers/punkt' )
except LookupError:
	nltk.download( 'punkt' )
	nltk.download( 'punkt_tab' )
	nltk.download( 'stopwords' )
	nltk.download( 'wordnet' )
	nltk.download( 'omw-1.4' )
	nltk.download( 'words' )

def throw_if( name: str, value: object ):
	if value is None:
		raise Exception( f'Argument "{name}" cannot be empty!' )

class Loader( ):
	'''

		Purpose:
		--------
		Base class providing shared utilities for concrete loader wrappers.
		Encapsulates file validation, path resolution, and document splitting.

	'''
	documents: Optional[ List[ Document ] ]
	file_path: Optional[ str ]
	pattern: Optional[ str ]
	expanded: Optional[ List[ str ] ]
	candidates: Optional[ List[ str ] ]
	resolved: Optional[ List[ str ] ]
	splitter: Optional[ RecursiveCharacterTextSplitter ]
	chunk_size: Optional[ int ]
	overlap_amount: Optional[ int ]
	
	
	def __init__( self ) -> None:
		self.documents = [ ]
		self.candidates = [ ]
		self.resolved = [ ]
		self.expanded = [ ]
	
	def _verify_exists( self, path: str ) -> str | None:
		'''

			Purpose:
			--------
			Ensure the given file path exists.

			Parameters:
			-----------
			path (str): Path to a file on disk.

			Returns:
			--------
			str: The validated file path.

		'''
		try:
			throw_if( 'path', path )
			self.file_path = path
			if not os.path.isfile( self.file_path ):
				raise FileNotFoundError( f'File not found: {self.file_path}' )
			else:
				self.file_path = path
			return self.file_path
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	def _resolve_paths( self, pattern: str ) -> List[ str ] | None:
		'''

			Purpose:
			--------
			Normalize a string glob pattern or a list of paths to a list of real file paths.

			Parameters:
			-----------
			pattern (str | List[str]): Path pattern or list of file paths.

			Returns:
			--------
			List[str]: Validated list of file paths.

		'''
		try:
			throw_if( 'pattern', pattern )
			self.candidates.append( pattern )
			for p in self.candidates:
				if os.path.isfile( p ):
					self.resolved.append( p )
				else:
					for m in glob.glob( p ):
						if os.path.isfile( m ):
							self.resolved.append( m )
			
			if not self.resolved:
				raise FileNotFoundError( f'No files matched or existed for input: {pattern}' )
			return sorted( set( self.resolved ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	def _split_documents( self, docs: List[ Document ], chunk: int=1000, overlap: int=200 ) -> \
	List[ Document ] | None:
		'''

			Purpose:
			--------
			Split long Document objects into smaller chunks for better token management.

			Parameters:
			-----------
			docs (List[Document]): Input LangChain Document objects.
			chunk(int): Max characters in each chunk.
			overlap(int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Re-chunked list of Document objects.

		'''
		try:
			throw_if( 'docs', docs )
			self.documents = docs
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.splitter = RecursiveCharacterTextSplitter( chunk_size=self.chunk_size,
				chunk_overlap=self.overlap_amount )
			return self.splitter.split_documents( docs )
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'Loader'
			exception.method = '_split_documents( self, docs, chunk, overlap ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class Processor( ):
	'''
		
		Purpose:
		Base class for processing classes
		
	'''
	lemmatizer: Optional[ WordNetLemmatizer ]
	stemmer: Optional[ PorterStemmer ]
	file_path: Optional[ str ]
	lowercase: Optional[ str ]
	normalized: Optional[ str ]
	lemmatized: Optional[ str ]
	tokenized: Optional[ str ]
	encoding: Optional[ Encoding ]
	nlp: Optional[ Language ]
	parts_of_speech: Optional[ List[ Tuple[ str, str ] ] ]
	embedddings: Optional[ List[ np.ndarray ] ]
	chunk_size: Optional[ int ]
	corrected: Optional[ str ]
	raw_input: Optional[ str ]
	raw_html: Optional[ str ]
	raw_pages: Optional[ List[ str ] ]
	lines: Optional[ List[ str ] ]
	tokens: Optional[ List[ str ] ]
	lines: Optional[ List[ str ] ]
	files: Optional[ List[ str ] ]
	pages: Optional[ List[ str ] ]
	paragraphs: Optional[ List[ str ] ]
	ids: Optional[ List[ int ] ]
	cleaned_lines: Optional[ List[ str ] ]
	cleaned_tokens: Optional[ List[ str ] ]
	cleaned_pages: Optional[ List[ str ] ]
	cleaned_html: Optional[ str ]
	stop_words: Optional[ set ]
	vocabulary: Optional[ set ]
	corpus: Optional[ List[ List[ str ] ] ]
	removed: Optional[ List[ str ] ]
	frequency_distribution: Optional[ Dict ]
	conditional_distribution: Optional[ Dict ]
	
	def __init__( self ):
		self.lemmatizer = WordNetLemmatizer( )
		self.stemmer = PorterStemmer( )
		self.files = [ ]
		self.lines = [ ]
		self.tokens = List[ List[ str ] ]
		self.lines = [ ]
		self.pages = [ ]
		self.ids = [ ]
		self.paragraphs = List[ List[ str ] ]
		self.chunks = List[ List[ str ] ]
		self.chunk_size = 0
		self.cleaned_lines = [ ]
		self.cleaned_tokens = List[ List[ str ] ]
		self.cleaned_pages = List[ List[ str ] ]
		self.removed = [ ]
		self.raw_pages = List[ List[ str ] ]
		self.stop_words = set( )
		self.vocabulary = set( )
		self.frequency_distribution = { }
		self.conditional_distribution = { }
		self.encoding = None
		self.embedddings = None
		self.file_path = ''
		self.raw_input = ''
		self.normalized = ''
		self.lemmatized = ''
		self.tokenized = ''
		self.cleaned_text = ''
		self.cleaned_html = None
		self.corrected = None
		self.lowercase = None
		self.raw_html = None

# noinspection PyTypeChecker,PyArgumentList,DuplicatedCode
class Text( Processor ):
	'''

		Purpose:
		---------
		Class providing path preprocessing functionality

		Methods:
		--------
		split_lines( self, path: str ) -> list
		split_pages( self, path: str, delimit: str ) -> list
		collapse_whitespace( self, path: str ) -> str
		remove_punctuation( self, path: str ) -> str:
		remove_special( self, path: str, keep_spaces: bool ) -> str:
		remove_html( self, path: str ) -> str
		remove_errors( self, path: str ) -> str
		remove_markdown( self, path: str ) -> str
		remove_stopwords( self, path: str ) -> str
		remove_headers( self, pages, min: int=3 ) -> str
		normalize_text( path: str ) -> str
		lemmatize_tokens( words: List[ str ] ) -> str
		tokenize_text( path: str ) -> str
		tokenize_words( path: str ) -> List[ str ]
		tokenize_sentences( path: str ) -> str
		chunk_text( self, path: str, max: int=800 ) -> List[ str ]
		chunk_words( self, path: str, max: int=800, over: int=50 ) -> List[ str ]
		split_paragraphs( self, path: str ) -> List[ str ]
		compute_frequency_distribution( self, words: List[ str ], proc: bool=True ) -> List[ str ]
		compute_conditional_distribution( self, words: List[ str ], condition: str=None,
		proc: bool=True ) -> List[ str ]
		create_vocabulary( self, frequency, min: int=1 ) -> List[ str ]
		create_wordbag( words: List[ str ] ) -> dict
		create_word2vec( sentences: List[ str ], vector_size=100, window=5, min_count=1 ) ->
		Word2Vec
		create_tfidf( words: List[ str ], max_features=1000, prep=True ) -> tuple

	'''
	
	def __init__( self ):
		'''

			Purpose:
			---------
			Constructor for 'Text' objects

		'''
		super( ).__init__( )
		self.lemmatizer = WordNetLemmatizer( )
		self.stemmer = PorterStemmer( )
		self.lines = [ ]
		self.tokens = List[ List[ str ] ]
		self.lines = [ ]
		self.pages = List[ List[ str ] ]
		self.ids = [ ]
		self.paragraphs = List[ List[ str ] ]
		self.chunks = List[ List[ str ] ]
		self.chunk_size = 0
		self.cleaned_lines = [ ]
		self.cleaned_tokens = List[ List[ str ] ]
		self.cleaned_pages = List[ List[ str ] ]
		self.removed = [ ]
		self.raw_pages = [ ]
		self.stop_words = set( )
		self.vocabulary = [ ]
		self.frequency_distribution = { }
		self.conditional_distribution = { }
		self.encoding = None
		self.file_path = ''
		self.raw_input = ''
		self.normalized = ''
		self.lemmatized = ''
		self.tokenized = ''
		self.cleaned_text = ''
		self.cleaned_html = None
		self.corrected = None
		self.lowercase = None
		self.raw_html = None
		self.translator = None
		self.tokenizer = None
		self.vectorizer = None
	
	def __dir__( self ) -> List[ str ] | None:
		'''

			Purpose:
			---------
			Provides a list of strings representing class members.


			Parameters:
			-----------
			- self

			Returns:
			--------
			- List[ str ] | None

		'''
		return [ 'file_path', 'raw_input', 'raw_pages', 'normalized', 'lemmatized', 'tokenized',
			'corrected', 'cleaned_text', 'words', 'paragraphs', 'words', 'pages', 'chunks',
			'chunk_size', 'cleaned_pages', 'stop_words', 'cleaned_lines', 'removed', 'lowercase',
			'encoding', 'vocabulary', 'translator', 'lemmatizer', 'stemmer', 'tokenizer',
			'vectorizer', 'split_lines', 'split_pages', 'collapse_whitespace', 'remove_punctuation',
			'remove_special', 'remove_html', 'remove_markdown', 'remove_stopwords',
			'remove_headers', 'tiktokenize', 'normalize_text', 'tokenize_text', 'tokenize_words',
			'tokenize_sentences', 'chunk_text', 'chunk_words', 'chunk_files', 'create_wordbag',
			'create_word2vec', 'chunk_data', 'remove_errors',
			'create_tfidf', 'clean_files', 'convert_jsonl', 'conditional_distribution' ]
	
	def load_text( self, filepath: str ) -> str | None:
		"""
	`
			Purpose:
				Loads raw text from a file.
				
			Parameters:
				file_path (str): Path to the .txt file.
				
			Returns:
				str: Raw file content as string.`
			
		"""
		try:
			throw_if( 'filepath', filepath )
			if not os.path.exists( filepath ):
				raise FileNotFoundError( f'File not found: {filepath}' )
			else:
				self.file_path = filepath
			raw_input = open( self.file_path, mode='r', encoding='utf-8', errors='ignore' ).read()
			return raw_input
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'load_text( self, file_path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def collapse_whitespace( self, text: str ) -> str | None:
		"""

			Purpose:
			-----------
			Removes extra lines from the string 'text'.

			Parameters:
			-----------
			- text : str

			Returns:
			--------
			A string with:
				- Consecutive whitespace reduced to a single space
				- Leading/trailing spaces removed
				- Blank words removed

		"""
		try:
			throw_if( 'text', text )
			self.raw_input = text
			extra_lines = re.sub( r'[\t]+', '', self.raw_input )
			self.cleaned_lines = [ line for line in extra_lines ]
			return ''.join( self.cleaned_lines )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'collapse_whitespace( self, path: str ) -> str:'
			error = ErrorDialog( exception )
			error.show( )
	
	def compress_whitespace( self, text: str ) -> str | None:
		"""

			Purpose:
			-----------
			Removes extra spaces and blank words from the string 'text'.

			Parameters:
			-----------
			- text : str

			Returns:
			--------
			A string with:
				- Consecutive whitespace reduced to a single space
				- Leading/trailing spaces removed
				- Blank words removed

		"""
		try:
			throw_if( 'text', text )
			self.raw_input = text
			extra_spaces = re.sub( r'\r\n\t+', '', self.raw_input )
			self.cleaned_lines = [ line for line in extra_spaces ]
			return ''.join( self.cleaned_lines )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'collapse_whitespace( self, path: str ) -> str:'
			error = ErrorDialog( exception )
			error.show( )
	
	def remove_punctuation( self, text: str ) -> str:
		"""
		
			Purpose:
			--------
			Removes all punctuation characters from the input text using NLTK's tokenizer.
		
			This function tokenizes the input into words and filters out any tokens that
			are composed entirely of punctuation characters.
		
			Parameters
			----------
			text : str
				The raw input text from which to remove punctuation.
		
			Returns
			-------
			str
				A cleaned string with all punctuation removed and original word spacing preserved.
		
			Example
			-------
			remove_punctuation("Hello, world! How's it going?")
			'Hello world Hows it going'
			
		"""
		try:
			throw_if( 'text', text )
			_tokens = text.split( ' ' )
			self.cleaned_tokens = [ t for t in _tokens if t not in string.punctuation ]
			return ' '.join( self.cleaned_tokens )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'remove_punctuation( self, text: str ) -> str:'
			error = ErrorDialog( exception )
			error.show( )
	
	def normalize_text( self, text: str ) -> str | None:
		"""

			Purpose:
			-----------
			Converts the input 'text' to lower case

			This function:
			  - Converts the input 'text' to lower case

			Parameters:
			-----------
			- text : str
				The raw text

			Returns:
			--------
			- str
				A cleaned_lines path containing
				only letters, numbers, and spaces.

		"""
		try:
			throw_if( 'text', text )
			lower_cased = [ ]
			tokens = text.split( )
			for char in tokens:
				lower = char.lower( )
				lower_cased.append( lower )
			return ' '.join( lower_cased )
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'Text'
			exception.method = 'normalize_text( self, text: str ) -> str:'
			error = ErrorDialog( exception )
			error.show( )
	
	def remove_fragments( self, text: str ) -> str | None:
		"""

			Purpose:
			-----------
			Removes strings less that 4 chars in length


			Parameters:
			-----------
			- pages : str
				The raw path path path potentially
				containing special characters.

			Returns:
			--------
			- str
				A cleaned_lines path containing
				only letters, numbers, and spaces.

		"""
		try:
			throw_if( 'text', text )
			cleaned = [ ]
			fragments = text.split( ' ' )
			for char in fragments:
				if len( char) > 2:
					cleaned.append( char )
			return ' '.join( cleaned )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'remove_special( self, text: str ) -> str:'
			error = ErrorDialog( exception )
			error.show( )
	
	def remove_special( self, text: str ) -> str | None:
		"""

			Purpose:
			-----------
			Removes special characters from the path path path.

			This function:
			  - Retains only alphanumeric characters and whitespace
			  - Removes symbols like @, #, $, %, &, etc.
			  - Preserves letters, numbers, and spaces

			Parameters:
			-----------
			- pages : str
				The raw path path path potentially
				containing special characters.

			Returns:
			--------
			- str
				A cleaned_lines path containing
				only letters, numbers, and spaces.

		"""
		try:
			throw_if( 'text', text )
			cleaned = [ ]
			keepers = [ '(', ')', '$', '. ', '! ', '? ', ': ', '; ', '-',  ]
			tokens = text.split( ' ' )
			for char in tokens:
				if char.isalpha( ) or char.isnumeric( ) or char in keepers:
					cleaned.append( char )
			return ' '.join( cleaned )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'remove_special( self, text: str ) -> str:'
			error = ErrorDialog( exception )
			error.show( )
	
	def remove_html( self, text: str ) -> str | None:
		"""

			Purpose:
			--------
			Removes HTML tags from the path path path.

			This function:
			  - Parses the path as HTML
			  - Extracts and returns only the visible content without tags

			Parameters:
			-----------
			- pages : str
				The path path containing HTML tags.

			Returns:
			--------
			- str
				A cleaned_lines path with all HTML tags removed.

		"""
		try:
			throw_if( 'text', text )
			self.raw_html = text
			self.cleaned_html = BeautifulSoup( self.raw_html, 'html.parser' ).get_text( )
			return self.cleaned_html
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'remove_html( self, text: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def remove_markdown( self, text: str ) -> str | None:
		"""


			Purpose:
			-----------
			Removes Markdown syntax (e.g., *, #, [], etc.)

			Parameters:
			-----------
			- pages : str
				The formatted path pages.

			Returns:
			--------
			- str
				A cleaned_lines version of the pages with formatting removed.

		"""
		try:
			throw_if( 'text', text )
			self.raw_input = text
			self.cleaned_text = re.sub( r'\[.*?]\(.*?\)', '', text )
			self.corrected = re.sub( r'[`_*#~>-]', '', self.cleaned_text )
			_retval = re.sub( r'!\[.*?]\(.*?\)', '', self.corrected )
			return _retval
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'remove_markdown( self, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def remove_stopwords( self, text: str ) -> str | None:
		"""

			Purpose:
			-----------
			This function:
			  - Removes English stopwords from the path pages path.
			  - Tokenizes the path pages
			  - Removes common stopwords (e.g., "the", "is", "and", etc.)
			  - Returns the pages with only meaningful words


			Parameters:
			-----------
			- pages : str
				The text string.

			Returns:
			--------
			- str
				A text string without stopwords.

		"""
		try:
			throw_if( 'text', text )
			self.stop_words = set( stopwords.words( 'english' ) )
			_words = text.split( ' ' )
			tokens = [ t for t in _words ]
			self.cleaned_tokens = [ w for w in tokens if w not in self.stop_words ]
			self.cleaned_text = ' '.join( self.cleaned_tokens )
			return self.cleaned_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'remove_stopwords( self, text: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def remove_encodings( self, text: str ) -> str | None:
		"""

			Purpose:
			---------
			Cleans text of encoding artifacts by resolving HTML entities,
			Unicode escape sequences, and over-encoded byte strings.

			Parameters
			----------
			text : str
			Input string potentially containing encoded characters.

			Returns
			-------
			str
			Cleaned Unicode-normalized text.

		"""
		try:
			throw_if( 'text', text )
			self.raw_input = text
			# Decode HTML entities (&amp;, &quot;, &#8217;)
			_html = html.unescape( self.raw_input )
			# Normalize accents and symbols (e.g., é → e, ü → u)
			_norm = unicodedata.normalize( 'NFKC', _html )
			# Strip out stray control characters
			_chars = re.sub( r'[\x00-\x1F\x7F]', "", _norm )
			self.cleaned_text = _chars.strip( )
			return self.cleaned_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'remove_encodings( self, text: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def remove_headers( self, filepath: str, lines_per_page: int=55,
		header_lines: int=3, footer_lines: int=3, ) -> str:
		"""
		
			Purpose:
			--------
			Remove repetitive headers and footers from a text document by identifying recurring
			line blocks at page boundaries and stripping the most frequent candidates.
	
			Parameters:
			-----------
			filepath (str):
			Path to the plain-text document to clean.
			lines_per_page (int, optional):
			Approximate number of lines per page. Defaults to 50. Tune for your source.
			header_lines (int, optional):
			Number of top lines per page to consider a header candidate. Defaults to 3.
			footer_lines (int, optional):
			Number of bottom lines per page to consider a footer candidate. Defaults to 3.
	
			Returns:
			--------
			str:
			Cleaned document text with repetitive headers and footers removed.
				
		"""
		try:
			throw_if( 'filepath', filepath )
			if lines_per_page < 6:
				raise ValueError( "Argument \"lines_per_page\" should be at least 6." )
			if header_lines < 0 or footer_lines < 0:
				msg = "Arguments \"header_lines\" and \"footer_lines\" must be non-negative."
				raise ValueError( msg )
			
			with open( filepath, 'r', encoding='utf-8', errors='ignore' ) as fh:
				all_lines: List[ str ] = fh.readlines( )
			
			pages = [ all_lines[ i: i + lines_per_page ] for i in
				range( 0, len( all_lines ), lines_per_page ) ]
			
			header_counts = { }
			footer_counts = { }
			
			for page in pages:
				n = len( page )
				if n == 0:
					continue
				
				if header_lines > 0 and n >= header_lines:
					hdr = tuple( page[ :header_lines ] )
					if hdr in header_counts:
						header_counts[ hdr ] += 1
					else:
						header_counts[ hdr ] = 1
				
				if footer_lines > 0 and n >= footer_lines:
					ftr = tuple( page[ -footer_lines: ] )
					if ftr in footer_counts:
						footer_counts[ ftr ] += 1
					else:
						footer_counts[ ftr ] = 1
			
			common_header = ( )
			if header_counts:
				common_header = max( header_counts.items( ), key=lambda kv: kv[ 1 ] )[ 0 ]
			
			common_footer = ( )
			if footer_counts:
				common_footer = max( footer_counts.items( ), key=lambda kv: kv[ 1 ] )[ 0 ]
			
			cleaned_pages = [ ]
			for page in pages:
				lines = list( page )
				
				if common_header and len( lines ) >= len( common_header ):
					if tuple( lines[ : len( common_header ) ] ) == common_header:
						lines = lines[ len( common_header ): ]
				
				if common_footer and len( lines ) >= len( common_footer ):
					if tuple( lines[ -len( common_footer ): ] ) == common_footer:
						lines = lines[ : -len( common_footer ) ]
				
				cleaned_pages.append( ''.join( lines ) )
			return '\n'.join( cleaned_pages )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'remove_headers( self, filepath: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def remove_errors( self, text: str  ) -> str:
		"""
		
			Purpose:
			----------
			Removes tokens that are not recognized as valid English words
			using the NLTK `words` corpus as a reference dictionary.
	
			This function is useful for cleaning text from OCR output, web-scraped data,
			or noisy documents by removing pseudo-words, typos, and out-of-vocabulary items.
	
			Parameters
			----------
			text : str
			The raw input text
	
			Returns
			-------
			str
			The raw input text without errors
	
			
		"""
		try:
			throw_if( 'text', text )
			wordlist = [ ]
			vocab = words.words( 'en' )
			tokens = text.split( ' ' )
			for word in tokens:
				if word.isnumeric( ) or word in vocab:
					wordlist.append( word )
			_data = ' '.join( wordlist )
			return _data
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'correct_data( text: str )'
			error = ErrorDialog( exception )
			error.show( )
	
	def lemmatize( self, text: str ) -> List[ str ] | None:
		"""

			Purpose:
			--------
			Tokenizes input text into a list of word tokens.


			Parameters:
			-----------
			text (str): Input text to tokenize.


			Returns:
			--------
			List[str]: List of token strings.

		"""
		try:
			throw_if( 'text', text )
			self.nlp = spacy.load( 'en_core_web_sm' )
			self.tokens = [ token.text for token in self.nlp( text ) ]
			return self.tokens
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = ('tokenize( self, text: str ) -> List[ str ] ')
			error = ErrorDialog( exception )
			error.show( )
	
	def lemmatize_tokens( self, tokens: list[ str ] ) -> list:
		"""
			
			Purpose:
			---------
			Lemmatize each token using WordNetLemmatizer.
	
			Parameters:
			tokens (list): List of word tokens.
	
			Returns:
			list: Lemmatized tokens.
		
		"""
		try:
			throw_if( 'tokens', tokens )
			return [ self.lemmatizer.lemmatize( token ) for token in tokens ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = ('tokenize( self, text: str ) -> List[ str ] ')
			error = ErrorDialog( exception )
			error.show( )
	
	def filter_tokens( self, tokens: List[ List[ str ] ] ) -> DataFrame:
		"""
		
			Purpose:
			Removes stopwords and short tokens.
			
			Parameters:
			tokenized_sentences (list[list[str]]): Tokenized text.
			
			Returns:
			list[list[str]]: Filtered sentences.
			
		"""
		try:
			throw_if( 'tokens', tokens )
			_num = len( tokens )
			_processed = [ ]
			_datamap = [ ]
			for _tkns in tokens:
				_words = [ t for t in _tkns if t not in self.stop_words and len( t ) > 4 ]
				_datamap.append( _words )
			
			_processed.append( _datamap )
			_data = pd.DataFrame( _processed )
			return _data
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = ('filter_tokens( self, tokens: List[ List[ str ] ] ) -> List[ [ ]')
			error = ErrorDialog( exception )
			error.show( )
			
	def tokenize_text( self, text: str ) -> str:
		'''

			Purpose:
			---------
			Splits the raw path removes non-words and returns words

			Parameters:
			-----------
			- cleaned_line: (str) - clean documents.

			Returns:
			- list: Cleaned and normalized documents.

		'''
		try:
			throw_if( 'text', text )
			_tokens = nltk.word_tokenize( text )
			_words = [ t for t in _tokens ]
			_tokenlist = [ re.sub( r'[^\w"-]', '', w ) for w in _words ]
			_data = ' '.join( _tokenlist )
			return _data
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'tokenize_text( self, path: str ) -> List[ str ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def tiktokenize( self, text: str, encoding: str='cl100k_base' ) -> DataFrame:
		"""

			Purpose:
			---------
			Tokenizes text text into subword words using OpenAI's tiktoken tokenizer.
			This function leverages the tiktoken library, which provides byte-pair encoding (BPE)
			tokenization used in models such as GPT-3.5 and GPT-4. Unlike standard word
			tokenization,
			this function splits text into model-specific subword units.

			Parameters
			----------
			- text : str
				The text string to be tokenized.

			- model : str, optional
				The tokenizer model to use. Examples include 'cl100k_base' (default),
				'gpt-3.5-turbo', or 'gpt-4'. Ensure the model is supported by tiktoken.

			Returns
			-------
			- List[str]
				A list of string words representing BPE subword units.

		"""
		try:
			throw_if( 'text', text )
			self.encoding = tiktoken.get_encoding( encoding )
			token_ids = self.encoding.encode( text )
			_data = pd.DataFrame( token_ids )
			return _data
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = ('tiktokenize( self, text, encoding) -> List[ int ]')
			error = ErrorDialog( exception )
			error.show( )
	
	def tokenize_words( self, text: str ) -> DataFrame:
		"""

			Purpose:
			-----------
			  - Tokenizes the path pages path into individual word words.
			  - Converts pages to lowercase
			  - Uses NLTK's word_tokenize to split
			  the pages into words and punctuation words

			Parameters:
			-----------
			- words : List[ str ]
				A list of strings to be tokenized.

			Returns:
			--------
			- A list of token strings (words and punctuation) extracted from the pages.

		"""
		try:
			throw_if( 'text', text )
			self.tokens = text.split( ' ' )
			_tokens = [ ]
			for w in self.tokens:
				token = nltk.word_tokenize( w )
				_tokens.append( token )
			_data = pd.DataFrame( _tokens )
			return _data
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'tokenize_words( self, path: str ) -> List[ str ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def speech_tagging( self, text: str ) -> List[ Tuple[ str, str ] ] | None:
		"""
		
			Purpose:
			--------
			Performs part-of-speech tagging.
		
		
			Parameters:
			-----------
			text (str): Text to process.
		
		
			Returns:
			--------
			List[Tuple[str, str]]: List of (token, POS tag) pairs.
			
		"""
		try:
			throw_if( 'text', text )
			self.nlp = spacy.load( 'en_core_web_sm' )
			self.parts_of_speech = [ ( token.text, token.pos_ ) for token in self.nlp( text ) ]
			return self.parts_of_speech
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'speech_tagging( self, text: str ) -> List[ Tuple[ str, str ] ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def chunk_text( self, text: str, size: int=15 ) -> str:
		"""

			Purpose:
			-----------
			Tokenizes cleaned_lines pages and breaks it into chunks for downstream vectors.
			  - Converts pages to lowercase
			  - Tokenizes pages using NLTK's word_tokenize
			  - Breaks words into chunks of a specified size
			  - Optionally joins words into strings (for transformer models)

			Parameters:
			-----------
			- pages : str
				The cleaned_lines path pages to be tokenized and chunked.

			- size : int, optional (default=50)
				Number of words per chunk_words.

			- return_as_string : bool, optional (default=True)
				If True, returns each chunk_words as a path; otherwise, returns a get_list of
				words.

			Returns:
			--------
			- a list

		"""
		try:
			throw_if( 'text', text )
			self.lines = text.split( ' ' )
			_chunks = [ self.lines[ i: i + size ] for i in range( 0, len( self.lines ), size ) ]
			_map = [ ]
			for index, chunk in enumerate( _chunks ):
				_map = ' '.join( chunk )
			_data = ' '.join( _map )
			return _data
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'chunk_sentences( self, text: str, max: int=20 ) -> DataFrame'
			error = ErrorDialog( exception )
			error.show( )
	
	def chunk_sentences( self, text: str, size: int=20 ) -> DataFrame:
		"""

			Purpose:
			-----------
			Tokenizes cleaned_lines pages and breaks it into chunks for downstream vectors.
			  - Converts pages to lowercase
			  - Tokenizes pages using NLTK's word_tokenize
			  - Breaks words into chunks of a specified size
			  - Optionally joins words into strings (for transformer models)

			Parameters:
			-----------
			- pages : str
				The cleaned_lines path pages to be tokenized and chunked.

			- size : int, optional (default=50)
				Number of words per chunk_words.

			- return_as_string : bool, optional (default=True)
				If True, returns each chunk_words as a path; otherwise, returns a get_list of
				words.

			Returns:
			--------
			- a list

		"""
		try:
			throw_if( 'text', text )
			_tokens = text.split( ' ' )
			_chunks = [ _tokens[ i: i + size ] for i in range( 0, len( _tokens ), size ) ]
			_datamap = [ ]
			for index, chunk in enumerate( _chunks ):
				_value = ' '.join( chunk )
				_item = f'{_value}'
				_datamap.append( _item )
			_data = pd.DataFrame( _datamap  )
			return _data
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'chunk_text( self, text: str, max: int=512 ) -> DataFrame'
			error = ErrorDialog( exception )
			error.show( )
	
	def chunk_words( self, tokens: List[ str ], size: int=15 ) -> DataFrame:
		"""

			Purpose:
			-----------
			Breaks a list of words/tokens into a List[ List[ str ] ] or a string.

			This function:
			- Groups words into chunks of min `size`
			- Returns a a List[ List[ str ] or string

			Parameters:
			-----------
			- words : a list of tokenizd words

			- size : int, optional (default=50)
			Number of words per chunk_words.

			- as_string : bool, optional (default=True)
			Returns a string if True, else a List[ List[ str ] ] if False.

			Returns:
			--------
			- List[ List[ str ] ]
			A list of a list of token chunks. Each chunk is a list of words.

		"""
		try:
			throw_if( 'tokens', tokens )
			processed = [ ]
			wordlist = [ ]
			for s in tokens:
				if len( s ) > 4:
					wordlist.append( s )
			self.chunks = [ wordlist[ i: i + size ] for i in range( 0, len( wordlist ), size ) ]
			nums = list( range( 0, len( self.chunks ) ) )
			for i, c in enumerate( self.chunks ):
				_item =  ' '.join( c )
				processed.append( _item )
			_data = pd.DataFrame( processed, columns=[ 'Text' ] )
			_data.reset_index( )
			_data[ 'Line' ] = nums
			_data.set_index( 'Line' )
			return _data
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Token'
			exception.method = ('chunk_words( self, words: list[ str ], size: int=512)')
			error = ErrorDialog( exception )
			error.show( )
	
	def split_sentences( self, text: str ) -> List[ str ] | None:
		"""

			Purpose:
			________
			Splits the text string into a list of
			strings using NLTK's Punkt sentence tokenizer.
			This function is useful for preparing text for further processing,
			such as tokenization, parsing, or named entity recognition.

			Parameters
			----------
			- text : str
			The raw text string to be segmented into sentences.

			Returns
			-------
			- List[ str ]
			A list of sentence strings, each corresponding to a single sentence detected
			in the text text.

		"""
		try:
			throw_if( 'text', text )
			self.lines = nltk.sent_tokenize( text )
			return self.lines
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'split_sentences( self, text: str ) -> List[ str ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split_pages( self, filepath: str, num_line: int=120 ) -> List[ str ] | None:
		"""
		    
		    Purpose:
		    ---------
	        Splits a plain-text document into a list of pages using either form-feed characters
	        or fixed line count windows when form-feeds are absent.
		
		    Parameters:
		    -----------
	        file_path (str): Path to the text document.
	        lines_per_page (int): Approximate number of lines per page (default: 55).
		
		    Returns:
		    ---------
	        List[str]: List of page-level string segments.
		

		"""
		try:
			throw_if( 'filepath', filepath )
			if not os.path.exists( filepath ):
				raise FileNotFoundError( f'File not found: {filepath}' )
			else:
				self.file_path = filepath
			with open( self.file_path, 'r', encoding='utf-8', errors='ignore' ) as file:
				content = file.read( )
			if '\f' in content:
				return [ page.strip( ) for page in content.split( '\f' ) if page.strip( ) ]
			self.lines = content.splitlines( )
			i = 0
			n = len( self.lines )
			while i < n:
				page_lines = self.lines[ i: i + num_line ]
				page_text = '\n'.join( page_lines ).strip( )
				if page_text:
					self.pages.append( page_text )
				i += num_line
			return self.pages
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'split_pages( file_path )'
			error = ErrorDialog( exception )
			error.show( )
		
	def split_paragraphs( self, filepath: str ) -> DataFrame:
		"""

			Purpose:
			---------
			Reads  a file and splits it into paragraphs. A paragraph is defined as a block
			of path separated by one or more empty lines (eg, '\n\n').

			Parameters:
			-----------
			- path (str): Path to the path file.

			Returns:
			---------
			- list of str: List of paragraph strings.

		"""
		try:
			throw_if( 'filepath', filepath )
			if not os.path.exists( filepath ):
				raise FileNotFoundError( f'File not found: {filepath}' )
			else:
				self.file_path = filepath
			with open( self.file_path, 'r', encoding='utf-8', errors='ignore' ) as file:
				_input = file.read( )
				_paragraphs = [ pg.strip( ) for pg in _input.split( ' ' ) if pg.strip( ) ]
				_data = pd.DataFrame( _paragraphs )
				return _data
		except UnicodeDecodeError:
			with open( self.file_path, 'r', encoding='latin1', errors='ignore' ) as file:
				_input = file.read( )
				_paragraphs = [ pg.strip( ) for pg in _input.split( ' ' ) if pg.strip( ) ]
				_data = pd.DataFrame( _paragraphs )
				return _data
	
	def compute_frequency_distribution( self, tokens: List[ str ] ) -> FreqDist:
		"""

			Purpose:
			--------
			Creates a word frequency freq_dist from a list of documents.

			Parameters:
			-----------
			- lines (list): List of raw or preprocessed path documents.
			- process (bool): Applies normalization, tokenization, stopword removal,
			and lemmatization.

			Returns:
			- dict: Dictionary of words and their corresponding frequencies.

		"""
		try:
			throw_if( 'tokens', tokens )
			processed = [ ]
			wordlist = [ str ]
			for t in tokens:
				if len( t ) > 4:
					wordlist.append( t )
			self.frequency_distribution = FreqDist( dict( Counter( wordlist ) ) )
			return self.frequency_distribution
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = ('compute_frequency_distribution( self, documents: list, process: '
			                    'bool=True) -> FreqDist')
			error = ErrorDialog( exception )
			error.show( )
	
	def compute_conditional_distribution( self, tokens: List[ str ], condition=None,
		process: bool=True ) -> ConditionalFreqDist:
		"""

			Purpose:
			--------
			Computes a Conditional Frequency Distribution (CFD)
			 over a collection of documents.

			Parameters:
			-----------
			- documents (list):
				A list of path sections (pages, paragraphs, etc.).

			- condition (function):
				A function to determine the condition/grouping. If None, uses document index.

			- process (bool):
				If True, applies normalization, tokenization,
				stopword removal, and lemmatization.

			Returns:
			- ConditionalFreqDist:
				An NLTK ConditionalFreqDist object mapping conditions to word frequencies.

		"""
		try:
			throw_if( 'tokens', tokens )
			self.tokens = tokens
			cfd = ConditionalFreqDist( )
			for idx, line in enumerate( self.tokens ):
				key = condition( line ) if condition else f'Line-{idx}'
				toks = self.tokenize_text( self.normalize_text( line ) if process else line )
				for t in toks.split( ):
					cfd[ key ][ t ] += 1
			self.conditional_distribution = cfd
			return self.conditional_distribution
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'compute_conditional_distribution( self, words, cond, proc  )'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_vocabulary( self, freq: Dict, size: int=1 ) -> DataFrame:
		"""

			Purpose:
			---------
			Builds a vocabulary list from a frequency
			distribution by applying a minimum frequency threshold.

			Parameters:
			-----------
			- freq_dist (dict):
				A dictionary mapping words to their frequencies.

			- min (int): Minimum num
				of occurrences required for a word to be included.

			Returns:
			--------
			- list: Sorted list of unique vocabulary words.

		"""
		try:
			throw_if( 'freq', freq )
			self.frequency_distribution = freq
			self.vocabulary = [ word for word, freq in freq.items( ) if freq >= size ]
			_data = pd.DataFrame( self.vocabulary )
			return _data
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = ('create_vocabulary( self, freq_dist: dict, min: int=1 ) -> List['
			                    'str]')
			error = ErrorDialog( exception )
			error.show( )
	
	def create_wordbag( self, tokens: List[ str ] ) -> Dict | None:
		"""

			Purpose:
			--------
			Construct a Bag-of-Words (BoW) frequency dictionary from a list of strings.

			Parameters:
			-----------
			- words (list): List of words from a document.

			Returns:
			--------
			- dict: Word frequency dictionary.

		"""
		try:
			throw_if( 'tokens', tokens )
			self.tokens = tokens
			return dict( Counter( self.tokens ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'create_wordbag( self, words: List[ str ] ) -> dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_vectors( self, tokens: List[ str ] ) -> Dict[ str, np.ndarray ]:
		"""
		
			Purpose:
			----------
			Generates word embeddings using TF-IDF vectors from a list of tokens.
			This function treats each token as its own "document" to simulate a corpus
			and calculates a TF-IDF vector for each word based on term statistics.
		
			Parameters
			----------
			tokens : List[str]
				A list of individual word tokens to be embedded.
		
			Returns
			-------
			Dict[str, np.ndarray]
				A dictionary mapping each word to its corresponding TF-IDF vector.
		
			Example
			-------
			create_tfidf_word_embeddings(["nlp", "text", "nlp", "model"])
			{
				'nlp': array([0.7071, 0.0, 0.7071]),
				'text': array([0.0, 1.0, 0.0]),
				'model': array([0.0, 0.0, 1.0])
			}
		"""
		try:
			# Each word is treated as a single "document" for TF-IDF
			fake_docs = [ [ word ] for word in tokens ]
			joined_docs = [ ' '.join( doc ) for doc in fake_docs ]
			vectorizer = TfidfVectorizer( )
			X = vectorizer.fit_transform( joined_docs )
			feature_names = vectorizer.get_feature_names_out( )
			embeddings = { }
			for idx, word in enumerate( tokens ):
				vector = X[ idx ].toarray( ).flatten( )
				embeddings[ word ] = vector
			
			return embeddings
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = ('create_vectors( self, tokens: List[str]) -> Dict[str, '
			                    'np.ndarray]')
			error = ErrorDialog( exception )
			error.show( )
			
	def clean_file( self, src: str ) -> str:
		"""

			Purpose:
			________
			Cleans text files given a source directory (src) and destination directory (dest)

			Parameters:
			----------
			- src (str): Source directory

			Returns:
			--------
			- None

		"""
		try:
			throw_if( 'src', src )
			_source = src
			_text = open( _source, 'r', encoding='utf-8', errors='ignore' ).read( )
			_collapse = self.collapse_whitespace( _text )
			_normal = self.normalize_text( _collapse )
			_special = self.remove_special( _normal )
			_fragments = self.remove_fragments( _special )
			_compress = self.compress_whitespace( _fragments )
			return _compress
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'clean_file( self, src: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def clean_files( self, src: str, dest: str ) -> None:
		"""

			Purpose:
			________
			Cleans text files given a source directory (src) and destination directory (dest)

			Parameters:
			----------
			- src (str): Source directory
			- dest (str): Destination directory

			Returns:
			--------
			- None

		"""
		try:
			throw_if( 'src', src )
			throw_if( 'dest', dest )
			source = src
			dest_path = dest
			files = os.listdir( source )
			for f in files:
				processed = [ ]
				filename = os.path.basename( f )
				source_path = source + '\\' + filename
				text = open( source_path, 'r', encoding='utf-8', errors='ignore' ).read( )
				collapse = self.collapse_whitespace( text )
				compress = self.compress_whitespace( collapse )
				tokens = self.tokenize_text( compress )
				normal = self.normalize_text( tokens )
				special = self.remove_special( normal )
				fragments = self.remove_fragments( special )
				sentences = self.chunk_sentences( fragments )
				destination = dest_path + '\\' + filename
				clean = open( destination, 'wt', encoding='utf-8', errors='ignore' )
				text = ' '.join( sentences )
				clean.write( text )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'clean_files( self, src: str, dest: str )'
			error = ErrorDialog( exception )
			error.show( )
	
	def chunk_files( self, source: str, destination: str, size: int=20 ) -> None:
		"""

			Purpose:
			________
			Chunks cleaned text files given a _source directory and destination directory

			Parameters:
			----------
			- src (str): Source directory
			- dest (str): Destination directory

			Returns:
			--------
			- None

		"""
		try:
			throw_if( 'src', source )
			throw_if( 'dest', destination )
			_source = source
			_destination = destination
			files = os.listdir( _source )
			wordlist = [ ]
			for f in files:
				processed = [ ]
				filename = os.path.basename( f )
				source_path = _source + '\\' + filename
				text = open( source_path, 'r', encoding='utf-8', errors='ignore' ).read( )
				_tokens =  text.split( ' ' )
				_chunks = [ _tokens[ i: i + size ] for i in range( 0, len( _tokens ), size ) ]
				_datamap = [ ]
				for i, c in enumerate( _chunks ):
					_value = '{ ' +f' {i} : [ ' + ' '.join( c ) + ' ] }, ' + "\n"
					_datamap.append( _value )
					
				for s in _datamap:
					processed.append( s )
				
				_final = _destination + '\\' + filename
				clean = open( _final, 'wt', encoding='utf-8', errors='ignore' )
				for p in processed:
					clean.write( p )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'chunk_files( self, src: str, dest: str )'
			error = ErrorDialog( exception )
			error.show( )
	
	def chunk_data( self, filepath: str, size: int=20  ) -> DataFrame:
		"""

			Purpose:
			________
			Chunks cleaned text files given a source directory and destination directory

			Parameters:
			----------
			- src (str): Source directory

			Returns:
			--------
			- DataFrame

		"""
		try:
			throw_if( 'src', filepath )
			_source = filepath
			processed = [ ]
			wordlist = [ ]
			vocab = words.words( 'en' )
			text = open( _source, 'r', encoding='utf-8', errors='ignore' ).read( )
			tokens = text.split( )
			for s in tokens:
				if s.isalpha( ) and s in vocab:
					wordlist.append( s )
			self.chunks = [ wordlist[ i: i + size ] for i in range( 0, len( wordlist ), size ) ]
			for i, c in enumerate( self.chunks ):
				_item =  ' '.join( c )
				processed.append( _item )
			_data = pd.DataFrame( processed )
			return _data
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'chunk_data( self, filepath: str, size: int=512  ) -> DataFrame'
			error = ErrorDialog( exception )
			error.show( )

	def convert_jsonl( self, source: str, destination: str, size: int=20 ) -> None:
		"""

			Purpose:
			________
			Coverts text files to JSONL format given a source directory (Source)
			 and destination directory (destination)

			Parameters:
			--------
			- source (str): Source directory
			- destination (str): Destination directory

			Returns:
			--------
			- None

		"""
		try:
			throw_if( 'src', source )
			throw_if( 'dest', destination )
			_source = source
			dest_path = destination
			files = os.listdir( _source )
			wordlist = [ ]
			for f in files:
				processed = [ ]
				filename = os.path.basename( f )
				source_path = _source + '\\' + filename
				text = open( source_path, 'r', encoding='utf-8', errors='ignore' ).read( )
				_tokens =  text.split( ' ' )
				_chunks = [ _tokens[ i: i + size ] for i in range( 0, len( _tokens ), size ) ]
				_datamap = [ ]
				for i, c in enumerate( _chunks ):
					_value = '{ ' +f' {i} : [ ' + ' '.join( c ) + ' ] }, ' + "\n"
					_datamap.append( _value )
					
				for s in _datamap:
					processed.append( s )
				
				destination = dest_path + '\\' + filename.replace( '.txt', '.jsonl' )
				clean = open( destination, 'wt', encoding='utf-8', errors='ignore' )
				for p in processed:
					clean.write( p )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'convert_jsonl( self, source: str, desination: str )'
			error = ErrorDialog( exception )
			error.show( )
	
	def encode_sentences( self, sentences: List[ str ], model: str= 'all-MiniLM-L6-v2' ) -> \
			Tuple[ List[ str ], np.ndarray ]:
		"""
		
			Purpose:
			Generate contextual sentence embeddings using SentenceTransformer.
			
			Parameters:
			sentences (list[str]): List of raw sentence strings.
			model_name (str): Model to use for encoding.
			
			Returns:
			tuple: (Cleaned sentences, Embeddings as np.ndarray)
			
		"""
		try:
			throw_if( 'sentences', sentences )
			throw_if( 'model', model )
			_transformer = SentenceTransformer( model )
			_tokens = self.lemmatize_tokens( sentences )
			_encoding = _transformer.encode( _tokens, show_progress_bar=True )
			return ( self.cleaned_tokens, np.array( _encoding ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'encode_sentences( self, sentences: List[ str ], model_name ) -> ( )'
			error = ErrorDialog( exception )
			error.show( )
	
	def semantic_search( self, query: str, tokens: List[ str ], embeddings: np.ndarray,
			model: SentenceTransformer, top: int=5 ) -> List[ tuple[ str, float ] ]:
		"""
			Purpose:
				Perform semantic search over embedded corpus using query.
				
			Parameters:
				query (str): Natural language input.
				tokens (list[str]): Corpus sentences.
				embeddings (np.ndarray): Sentence embeddings.
				model (SentenceTransformer): Same model used for encoding.
				top (int): Number of matches to return.
				
			Returns:
				list[tuple[str, float]]: Top-k (sentence, similarity) pairs.
		"""
		try:
			throw_if( 'query', query )
			throw_if( 'tokens', tokens )
			throw_if( 'embedding', embeddings )
			throw_if( 'model', model )
			query_vec = model.encode( [ query ] )
			sims = cosine_similarity( query_vec, embeddings )[ 0 ]
			top_indices = sims.argsort( )[ ::-1 ][ : top ]
			return [ ( tokens[ i ], sims[ i ] ) for i in top_indices ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = ('semantic_search( self, query: str, tokens: List[ str ], '
			                    'embeddings: np.ndarray, model: SentenceTransformer,  '
			                    'top_k: int=5 ) -> List[ tuple[ str, float ] ]')
			error = ErrorDialog( exception )
			error.show( )
	

class Word( Processor ):
	"""

		Purpose:
		--------
		A class to extract, clean, and analyze text from Microsoft Word documents.

		Methods:
		--------
		split_sentences( self ) -> None
		clean_sentences( self ) -> None
		create_vocabulary( self ) -> None
		compute_frequency_distribution( self ) -> None
		summarize( self ) -> None:

	"""
	sentences: Optional[ List[ str ] ]
	cleaned_sentences: Optional[ List[ str ] ]
	document: Optional[ Docx ]
	raw_text: Optional[ str ]
	paragraphs: Optional[ List[ str ] ]
	file_path: Optional[ str ]
	vocabulary: Optional[ set ]
	document: Optional[ Docx ]
	
	def __init__( self, filepath: str ) -> None:
		"""

			Purpose:
			--------
			Initializes the WordTextProcessor with the path to the .docx file.

			Parameters:
			----------
			:param filepath: Path to the Microsoft Word document (.docx)

		"""
		super( ).__init__( )
		self.file_path = filepath
		self.raw_text = ''
		self.paragraphs = [ ]
		self.sentences = [ ]
		self.cleaned_sentences = [ ]
		self.vocabulary = set( )
	
	def __dir__( self ) -> List[ str ] | None:
		'''

			Purpose:
			---------
			Provides a list of strings representing class members.


			Parameters:
			-----------
			- self

			Returns:
			--------
			- List[ str ] | None

		'''
		return [ 'extract_text', 'split_sentences', 'clean_sentences', 'create_vocabulary',
			'compute_frequency_distribution', 'summarize', 'filepath', 'raw_text', 'paragraphs',
			'sentences', 'cleaned_sentences', 'vocabulary', 'freq_dist' ]
	
	def extract_text( self ) -> str | None:
		"""

			Purpose:
			--------
			Extracts raw text and paragraphs from the .docx file.

		"""
		try:
			self.document = Docx( self.file_path )
			self.paragraphs = [ para.text.strip( ) for para in self.document.paragraphs if
				para.text.strip( ) ]
			self.raw_text = '\n'.join( self.paragraphs )
			return self.raw_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Word'
			exception.method = 'extract_text( self ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def split_sentences( self ) -> List[ str ] | None:
		"""

			Purpose:
			--------
			Splits the raw text into sentences.

		"""
		try:
			self.sentences = sent_tokenize( self.raw_text )
			return self.sentences
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Word'
			exception.method = 'split_sentences( self ) -> List[ str ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def clean_sentences( self ) -> List[ str ] | None:
		"""

			Purpose:
			-------
			Cleans each _sentence: removes extra whitespace, punctuation, and lowers the text.

		"""
		try:
			for _sent in self.sentences:
				_sent = re.sub( r'[\r\n\t]+', ' ', _sent )
				_sent = re.sub( r"[^a-zA-Z0-9\s']", '', _sent )
				_sent = re.sub( r'\s{2,}', ' ', _sent ).strip( ).lower( )
				self.cleaned_sentences.append( _sent )
			return self.cleaned_sentences
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Word'
			exception.method = 'clean_sentences( self ) -> List[ str ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_vocabulary( self ) -> set | None:
		"""

			Purpose:
			--------
			Computes vocabulary terms from cleaned sentences.

		"""
		try:
			all_words = [ ]
			self.stop_words = set( stopwords.words( 'english' ) )
			for _sentence in self.cleaned_sentences:
				_tokens = word_tokenize( _sentence )
				self.tokens = [ token for token in _tokens if
					token.isalpha( ) and token not in self.stop_words ]
				all_words.extend( self.tokens )
			self.vocabulary = set( all_words )
			return self.vocabulary
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Word'
			exception.method = 'create_vocabulary( self ) -> List[ str ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def compute_frequency_distribution( self ) -> Dict[ str, int ] | None:
		"""

			Purpose:
			-------
			Computes frequency distribution of the vocabulary.

		"""
		try:
			words = [ ]
			for sentence in self.cleaned_sentences:
				tokens = word_tokenize( sentence )
				tokens = [ token for token in tokens if token.isalpha( ) ]
				words.extend( tokens )
			self.frequency_distribution = dict( Counter( words ) )
			return self.frequency_distribution
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Word'
			exception.method = 'compute_frequency_distribution( self ) -> Dict[ str, int ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def summarize( self ) -> List[ str ] | None:
		"""

			Purpose:
			-------
			Prints a summary of extracted and processed text.

		"""
		print( f'Document: {self.file_path}' )
		print( f'Paragraphs: {len( self.paragraphs )}' )
		print( f'Sentences: {len( self.sentences )}' )
		print( f'Vocabulary Size: {len( self.vocabulary )}' )
		print( f'Top 10 Frequent Words: {Counter( self.frequency_distribution ).most_common( 10 )}' )

class PDF( Processor ):
	"""

		Purpose:
		--------
		A utility class for extracting clean pages from PDF files into a list of strings.
		Handles nuances such as layout artifacts, page separation, optional filtering,
		and includes df detection capabilities.


		Methods:
		--------
		extract_lines( self, path, max: int=None) -> List[ str ]
		extract_text( self, path, max: int=None) -> str
		export_csv( self, tables: List[ pd.DataFrame ], filename: str=None ) -> None
		export_text( self, words: List[ str ], path: str=None ) -> None
		export_excel( self, tables: List[ pd.DataFrame ], path: str=None ) -> None

	"""
	strip_headers: Optional[ bool ]
	minimum_length: Optional[ int ]
	extract_tables: Optional[ bool ]
	extracted_lines: Optional[ List ]
	extracted_tables: Optional[ List ]
	extracted_pages: Optional[ List ]
	
	def __init__( self, headers: bool=False, size: int=10, tables: bool=True ) -> None:
		"""

			Purpose:
			-----------
			Initialize the PDF pages extractor with configurable settings.

			Parameters:
			-----------
			- headers (bool): If True, attempts to strip recurring headers/footers.
			- min (int): Minimum num of characters for a line to be included.
			- tables (bool): If True, extract pages from detected tables using block
			grouping.

		"""
		super( ).__init__( )
		self.strip_headers = headers
		self.minimum_length = size
		self.extract_tables = tables
		self.pages = [ ]
		self.lines = [ ]
		self.clean_lines = [ ]
		self.extracted_lines = [ ]
		self.extracted_tables = [ ]
		self.extracted_pages = [ ]
		self.tables = None
		self.file_path = ''
		self.page = ''
	
	def __dir__( self ) -> List[ str ] | None:
		'''

			Purpose:
			---------
			Provides a list of strings representing class members.


			Parameters:
			-----------
			- self

			Returns:
			--------
			- List[ str ] | None

		'''
		return [ 'strip_headers', 'minimum_length', 'extract_tables', 'file_path', 'page', 'pages',
			'words', 'clean_lines', 'extracted_lines', 'extracted_tables', 'extracted_pages',
			'extract_lines', 'extract_text', 'export_csv', 'export_text', 'export_excel' ]
	
	def extract_lines( self, path: str, size: Optional[ int ]=None ) -> List[ str ] | None:
		"""

			Purpose:
			--------
			Extract words of pages from a PDF,
			optionally limiting to the first N pages.

			Parameters:
			----------
			- path (str): Path to the PDF file
			- max (Optional[int]): Max num of pages to process (None for all pages)

			Returns:
			--------
			- List[str]: Cleaned list of non-empty words

		"""
		try:
			throw_if( 'path', path )
			self.file_path = path
			with fitz.open( self.file_path ) as doc:
				for i, page in enumerate( doc ):
					if size is not None and i >= size:
						break
					if self.extract_tables:
						page_lines = self._extract_tables( page )
					else:
						_text = page.get_text( 'text' )
						page_lines = _text.splitlines( )
					filtered = self._filter_lines( page_lines )
					self.extracted_lines.extend( filtered )
			return self.extracted_lines
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'PDF'
			exception.method = ('extract_lines( self, path: str, max: Optional[ int ]=None ) -> '
			                    'List[ str ]')
			error = ErrorDialog( exception )
			error.show( )
	
	def _extract_tables( self, page: Page ) -> List[ str ] | None:
		"""

			Purpose:
			--------
			Attempt to extract structured blocks
			such as tables using spatial grouping.

			Parameters:
			----------
			- page: PyMuPDF page object

			Returns:
			--------
			- List[str]: Grouped blocks including potential tables

		"""
		try:
			throw_if( 'page', page )
			tf = page.find_tables( )
			lines = [ ]
			for t in getattr( tf, 'tables', [ ] ):
				df = pd.DataFrame( t.extract( ) )  # or t.to_pandas()
				for row in df.values.tolist( ):
					lines.append( ' '.join( map( str, row ) ) )
			return lines
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'PDF'
			exception.method = '_extract_tables( self, page ) -> List[ str ]:'
			error = ErrorDialog( exception )
			error.show( )
	
	def _filter_lines( self, lines: List[ str ] ) -> List[ str ] | None:
		"""

			Purpose:
			-----------
			Filter and clean words from a page of pages.

			Parameters:
			- words (List[str]): Raw words of pages

			Returns:
			--------
			- List[str]: Filtered, non-trivial words

		"""
		try:
			if lines is None:
				raise Exception( 'The argument "lines" is None' )
			else:
				self.lines = lines
				clean = [ ]
				for line in lines:
					line = line.strip( )
					if len( line ) < self.minimum_length:
						continue
					if self.strip_headers and self._has_header( line ):
						continue
					clean.append( line )
				return clean
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'PDF'
			exception.method = '_filter_lines( self, words: List[ str ] ) -> List[ str ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def _has_header( self, line: str ) -> bool | None:
		"""

			Purpose:
			--------
			Heuristic to detect common headers/footers (basic implementation).

			Parameters:
			- line (str): A line of pages

			Returns:
			--------
			- bool: True if line is likely a header or footer

		"""
		try:
			throw_if( 'line', line )
			_keywords = [ 'page', 'public law', 'u.s. government', 'united states' ]
			return any( kw in line.lower( ) for kw in _keywords )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'PDF'
			exception.method = '_has_repeating_header( self, line: str ) -> bool'
			error = ErrorDialog( exception )
			error.show( )
	
	def extract_text( self, path: str, size: Optional[ int ]=None ) -> str | None:
		"""

			Purpose:
			---------
			Extract the entire pages from a PDF into one continuous path.

			Parameters:
			-----------
			- path (str): Path to the PDF file
			- max (Optional[int]): Maximum num of pages to process

			Returns:
			--------
			- str: Full concatenated pages

		"""
		try:
			throw_if( 'path', path )
			if size is not None and size > 0:
				self.file_path = path
				self.lines = self.extract_lines( self.file_path, size=size )
				return '\n'.join( self.lines )
			elif size is None or size <= 0:
				self.file_path = path
				self.lines = self.extract_lines( self.file_path )
				return '\n'.join( self.lines )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'PDF'
			exception.method = 'extract_text( self, path: str, max: Optional[ int ]=None ) -> str:'
			error = ErrorDialog( exception )
			error.show( )
	
	def extract_tables( self, path: str, size: Optional[ int ]=None ) -> (
			List[ pd.DataFrame ] | None):
		"""

			Purpose:
			-----------
			Extract tables from the PDF and return them as a list of DataFrames.

			Parameters:
			- path (str): Path to the PDF file
			- max (Optional[int]): Maximum num of pages to process

			Returns:
			--------
			- List[pd.DataFrame]: List of DataFrames representing detected tables

		"""
		try:
			throw_if( 'path', path )
			throw_if( 'max', size )
			self.file_path = path
			self.tables = [ ]
			with fitz.open( self.file_path ) as doc:
				for i, page in enumerate( doc ):
					if size is not None and i >= size:
						break
					tf = page.find_tables( )
					for t in getattr( tf, 'tables', [ ] ):
						self.tables.append( pd.DataFrame( t.extract( ) ) )
			return self.tables
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'PDF'
			exception.method = (
				'extract_tables( self, path: str, max: Optional[ int ] = None ) -> List[ '
				'pd.DataFrame ]')
			error = ErrorDialog( exception )
			error.show( )
	
	def export_csv( self, tables: List[ pd.DataFrame ], filename: str ) -> None:
		"""

			Purpose:
			-----------
			Export a list of DataFrames (tables) to individual CSV files.

			Parameters:
			- tables (List[pd.DataFrame]): List of tables to export
			- filename (str): Prefix for output filenames (e.g., 'output_table')

		"""
		try:
			throw_if( 'tables', tables )
			throw_if( 'filename', filename )
			self.tables = tables
			for i, df in enumerate( self.tables ):
				df.to_csv( f'{filename}_{i + 1}.csv', index=False )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'PDF'
			exception.method = (
				'export_csv( self, tables: List[ pd.DataFrame ], filename: str ) -> '
				'None')
			error = ErrorDialog( exception )
			error.show( )
	
	def export_text( self, lines: List[ str ], path: str ) -> None:
		"""

			Export extracted words of
			pages to a plain pages file.

			Parameters:
			-----------
			- words (List[str]): List of pages words
			- path (str): Path to output pages file

		"""
		try:
			throw_if( 'path', path )
			throw_if( 'lines', lines )
			self.file_path = path
			self.lines = lines
			with open( self.file_path, 'w', encoding='utf-8', errors='ignore' ) as f:
				for line in self.lines:
					f.write( line + '\n' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'PDF'
			exception.method = 'export_text( self, lines: List[ str ], path: str ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def export_excel( self, tables: List[ pd.DataFrame ], path: str ) -> None:
		"""

			Export all extracted tables into a single
			Excel workbook with one sheet per df.

			Parameters:
			-----------
			- tables (List[pd.DataFrame]): List of tables to export
			- path (str): Path to the output Excel file

		"""
		try:
			throw_if( 'tables', tables )
			self.tables = tables
			self.file_path = path
			with pd.ExcelWriter( self.file_path, engine='xlsxwriter' ) as _writer:
				for i, df in enumerate( self.tables ):
					_sheet = f'Table_{i + 1}'
					df.to_excel( _writer, sheet_name=_sheet, index=False )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'PDF'
			exception.method = 'export_excel( self, tables: List[ pd.DataFrame ], path: str )->None'
			error = ErrorDialog( exception )
			error.show( )

class CSV( Loader ):
	'''

		Purpose:
		--------
		Wrap LangChain's CSVLoader to parse CSV files into Document objects.

	'''
	loader: Optional[ CSVLoader ]
	documents: Optional[ List[ Document ] ]
	file_path: Optional[ str ]
	pattern: Optional[ List[ str ] ]
	expanded: Optional[ List[ str ] ]
	candidates: Optional[ List[ str ] ]
	resolved: Optional[ List[ str ] ]
	encoding: Optional[ str ]
	csv_args: Optional[ Dict[ str, Any ] ]
	source_column: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.encoding = None
		self.source_column = None
		self.documents = [ ]
		self.csv_args = { }
		self.pattern = [ ]
		self.expanded = [ ]
		self.candidates = [ ]
		self.resolved = [ ]
	
	def load( self, path: str, encoding: str=None, csv_args: Dict[ str, Any ]=None,
			source_column: str=None ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load a CSV file into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the CSV file.
			encoding (Optional[str]): File encoding (e.g., 'utf-8') if known.
			csv_args (Optional[Dict[str, Any]]): Additional CSV parsing arguments.
			source_column (Optional[str]): Column name used for source attribution.

			Returns:
			--------
			List[Document]: List of LangChain Document objects parsed from the CSV.

		'''
		try:
			self.file_path = self._verify_exists( path )
			self.encoding = encoding
			self.csv_args = csv_args
			self.source_column = source_column
			self.loader = CSVLoader( file_path=self.file_path, encoding=self.encoding,
				csv_args=self.csv_args, source_column=self.source_column )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'CSV'
			exception.method = 'load( self, path: str ) -> List[ Documents ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, size: int=1000, amount: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded CSV documents into smaller text chunks.

			Parameters:
			-----------
			chunk_size (int): Maximum number of characters per chunk.
			chunk_overlap (int): Number of overlapping characters between chunks.

			Returns:
			--------
			List[Document]: List of split Document chunks.

		'''
		try:
			throw_if( 'documenys', self.documents )
			self.documents = self._split_documents( self.documents, chunk=size, overlap=amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'CSV'
			exception.method = 'split( self, size: int=1000, amount: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class Web( Loader ):
	'''

		Purpose:
		--------
		Wrap LangChain's WebBaseLoader to retrieve and parse HTML content into Document objects.

	'''
	loader: Optional[ WebBaseLoader ]
	urls: Optional[ List[ str ] ]
	documents: Optional[ List[ Document ] ]
	encoding: Optional[ str ]
	file_path: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.encoding = None
		self.urls = None
	
	def load( self, urls: List[ str ] ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load one or more web pages and convert to Document objects.

			Parameters:
			-----------
			urls (str | List[str]): A single URL string or list of URL strings.

			Returns:
			--------
			List[Document]: Parsed Document objects from fetched HTML content.

		'''
		try:
			throw_if( 'urls', urls )
			self.urls = urls
			self.loader = WebBaseLoader( web_path=self.urls )
			self.documents = loader.load( path=self.file_path )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''
		
			Purpose:
			--------
			Split loaded web documents into smaller chunks for better LLM processing.

			Parameters:
			-----------
			chunk(int): Max characters per chunk.
			overlap(int): Overlap between chunks in characters.

			Returns:
			--------
			List[ Document ]: Chunked Document objects.

		'''
		try:
			throw_if( 'documents', self.documents )
			self.documents = self._split_documents( self.documents, chunk=chunk, overlap=overlap )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )

class DOCX( Loader ):
	'''

		Purpose:
		--------
		Wrap LangChain's Docx2txtLoader to convert Word .docx files into Document objects.

	'''
	loader: Optional[ Docx2txtLoader ]
	file_path: str | None
	documents: List[ Document ] | None
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.encoding = None
	
	def load( self, path: str ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load the contents of a Word .docx file into LangChain Document objects.

			Parameters:
			-----------
			path (str): File path to the .docx document.

			Returns:
			--------
			List[Document]: Parsed Document list from Word file.

		'''
		try:
			self.file_path = self._verify_exists( path )
			self.loader = Docx2txtLoader( self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split Word documents into text chunks suitable for LLM processing.

			Parameters:
			-----------
			chunk(int): Maximum characters per chunk.
			overlap(int): Overlap between chunks in characters.

			Returns:
			--------
			List[Document]: Chunked list of Document objects.

		'''
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self._split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.docs
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )

class Markdown( Loader ):
	'''
	
		Purpose:
		--------
		Wrap LangChain's UnstructuredMarkdownLoader to parse
		Markdown files into Document objects.
	
	
	'''
	loader: Optional[ UnstructuredMarkdownLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = [ ]
		self.loader = None
	
	def load( self, path: str ) -> List[ Document ] | None:
		'''
		
			Purpose:
			--------
			Load a Markdown (.md) file into LangChain Document objects.

			Parameters:
			-----------
			path (str): File path to the Markdown file.

			Returns:
			--------
			List[Document]: List of parsed Document objects from the Markdown file.
			
		'''
		try:
			self.file_path = self._verify_exists( path )
			self.loader = UnstructuredMarkdownLoader( file_path=self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded Markdown content into text chunks for LLM consumption.

			Parameters:
			-----------
			chunk(int): Max characters per chunk.
			overlap(int): Number of characters that overlap between chunks.

			Returns:
			--------
			List[Document]: Split Document chunks from the original Markdown content.

		'''
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self._split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )

class HTML( Loader ):
	'''

		Purpose:
		--------
		Wrap LangChain's UnstructuredHTMLLoader to parse
		HTML files into Document objects.

	'''
	loader: Optional[ UnstructuredHTMLLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = [ ]
		self.loader = None
	
	def load( self, path: str ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load an HTML file and convert its contents into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			self.file_path = self._verify_exists( path )
			self.loader = UnstructuredHTMLLoader( file_path=self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded HTML documents into manageable text chunks.

			Parameters:
			-----------
			chunk(int): Max characters per chunk.
			overlap(int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Chunked list of LangChain Document objects.

		'''
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self._split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )

class Fetch( ):
	"""

		Purpose:
		Provides a unified conversational system with explicit methods for
		querying structured data (SQL), unstructured documents, or free-form
		chat with an OpenAI LLM. Each method is deterministic and isolates
		a specific capability.

		Parameters:
		db_uri (str):
		URI string for the SQLite database connection.
		doc_paths (List[str]):
		File paths to documents (txt, pdf, csv, html) for ingestion.
		model (str, optional):
		OpenAI model to use (default: 'gpt-4o-mini').
		temperature (float, optional):
		Sampling temperature for the LLM (default: 0.8).

		Attributes:
		model (str): OpenAI model identifier.
		temperature (float): Temperature setting for sampling.
		llm (ChatOpenAI): Instantiated OpenAI-compatible chat model.
		db_uri (str): SQLite database URI.
		doc_paths (List[str]): Paths to local document sources.
		memory (ConversationBufferMemory): LangChain conversation buffer.
		sql_tool (Optional[Tool]): SQL query tool.
		doc_tool (Optional[Tool]): Vector document retrieval tool.
		api_tools (List[Tool]): List of custom API tools.
		agent (AgentExecutor): LangChain multi-tool agent.
		__tools (List[Tool]): Consolidated tool list used by agent.
		documents (List[str]): Cached document source text or metadata.
		db_toolkit (Optional[object]): SQLDatabaseToolkit instance.
		database (Optional[object]): Underlying SQLAlchemy database.
		loader (Optional[object]): Last-used document loader.
		tool (Optional[object]): Active retrieval tool.
		extension (Optional[str]): File extension for routing.

	"""
	model: Optional[ str ]
	temperature: Optional[ float ]
	llm: Optional[ ChatOpenAI ]
	db_uri: Optional[ str ]
	doc_paths: Optional[ List[ str ] ]
	memory: Optional[ ConversationBufferMemory ]
	sql_tool: Optional[ Tool ]
	doc_tool: Optional[ Tool ]
	api_tools: List[ Tool ]
	agent: Optional[ AgentExecutor ]
	__tools: List[ Tool ]
	documents: List[ str ]
	db_toolkit: Optional[ object ]
	database: Optional[ object ]
	loader: Optional[ object ]
	tool: Optional[ object ]
	extension: Optional[ str ]
	answer: Optional[ Dict ]
	sources: Optional[ Dict[ str, str ] ]
	
	def __init__( self, db_uri: str, doc_paths: List[ str ], model: str='gpt-4o-mini',
			temperature: float=0.8 ):
		"""

			Purpose:
			-------
			Initializes the Fetch system and configures tools for SQL,
			document retrieval, and conversational use.

			Parameters:
			-----------
			db_uri (str): Path or URI to SQLite database.
			doc_paths (List[str]): Files to be processed for retrieval.
			model (str): LLM model name (default: gpt-4o-mini).
			temperature (float): Sampling diversity (default: 0.8).

			Returns:
				None

		"""
		self.model = model
		self.temperature = temperature
		self.llm = ChatOpenAI( model=self.model, temperature=self.temperature, streaming=True )
		self.db_uri = db_uri
		self.doc_paths = doc_paths
		self.memory = ConversationBufferMemory( memory_key='chat_history', return_messages=True )
		self.sql_tool = self._init_sql_tool( )
		self.doc_tool = self._init_doc_tool( )
		self.api_tools = self._init_api_tools( )
		self.documents = [ ]
		self.db_toolkit = None
		self.database = None
		self.loader = None
		self.tool = None
		self.extension = None
		self.answer = { }
		self.__tools = [ t for t in [ self.sql_tool, self.doc_tool ] + self.api_tools if
		                 t is not None ]
		self.agent = initialize_agent( tools=self.__tools, llm=self.llm, memory=self.memory,
			agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True )
	
	def query_sql( self, question: str ) -> str | None:
		"""

			Purpose:
			Answer a question using ONLY the SQL database tool.

			Parameters:
			question (str): Natural language SQL-like question.

			Returns:
			str: Answer from the SQL query tool.

		"""
		try:
			throw_if( 'question', question )
			return self.sql_tool.func( question )
		except Exception as e:
			exception = Error( e )
			exception.module = 'sketchy'
			exception.cause = 'Fetch'
			exception.method = 'query_sql(self, question)'
			error = ErrorDialog( exception )
			error.show( )
	
	def query_docs( self, question: str, with_sources: bool=False ) -> str | None:
		"""

			Purpose:
			Answer a question using ONLY the document retrieval tool.

			Parameters:
			question (str):
			Natural language question grounded in the loaded documents.
			
			with_sources (bool):
			If True, returns sources alongside the answer.

			Returns:
				str:
					Response from the document retriever. Includes sources if available.

		"""
		try:
			throw_if( 'question', question )
			if with_sources:
				if self.doc_chain_with_sources is None:
					raise RuntimeError( 'Document chain with sources is not available' )
				
				result = self.doc_chain_with_sources( { 'question': question } )
				if 'answer' not in result or 'sources' not in result:
					raise RuntimeError( 'Malformed response from doc_chain_with_sources' )
				
				answer = result[ 'answer' ]
				sources = result[ 'sources' ]
				
				if sources:
					return f"{answer}\n\nSOURCES:\n{sources}"
				return answer
			else:
				return self.doc_tool.func( question )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'sketchy'
			exception.cause = 'Fetch'
			exception.method = 'query_docs(self, question, with_sources)'
			error = ErrorDialog( exception )
			error.show( )
	
	def query_chat( self, prompt: str ) -> str | None:
		"""

			Purpose:
				Send a general-purpose prompt directly to the LLM without using
				tools, but with full memory context.

			Parameters:
				prompt (str): User message for free-form reasoning.

			Returns:
				str: LLM-generated conversational response.

		"""
		try:
			throw_if( 'prompt', prompt )
			return self.llm.invoke( prompt ).content
		except Exception as e:
			exception = Error( e )
			exception.module = 'sketchy'
			exception.cause = 'Fetch'
			exception.method = 'query_chat(self, prompt)'
			error = ErrorDialog( exception )
			error.show( )
