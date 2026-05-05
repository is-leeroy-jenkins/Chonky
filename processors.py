'''
  ******************************************************************************************
	  Assembly:                Chonky
	  Filename:                processors.py
	  Author:                  Terry D. Eppler
	  Created:                 05-31-2022

	  Last Modified By:        Terry D. Eppler
	  Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="processors.py" company="Terry D. Eppler">

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
	processors.py
  </summary>
  ******************************************************************************************
  '''
from __future__ import annotations
import string

import docx
from pymupdf import Page, Document
from sklearn.feature_extraction.text import TfidfVectorizer
from boogr import Error
from bs4 import BeautifulSoup
import chromadb
from chromadb.config import Settings
from collections import Counter
import html
import fitz
import glob
from gensim.models import Word2Vec
import json
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk import FreqDist, ConditionalFreqDist
from nltk.corpus import stopwords, wordnet, words
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import os
import pandas as pd
from pandas import DataFrame, Series
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
import re
import spacy
from spacy import Language
import sqlite3
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from typing import List, Optional, Dict, Tuple, Any, Set
import tiktoken
from tiktoken.core import Encoding
import unicodedata
from lxml import etree

DELIMITERS: Set[ str ] = { '. ', '; ', '? ', '! ', ', ' }

SYMBOLS: Set[ str ] = {
		"@",
		"#",
		"$",
		"^",
		"*",
		"<",
		">",
		"+",
		"=",
		"|",
		"\\",
		"<",
		">",
		":",
		"[",
		"]",
		"{",
		"}",
		"(",
		")",
		"`",
		"~",
		"-",
		"_",
		'"',
		"'",
		".",
}

ASCII_LETTERS: Set[ str ] = set( string.ascii_letters )

DIGITS: Set[ str ] = set( string.digits )

PUNCTUATION: Set[ str ] = set( string.punctuation )

WHITESPACE: Set[ str ] = {
		" ", "\t", "\n", "\r", "\v", "\f"
}

CONTROL_CHARACTERS: Set[ str ] = {
		chr( i ) for i in range( 0x00, 0x20 )
}.union( { chr( 0x7F ) } )

NUMERALS = (r"\bM{0,4}(CM|CD|D?C{0,3})"
            r"(XC|XL|L?X{0,3})"
            r"(IX|IV|V?I{0,3})\b")

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

class Processor( ):
	'''
		
		Purpose:
		Base class for processing classes
		
	'''
	lemmatizer: Optional[ WordNetLemmatizer ]
	stemmer: Optional[ PorterStemmer ]
	file_path: Optional[ str ]
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
	stop_words: Optional[ set ]
	vocabulary: Optional[ set ]
	corpus: Optional[ DataFrame ]
	removed: Optional[ List[ str ] ]
	frequency_distribution: Optional[ DataFrame ]
	
	def __init__( self ):
		self.lemmatizer = WordNetLemmatizer( )
		self.stemmer = PorterStemmer( )
		self.files = [ ]
		self.lines = [ ]
		self.tokens = [ ]
		self.lines = [ ]
		self.pages = [ ]
		self.ids = [ ]
		self.chunks = [ ]
		self.chunk_size = 0
		self.paragraphs = [ ]
		self.embedddings = [ ]
		self.stop_words = set( )
		self.vocabulary = set( )
		self.frequency_distribution = { }
		self.encoding = None
		self.corrected = None
		self.lowercase = None
		self.raw_html = None
		self.corpus = None
		self.file_path = ''
		self.raw_input = ''
		self.normalized = ''
		self.lemmatized = ''
		self.tokenized = ''
		self.cleaned_text = ''

class TextParser( Processor ):
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
	lowercase: Optional[ str ]
	cleaned_text: Optional[ str ]
	cleaned_lines: Optional[ List[ str ] ]
	cleaned_tokens: Optional[ List[ str ] ]
	cleaned_pages: Optional[ List[ str ] ]
	cleaned_html: Optional[ str ]
	conditional_distribution: Optional[ DataFrame ]
	PUNCTUATION: Optional[ Set[ str ] ]
	CONTROL_CHARACTERS: Optional[ Set[ str ] ]
	DELIMITERS: Optional[ Set[ str ] ]
	DIGITS: Optional[ Set[ str ] ]
	SYMBOLS: Optional[ Set[ str ] ]
	NUMERALS: Optional[ str ]

	def __init__( self ):
		'''

			Purpose:
			---------
			Constructor for 'Text' objects

		'''
		super( ).__init__( )
		self.PUNCTUATION = PUNCTUATION
		self.CONTROL_CHARACTERS = ( {chr(i) for i in range(0x00, 0x20)} | {chr(0x7F)} )
		self.DELIMITERS = DELIMITERS
		self.DIGITS = DIGITS
		self.SYMBOLS = SYMBOLS
		self.NUMERALS = NUMERALS
		self.lemmatizer = WordNetLemmatizer( )
		self.stemmer = PorterStemmer( )
		self.encoding = tiktoken.get_encoding( 'cl100k_base' )
		self.lines = [ ]
		self.tokens = [ ]
		self.pages = [ ]
		self.ids = [ ]
		self.paragraphs = [ ]
		self.chunks = [ ]
		self.chunk_size = 10
		self.raw_pages = [ ]
		self.stop_words = set( )
		self.frequency_distribution = { }
		self.file_path = ''
		self.raw_input = ''
		self.normalized = ''
		self.lemmatized = ''
		self.tokenized = ''
		self.cleaned_text = ''
		self.vocabulary = None
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
		return [ # Attributes
				'file_path',
				'raw_input',
				'raw_pages',
				'normalized',
				'lemmatized',
				'tokenized',
				'corrected',
				'cleaned_text',
				'words',
				'paragraphs',
				'words',
				'pages',
				'chunks',
				'chunk_size',
				'stop_words',
				'removed',
				'lowercase',
				'encoding',
				'vocabulary',
				'translator',
				'lemmatizer',
				'stemmer',
				'tokenizer',
				'vectorizer',
				'conditional_distribution',
				# Methods
				'split_sentences',
				'split_pages',
				'collapse_whitespace',
				'compress_whitespace',
				'remove_punctuation',
				'remove_numbers',
				'remove_special',
				'remove_html',
				'remove_markdown',
				'remove_stopwords',
				'remove_formatting',
				'remove_headers',
				'remove_encodings',
				'tiktokenize',
				'lemmatize_text',
				'normalize_text',
				'chunk_text',
				'chunk_sentences',
				'chunk_files',
				'chunk_data',
				'chunk_datasets',
				'create_wordbag',
				'clean_file',
				'clean_files',
				'convert_jsonl',
				'speech_tagging',
				'split_paragraphs',
				'calculate_frequency_distribution',
				'create_vocabulary',
				'create_wordbag',
				'create_vectors',
				'encode_sentences',
				'semantic_search' ]
	
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
			raw_text = open( self.file_path, mode='r', encoding='utf-8', errors='ignore' ).read()
			return raw_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'load_text( self, file_path: str ) -> str'
			raise exception
			
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
			_text = text.lower( )
			return ' '.join( _text.split( ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'collapse_whitespace( self, path: str ) -> str:'
			raise exception
		
	def remove_punctuation( self, text: str ) -> str:
		"""

			Purpose:
			--------
			Removes punctuation characters from text while preserving sentence
			delimiters used by sentence tokenization.

			Parameters:
			----------
			text : str
				The raw input text from which punctuation should be removed.

			Returns:
			-------
			str
				Cleaned text with non-delimiter punctuation removed and sentence
				delimiters preserved.

		"""
		try:
			throw_if( 'text', text )
			_text = text.lower( )
			_sentence_delimiters = { '.', '?', '!' }
			_chars = [ ]
			
			for char in _text:
				if char in _sentence_delimiters:
					_chars.append( char )
				elif char in self.PUNCTUATION:
					_chars.append( ' ' )
				else:
					_chars.append( char )
			
			_cleaned = ''.join( _chars )
			_cleaned = re.sub( r'\s+', ' ', _cleaned ).strip( )
			_cleaned = re.sub( r'\s+([.!?])', r'\1', _cleaned )
			_cleaned = re.sub( r'([.!?])(?=\w)', r'\1 ', _cleaned )
			return _cleaned
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'remove_punctuation( self, text: str ) -> str:'
			raise exception
			
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
			return text.lower( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'normalize_text( self, text: str ) -> str:'
			raise exception
		
	def remove_errors( self, text: str ) -> str:
		"""

			Purpose:
			----------
			Removes tokens that are not recognized as valid English words while
			preserving sentence delimiters used by sentence tokenization.

			Parameters:
			----------
			text : str
				The raw input text.

			Returns:
			-------
			str
				Text with out-of-vocabulary word tokens removed and sentence
				delimiters preserved.

		"""
		try:
			throw_if( 'text', text )
			_vocab = set( words.words( 'en' ) )
			_sentence_delimiters = { '.', '?', '!', ';', ':' }
			_text = text.lower( )
			_tokens = word_tokenize( _text )
			_cleaned = [ ]
			
			for token in _tokens:
				if token in _sentence_delimiters:
					_cleaned.append( token )
				elif token.isalpha( ) and token in _vocab:
					_cleaned.append( token )
			
			_data = ' '.join( _cleaned )
			_data = re.sub( r'\s+([.!?;:])', r'\1', _data )
			_data = re.sub( r'([.!?;:])(?=\w)', r'\1 ', _data )
			_data = re.sub( r'\s+', ' ', _data ).strip( )
			return _data
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'remove_errors( self, text: str  ) -> str'
			raise exception
	
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
			_text = text.lower( )
			_cleaned = [ ]
			_fragments = _text.split( )
			for char in _fragments:
				if len( char) > 2:
					_cleaned.append( char )
			return ' '.join( _cleaned )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'remove_fragments( self, text: str ) -> str:'
			raise exception
		
	def remove_symbols( self, text: str ) -> str | None:
		"""

			Purpose:
			-----------
			Removes configured symbol characters from text while preserving sentence
			delimiters used by sentence tokenization.

			Parameters:
			-----------
			text : str
				The raw text potentially containing special symbol characters.

			Returns:
			--------
			str | None
				Cleaned text with symbols removed and sentence delimiters preserved.

		"""
		try:
			throw_if( 'text', text )
			_text = text.lower( )
			_sentence_delimiters = { '.', '?', '!', ';', ':' }
			_symbols = self.SYMBOLS.difference( _sentence_delimiters )
			return ''.join( c for c in _text if c not in _symbols )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'remove_symbols( self, text: str ) -> str:'
			raise exception
			
	def remove_html( self, text: str ) -> str | None:
		"""

			Purpose:
			--------
			Removes HTML tags from the path.

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
			cleaned_html = BeautifulSoup( self.raw_html, 'html.parser' ).get_text( )
			return cleaned_html
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'remove_html( self, text: str ) -> str'
			raise exception
			
	def remove_xml( self, text: str ) -> str:
		"""
		
			Purpose:
			--------
			Remove XML tags from a string while preserving inner text content using
			lxml for robust, fast, and XPath-capable parsing.
	
			This function safely parses XML fragments by wrapping them in a synthetic
			root element, then extracts all textual content (element text and tails)
			while discarding tags and attributes.
	
			Parameters:
			-----------
			text : str
				Input text containing XML markup.
	
			Returns:
			--------
			str
				Text with XML tags removed and inner text preserved.
	
			Raises:
			-------
			RuntimeError
				Raised when XML parsing fails.
			
		"""
		throw_if( 'text', text )
		try:
			_text = text.lower( )
			wrapped_text = f"<root>{_text}</root>"
			parser = etree.XMLParser( recover=True, remove_comments=True,
				remove_blank_text=False )
			
			root = etree.fromstring( wrapped_text.encode( "utf-8" ), parser )
			text_parts = [ ]
			for element in root.iter( ):
				if element.text:
					text_parts.append( element.text )
				if element.tail:
					text_parts.append( element.tail )
			
			return "".join( text_parts )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'remove_xml( self, text: str ) -> str'
			raise exception
			
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
			self.raw_input = text.lower( )
			_text = re.sub( r'\[.*?]\(.*?\)', ' ', self.raw_input )
			_unmarked = re.sub( r'[`_*#~><-]', ' ', _text )
			self.cleaned_text = re.sub( r'!\[.*?]\(.*?\)', ' ', _unmarked )
			return self.cleaned_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'remove_markdown( self, path: str ) -> str'
			raise exception
		
	def remove_stopwords( self, text: str ) -> str:
		"""

			Purpose:
			--------
			Removes English stop words while preserving sentence delimiters used by
			sentence tokenization.

			Parameters:
			-----------
			text : str
				The raw input text from which stop words should be removed.

			Returns:
			--------
			str
				Text with stop words removed and sentence delimiters preserved.

		"""
		try:
			throw_if( 'text', text )
			_stop_words = set( stopwords.words( 'english' ) )
			_sentence_delimiters = { '.', '?', '!', ';', ':' }
			_text = text.lower( )
			_tokens = word_tokenize( _text )
			_filtered = [ ]
			
			for token in _tokens:
				if token in _sentence_delimiters:
					_filtered.append( token )
				elif token.isalnum( ) and token not in _stop_words:
					_filtered.append( token )
			
			_cleaned = ' '.join( _filtered )
			_cleaned = re.sub( r'\s+([.!?;:])', r'\1', _cleaned )
			_cleaned = re.sub( r'([.!?;:])(?=\w)', r'\1 ', _cleaned )
			_cleaned = re.sub( r'\s+', ' ', _cleaned ).strip( )
			return _cleaned
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'remove_stopwords( self, text: str ) -> str'
			raise exception
		
	def remove_encodings( self, text: str ) -> str | None:
		"""

			Purpose:
			---------
			Cleans encoding artifacts safely without corrupting already-decoded Unicode
			text extracted from PDFs, Word documents, HTML, or plain text files.

			Parameters:
			----------
			text : str
				Input string potentially containing HTML entities, literal Unicode escape
				sequences, control characters, or common mojibake artifacts.

			Returns:
			-------
			str | None
				Unicode-normalized text with encoding artifacts repaired or removed.

		"""
		try:
			throw_if( 'text', text )
			self.raw_input = text
			cleaned = html.unescape( self.raw_input )
			
			if re.search( r'\\u[0-9a-fA-F]{4}|\\x[0-9a-fA-F]{2}', cleaned ):
				try:
					cleaned = bytes( cleaned, 'utf-8' ).decode( 'unicode_escape' )
				except UnicodeDecodeError:
					pass
			
			replacements = {
					'â': '’',
					'â': '‘',
					'â': '“',
					'â': '”',
					'â': '–',
					'â': '—',
					'â¢': '•',
					'â': '⁄',
					'â¢': '™',
					'Â§': '§',
					'Â¶': '¶',
					'Â©': '©',
					'Â®': '®',
					'Ã': '×',
					'Â': '',
			}
			
			for bad, good in replacements.items( ):
				cleaned = cleaned.replace( bad, good )
			
			cleaned = unicodedata.normalize( 'NFKC', cleaned )
			cleaned = re.sub( r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned )
			cleaned = re.sub( r'[ \t]{2,}', ' ', cleaned )
			cleaned = re.sub( r' +\n', '\n', cleaned )
			cleaned = re.sub( r'\n +', '\n', cleaned )
			cleaned = re.sub( r'\n{3,}', '\n\n', cleaned )
			return cleaned.strip( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'remove_encodings( self, text: str ) -> str'
			raise exception
			
	def remove_headers( self, filepath: str, lines: int=50, headers: int=3, footers: int=3 ) -> str | None:
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
			if not os.path.exists( filepath ):
				raise FileNotFoundError( f'File not found: {filepath}' )
			else:
				self.file_path = filepath
			if lines < 6:
				raise ValueError( 'Argument \"lines_per_page\" should be at least 6.' )
			if headers < 0 or footers < 0:
				msg = 'Arguments \"header_lines\" and \"footer_lines\" must be non-negative.'
				raise ValueError( msg )
			
			with open( self.file_path, 'r', encoding='utf-8', errors='ignore' ) as fh:
				self.lines = fh.readlines( )
			
			self.pages = [ self.lines[ i: i + lines ] for i in
			          range( 0, len( self.lines ), lines ) ]
			
			header_counts = { }
			footer_counts = { }
			for page in self.pages:
				n = len( page )
				if n == 0:
					continue
				
				if headers > 0 and n >= headers:
					hdr = tuple( page[ :headers ] )
					if hdr in header_counts:
						header_counts[ hdr ] += 1
					else:
						header_counts[ hdr ] = 1
				
				if footers > 0 and n >= footers:
					ftr = tuple( page[ -footers: ] )
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
			for page in self.pages:
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
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'remove_headers( self, filepath: str ) -> str'
			raise exception
			
	def remove_numbers( self, text: str ) -> str | None:
		"""

			Purpose:
			---------
			Removes the numbers 0 through 9 from the input text.

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
			_text = text.lower( )
			return re.sub( r'\d+', '', _text )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'remove_encodings( self, text: str ) -> str'
			raise exception
			
	def remove_numerals( self, text: str ) -> str | None:
		"""

			Purpose:
			---------
			Removes the numbers 0 through 9 from the input text.

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
			self.raw_input = text.lower( )
			self.cleaned_text = re.sub( self.NUMERALS, ' ', self.raw_input, flags=re.IGNORECASE, )
			return self.cleaned_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'remove_numerals( self, text: str ) -> str'
			raise exception
			
	def remove_images( self, text: str ) -> str:
		"""
			Purpose:
			--------
			Remove image references from text, including Markdown images, HTML <img> tags,
			and standalone image URLs.
	
			Parameters:
			-----------
			text : str
				Input text.
	
			Returns:
			--------
			str
				Text with image references removed.
	
			Raises:
			-------
			RuntimeError
				Raised when processing fails.
		"""
		throw_if( "text", text )
		
		try:
			self.raw_input = text
			
			# Remove Markdown images: ![alt](path)
			without_markdown_images = re.sub(
				r"!\[[^\]]*]\([^)]*\)",
				" ",
				self.raw_input
			)
			
			# Remove HTML <img> tags
			without_html_images = re.sub(
				r"<img\b[^>]*>",
				" ",
				without_markdown_images,
				flags=re.IGNORECASE
			)
			
			# Remove standalone image URLs
			self.parsed_text = re.sub(
				r"https?://\S+\.(png|jpg|jpeg|gif|bmp|svg|webp)",
				" ",
				without_html_images,
				flags=re.IGNORECASE
			)
			
			return self.parsed_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = ('remove_formatting( self, text: str ) -> str')
			raise exception
			
	def tiktokenize( self, text: str, encoding: str='cl100k_base' ) -> DataFrame | None:
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
			_text = text.lower( )
			self.encoding = tiktoken.get_encoding( encoding )
			token_ids = self.encoding.encode( _text )
			_data = pd.DataFrame( token_ids )
			return _data
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = ('tiktokenize( self, text, encoding) -> List[ int ]')
			raise exception
			
	def split_sentences( self, text: str  ) -> List[ str ] | None:
		"""

			Purpose:
			________
			Splits the text string into a list of strings using NLTK's Punkt sentence tokenizer.
			This function is useful for preparing text for further processing,
			such as tokenization, parsing, or named entity recognition.

			Parameters
			----------
			- text : str
			The raw text string to be segmented into sentences.
			
			- size : int
			THe chunk size

			Returns
			-------
			- List of strings

		"""
		try:
			throw_if( 'text', text )
			_text = text.lower( )
			_sentences = sent_tokenize( _text )
			return _sentences
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'split_sentences( self, text: str ) -> DataFrame'
			raise exception
			
	def split_pages( self, filepath: str, num: int=50 ) -> List[ str ] | None:
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
				page_lines = self.lines[ i: i + num ]
				page_text = '\n'.join( page_lines ).strip( )
				if page_text:
					self.pages.append( page_text )
				i += num
			return self.pages
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'split_pages( file_path )'
			raise exception
			
	def split_paragraphs( self, filepath: str ) -> DataFrame | None:
		"""

			Purpose:
			---------
			Reads  a file and splits it into paragraphs. A paragraph is defined as a block
			of text separated by one or more empty lines (eg, '\n\n').

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
	
	def create_frequency_distribution( self, tokens: List[ str ] ) -> DataFrame | None:
		"""

			Purpose:
			--------
			Creates a word frequency freq_dist from a list of tokens.

			Parameters:
			-----------
			- tokens: List[ str ]

			Returns:
			- dict: Dictionary of words and their corresponding frequencies.

		"""
		try:
			throw_if( 'tokens', tokens )
			self.tokens = tokens
			_freqdist = FreqDist( dict( Counter( self.tokens ) ) )
			_words = _freqdist.items( )
			_data = pd.DataFrame( _words, columns=[ 'Word', 'Frequency' ] )
			_data.index.name = 'ID'
			return _data
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'create_frequency_distribution(self, tokens: List[ str ])->DataFrame'
			raise exception
			
	def create_vocabulary( self, tokens: List[ str ] ) -> Series | None:
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
			throw_if( 'tokens', tokens )
			self.tokens = tokens
			_freqdist = FreqDist( dict( Counter( self.tokens ) ) )
			_vocab = _freqdist.items( )
			_vocabulary = pd.DataFrame( _vocab, columns=[ 'Word', 'Frequency' ] )
			_words = _vocabulary.iloc[ :, 0 ]
			return _words
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = ('create_vocabulary(self, freq_dist: dict, min: int=1)->List[str]')
			raise exception
			
	def create_wordbag( self, tokens: List[ str ] ) -> DataFrame | None:
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
			_freqdist = FreqDist( dict( Counter( self.tokens ) ) )
			_words = _freqdist.keys( )
			_data = pd.DataFrame( _words )
			return _data
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'create_wordbag( self, words: List[ str ] ) -> dict'
			raise exception
			
	def create_vectors( self, tokens: List[ str ] ) -> DataFrame | None:
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
			fake_docs = [ [ word ] for word in tokens ]
			joined_docs = [ ' '.join( doc ) for doc in fake_docs ]
			vectorizer = TfidfVectorizer( )
			X = vectorizer.fit_transform( joined_docs )
			feature_names = vectorizer.get_feature_names_out( )
			embeddings = { }
			for idx, word in enumerate( tokens ):
				vector = X[ idx ].toarray( ).flatten( )
				embeddings[ word ] = vector
			
			_data = pd.DataFrame( data=embeddings, columns=feature_names )
			return embeddings
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'create_vectors( self, tokens: List[str]) -> Dict[str, np.ndarray]'
			raise exception
			
	def clean_file( self, filepath: str ) -> str | None:
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
			throw_if( 'filepath', filepath )
			if not os.path.exists( filepath ):
				raise FileNotFoundError( f'File not found: {filepath}' )
			else:
				_sourcepath = filepath
				_text = open( _sourcepath, 'r', encoding='utf-8', errors='ignore' ).read( )
				_collapsed = self.collapse_whitespace( _text )
				_compressed = self.compress_whitespace( _collapsed )
				_normalized = self.normalize_text( _compressed )
				_encoded = self.remove_encodings( _normalized )
				_special = self.remove_symbols( _encoded )
				_cleaned = self.remove_fragments( _special )
				_recompress = self.compress_whitespace( _cleaned )
				_lemmatized = self.lemmatize_text( _recompress )
				_stops = self.remove_stopwords( _lemmatized )
				return _stops
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'clean_file( self, src: str ) -> str'
			raise exception
			
	def clean_files( self, source: str, destination: str ) -> None:
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
			throw_if( 'src', source )
			throw_if( 'dest', destination )
			if not os.path.exists( source ):
				raise FileNotFoundError( f'File not found: {source}' )
			elif not os.path.exists( destination ):
				raise FileNotFoundError( f'File not found: {destination}' )
			else:
				_source = source
				_destpath = destination
				_files = os.listdir( _source )
				for f in _files:
					_processed = [ ]
					_filename = os.path.basename( f )
					_sourcepath = _source + '\\' + _filename
					_text = open( _sourcepath, 'r', encoding='utf-8', errors='ignore' ).read( )
					_collapsed = self.collapse_whitespace( _text )
					_compressed = self.compress_whitespace( _collapsed )
					_normalized = self.normalize_text( _compressed )
					_encoded = self.remove_encodings( _normalized )
					_special = self.remove_symbols( _encoded )
					_cleaned = self.remove_fragments( _special )
					_recompress = self.compress_whitespace( _cleaned )
					_lemmatized = self.lemmatize_text( _recompress )
					_stops = self.remove_stopwords( _lemmatized )
					_sentences = self.split_sentences( _stops )
					_destination = _destpath + '\\' + _filename
					_clean = open( _destination, 'wt', encoding='utf-8', errors='ignore' )
					_lines = ' '.join( _sentences )
					_clean.write( _lines )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'clean_files( self, src: str, dest: str )'
			raise exception
			
	def chunk_files( self, source: str, destination: str ) -> None:
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
			if not os.path.exists( source ):
				raise FileNotFoundError( f'File not found: {source}' )
			elif not os.path.exists( destination ):
				raise FileNotFoundError( f'File not found: {destination}' )
			else:
				_source = source
				_destination = destination
				_files = os.listdir( _source )
				_words = [ ]
				for f in _files:
					_processed = [ ]
					_filename = os.path.basename( f )
					_sourcepath = _source + '\\' + _filename
					_text = open( _sourcepath, 'r', encoding='utf-8', errors='ignore' ).read( )
					_sentences =  self.split_sentences( _text )
					_datamap = [ ]
					for v in _sentences:
						_datamap.append( v )
						
					for s in _datamap:
						_processed.append( s )
					
					_final = _destination + '\\' + _filename
					_clean = open( _final, 'wt', encoding='utf-8', errors='ignore' )
					for p in _processed:
						_clean.write( p )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'chunk_files( self, src: str, dest: str )'
			raise exception
			
	def chunk_data( self, filepath: str, size: int=10  ) -> DataFrame | None:
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
			throw_if( 'filepath', filepath )
			if not os.path.exists( filepath ):
				raise FileNotFoundError( f'File not found: {filepath}' )
			else:
				_source = filepath
				_processed = [ ]
				_wordlist = [ ]
				_vocab = words.words( 'en' )
				_text = open( _source, 'r', encoding='utf-8', errors='ignore' ).read( )
				_lower = _text.lower( )
				_tokens = _lower.split( )
				for s in _tokens:
					if s.isalpha( ) and s in _vocab:
						_wordlist.append( s )
				self.chunks = [ _wordlist[ i: i + size ] for i in range( 0, len( _wordlist ), size ) ]
				for i, c in enumerate( self.chunks ):
					_item =  '[' + ' '.join( c ) + '],'
					_processed.append( _item )
				_data = pd.DataFrame( _processed )
				return _data
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'chunk_data( self, filepath: str, size: int=512  ) -> DataFrame'
			raise exception
			
	def chunk_datasets( self, source: str, destination: str, size: int=10 ) -> DataFrame:
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
			throw_if( 'filepath', source )
			throw_if( 'destination', destination )
			if not os.path.exists( source ):
				raise FileNotFoundError( f'File not found: {source}' )
			elif not os.path.exists( destination ):
				raise FileNotFoundError( f'File not found: {destination}' )
			else:
				_src = source
				_destination = destination
				_files = os.listdir( _src )
				_words = [ ]
				for f in _files:
					_processed = [ ]
					_filename = os.path.basename( f )
					_sourcepath = _src+ '\\' + _filename
					_text = open( _sourcepath, 'r', encoding='utf-8', errors='ignore' ).read( )
					_collapsed = self.collapse_whitespace( _text )
					_compressed = self.compress_whitespace( _collapsed )
					_normalized = self.normalize_text( _compressed )
					_encoded = self.remove_encodings( _normalized )
					_special = self.remove_symbols( _encoded )
					_cleaned = self.remove_fragments( _special )
					_recompress = self.compress_whitespace( _cleaned )
					_lemmatized = self.lemmatize_text( _recompress )
					_stops = self.remove_stopwords( _lemmatized )
					_tokens = _stops.split( None )
					_chunks = [ _tokens[ i: i + size ] for i in range( 0, len( _tokens ), size ) ]
					_datamap = [ ]
					for i, c in enumerate( _chunks ):
						_row = ' '.join( c )
						_datamap.append( _row )
						
					for s in _datamap:
						_processed.append( s )
					
					_name = _filename.replace( '.txt', '.xlsx' )
					_savepath = ( _destination + f'\\' + _name )
					_data = pd.DataFrame( _processed, columns=[ 'Data', ] )
					_data.to_excel( _savepath, sheet_name='Dataset', index=False, columns=[ 'Data', ] )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'chunk_data( self, filepath: str, size: int=15  ) -> DataFrame'
			raise exception
			
	def convert_jsonl( self, source: str, destination: str, size: int=10 ) -> None:
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
			throw_if( 'source', source )
			throw_if( 'destination', destination )
			if not os.path.exists( source ):
				raise FileNotFoundError( f'File not found: {source}' )
			elif not os.path.exists( destination ):
				raise FileNotFoundError( f'File not found: {destination}' )
			else:
				_source = source
				_destpath = destination
				_files = os.listdir( _source )
				_wordlist = [ ]
				for f in _files:
					_processed = [ ]
					_filename = os.path.basename( f )
					_sourcepath = _source + '\\' + _filename
					_text = open( _sourcepath, 'r', encoding='utf-8', errors='ignore' ).read( )
					_tokens =  _text.split( ' ' )
					_chunks = [ _tokens[ i: i + size ] for i in range( 0, len( _tokens ), size ) ]
					_datamap = [ ]
					for i, c in enumerate( _chunks ):
						_value = '{ ' +f' {i} : [ ' + ' '.join( c ) + ' ] }, ' + "\n"
						_datamap.append( _value )
						
					for s in _datamap:
						_processed.append( s )
					
					_destination = _destpath + '\\' + _filename.replace( '.txt', '.jsonl' )
					_clean = open( _destination, 'wt', encoding='utf-8', errors='ignore' )
					for p in _processed:
						_clean.write( p )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'convert_jsonl( self, source: str, desination: str )'
			raise exception
			
	def encode_sentences( self, tokens: List[ str ], model: str='all-MiniLM-L6-v2' ) -> \
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
			throw_if( 'tokens', tokens )
			throw_if( 'model', model )
			_transformer = SentenceTransformer( model )
			_tokens = [ self.lemmatizer.lemmatize( t ) for t in tokens ]
			_encoding = _transformer.encode( _tokens, show_progress_bar=True )
			return ( self.cleaned_tokens, np.array( _encoding ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'encode_sentences( self, sentences: List[ str ], model_name ) -> ( )'
			raise exception
			
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
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = ('semantic_search( self, query: str, tokens: List[ str ], '
			                    'embeddings: np.ndarray, model: SentenceTransformer,  '
			                    'top_k: int=5 ) -> List[ tuple[ str, float ] ]')
			raise exception

class NltkParser( Processor ):
	'''

		Purpose:
		---------
		Class providing NLTK-based natural language processing functionality for text that
		has already been cleaned and normalized by the Text Processing expander.

		Methods:
		--------
		word_tokenizer( self, text: str ) -> List [ str ]
		sentence_tokenizer( self, text: str ) -> List [ str ]
		word_stemmer( self, text: str ) -> List [ str ]
		word_lemmatizer( self, text: str ) -> List [ str ]
		pos_tagger( self, text: str ) -> List[ Tuple[ str, str ] ]
		named_entity_recognition( self, text: str ) -> List[ Tuple[ str, str ] ]

	'''
	word_tokens: Optional[ List[ str ] ]
	sentence_tokens: Optional[ List[ str ] ]
	stemmed_tokens: Optional[ List[ str ] ]
	lemmatized_tokens: Optional[ List[ str ] ]
	tagged_tokens: Optional[ List[ Tuple[ str, str ] ] ]
	named_entities: Optional[ List[ Tuple[ str, str ] ] ]
	
	def __init__( self ) -> None:
		'''

			Purpose:
			---------
			Initializes the NltkParser and prepares internal containers used by the
			NLTK processing methods.

			Parameters:
			-----------
			- self

			Returns:
			--------
			- None

		'''
		super( ).__init__( )
		self.initialize_resources( )
		self.word_tokens = [ ]
		self.sentence_tokens = [ ]
		self.stemmed_tokens = [ ]
		self.lemmatized_tokens = [ ]
		self.tagged_tokens = [ ]
		self.named_entities = [ ]
	
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
		return [ 'initialize_resources', 'word_tokenizer', 'sentence_tokenizer', 'word_stemmer',
		         'word_lemmatizer', 'pos_tagger', 'named_entity_recognition', 'word_tokens',
		         'sentence_tokens', 'stemmed_tokens', 'lemmatized_tokens', 'tagged_tokens',
		         'named_entities' ]
	
	def initialize_resources( self ) -> None:
		'''

			Purpose:
			---------
			Ensures the NLTK tokenizers, taggers, and corpora required by this class
			are available before processing begins.

			Parameters:
			-----------
			- self

			Returns:
			--------
			- None

		'''
		try:
			required_resources: List[ Tuple[ str, str ] ] = [
					('tokenizers/punkt', 'punkt'),
					('tokenizers/punkt_tab', 'punkt_tab'),
					('corpora/wordnet', 'wordnet'),
					('corpora/omw-1.4', 'omw-1.4'),
					('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
					('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
					('chunkers/maxent_ne_chunker', 'maxent_ne_chunker'),
					('chunkers/maxent_ne_chunker_tab', 'maxent_ne_chunker_tab'),
					('corpora/words', 'words'), ]
			
			for resource_path, resource_name in required_resources:
				try:
					nltk.data.find( resource_path )
				except LookupError:
					nltk.download( resource_name )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'NltkParser'
			exception.method = 'NltkParser._ensure_nltk_resources( self ) -> None'
			raise exception
	
	def word_tokenizer( self, text: str ) -> List[ str ] | None:
		'''

			Purpose:
			---------
			Tokenizes the input text into word tokens and returns a display-ready
			string with one token per line.

			Parameters:
			-----------
			- text: str
				The cleaned text to tokenize into words.

			Returns:
			--------
			- str

		'''
		try:
			throw_if( 'text', text )
			_text = text.lower( )
			self.word_tokens = word_tokenize( _text )
			words = [ token for token in self.word_tokens ]
			return words
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'NltkParser'
			exception.method = 'word_tokenizer( self, text: str ) -> List[ str ]'
			raise exception
	
	def sentence_tokenizer( self, text: str ) -> List[ str ] | None:
		'''

			Purpose:
			---------
			Tokenizes the input text into sentences and returns a display-ready
			numbered string.

			Parameters:
			-----------
			- text: str
				The cleaned text to tokenize into sentences.

			Returns:
			--------
			- str

		'''
		try:
			throw_if( 'text', text )
			_text = text.lower( )
			self.sentence_tokens = sent_tokenize( _text )
			return self.sentence_tokens
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'NltkParser'
			exception.method = 'tokenize_sentences( self, text: str ) -> str'
			raise exception
	
	def word_stemmer( self, text: str ) -> List[ str ] | None:
		'''

			Purpose:
			---------
			Applies stemming to the input text and returns a whitespace-joined
			display-ready string of stemmed tokens.

			Parameters:
			-----------
			- text: str
				The cleaned text to stem.

			Returns:
			--------
			- str

		'''
		try:
			throw_if( 'text', text )
			_text = text.lower( )
			self.word_tokens = word_tokenize( _text )
			self.stemmed_tokens = [ self.stemmer.stem( t ) for t in self.word_tokens
			                        if isinstance( t, str ) and t.strip( ) ]
			
			return self.stemmed_tokens
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'NltkParser'
			exception.method = 'stemmer( self, text: str ) -> str'
			raise exception
	
	def word_lemmatizer( self, text: str ) -> List[ str ] | None:
		'''

			Purpose:
			---------
			Applies WordNet lemmatization to the input text and returns a
			whitespace-joined display-ready string of lemmatized tokens.

			Parameters:
			-----------
			- text: str
				The cleaned text to lemmatize.

			Returns:
			--------
			- str

		'''
		try:
			throw_if( 'text', text )
			_text = text.lower( )
			self.word_tokens = word_tokenize( _text )
			self.lemmatized_tokens = [ self.lemmatizer.lemmatize( t ) for t in self.word_tokens
			                           if isinstance( t, str ) and t.strip( ) ]
			
			return self.lemmatized_tokens
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'NltkParser'
			exception.method = 'lemmatizer( self, text: str ) -> str'
			raise exception
	
	def pos_tagger( self, text: str ) -> List[ Tuple[ str, str ] ] | None:
		'''

			Purpose:
			---------
			Applies part-of-speech tagging to the input text and returns a
			display-ready string containing one token-tag pair per line.

			Parameters:
			-----------
			- text: str
				The cleaned text to tag.

			Returns:
			--------
			- str

		'''
		try:
			throw_if( 'text', text )
			_text = text.lower( )
			self.word_tokens = word_tokenize( _text )
			self.tagged_tokens = nltk.pos_tag( self.word_tokens )
			return self.tagged_tokens
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'NltkParser'
			exception.method = 'pos_tagger( self, text: str ) -> str'
			raise exception
	
	def named_entity_recognition( self, text: str ) -> List[ Tuple[ str, str ] ] | None:
		'''

			Purpose:
			---------
			Applies named entity recognition to the input text and returns a
			display-ready string containing one extracted entity per line.

			Parameters:
			-----------
			- text: str
				The cleaned text to analyze for named entities.

			Returns:
			--------
			- str

		'''
		try:
			throw_if( 'text', text )
			_text = text.lower( )
			self.word_tokens = word_tokenize( _text )
			self.tagged_tokens = nltk.pos_tag( self.word_tokens )
			tree = nltk.ne_chunk( self.tagged_tokens )
			self.named_entities = [ ]
			for node in tree:
				if hasattr( node, 'label' ):
					label = node.label( )
					entity_text = ' '.join( token for token, _ in node.leaves( )
					                        if isinstance( token, str ) and token.strip( ) )
					
					if entity_text:
						self.named_entities.append( (entity_text, label) )
			
			return self.named_entities
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'NltkParser'
			exception.method = 'named_entity_recogniztion( self, text: str ) -> str'
			raise exception
	
	def chunk_words( self, text: str, size: int=5 ) -> DataFrame | None:
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
			_text = text.lower( )
			_tokens = nltk.word_tokenize( _text )
			_sentences = [ _tokens[ i: i + size ] for i in range( 0, len( _tokens ), size ) ]
			_datamap = [ ]
			for index, chunk in enumerate( _sentences ):
				_item = ' '.join( chunk )
				_datamap.append( _item )
			
			_data = pd.DataFrame( _datamap )
			return _data
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'chunk_sentences( self, text: str, max: int=10 ) -> DataFrame'
			raise exception
	
	def chunk_sentences( self, text: str, size: int=15 ) -> DataFrame | None:
		"""

			Purpose:
			-----------
			Tokenizes cleaned_lines pages and breaks it into chunks for downstream vectors.
			  - Converts pages to lowercase
			  - Tokenizes pages using NLTK's sent_tokenize
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
			_text = text.lower( )
			_tokens = sent_tokenize( _text )
			_sentences = [ _tokens[ i: i + size ] for i in range( 0, len( _tokens ), size ) ]
			_datamap = [ ]
			for i, c in enumerate( _sentences ):
				_item = ' '.join( c )
				_datamap.append( _item )
			
			_data = pd.DataFrame( _datamap )
			return _data
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'NltkParser'
			exception.method = 'chunk_sentences( self, text: str, max: int=512 ) -> DataFrame'
			raise exception

class WordParser( Processor ):
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
	document: Optional[ Document ]
	raw_text: Optional[ str ]
	paragraphs: Optional[ List[ str ] ]
	file_path: Optional[ str ]
	vocabulary: Optional[ set ]
	document: Optional[ Document ]
	
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
		self.document = Document( open( self.file_path, 'rt+' ) )
		self.page_text = ''
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
		return [ 'extract_text',
		         'split_sentences',
		         'clean_sentences',
		         'create_vocabulary',
		         'compute_frequency_distribution',
		         'summarize',
		         'filepath',
		         'raw_text',
		         'paragraphs',
		         'sentences',
		         'cleaned_sentences',
		         'vocabulary',
		         'freq_dist' ]
	
	def extract_text( self, num: int=1 ) -> str | None:
		"""

			Purpose:
			--------
			Extracts raw text and paragraphs from the .docx file.

		"""
		try:
			self.page_text = self.document.get_page_text( pno=num )
			return self.page_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'Word'
			exception.method = 'extract_text( self, num: int ) -> str'
			raise exception
			
	def split_sentences( self ) -> List[ str ] | None:
		"""

			Purpose:
			--------
			Splits the raw text into sentences.

		"""
		try:
			_text = self.page_text.lower( )
			self.sentences = sent_tokenize(_text )
			return self.sentences
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'Word'
			exception.method = 'split_sentences( self ) -> List[ str ]'
			raise exception
			
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
			exception.module = 'processors'
			exception.cause = 'Word'
			exception.method = 'clean_sentences( self ) -> List[ str ]'
			raise exception
			
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
			exception.module = 'processors'
			exception.cause = 'Word'
			exception.method = 'create_vocabulary( self ) -> List[ str ]'
			raise exception
			
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
			exception.module = 'processors'
			exception.cause = 'Word'
			exception.method = 'compute_frequency_distribution( self ) -> Dict[ str, int ]'
			raise exception
			
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

class PdfParser( Processor ):
	"""

		Purpose:
		--------
		Provides general-purpose PDF extraction and PDF text-cleanup utilities. The
		class separates extraction from processing: geometry methods extract page and
		block content, while cleanup methods are invoked explicitly by the Processing
		tab.

		Attributes:
		-----------
		strip_headers : Optional[ bool ]
			Legacy extraction flag retained for compatibility.
		minimum_length : Optional[ int ]
			Minimum line length retained by line extraction helpers.
		extract_tables_enabled : Optional[ bool ]
			Indicates whether table extraction should be attempted where supported.
		pages : Optional[ List ]
			Page-level geometry records.
		lines : Optional[ List ]
			Line-level extraction records.
		blocks : Optional[ List ]
			Block-level extraction records.
		clean_lines : Optional[ List ]
			Cleaned line records.
		extracted_lines : Optional[ List ]
			Extracted PDF lines.
		extracted_tables : Optional[ List ]
			Extracted PDF tables.
		extracted_pages : Optional[ List ]
			Extracted PDF page records.
		file_path : Optional[ str ]
			Active PDF file path.
		page : Optional[ str ]
			Current page text.

		Methods:
		--------
		geometric_extract( self, path: str, count: Optional[ int ]=None,
			header_ratio: float=0.08, footer_ratio: float=0.08,
			preserve_page_breaks: bool=False ) -> str | None
		extract_pages( self, path: str, count: Optional[ int ]=None,
			header_ratio: float=0.08, footer_ratio: float=0.08 ) -> List[ dict ] | None
		remove_repeats( self, pages: List[ dict ], minimum_repeats: int=3 )
			-> List[ dict ] | None
		clean_artifacts( self, text: str ) -> str
		repair_spacing( self, text: str ) -> str
		rejoin_hyphenation( self, text: str, repair_embedded: bool=True ) -> str
		rebuild_pages( self, pages: List[ dict ], preserve_page_breaks: bool=False ) -> str
		extract_lines( self, path: str, count: Optional[ int ]=None ) -> List[ str ] | None
		extract_text( self, path: str, count: Optional[ int ]=None ) -> str | None
		extract_tables( self, path: str, count: Optional[ int ]=None )
			-> List[ pd.DataFrame ] | None
		export_csv( self, tables: List[ pd.DataFrame ], filename: str ) -> None
		export_text( self, lines: List[ str ], path: str ) -> None
		export_excel( self, tables: List[ pd.DataFrame ], path: str ) -> None

	"""
	strip_headers: Optional[ bool ]
	minimum_length: Optional[ int ]
	extract_tables_enabled: Optional[ bool ]
	extracted_lines: Optional[ List ]
	extracted_tables: Optional[ List ]
	extracted_pages: Optional[ List ]
	
	def __init__( self, headers: bool = False, size: int = 10, tables: bool = True ) -> None:
		"""

			Purpose:
			--------
			Initializes the PDF parser.

			Parameters:
			-----------
			headers : bool
				Legacy header flag retained for compatibility.
			size : int
				Minimum line length retained by line-oriented helpers.
			tables : bool
				Indicates whether table extraction should be attempted where supported.

			Returns:
			--------
			None

		"""
		super( ).__init__( )
		self.strip_headers = headers
		self.minimum_length = size
		self.extract_tables_enabled = tables
		self.pages = [ ]
		self.lines = [ ]
		self.blocks = [ ]
		self.clean_lines = [ ]
		self.extracted_lines = [ ]
		self.extracted_tables = [ ]
		self.extracted_pages = [ ]
		self.tables = None
		self.file_path = ''
		self.page = ''
	
	def __dir__( self ) -> List[ str ] | None:
		"""

			Purpose:
			--------
			Provides a list of public attributes and methods.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None
				Class attributes and methods exposed by the parser.

		"""
		return [
				'strip_headers',
				'minimum_length',
				'extract_tables_enabled',
				'file_path',
				'page',
				'pages',
				'lines',
				'blocks',
				'clean_lines',
				'extracted_lines',
				'extracted_tables',
				'extracted_pages',
				'geometric_extract',
				'extract_pages',
				'remove_repeats',
				'clean_artifacts',
				'repair_spacing',
				'rejoin_hyphenation',
				'rebuild_pages',
				'extract_lines',
				'extract_text',
				'extract_tables',
				'export_csv',
				'export_text',
				'export_excel'
		]
	
	def geometric_extract( self, path: str, count: Optional[ int ] = None,
			header_ratio: float = 0.08, footer_ratio: float = 0.08,
			preserve_page_breaks: bool = False ) -> str | None:
		"""

			Purpose:
			--------
			Extracts PDF text using PyMuPDF page geometry. This method performs
			extraction only. It does not remove repeated marginalia, clean artifacts,
			repair spacing, or rejoin hyphenation.

			Parameters:
			-----------
			path : str
				Path to the PDF file.
			count : Optional[ int ]
				Maximum number of pages to process. None processes all pages.
			header_ratio : float
				Top-of-page height ratio used only to classify candidate header blocks.
			footer_ratio : float
				Bottom-of-page height ratio used only to classify candidate footer blocks.
			preserve_page_breaks : bool
				If true, inserts explicit page-break markers between extracted pages.

			Returns:
			--------
			str | None
				Geometry-extracted PDF text.

		"""
		try:
			throw_if( 'path', path )
			pages = self.extract_pages( path=path, count=count,
				header_ratio=header_ratio, footer_ratio=footer_ratio ) or [ ]
			
			return self.rebuild_pages( pages=pages,
				preserve_page_breaks=preserve_page_breaks )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'PdfParser'
			exception.method = 'geometric_extract( self, path: str ) -> str'
			raise exception
	
	def extract_pages( self, path: str, count: Optional[ int ] = None,
			header_ratio: float = 0.08, footer_ratio: float = 0.08 ) -> List[ dict ] | None:
		"""

			Purpose:
			--------
			Extracts page-level text blocks with PyMuPDF geometry and classifies each
			block as header-band, body-band, or footer-band without removing content.

			Parameters:
			-----------
			path : str
				Path to the PDF file.
			count : Optional[ int ]
				Maximum number of pages to process. None processes all pages.
			header_ratio : float
				Top-of-page height ratio used to classify candidate header blocks.
			footer_ratio : float
				Bottom-of-page height ratio used to classify candidate footer blocks.

			Returns:
			--------
			List[ dict ] | None
				Page dictionaries containing ordered block dictionaries.

		"""
		try:
			throw_if( 'path', path )
			
			if header_ratio < 0.0 or header_ratio > 0.30:
				raise ValueError( 'Argument "header_ratio" must be between 0.00 and 0.30.' )
			
			if footer_ratio < 0.0 or footer_ratio > 0.30:
				raise ValueError( 'Argument "footer_ratio" must be between 0.00 and 0.30.' )
			
			self.file_path = path
			self.pages = [ ]
			
			with fitz.open( self.file_path ) as doc:
				for page_index, page in enumerate( doc ):
					if count is not None and page_index >= count:
						break
					
					page_height = float( page.rect.height )
					header_limit = page_height * header_ratio
					footer_limit = page_height * (1.0 - footer_ratio)
					raw_blocks = page.get_text( 'blocks' ) or [ ]
					page_blocks = [ ]
					
					for block_index, block in enumerate( raw_blocks ):
						if len( block ) < 5:
							continue
						
						x0 = float( block[ 0 ] )
						y0 = float( block[ 1 ] )
						x1 = float( block[ 2 ] )
						y1 = float( block[ 3 ] )
						text = block[ 4 ]
						block_type = block[ 6 ] if len( block ) > 6 else 0
						
						if block_type != 0:
							continue
						
						if not isinstance( text, str ) or not text.strip( ):
							continue
						
						midpoint = (y0 + y1) / 2.0
						zone = 'body'
						
						if midpoint <= header_limit:
							zone = 'header'
						elif midpoint >= footer_limit:
							zone = 'footer'
						
						page_blocks.append(
							{
									'page': page_index + 1,
									'index': block_index,
									'x0': x0,
									'y0': y0,
									'x1': x1,
									'y1': y1,
									'midpoint': midpoint,
									'zone': zone,
									'text': text,
									'drop': False,
							}
						)
					
					page_blocks.sort( key=lambda item: (item[ 'y0' ], item[ 'x0' ]) )
					
					self.pages.append(
						{
								'page': page_index + 1,
								'width': float( page.rect.width ),
								'height': page_height,
								'blocks': page_blocks,
						}
					)
			
			return self.pages
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'PdfParser'
			exception.method = 'extract_pages( self, path: str ) -> List[ dict ]'
			raise exception
	
	def remove_repeats( self, pages: List[ dict ], minimum_repeats: int = 3 ) -> List[ dict ] | None:
		"""

			Purpose:
			--------
			Marks repeated candidate marginalia for removal using page-band location and
			normalized recurrence. This method does not use document titles, agencies,
			legal labels, or document-family vocabulary.

			Parameters:
			-----------
			pages : List[ dict ]
				Page dictionaries produced by extract_pages.
			minimum_repeats : int
				Minimum occurrences required before a marginal block is removed.

			Returns:
			--------
			List[ dict ] | None
				Page dictionaries with repeated marginal blocks marked for removal.

		"""
		try:
			throw_if( 'pages', pages )
			
			def normalize_candidate( value: str ) -> str:
				candidate = value.strip( ).lower( )
				candidate = re.sub( r'\s+', ' ', candidate )
				candidate = re.sub( r'\b\d{1,5}\b', '#', candidate )
				candidate = re.sub( r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '#date#', candidate )
				candidate = re.sub( r'\s+', ' ', candidate ).strip( )
				return candidate
			
			def is_page_label( value: str ) -> bool:
				candidate = value.strip( )
				
				if re.fullmatch( r'\d{1,5}', candidate ):
					return True
				
				if re.fullmatch( r'[ivxlcdm]{1,10}', candidate, flags=re.IGNORECASE ):
					return True
				
				if re.fullmatch( r'(?:page|p\.)\s+\d+(?:\s+of\s+\d+)?',
						candidate, flags=re.IGNORECASE ):
					return True
				
				return False
			
			counts = { }
			threshold = max( 2, int( minimum_repeats ) )
			
			for page in pages:
				for block in page.get( 'blocks', [ ] ):
					if block.get( 'zone' ) not in { 'header', 'footer' }:
						continue
					
					text = block.get( 'text' )
					
					if not isinstance( text, str ) or not text.strip( ):
						continue
					
					key = normalize_candidate( text )
					
					if key:
						counts[ key ] = counts.get( key, 0 ) + 1
			
			repeated = { key for key, value in counts.items( ) if value >= threshold }
			cleaned_pages = [ ]
			
			for page in pages:
				page_copy = dict( page )
				new_blocks = [ ]
				
				for block in page.get( 'blocks', [ ] ):
					block_copy = dict( block )
					text = block_copy.get( 'text' )
					
					if isinstance( text, str ) and block_copy.get( 'zone' ) in { 'header',
					                                                             'footer' }:
						key = normalize_candidate( text )
						
						if key in repeated or is_page_label( text ):
							block_copy[ 'drop' ] = True
					
					new_blocks.append( block_copy )
				
				page_copy[ 'blocks' ] = new_blocks
				cleaned_pages.append( page_copy )
			
			return cleaned_pages
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'PdfParser'
			exception.method = 'remove_repeats( self, pages: List[ dict ] )'
			raise exception
	
	def clean_artifacts( self, text: str ) -> str:
		"""

			Purpose:
			--------
			Removes generic parser and PDF extraction artifacts without using document
			family, agency, legal, regulatory, or document-title vocabulary.

			Parameters:
			-----------
			text : str
				Extracted PDF text.

			Returns:
			--------
			str
				Text with generic parser artifacts removed.

		"""
		try:
			throw_if( 'text', text )
			cleaned = text
			
			cleaned = re.sub(
				r'<parsed\s+text\s+for\s+page:\s*\d+\s*/\s*\d+>',
				' ',
				cleaned,
				flags=re.IGNORECASE
			)
			
			cleaned = re.sub(
				r'<image\s+for\s+page:\s*\d+\s*/\s*\d+>',
				' ',
				cleaned,
				flags=re.IGNORECASE
			)
			
			cleaned = re.sub(
				r'</?[a-z][a-z0-9_-]{1,20}>',
				' ',
				cleaned,
				flags=re.IGNORECASE
			)
			
			cleaned = re.sub(
				r'\b[a-z]:\\[^\s<>]*(?:\.[a-z0-9]{2,5})\b',
				' ',
				cleaned,
				flags=re.IGNORECASE
			)
			
			cleaned = re.sub(
				r'\b(?:/[^/\s<>]+)+/(?:[^/\s<>]+\.[a-z0-9]{2,5})\b',
				' ',
				cleaned,
				flags=re.IGNORECASE
			)
			
			cleaned = re.sub(
				r'(?im)^\s*(?:endobj|obj|xref|trailer|startxref|%%eof)\s*$',
				' ',
				cleaned
			)
			
			cleaned = re.sub(
				r'(?im)^\s*[a-z][a-z0-9_.-]{1,40}\s+on\s+[a-z0-9_.-]{6,}'
				r'(?:\s+with\s+[a-z0-9_.-]+)?\s*$',
				' ',
				cleaned
			)
			
			cleaned = re.sub( r'(?<=\S)\s*\.{4,}\s*(?=\S)', ' ', cleaned )
			cleaned = re.sub( r'(?<=\S)\s*(?:\.\s*){4,}(?=\S)', ' ', cleaned )
			cleaned = re.sub( r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned )
			cleaned = re.sub( r'[ \t]{2,}', ' ', cleaned )
			cleaned = re.sub( r' +\n', '\n', cleaned )
			cleaned = re.sub( r'\n +', '\n', cleaned )
			cleaned = re.sub( r'\n{3,}', '\n\n', cleaned )
			
			return cleaned.strip( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'PdfParser'
			exception.method = 'clean_artifacts( self, text: str ) -> str'
			raise exception
	
	def repair_spacing( self, text: str ) -> str:
		"""

			Purpose:
			--------
			Repairs generic spacing defects introduced by PDF extraction without using
			document-family-specific terms, titles, headings, agencies, legal labels, or
			regulatory vocabulary.

			Parameters:
			-----------
			text : str
				Extracted PDF text.

			Returns:
			--------
			str
				Text with generic spacing normalized.

		"""
		try:
			throw_if( 'text', text )
			cleaned = text
			
			def repair_bracketed( match: re.Match ) -> str:
				value = match.group( 1 )
				
				if re.fullmatch( r'[A-Z\s\-\n\r]{3,}', value ):
					value = re.sub( r'[\s\-]+', '', value )
					return f'[{value}]'
				
				return match.group( 0 )
			
			def repair_letter_spaced_line( match: re.Match ) -> str:
				line = match.group( 0 )
				tokens = line.split( )
				single_letters = [ t for t in tokens if re.fullmatch( r'[A-Z]', t ) ]
				
				if len( single_letters ) >= 3 and len( single_letters ) >= len( tokens ) / 2:
					return re.sub( r'\b([A-Z])(?:\s+)(?=[A-Z]\b)', r'\1', line )
				
				return line
			
			cleaned = re.sub( r'\[([A-Z\s\-\n\r]{3,})]', repair_bracketed, cleaned )
			
			cleaned = re.sub(
				r'(?m)^[A-Z](?:\s+[A-Z]){2,}(?:\s+[A-Z]{2,})*\s*$',
				repair_letter_spaced_line,
				cleaned
			)
			
			cleaned = re.sub( r'[ \t]+([,.;:!?])', r'\1', cleaned )
			cleaned = re.sub( r'([!?;:])(?=\S)', r'\1 ', cleaned )
			cleaned = re.sub( r'(?<=[a-z0-9])\.(?=[A-Z][a-z])', '. ', cleaned )
			cleaned = re.sub( r'[ \t]{2,}', ' ', cleaned )
			cleaned = re.sub( r' +\n', '\n', cleaned )
			cleaned = re.sub( r'\n +', '\n', cleaned )
			cleaned = re.sub( r'\n{3,}', '\n\n', cleaned )
			
			return cleaned.strip( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'PdfParser'
			exception.method = 'repair_spacing( self, text: str ) -> str'
			raise exception
	
	def rejoin_hyphenation( self, text: str, repair_embedded: bool = True ) -> str:
		"""

			Purpose:
			--------
			Repairs line-break hyphenation, soft-hyphen artifacts, and conservative
			embedded PDF extraction splits without document-specific vocabulary.

			Parameters:
			-----------
			text : str
				Extracted PDF text.
			repair_embedded : bool
				If true, repairs embedded alphabetic splits only when morphology,
				dictionary lookup, or token-shape evidence supports rejoining.

			Returns:
			--------
			str
				Text with broken PDF hyphenation repaired.

		"""
		try:
			throw_if( 'text', text )
			cleaned = text
			
			try:
				from nltk.corpus import words as nltk_words
				
				vocabulary = set( nltk_words.words( 'en' ) )
			except Exception:
				vocabulary = set( )
			
			try:
				from nltk.corpus import wordnet as nltk_wordnet
			except Exception:
				nltk_wordnet = None
			
			def is_known_word( value: str ) -> bool:
				token = value.lower( )
				
				if token in vocabulary:
					return True
				
				if nltk_wordnet is not None:
					try:
						if nltk_wordnet.synsets( token ):
							return True
						
						if nltk_wordnet.morphy( token ) is not None:
							return True
					except Exception:
						return False
				
				return False
			
			def repair_line_break( match: re.Match ) -> str:
				left = match.group( 1 )
				right = match.group( 2 )
				
				if '-' in right:
					return f'{left}-{right}'
				
				return f'{left}{right}'
			
			def repair_embedded_split( match: re.Match ) -> str:
				left = match.group( 1 )
				right = match.group( 2 )
				combined = f'{left}{right}'
				
				combined_known = is_known_word( combined )
				
				if combined_known:
					return combined
				
				if re.fullmatch(
						r'(?:able|ible|ally|ance|ancy|ence|ency|ation|ations|'
						r'cation|cations|cies|tion|tions|sion|sions|ment|ments|'
						r'ness|less|ship|ships|ing|ings|ed|er|ers|or|ors|ies|'
						r'ive|ives|ity|ities|al|als|ary|ory|ories|ous|ious|eous)',
						right,
						flags=re.IGNORECASE ):
					return combined
				
				return match.group( 0 )
			
			cleaned = re.sub(
				r'(?<=[A-Za-z])[\u00AD\uFFFC\uFFFD\uFFFE]\s*(?=[A-Za-z])',
				'',
				cleaned
			)
			
			cleaned = re.sub(
				r'\b([A-Za-z]{2,})-\s*\n\s*([A-Za-z][A-Za-z-]*)\b',
				repair_line_break,
				cleaned
			)
			
			cleaned = re.sub(
				r'\b([A-Za-z]{2,})-\s+([A-Za-z][A-Za-z-]*)\b',
				repair_line_break,
				cleaned
			)
			
			if repair_embedded:
				cleaned = re.sub(
					r'\b([A-Za-z]{2,})-([a-z]{2,})\b',
					repair_embedded_split,
					cleaned
				)
			
			cleaned = re.sub( r'[ \t]{2,}', ' ', cleaned )
			cleaned = re.sub( r' +\n', '\n', cleaned )
			cleaned = re.sub( r'\n +', '\n', cleaned )
			cleaned = re.sub( r'\n{3,}', '\n\n', cleaned )
			
			return cleaned.strip( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'PdfParser'
			exception.method = 'rejoin_hyphenation( self, text: str ) -> str'
			raise exception
	
	def rebuild_pages( self, pages: List[ dict ], preserve_page_breaks: bool = False ) -> str:
		"""

			Purpose:
			--------
			Rebuilds text from page/block geometry. Blocks marked with drop=True are
			excluded; all other blocks are preserved in y/x reading order.

			Parameters:
			-----------
			pages : List[ dict ]
				Page dictionaries produced by extract_pages or remove_repeats.
			preserve_page_breaks : bool
				If true, inserts explicit page-break markers between pages.

			Returns:
			--------
			str
				Rebuilt PDF text.

		"""
		try:
			throw_if( 'pages', pages )
			page_texts = [ ]
			
			for page in pages:
				page_number = page.get( 'page' )
				blocks = page.get( 'blocks', [ ] )
				parts = [ ]
				
				for block in blocks:
					if block.get( 'drop' ) is True:
						continue
					
					text = block.get( 'text' )
					
					if not isinstance( text, str ) or not text.strip( ):
						continue
					
					block_lines = [
							line.strip( )
							for line in text.splitlines( )
							if isinstance( line, str ) and line.strip( )
					]
					
					if block_lines:
						parts.append( '\n'.join( block_lines ) )
				
				page_text = '\n\n'.join( parts )
				page_text = re.sub( r'[ \t]{2,}', ' ', page_text )
				page_text = re.sub( r' +\n', '\n', page_text )
				page_text = re.sub( r'\n +', '\n', page_text )
				page_text = re.sub( r'\n{3,}', '\n\n', page_text ).strip( )
				
				if not page_text:
					continue
				
				if preserve_page_breaks:
					page_texts.append( f'<<<PAGE_BREAK:{page_number}>>>\n{page_text}' )
				else:
					page_texts.append( page_text )
			
			return '\n\n'.join( page_texts ).strip( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'PdfParser'
			exception.method = 'rebuild_pages( self, pages: List[ dict ] ) -> str'
			raise exception
	
	def extract_lines( self, path: str, count: Optional[ int ] = None ) -> List[ str ] | None:
		"""

			Purpose:
			--------
			Extracts non-empty text lines from a PDF using geometry extraction only.

			Parameters:
			-----------
			path : str
				Path to the PDF file.
			count : Optional[ int ]
				Maximum number of pages to process. None processes all pages.

			Returns:
			--------
			List[ str ] | None
				List of extracted non-empty lines.

		"""
		try:
			throw_if( 'path', path )
			text = self.geometric_extract( path=path, count=count ) or ''
			self.extracted_lines = [
					line.strip( )
					for line in text.splitlines( )
					if isinstance( line, str ) and line.strip( )
			]
			return self.extracted_lines
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'PdfParser'
			exception.method = 'extract_lines( self, path: str, count: Optional[ int ]=None )'
			raise exception
	
	def extract_text( self, path: str, count: Optional[ int ] = None ) -> str | None:
		"""

			Purpose:
			--------
			Extracts full PDF text using geometry extraction only.

			Parameters:
			-----------
			path : str
				Path to the PDF file.
			count : Optional[ int ]
				Maximum number of pages to process. None processes all pages.

			Returns:
			--------
			str | None
				Extracted PDF text.

		"""
		try:
			throw_if( 'path', path )
			return self.geometric_extract( path=path, count=count )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'PdfParser'
			exception.method = 'extract_text( self, path: str, count: Optional[ int ]=None )'
			raise exception
	
	def extract_tables( self, path: str, count: Optional[ int ] = None ) -> List[ pd.DataFrame ] | None:
		"""

			Purpose:
			--------
			Extracts tables from a PDF and returns them as DataFrames.

			Parameters:
			-----------
			path : str
				Path to the PDF file.
			count : Optional[ int ]
				Maximum number of pages to process. None processes all pages.

			Returns:
			--------
			List[ pd.DataFrame ] | None
				Extracted table DataFrames.

		"""
		try:
			throw_if( 'path', path )
			self.file_path = path
			self.tables = [ ]
			
			with fitz.open( self.file_path ) as doc:
				for page_index, page in enumerate( doc ):
					if count is not None and page_index >= count:
						break
					
					tables = page.find_tables( )
					
					for table in getattr( tables, 'tables', [ ] ):
						self.tables.append( pd.DataFrame( table.extract( ) ) )
			
			return self.tables
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'PdfParser'
			exception.method = 'extract_tables( self, path: str, count: Optional[ int ]=None )'
			raise exception
	
	def export_csv( self, tables: List[ pd.DataFrame ], filename: str ) -> None:
		"""

			Purpose:
			--------
			Exports a list of DataFrames to individual CSV files.

			Parameters:
			-----------
			tables : List[ pd.DataFrame ]
				List of tables to export.
			filename : str
				Output filename prefix.

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'tables', tables )
			throw_if( 'filename', filename )
			self.tables = tables
			
			for index, df_table in enumerate( self.tables ):
				df_table.to_csv( f'{filename}_{index + 1}.csv', index=False )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'PdfParser'
			exception.method = 'export_csv( self, tables: List[ pd.DataFrame ], filename: str )'
			raise exception
	
	def export_text( self, lines: List[ str ], path: str ) -> None:
		"""

			Purpose:
			--------
			Exports extracted lines to a plain text file.

			Parameters:
			-----------
			lines : List[ str ]
				Lines to export.
			path : str
				Output text file path.

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'lines', lines )
			throw_if( 'path', path )
			self.lines = lines
			self.file_path = path
			
			with open( self.file_path, 'w', encoding='utf-8', errors='ignore' ) as file:
				for line in self.lines:
					file.write( line + '\n' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'PdfParser'
			exception.method = 'export_text( self, lines: List[ str ], path: str )'
			raise exception
	
	def export_excel( self, tables: List[ pd.DataFrame ], path: str ) -> None:
		"""

			Purpose:
			--------
			Exports all extracted tables into a single Excel workbook.

			Parameters:
			-----------
			tables : List[ pd.DataFrame ]
				List of tables to export.
			path : str
				Output Excel workbook path.

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'tables', tables )
			throw_if( 'path', path )
			self.tables = tables
			self.file_path = path
			
			with pd.ExcelWriter( self.file_path, engine='xlsxwriter' ) as writer:
				for index, df_table in enumerate( self.tables ):
					sheet = f'Table_{index + 1}'
					df_table.to_excel( writer, sheet_name=sheet, index=False )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'PdfParser'
			exception.method = 'export_excel( self, tables: List[ pd.DataFrame ], path: str )'
			raise exception
			


