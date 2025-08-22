'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                processing.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="tigrr.py" company="Terry D. Eppler">

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

import os
import re
import string
from collections import Counter, defaultdict
from typing import List, Optional, Dict

import fitz
import nltk
import pandas as pd
import tiktoken
import unicodedata
from bs4 import BeautifulSoup
from docx import Document as Docx
from gensim.models import Word2Vec
from nltk import FreqDist, ConditionalFreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from tiktoken.core import Encoding

from boogr import Error, ErrorDialog

try:
	nltk.data.find( 'tokenizers/punkt' )
except LookupError:
	nltk.download( 'punkt' )
	nltk.download( 'punkt_tab' )
	nltk.download( 'stopwords' )
	nltk.download( 'wordnet' )
	nltk.download( 'omw-1.4' )
	nltk.download( 'words' )

class Processor( ):
	'''
		
		Purpose:
		Base class for processing classes
		
	'''
	lemmatizer: WordNetLemmatizer
	stemmer: PorterStemmer
	file_path: Optional[ str ]
	lowercase: Optional[ str ]
	normalized: Optional[ str ]
	lemmatized: Optional[ str ]
	tokenized: Optional[ str ]
	encoding: Optional[ Encoding ]
	chunk_size: Optional[ int ]
	corrected: Optional[ str ]
	raw_input: Optional[ str ]
	raw_html: Optional[ str ]
	raw_pages: Optional[ List[ str ] ]
	words: Optional[ List[ str ] ]
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
	vocabulary: Optional[ List[ str ] ]
	removed: Optional[ List[ str ] ]
	frequency_distribution: Optional[ Dict ]
	conditional_distribution: Optional[ Dict ]

	def __init__( self ):
		self.lemmatizer = WordNetLemmatizer( )
		self.stemmer = PorterStemmer( )
		self.files = [ ]
		self.words = [ ]
		self.tokens = [ ]
		self.lines = [ ]
		self.pages = [ ]
		self.ids = [ ]
		self.paragraphs = [ ]
		self.chunks = [ ]
		self.chunk_size = 0
		self.cleaned_lines = [ ]
		self.cleaned_tokens = [ ]
		self.cleaned_pages = [ ]
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

# noinspection PyTypeChecker
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
		self.words = [ ]
		self.tokens = [ ]
		self.lines = [ ]
		self.pages = [ ]
		self.ids = [ ]
		self.paragraphs = [ ]
		self.chunks = [ ]
		self.chunk_size = 0
		self.cleaned_lines = [ ]
		self.cleaned_tokens = [ ]
		self.cleaned_pages = [ ]
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
		return [ 'file_path', 'raw_input', 'raw_pages', 'normalized', 'lemmatized',
		         'tokenized', 'corrected', 'cleaned_text', 'words', 'paragraphs',
		         'words', 'pages', 'chunks', 'chunk_size', 'cleaned_pages',
		         'stop_words', 'cleaned_lines', 'removed', 'lowercase', 'encoding', 'vocabulary',
		         'translator', 'lemmatizer', 'stemmer', 'tokenizer', 'vectorizer',
		         'split_lines', 'split_pages', 'collapse_whitespace',
		         'remove_punctuation', 'remove_special', 'remove_html',
		         'remove_markdown', 'remove_stopwords', 'remove_headers', 'tiktokenize',
		         'normalize_text', 'tokenize_text', 'tokenize_words',
		         'tokenize_sentences', 'chunk_text', 'chunk_words',
		         'create_wordbag', 'create_word2vec', 'create_tfidf',
		         'clean_files', 'convert_jsonl', 'conditional_distribution' ]

	def collapse_whitespace( self, text: str ) -> str | None:
		"""

			Purpose:
			-----------
			Removes extra spaces and blank words from the string 'text'.

			Parameters:
			-----------
			- text : str

			Returns:
			--------
			A cleaned_lines path path with:
				- Consecutive whitespace reduced to a single space
				- Leading/trailing spaces removed
				- Blank words removed

		"""
		try:
			if text is None:
				raise Exception( 'The argument "text" is required.' )
			else:
				self.raw_input = text
				self.cleaned_text = re.sub( r'[ \t]+', ' ', self.raw_input )
				self.cleaned_lines = [ line.strip( ) for line in self.cleaned_text.splitlines( ) ]
				self.lines = [ line for line in self.cleaned_lines if line ]
				return ' '.join( self.lines )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'collapse_whitespace( self, path: str ) -> str:'
			error = ErrorDialog( exception )
			error.show( )

	def remove_punctuation( self, text: str ) -> str | None:
		"""

			Purpose:
			-----------
			Removes all punctuation characters from the path path path.

			Parameters:
			-----------
			- pages : str
				The path path path to be cleaned_lines.

			Returns:
			--------
			- str
				The path path with all punctuation removed.

		"""
		try:
			if text is None:
				raise Exception( 'The argument "text" is required.' )
			else:
				self.raw_input = text
				self.translator = str.maketrans( '', '', string.punctuation )
				self.cleaned_text = self.raw_input.translate( self.translator )
				return self.cleaned_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'remove_punctuation( self, text: str ) -> str:'
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
			if text is None:
				raise Exception( 'The argument "text" is required.' )
			else:
				cleaned = [ ]
				keepers = [ '(', ')', '$', '. ', ';', ':' ]
				for char in text:
					if char in keepers:
						cleaned.append( char )
					elif char.isalnum( ) or char.isspace( ):
						cleaned.append( char )
				return ''.join( cleaned )
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
			if text is None:
				raise Exception( 'The argument "text" is required.' )
			else:
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
			if text is None:
				raise Exception( 'The argument "text" is required.' )
			else:
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
			if text is None:
				raise Exception( 'The argument "text" is required.' )
			else:
				self.raw_input = text.lower( )
				self.stop_words = stopwords.words( 'english' )
				self.tokens = nltk.word_tokenize( self.raw_input )
				self.cleaned_tokens = [ w for w in self.tokens if
				                        w.isalnum( ) and w not in self.stop_words ]
				self.cleaned_text = ' '.join( self.cleaned_tokens )
				return self.cleaned_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'remove_stopwords( self, text: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )

	def clean_space( self, text: str ) -> str | None:
		"""

			Purpose:
			_____________
			Removes extra spaces and blank words from the path pages.

			Parameters:
			-----------
			- pages : str
				The raw path pages path to be cleaned_lines.

			Returns:
			--------
			- str
				A cleaned_lines pages path with:
					- Consecutive whitespace reduced to a single space
					- Leading/trailing spaces removed
					- Blank words removed

		"""
		try:
			if text is None:
				raise Exception( 'The argument "text" is required.' )
			else:
				self.raw_input = text.lower( )
				tabs = re.sub( r'[ \t]+', ' ', text.lower( ) )
				collapsed = re.sub( r'\s+', ' ', tabs ).strip( )
				self.cleaned_text = collapsed
				return self.cleaned_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'remove_errors( self, text: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )

	def remove_headers( self, pages: List[ str ], min: int=3 ) -> List[ str ] | None:
		"""

			Purpose:
			--------
			Removes repetitive headers and footers across a list of pages by frequency analysis.

			Parameters:
			-----------
			- pages (list of str):
				A list where each element is the full path of one page.

			- min (int):
				Minimum num of times a line must appear at the top/bottom to be a header/footer.

			Returns:
			---------
			- list of str:
				List of cleaned_lines page words without detected headers/footers.

		"""
		try:
			if pages is None:
				raise Exception( 'The argument "pages" is required.' )
			else:
				_headers = defaultdict( int )
				_footers = defaultdict( int )
				self.pages = pages

				# First pass: collect frequency of top/bottom words
				for _page in self.pages:
					self.lines = _page.strip( ).splitlines( )
					if not self.lines:
						continue
					_headers[ self.lines[ 0 ].strip( ) ] += 1
					_footers[ self.lines[ -1 ].strip( ) ] += 1

				# Identify candidates for removal
				_head = { line for line, count in _headers.items( ) if
				          count >= min }
				_foot = { line for line, count in _footers.items( ) if
				          count >= min }

				# Second pass: clean pages
				for _page in self.pages:
					if not self.lines:
						continue
					self.lines = _page.strip( ).splitlines( )
					if not self.lines:
						self.cleaned_pages.append( _page )
						continue

					# Remove header
					if self.lines[ 0 ].strip( ) in _head:
						self.lines = self.lines[ 1: ]

					# Remove footer
					if self.lines and self.lines[ -1 ].strip( ) in _foot:
						self.lines = self.lines[ : -1 ]

					self.cleaned_pages.append( '\n'.join( self.lines ) )
				_retval = self.cleaned_pages
				return _retval
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = ('remove_headers( self, pages: List[ str ], min: int=3 ) -> List['
			                    'str]')
			error = ErrorDialog( exception )
			error.show( )

	def normalize_text( self, text: str ) -> str | None:
		"""

			Purpose:
			-----------
			Normalizes the path pages path.
			  - Converts pages to lowercase
			  - Removes accented characters (e.g., é -> e)
			  - Removes leading/trailing spaces
			  - Collapses multiple whitespace characters into a single space

			Parameters:
			-----------
			- pages : str
				The raw path pages path to be normalized.

			Returns:
			--------
			- str
				A normalized, cleaned_lines version of the path path.

		"""
		try:
			if text is None:
				raise Exception( 'The argument "text" is required.' )
			else:
				self.raw_input = text
				self.normalized = unicodedata.normalize( 'NFKD', text ).encode( 'ascii',
					'ignore' ).decode( 'utf-8' )
				self.normalized = re.sub( r'\s+', ' ', self.normalized ).strip( ).lower( )
				return self.normalized
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'normalize_text( self, text: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )

	def tokenize_text( self, text: str ) -> List[ str ] | None:
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
			if text is None:
				raise Exception( 'The argument "text" was None' )
			else:
				_tokens = nltk.word_tokenize( text )
				self.words = [ t for t in _tokens ]
				self.tokens = [ re.sub( r'[^\w"-]', '', w ) for w in self.words if w.strip( ) ]
				return self.tokens
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'tokenize_text( self, path: str ) -> List[ str ]'
			error = ErrorDialog( exception )
			error.show( )

	# noinspection PyTypeChecker
	def tiktokenize( self, text: str, encoding: str='cl100k_base' ) -> List[ str ] | None:
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
			if text is None:
				raise Exception( 'The argument "text" was None' )
			else:
				self.encoding = tiktoken.get_encoding( encoding )
				token_ids = self.encoding.encode( text )
				return token_ids  # or [self.encoding.decode_single_token_bytes(t) for t in token_ids]
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = ('tiktokenize( self, text: str, encoding: str="cl100k_base" ) -> '
			                    'List[ str ]')
			error = ErrorDialog( exception )
			error.show( )

	def tokenize_words( self, words: List[ str ] ) -> List[ List[ str ] ]| None:
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
			if words is None:
				raise Exception( 'The argument "words" was None' )
			else:
				self.words = words
				for w in self.words:
					_tokens = nltk.word_tokenize( w )
					self.tokens.append( _tokens )
				return self.tokens
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'tokenize_words( self, path: str ) -> List[ str ]'
			error = ErrorDialog( exception )
			error.show( )

	def chunk_text( self, text: str, size: int=50, return_as_string: bool=True ) -> (List[
		                                                                                    str ] |
	                                                                                     None):
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
			if text is None:
				raise Exception( 'The argument "text" is required.' )
			else:
				self.tokens = nltk.word_tokenize( text )
				self.chunks = [ self.tokens[ i: i + size ] for i in
				                range( 0, len( self.tokens ), size ) ]
				if return_as_string:
					return [ ' '.join( chunk ) for chunk in self.chunks ]
				else:
					return self.chunks
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'chunk_text( self, text: str, max: int=800 ) -> list[ str ]'
			error = ErrorDialog( exception )
			error.show( )

	def chunk_words( self, words: List[ str ], size: int=50, as_string: bool=True ) -> List[ str ] | List[ List[ str ] ] | None:
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
			if words is None:
				raise Exception( 'The argument "words" is required.' )
			else:
				self.tokens = [ token for sublist in words for token in sublist ]
				self.chunks = [ self.tokens[ i: i + size ]
				                for i in range( 0, len( self.tokens ), size ) ]
				if as_string:
					return [ ' '.join( chunk ) for chunk in self.chunks ]
				else:
					return self.chunks
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Token'
			exception.method = (
					'chunk_words( self, words: list[ str ], max: int=800, over: int=50 ) -> list[ '
					'str ]')
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
			if text is None:
				raise Exception( 'The argument "text" is required.' )
			else:
				self.lines = nltk.sent_tokenize( text )
				return self.lines
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'split_sentences( self, text: str ) -> List[ str ]'
			error = ErrorDialog( exception )
			error.show( )

	def split_pages( self, path: str, delimit: str = '\f' ) -> List[ str ] | None:
		"""

			Purpose:
			--------
			Reads path from a file, splits it into words,
			and groups them into path.

			Parameters:
			-----------
			- path (str): Path to the path file.
			- delimiter (str): Page separator path (default is '\f' for form feed).

			Returns:
			---------
			- list[ str ]  where each element is the path.

		"""
		try:
			if path is None:
				raise Exception( 'The argument "path" is required' )
			else:
				self.file_path = path
				with open( self.file_path, 'r', encoding = 'utf-8', errors = 'ignore' ) as _file:
					_content = _file.read( )
					self.raw_pages = _content.split( delimit )

				for _page in self.raw_pages:
					self.lines = _page.strip( ).splitlines( )
					self.cleaned_text = '\n'.join(
						[ l.strip( ) for l in self.lines if l.strip( ) ] )
					self.cleaned_pages.append( self.cleaned_text )
				return self.cleaned_pages
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'split_pages( self, path: str, delimit: str="\f" ) -> List[ str ]'
			error = ErrorDialog( exception )
			error.show( )

	def split_paragraphs( self, path: str ) -> List[ str ] | None:
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
			if path is None:
				raise Exception( 'The argument "path" is required.' )
			else:
				self.file_path = path
				with open( self.file_path, 'r', encoding='utf-8', errors='ignore' ) as _file:
					self.raw_input = _file.read( )
					self.paragraphs = [ pg.strip( ) for pg in self.raw_input.split( '\n\n' ) if
					                    pg.strip( ) ]

					return self.paragraphs
		except UnicodeDecodeError:
			with open( self.file_path, 'r', encoding = 'latin1', errors = 'ignore' ) as _file:
				self.raw_input = _file.read( )
				self.paragraphs = [ pg.strip( ) for pg in self.raw_input.split( '\n\n' ) if
				                    pg.strip( ) ]
				return self.paragraphs

	def compute_frequency_distribution( self, lines: List[ str ] ) -> FreqDist | None:
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
			if lines is None:
				raise Exception( 'The argument "words" is required.' )
			else:
				self.lines = lines
				all_tokens: list[ str ] = [ ]
				for _line in lines:
					toks = self.tokenize_text( _line )
					all_tokens.extend( toks )
				self.frequency_distribution = dict( Counter( all_tokens ) )
				return self.frequency_distribution
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = ('compute_frequency_distribution( self, documents: list, process: '
			                    'bool=True) -> FreqDist')
			error = ErrorDialog( exception )
			error.show( )

	def compute_conditional_distribution( self, lines: List[ str ], condition=None,
	                                      process: bool=True ) -> ConditionalFreqDist | None:
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
			if lines is None:
				raise Exception( 'The argument "words" is required.' )
			else:
				self.lines = lines
				cfd = ConditionalFreqDist( )
				for idx, line in enumerate( lines ):
					key = condition( line ) if condition else f'Doc_{idx}'
					toks = self.tokenize_text( self.normalize_text( line ) if process else line )
					for t in toks:
						cfd[ key ][ t ] += 1
				self.conditional_distribution = cfd
				return cfd
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = ('compute_conditional_distribution( self, words: List[ str ], '
			                    'condition=None, process: bool=True ) -> ConditionalFreqDist')
			error = ErrorDialog( exception )
			error.show( )

	def create_vocabulary( self, freq_dist: Dict, min: int=1 ) -> List[ str ] | None:
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
			if freq_dist is None:
				raise Exception( 'The argument "freq_dist" is required.' )
			else:
				self.frequency_distribution = freq_dist
				self.words = [ word for word, freq in freq_dist.items( ) if freq >= min ]
				self.vocabulary = sorted( self.words )
				return self.vocabulary
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = ('create_vocabulary( self, freq_dist: dict, min: int=1 ) -> List['
			                    'str]')
			error = ErrorDialog( exception )
			error.show( )

	def create_wordbag( self, words: List[ str ] ) -> dict | None:
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
			if words is None:
				raise Exception( 'The argument "words" is required.' )
			else:
				self.words = words
				return dict( Counter( self.words ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'create_wordbag( self, words: List[ str ] ) -> dict'
			error = ErrorDialog( exception )
			error.show( )

	def create_word2vec( self, words: List[ List[ str ] ], size=100, window=5, min=1 ) -> Word2Vec | None:
		"""

			Purpose:
			--------
			Train a Word2Vec embedding small_model from tokenized sentences.

			Parameters:
			--------
			- sentences (get_list of get_list of str): List of tokenized sentences.
			- vector_size (int): Dimensionality of word vec.
			- window (int): Max distance between current and predicted word.
			- min_count (int): Minimum frequency for inclusion in vocabulary.

			Returns:
			-------
			- Word2Vec: Trained Gensim Word2Vec small_model.

		"""
		try:
			if words is None:
				raise Exception( 'The argument "words" is required.' )
			else:
				self.words = words
				return Word2Vec( sentences=self.words, vector_size=size,
					window=window, min_count=min )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = ('create_word2vec( self, words: List[ str ], '
			                    'size=100, window=5, min=1 ) -> Word2Vec')
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
			if src is None:
				raise Exception( 'The argument "src" is required.' )
			elif dest is None:
				raise Exception( 'The argument "dest" is required.' )
			else:
				source = src
				destination = dest
				files = os.listdir( source )
				for f in files:
					processed = [ ]
					filename = os.path.basename( f )
					source_path = source + '\\' + filename
					text = open( source_path, 'r', encoding='utf-8', errors='ignore' ).read( )
					punc = self.remove_special( text )
					dirty = self.split_sentences( punc )
					for d in dirty:
						if d != " ":
							lower = d.lower( )
							normal = self.normalize_text( lower )
							slim = self.collapse_whitespace( normal )
							processed.append( slim )

					dest_path = destination + '\\' + filename
					clean = open( dest_path, 'wt', encoding='utf-8', errors='ignore' )
					for p in processed:
						clean.write( p )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'clean_files( self, src: str, dest: str )'
			error = ErrorDialog( exception )
			error.show( )

	def convert_jsonl( self, source: str, desination: str ) -> None:
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
			if source is None:
				raise Exception( 'The argument "source" is required.' )
			elif desination is None:
				raise Exception( 'The argument "desination" is required.' )
			else:
				_source = source
				_destination = desination
				self.files = os.listdir( _source )
				_processed = [ ]
				for _f in self.files:
					_count = 0
					_basename = os.path.basename( _f )
					_sourcepath = _source + f'\\{_basename}'
					_text = open( _sourcepath, 'r', encoding='utf-8', errors='ignore' ).read( )
					_stops = self.remove_stopwords( _text )
					_tokens = self.tokenize_text( _stops )
					_chunks = self.chunk_text( _text )
					_filename = _basename.rstrip( '.txt' )
					_destinationpath = _destination + f'\\{_filename}.jsonl'
					_clean = open( _destinationpath, 'wt', encoding='utf-8', errors='ignore' )
					for _i in range( len( _chunks ) ):
						_list = _chunks[ _i ]
						_part = ''.join( _list )
						_row = '{' + f'\"Line-{_i}\":\"{_part}\"' + '}' + '\r'
						_processed.append( _row )

					for _t in _processed:
						_clean.write( _t )

					_clean.flush( )
					_clean.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Text'
			exception.method = 'convert_jsonl( self, source: str, desination: str )'
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
		return [ 'extract_text', 'split_sentences', 'clean_sentences',
		         'create_vocabulary', 'compute_frequency_distribution',
		         'summarize', 'filepath', 'raw_text', 'paragraphs',
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
			for _sentence in self.sentences:
				_sentence = re.sub( r'[\r\n\t]+', ' ', _sentence )
				_sentence = re.sub( r"[^a-zA-Z0-9\s']", '', _sentence )
				_sentence = re.sub( r'\s{2,}', ' ', _sentence ).strip( ).lower( )
				self.cleaned_sentences.append( _sentence )

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
				tokens = word_tokenize( _sentence )
				tokens = [ token for token in tokens if
				           token.isalpha( ) and token not in self.stop_words ]
				all_words.extend( tokens )
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
		print(
			f'Top 10 Frequent Words: {Counter( self.frequency_distribution ).most_common( 10 )}' )

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

	def __init__( self, headers: bool=False, min: int=10, tables: bool=True ) -> None:
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
		self.minimum_length = min
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
		return [ 'strip_headers', 'minimum_length', 'extract_tables',
		         'path', 'page', 'pages', 'words', 'clean_lines', 'extracted_lines',
		         'extracted_tables', 'extracted_pages', 'extract_lines',
		         'extract_text', 'export_csv', 'export_text', 'export_excel' ]

	def extract_lines( self, path: str, max: Optional[ int ]=None ) -> List[ str ] | None:
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
			if path is None:
				raise Exception( 'The argument "path" must be specified' )
			else:
				self.file_path = path
				self.extracted_lines = [ ]
				with fitz.open( self.file_path ) as doc:
					for i, page in enumerate( doc ):
						if max is not None and i >= max:
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
			if page is None:
				raise Exception( 'The argument "page" cannot be None' )
			else:
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
					if self.strip_headers and self._has_repeating_header( line ):
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

	def _has_repeating_header( self, line: str ) -> bool | None:
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
			if line is None:
				raise Exception( 'The argument "line" is None' )
			else:
				_keywords = [ 'page', 'public law', 'u.s. government', 'united states' ]
				return any( kw in line.lower( ) for kw in _keywords )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'PDF'
			exception.method = '_has_repeating_header( self, line: str ) -> bool'
			error = ErrorDialog( exception )
			error.show( )

	def extract_text( self, path: str, max: Optional[ int ]=None ) -> str | None:
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
			if path is None:
				raise Exception( 'The argument "path" must be specified' )
			else:
				if max is not None and max > 0:
					self.file_path = path
					self.lines = self.extract_lines( self.file_path, max=max )
					return '\n'.join( self.lines )
				elif max is None or max <= 0:
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

	def extract_tables( self, path: str, max: Optional[ int ]=None ) -> (
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
			if path is None:
				raise Exception( 'The argument "path" must be specified' )
			elif max is None:
				raise Exception( 'The argument "max" must be specified' )
			else:
				self.file_path = path
				self.tables = [ ]
				with fitz.open( self.file_path ) as doc:
					for i, page in enumerate( doc ):
						if max is not None and i >= max:
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
			if tables is None:
				raise Exception( 'The argument "tables" must not be None' )
			elif filename is None:
				raise Exception( 'The argument "filename" must not be None' )
			else:
				self.tables = tables
				for i, df in enumerate( self.tables ):
					df.to_csv( f'{filename}_{i + 1}.csv', index = False )
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
			if lines is None:
				raise Exception( 'The argument "words" must be provided.' )
			elif path is None:
				raise Exception( 'The argument "path" must be provided.' )
			else:
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
			if tables is None:
				raise Exception( 'The argument "tables" must not be None' )
			elif path is None:
				raise Exception( 'The argument "path" must not be None' )
			else:
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
			exception.method = ('export_excel( self, tables: List[ pd.DataFrame ], path: str ) -> '
			                    'None')
			error = ErrorDialog( exception )
			error.show( )
