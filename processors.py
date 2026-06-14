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
    Provides text-processing, parsing, tokenization, NLP, and PDF cleanup utilities for Chonky.

    Purpose:
        Defines the processing classes used after document ingestion and before embedding or
        vector persistence. The module supports text normalization, formatting cleanup,
        tokenization, lemmatization, stemming, vocabulary construction, frequency analysis,
        Word document parsing, and geometry-aware PDF text reconstruction. Existing wrapped
        exception handlers record validation and processing failures through the application
        logger before re-raising the project Error wrapper.
  </summary>
  ******************************************************************************************
  '''
from __future__ import annotations
import string

import docx
from pymupdf import Page, Document
from sklearn.feature_extraction.text import TfidfVectorizer
from boogr import Error, Logger
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
	"""Processor processing component.
	
	Purpose:
		Initializes shared state used by Chonky text, Word, NLTK, and PDF parser subclasses, including token caches, line/page buffers, vocabulary stores, NLP helpers, and cleaned-text fields used across the processing workflow.
	
	Attributes:
		lemmatizer: Runtime state used by ``Processor`` during Chonky processing operations.
		stemmer: Runtime state used by ``Processor`` during Chonky processing operations.
		file_path: Runtime state used by ``Processor`` during Chonky processing operations.
		normalized: Runtime state used by ``Processor`` during Chonky processing operations.
		lemmatized: Runtime state used by ``Processor`` during Chonky processing operations.
		tokenized: Runtime state used by ``Processor`` during Chonky processing operations.
		encoding: Runtime state used by ``Processor`` during Chonky processing operations.
		nlp: Runtime state used by ``Processor`` during Chonky processing operations.
		parts_of_speech: Runtime state used by ``Processor`` during Chonky processing operations.
		embedddings: Runtime state used by ``Processor`` during Chonky processing operations.
		chunk_size: Runtime state used by ``Processor`` during Chonky processing operations.
		corrected: Runtime state used by ``Processor`` during Chonky processing operations.
		raw_input: Runtime state used by ``Processor`` during Chonky processing operations.
		raw_html: Runtime state used by ``Processor`` during Chonky processing operations.
		raw_pages: Runtime state used by ``Processor`` during Chonky processing operations.
		lines: Runtime state used by ``Processor`` during Chonky processing operations.
		tokens: Runtime state used by ``Processor`` during Chonky processing operations.
		lines: Runtime state used by ``Processor`` during Chonky processing operations.
		files: Runtime state used by ``Processor`` during Chonky processing operations.
		pages: Runtime state used by ``Processor`` during Chonky processing operations.
		paragraphs: Runtime state used by ``Processor`` during Chonky processing operations.
		ids: Runtime state used by ``Processor`` during Chonky processing operations.
		stop_words: Runtime state used by ``Processor`` during Chonky processing operations.
		vocabulary: Runtime state used by ``Processor`` during Chonky processing operations.
		corpus: Runtime state used by ``Processor`` during Chonky processing operations.
		removed: Runtime state used by ``Processor`` during Chonky processing operations.
		frequency_distribution: Runtime state used by ``Processor`` during Chonky processing operations.
	"""
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
	"""TextParser processing component.
	
	Purpose:
		Provides text cleanup, normalization, tokenization, chunking, vocabulary, frequency-distribution, and semantic-preparation utilities used by the Text Processing, Analysis, and Tokenization tabs.
	
	Attributes:
		lowercase: Runtime state used by ``TextParser`` during Chonky processing operations.
		cleaned_text: Runtime state used by ``TextParser`` during Chonky processing operations.
		cleaned_lines: Runtime state used by ``TextParser`` during Chonky processing operations.
		cleaned_tokens: Runtime state used by ``TextParser`` during Chonky processing operations.
		cleaned_pages: Runtime state used by ``TextParser`` during Chonky processing operations.
		cleaned_html: Runtime state used by ``TextParser`` during Chonky processing operations.
		conditional_distribution: Runtime state used by ``TextParser`` during Chonky processing operations.
		PUNCTUATION: Runtime state used by ``TextParser`` during Chonky processing operations.
		CONTROL_CHARACTERS: Runtime state used by ``TextParser`` during Chonky processing operations.
		DELIMITERS: Runtime state used by ``TextParser`` during Chonky processing operations.
		DIGITS: Runtime state used by ``TextParser`` during Chonky processing operations.
		SYMBOLS: Runtime state used by ``TextParser`` during Chonky processing operations.
		NUMERALS: Runtime state used by ``TextParser`` during Chonky processing operations.
	"""
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
		"""Initialize the TextParser instance.
		
		Purpose:
			Initializes parser state, reusable helper objects, and runtime caches used by later processing methods.
		"""
		super( ).__init__( )
		self.PUNCTUATION = PUNCTUATION
		self.CONTROL_CHARACTERS = ({ chr( i ) for i in range( 0x00, 0x20 ) } | { chr( 0x7F ) })
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
		"""Dir.
		
		Purpose:
			Returns the public member names exposed by the parser for introspection, diagnostics, and MkDocs API documentation.
		
		Returns:
			List[ str ] | None: Result produced by the processing operation.
		"""
		return [  # Attributes
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
		"""Load text.
		
		Purpose:
			Executes the ``load text`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			filepath: Filepath value used by the processing operation. Expected type: ``str``.
		
		Returns:
			str | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
		"""
		try:
			throw_if( 'filepath', filepath )
			if not os.path.exists( filepath ):
				raise FileNotFoundError( f'File not found: {filepath}' )
			else:
				self.file_path = filepath
			raw_text = open( self.file_path, mode='r', encoding='utf-8', errors='ignore' ).read( )
			return raw_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'load_text( self, file_path: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def collapse_whitespace( self, text: str ) -> str | None:
		"""Collapse whitespace.
		
		Purpose:
			Executes the ``collapse whitespace`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
		
		Returns:
			str | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def remove_punctuation( self, text: str ) -> str:
		"""Remove punctuation.
		
		Purpose:
			Executes the ``remove punctuation`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
		
		Returns:
			str: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def reduce_repeats( self, text: str ) -> str:
		"""Reduce repeats.
		
		Purpose:
			Executes the ``reduce repeats`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
		
		Returns:
			str: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
		"""
		try:
			throw_if( 'text', text )
			_cleaned = re.sub( r'([^\w\s]){2,}', lambda match: match.group( 0 )[ 0 ], text )
			_cleaned = re.sub( r'([^\w\s])\s*(?=\S)', r'\1 ', _cleaned )
			return _cleaned
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'reduce_repeats( self, text: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def normalize_text( self, text: str ) -> str | None:
		"""Normalize text.
		
		Purpose:
			Executes the ``normalize text`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
		
		Returns:
			str | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
		"""
		try:
			throw_if( 'text', text )
			return text.lower( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'normalize_text( self, text: str ) -> str:'
			Logger( ).write( exception )
			raise exception
	
	def remove_errors( self, text: str ) -> str:
		"""Remove errors.
		
		Purpose:
			Executes the ``remove errors`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
		
		Returns:
			str: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def remove_fragments( self, text: str ) -> str | None:
		"""Remove fragments.
		
		Purpose:
			Executes the ``remove fragments`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
		
		Returns:
			str | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
		"""
		try:
			throw_if( 'text', text )
			_text = text.lower( )
			_cleaned = [ ]
			_fragments = _text.split( )
			for char in _fragments:
				if len( char ) > 2:
					_cleaned.append( char )
			return ' '.join( _cleaned )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'remove_fragments( self, text: str ) -> str:'
			Logger( ).write( exception )
			raise exception
	
	def remove_symbols( self, text: str ) -> str | None:
		"""Remove symbols.
		
		Purpose:
			Executes the ``remove symbols`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
		
		Returns:
			str | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def remove_html( self, text: str ) -> str | None:
		"""Remove html.
		
		Purpose:
			Executes the ``remove html`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
		
		Returns:
			str | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def remove_xml( self, text: str ) -> str:
		"""Remove xml.
		
		Purpose:
			Executes the ``remove xml`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
		
		Returns:
			str: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def remove_markdown( self, text: str ) -> str | None:
		"""Remove markdown.
		
		Purpose:
			Executes the ``remove markdown`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
		
		Returns:
			str | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def remove_stopwords( self, text: str ) -> str:
		"""Remove stopwords.
		
		Purpose:
			Executes the ``remove stopwords`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
		
		Returns:
			str: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def remove_encodings( self, text: str ) -> str | None:
		"""Remove encodings.
		
		Purpose:
			Executes the ``remove encodings`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
		
		Returns:
			str | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
		"""
		try:
			throw_if( 'text', text )
			try:
				_text = text.lower( )
				text = bytes( _text, 'utf-8' ).decode( 'unicode_escape' )
			except UnicodeDecodeError:
				pass
			
			self.raw_input = text
			_html = html.unescape( self.raw_input )
			_norm = unicodedata.normalize( 'NFKC', _html )
			_chars = re.sub( r'[\x00-\x1F\x7F]', '', _norm )
			cleaned_text = _chars.strip( )
			return cleaned_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'remove_encodings( self, text: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def remove_headers( self, filepath: str, lines: int = 50, headers: int = 3,
			footers: int = 3 ) -> str | None:
		"""Remove headers.
		
		Purpose:
			Executes the ``remove headers`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			filepath: Filepath value used by the processing operation. Expected type: ``str``.
			lines: Lines value used by the processing operation. Expected type: ``int``. Defaults to ``50``.
			headers: Headers value used by the processing operation. Expected type: ``int``. Defaults to ``3``.
			footers: Footers value used by the processing operation. Expected type: ``int``. Defaults to ``3``.
		
		Returns:
			str | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def remove_numbers( self, text: str ) -> str | None:
		"""Remove numbers.
		
		Purpose:
			Executes the ``remove numbers`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
		
		Returns:
			str | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def remove_numerals( self, text: str ) -> str | None:
		"""Remove numerals.
		
		Purpose:
			Executes the ``remove numerals`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
		
		Returns:
			str | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def remove_images( self, text: str ) -> str:
		"""Remove images.
		
		Purpose:
			Executes the ``remove images`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
		
		Returns:
			str: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def tiktokenize( self, text: str, encoding: str = 'cl100k_base' ) -> DataFrame | None:
		"""Tiktokenize.
		
		Purpose:
			Executes the ``tiktokenize`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
			encoding: Encoding value used by the processing operation. Expected type: ``str``. Defaults to ``'cl100k_base'``.
		
		Returns:
			DataFrame | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def split_sentences( self, text: str ) -> List[ str ] | None:
		"""Split sentences.
		
		Purpose:
			Executes the ``split sentences`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
		
		Returns:
			List[ str ] | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def split_pages( self, filepath: str, num: int = 50 ) -> List[ str ] | None:
		"""Split pages.
		
		Purpose:
			Executes the ``split pages`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			filepath: Filepath value used by the processing operation. Expected type: ``str``.
			num: Num value used by the processing operation. Expected type: ``int``. Defaults to ``50``.
		
		Returns:
			List[ str ] | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def split_paragraphs( self, filepath: str ) -> DataFrame | None:
		"""Split paragraphs.
		
		Purpose:
			Executes the ``split paragraphs`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			filepath: Filepath value used by the processing operation. Expected type: ``str``.
		
		Returns:
			DataFrame | None: Result produced by the processing operation.
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
		"""Create frequency distribution.
		
		Purpose:
			Executes the ``create frequency distribution`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			tokens: Tokens value used by the processing operation. Expected type: ``List[ str ]``.
		
		Returns:
			DataFrame | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def create_vocabulary( self, tokens: List[ str ] ) -> Series | None:
		"""Create vocabulary.
		
		Purpose:
			Executes the ``create vocabulary`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			tokens: Tokens value used by the processing operation. Expected type: ``List[ str ]``.
		
		Returns:
			Series | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def create_wordbag( self, tokens: List[ str ] ) -> DataFrame | None:
		"""Create wordbag.
		
		Purpose:
			Executes the ``create wordbag`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			tokens: Tokens value used by the processing operation. Expected type: ``List[ str ]``.
		
		Returns:
			DataFrame | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def create_vectors( self, tokens: List[ str ] ) -> DataFrame | None:
		"""Create vectors.
		
		Purpose:
			Executes the ``create vectors`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			tokens: Tokens value used by the processing operation. Expected type: ``List[ str ]``.
		
		Returns:
			DataFrame | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def clean_file( self, filepath: str ) -> str | None:
		"""Clean file.
		
		Purpose:
			Executes the ``clean file`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			filepath: Filepath value used by the processing operation. Expected type: ``str``.
		
		Returns:
			str | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def clean_files( self, source: str, destination: str ) -> None:
		"""Clean files.
		
		Purpose:
			Executes the ``clean files`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			source: Source value used by the processing operation. Expected type: ``str``.
			destination: Destination value used by the processing operation. Expected type: ``str``.
		
		Returns:
			None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def chunk_files( self, source: str, destination: str ) -> None:
		"""Chunk files.
		
		Purpose:
			Executes the ``chunk files`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			source: Source value used by the processing operation. Expected type: ``str``.
			destination: Destination value used by the processing operation. Expected type: ``str``.
		
		Returns:
			None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
					_sentences = self.split_sentences( _text )
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
			Logger( ).write( exception )
			raise exception
	
	def chunk_data( self, filepath: str, size: int = 10 ) -> DataFrame | None:
		"""Chunk data.
		
		Purpose:
			Executes the ``chunk data`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			filepath: Filepath value used by the processing operation. Expected type: ``str``.
			size: Size value used by the processing operation. Expected type: ``int``. Defaults to ``10``.
		
		Returns:
			DataFrame | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
				self.chunks = [ _wordlist[ i: i + size ] for i in
				                range( 0, len( _wordlist ), size ) ]
				for i, c in enumerate( self.chunks ):
					_item = '[' + ' '.join( c ) + '],'
					_processed.append( _item )
				_data = pd.DataFrame( _processed )
				return _data
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'chunk_data( self, filepath: str, size: int=512  ) -> DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def chunk_datasets( self, source: str, destination: str, size: int = 10 ) -> DataFrame:
		"""Chunk datasets.
		
		Purpose:
			Executes the ``chunk datasets`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			source: Source value used by the processing operation. Expected type: ``str``.
			destination: Destination value used by the processing operation. Expected type: ``str``.
			size: Size value used by the processing operation. Expected type: ``int``. Defaults to ``10``.
		
		Returns:
			DataFrame: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
					_sourcepath = _src + '\\' + _filename
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
					_savepath = (_destination + f'\\' + _name)
					_data = pd.DataFrame( _processed, columns=[ 'Data', ] )
					_data.to_excel( _savepath, sheet_name='Dataset', index=False,
						columns=[ 'Data', ] )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'chunk_data( self, filepath: str, size: int=15  ) -> DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def convert_jsonl( self, source: str, destination: str, size: int = 10 ) -> None:
		"""Convert jsonl.
		
		Purpose:
			Executes the ``convert jsonl`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			source: Source value used by the processing operation. Expected type: ``str``.
			destination: Destination value used by the processing operation. Expected type: ``str``.
			size: Size value used by the processing operation. Expected type: ``int``. Defaults to ``10``.
		
		Returns:
			None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
					_tokens = _text.split( ' ' )
					_chunks = [ _tokens[ i: i + size ] for i in range( 0, len( _tokens ), size ) ]
					_datamap = [ ]
					for i, c in enumerate( _chunks ):
						_value = '{ ' + f' {i} : [ ' + ' '.join( c ) + ' ] }, ' + "\n"
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
			Logger( ).write( exception )
			raise exception
	
	def encode_sentences( self, tokens: List[ str ], model: str = 'all-MiniLM-L6-v2' ) -> \
			Tuple[ List[ str ], np.ndarray ]:
		"""Encode sentences.
		
		Purpose:
			Executes the ``encode sentences`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			tokens: Tokens value used by the processing operation. Expected type: ``List[ str ]``.
			model: Model value used by the processing operation. Expected type: ``str``. Defaults to ``'all-MiniLM-L6-v2'``.
		
		Returns:
			Tuple[ List[ str ], np.ndarray ]: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
		"""
		try:
			throw_if( 'tokens', tokens )
			throw_if( 'model', model )
			_transformer = SentenceTransformer( model )
			_tokens = [ self.lemmatizer.lemmatize( t ) for t in tokens ]
			_encoding = _transformer.encode( _tokens, show_progress_bar=True )
			return (self.cleaned_tokens, np.array( _encoding ))
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = 'encode_sentences( self, sentences: List[ str ], model_name ) -> ( )'
			Logger( ).write( exception )
			raise exception
	
	def semantic_search( self, query: str, tokens: List[ str ], embeddings: np.ndarray,
			model: SentenceTransformer, top: int = 5 ) -> List[ tuple[ str, float ] ]:
		"""Semantic search.
		
		Purpose:
			Executes the ``semantic search`` operation for the ``TextParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			query: Query value used by the processing operation. Expected type: ``str``.
			tokens: Tokens value used by the processing operation. Expected type: ``List[ str ]``.
			embeddings: Embeddings value used by the processing operation. Expected type: ``np.ndarray``.
			model: Model value used by the processing operation. Expected type: ``SentenceTransformer``.
			top: Top value used by the processing operation. Expected type: ``int``. Defaults to ``5``.
		
		Returns:
			List[ tuple[ str, float ] ]: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
		"""
		try:
			throw_if( 'query', query )
			throw_if( 'tokens', tokens )
			throw_if( 'embedding', embeddings )
			throw_if( 'model', model )
			query_vec = model.encode( [ query ] )
			sims = cosine_similarity( query_vec, embeddings )[ 0 ]
			top_indices = sims.argsort( )[ ::-1 ][ : top ]
			return [ (tokens[ i ], sims[ i ]) for i in top_indices ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'TextParser'
			exception.method = ('semantic_search( self, query: str, tokens: List[ str ], '
			                    'embeddings: np.ndarray, model: SentenceTransformer,  '
			                    'top_k: int=5 ) -> List[ tuple[ str, float ] ]')
			Logger( ).write( exception )
			raise exception

class NltkParser( Processor ):
	"""NltkParser processing component.
	
	Purpose:
		Provides NLTK-backed parsing utilities for corpus tokenization, stemming, lemmatization, part-of-speech tagging, named-entity handling, and lexical diagnostics used by Chonky analysis workflows.
	
	Attributes:
		word_tokens: Runtime state used by ``NltkParser`` during Chonky processing operations.
		sentence_tokens: Runtime state used by ``NltkParser`` during Chonky processing operations.
		stemmed_tokens: Runtime state used by ``NltkParser`` during Chonky processing operations.
		lemmatized_tokens: Runtime state used by ``NltkParser`` during Chonky processing operations.
		tagged_tokens: Runtime state used by ``NltkParser`` during Chonky processing operations.
		named_entities: Runtime state used by ``NltkParser`` during Chonky processing operations.
	"""
	word_tokens: Optional[ List[ str ] ]
	sentence_tokens: Optional[ List[ str ] ]
	stemmed_tokens: Optional[ List[ str ] ]
	lemmatized_tokens: Optional[ List[ str ] ]
	tagged_tokens: Optional[ List[ Tuple[ str, str ] ] ]
	named_entities: Optional[ List[ Tuple[ str, str ] ] ]
	
	def __init__( self ) -> None:
		"""Initialize the NltkParser instance.
		
		Purpose:
			Initializes parser state, reusable helper objects, and runtime caches used by later processing methods.
		"""
		super( ).__init__( )
		self.initialize_resources( )
		self.word_tokens = [ ]
		self.sentence_tokens = [ ]
		self.stemmed_tokens = [ ]
		self.lemmatized_tokens = [ ]
		self.tagged_tokens = [ ]
		self.named_entities = [ ]
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		Purpose:
			Returns the public member names exposed by the parser for introspection, diagnostics, and MkDocs API documentation.
		
		Returns:
			List[ str ] | None: Result produced by the processing operation.
		"""
		return [ 'initialize_resources', 'word_tokenizer', 'sentence_tokenizer', 'word_stemmer',
		         'word_lemmatizer', 'pos_tagger', 'named_entity_recognition', 'word_tokens',
		         'sentence_tokens', 'stemmed_tokens', 'lemmatized_tokens', 'tagged_tokens',
		         'named_entities' ]
	
	def initialize_resources( self ) -> None:
		"""Initialize resources.
		
		Purpose:
			Executes the ``initialize resources`` operation for the ``NltkParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Returns:
			None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
		"""
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
			Logger( ).write( exception )
			raise exception
	
	def word_tokenizer( self, text: str ) -> List[ str ] | None:
		"""Word tokenizer.
		
		Purpose:
			Executes the ``word tokenizer`` operation for the ``NltkParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
		
		Returns:
			List[ str ] | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
		"""
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
			Logger( ).write( exception )
			raise exception
	
	def sentence_tokenizer( self, text: str ) -> List[ str ] | None:
		"""Sentence tokenizer.
		
		Purpose:
			Executes the ``sentence tokenizer`` operation for the ``NltkParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
		
		Returns:
			List[ str ] | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
		"""
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
			Logger( ).write( exception )
			raise exception
	
	def word_stemmer( self, text: str ) -> List[ str ] | None:
		"""Word stemmer.
		
		Purpose:
			Executes the ``word stemmer`` operation for the ``NltkParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
		
		Returns:
			List[ str ] | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
		"""
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
			Logger( ).write( exception )
			raise exception
	
	def word_lemmatizer( self, text: str ) -> List[ str ] | None:
		"""Word lemmatizer.
		
		Purpose:
			Executes the ``word lemmatizer`` operation for the ``NltkParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
		
		Returns:
			List[ str ] | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
		"""
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
			Logger( ).write( exception )
			raise exception
	
	def pos_tagger( self, text: str ) -> List[ Tuple[ str, str ] ] | None:
		"""Pos tagger.
		
		Purpose:
			Executes the ``pos tagger`` operation for the ``NltkParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
		
		Returns:
			List[ Tuple[ str, str ] ] | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
		"""
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
			Logger( ).write( exception )
			raise exception
	
	def named_entity_recognition( self, text: str ) -> List[ Tuple[ str, str ] ] | None:
		"""Named entity recognition.
		
		Purpose:
			Executes the ``named entity recognition`` operation for the ``NltkParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
		
		Returns:
			List[ Tuple[ str, str ] ] | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
		"""
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
			Logger( ).write( exception )
			raise exception
	
	def chunk_words( self, text: str, size: int = 5 ) -> DataFrame | None:
		"""Chunk words.
		
		Purpose:
			Executes the ``chunk words`` operation for the ``NltkParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
			size: Size value used by the processing operation. Expected type: ``int``. Defaults to ``5``.
		
		Returns:
			DataFrame | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def chunk_sentences( self, text: str, size: int = 15 ) -> DataFrame | None:
		"""Chunk sentences.
		
		Purpose:
			Executes the ``chunk sentences`` operation for the ``NltkParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
			size: Size value used by the processing operation. Expected type: ``int``. Defaults to ``15``.
		
		Returns:
			DataFrame | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception

class WordParser( Processor ):
	"""WordParser processing component.
	
	Purpose:
		Provides Microsoft Word document parsing utilities that extract paragraphs, tables, metadata, and cleaned text for downstream Chonky processing and analysis.
	
	Attributes:
		sentences: Runtime state used by ``WordParser`` during Chonky processing operations.
		cleaned_sentences: Runtime state used by ``WordParser`` during Chonky processing operations.
		document: Runtime state used by ``WordParser`` during Chonky processing operations.
		raw_text: Runtime state used by ``WordParser`` during Chonky processing operations.
		paragraphs: Runtime state used by ``WordParser`` during Chonky processing operations.
		file_path: Runtime state used by ``WordParser`` during Chonky processing operations.
		vocabulary: Runtime state used by ``WordParser`` during Chonky processing operations.
		document: Runtime state used by ``WordParser`` during Chonky processing operations.
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
		"""Initialize the WordParser instance.
		
		Purpose:
			Initializes parser state, reusable helper objects, and runtime caches used by later processing methods.
		
		Args:
			filepath: Filepath value used by the processing operation. Expected type: ``str``.
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
		"""Dir.
		
		Purpose:
			Returns the public member names exposed by the parser for introspection, diagnostics, and MkDocs API documentation.
		
		Returns:
			List[ str ] | None: Result produced by the processing operation.
		"""
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
	
	def extract_text( self, num: int = 1 ) -> str | None:
		"""Extract text.
		
		Purpose:
			Executes the ``extract text`` operation for the ``WordParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			num: Num value used by the processing operation. Expected type: ``int``. Defaults to ``1``.
		
		Returns:
			str | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
		"""
		try:
			self.page_text = self.document.get_page_text( pno=num )
			return self.page_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'Word'
			exception.method = 'extract_text( self, num: int ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def split_sentences( self ) -> List[ str ] | None:
		"""Split sentences.
		
		Purpose:
			Executes the ``split sentences`` operation for the ``WordParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Returns:
			List[ str ] | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
		"""
		try:
			_text = self.page_text.lower( )
			self.sentences = sent_tokenize( _text )
			return self.sentences
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'Word'
			exception.method = 'split_sentences( self ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def clean_sentences( self ) -> List[ str ] | None:
		"""Clean sentences.
		
		Purpose:
			Executes the ``clean sentences`` operation for the ``WordParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Returns:
			List[ str ] | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def create_vocabulary( self ) -> set | None:
		"""Create vocabulary.
		
		Purpose:
			Executes the ``create vocabulary`` operation for the ``WordParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Returns:
			set | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def compute_frequency_distribution( self ) -> Dict[ str, int ] | None:
		"""Compute frequency distribution.
		
		Purpose:
			Executes the ``compute frequency distribution`` operation for the ``WordParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Returns:
			Dict[ str, int ] | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def summarize( self ) -> List[ str ] | None:
		"""Summarize.
		
		Purpose:
			Executes the ``summarize`` operation for the ``WordParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Returns:
			List[ str ] | None: Result produced by the processing operation.
		"""
		print( f'Document: {self.file_path}' )
		print( f'Paragraphs: {len( self.paragraphs )}' )
		print( f'Sentences: {len( self.sentences )}' )
		print( f'Vocabulary Size: {len( self.vocabulary )}' )
		print(
			f'Top 10 Frequent Words: {Counter( self.frequency_distribution ).most_common( 10 )}' )

class PdfParser( Processor ):
	"""PdfParser processing component.
	
	Purpose:
		Provides geometry-aware PDF parsing and cleanup utilities that extract page blocks, remove repeating headers and footers, repair line spacing, normalize artifacts, and rebuild PDF text for downstream processing.
	
	Attributes:
		strip_headers: Runtime state used by ``PdfParser`` during Chonky processing operations.
		minimum_length: Runtime state used by ``PdfParser`` during Chonky processing operations.
		extract_tables_enabled: Runtime state used by ``PdfParser`` during Chonky processing operations.
		extracted_lines: Runtime state used by ``PdfParser`` during Chonky processing operations.
		extracted_tables: Runtime state used by ``PdfParser`` during Chonky processing operations.
		extracted_pages: Runtime state used by ``PdfParser`` during Chonky processing operations.
	"""
	strip_headers: Optional[ bool ]
	minimum_length: Optional[ int ]
	extract_tables_enabled: Optional[ bool ]
	extracted_lines: Optional[ List ]
	extracted_tables: Optional[ List ]
	extracted_pages: Optional[ List ]
	
	def __init__( self, headers: bool = False, size: int = 10, tables: bool = True ) -> None:
		"""Initialize the PdfParser instance.
		
		Purpose:
			Initializes parser state, reusable helper objects, and runtime caches used by later processing methods.
		
		Args:
			headers: Headers value used by the processing operation. Expected type: ``bool``. Defaults to ``False``.
			size: Size value used by the processing operation. Expected type: ``int``. Defaults to ``10``.
			tables: Tables value used by the processing operation. Expected type: ``bool``. Defaults to ``True``.
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
		"""Dir.
		
		Purpose:
			Returns the public member names exposed by the parser for introspection, diagnostics, and MkDocs API documentation.
		
		Returns:
			List[ str ] | None: Result produced by the processing operation.
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
		"""Geometric extract.
		
		Purpose:
			Executes the ``geometric extract`` operation for the ``PdfParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			path: Path value used by the processing operation. Expected type: ``str``.
			count: Count value used by the processing operation. Expected type: ``Optional[ int ]``. Defaults to ``None``.
			header_ratio: Header ratio value used by the processing operation. Expected type: ``float``. Defaults to ``0.08``.
			footer_ratio: Footer ratio value used by the processing operation. Expected type: ``float``. Defaults to ``0.08``.
			preserve_page_breaks: Preserve page breaks value used by the processing operation. Expected type: ``bool``. Defaults to ``False``.
		
		Returns:
			str | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def extract_pages( self, path: str, count: Optional[ int ] = None,
			header_ratio: float = 0.08, footer_ratio: float = 0.08 ) -> List[ dict ] | None:
		"""Extract pages.
		
		Purpose:
			Executes the ``extract pages`` operation for the ``PdfParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			path: Path value used by the processing operation. Expected type: ``str``.
			count: Count value used by the processing operation. Expected type: ``Optional[ int ]``. Defaults to ``None``.
			header_ratio: Header ratio value used by the processing operation. Expected type: ``float``. Defaults to ``0.08``.
			footer_ratio: Footer ratio value used by the processing operation. Expected type: ``float``. Defaults to ``0.08``.
		
		Returns:
			List[ dict ] | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def remove_repeats( self, pages: List[ dict ], minimum_repeats: int = 3 ) -> List[
		                                                                             dict ] | None:
		"""Remove repeats.
		
		Purpose:
			Executes the ``remove repeats`` operation for the ``PdfParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			pages: Pages value used by the processing operation. Expected type: ``List[ dict ]``.
			minimum_repeats: Minimum repeats value used by the processing operation. Expected type: ``int``. Defaults to ``3``.
		
		Returns:
			List[
				                                                                             dict ] | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def clean_artifacts( self, text: str ) -> str:
		"""Clean artifacts.
		
		Purpose:
			Executes the ``clean artifacts`` operation for the ``PdfParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
		
		Returns:
			str: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def repair_spacing( self, text: str ) -> str:
		"""Repair spacing.
		
		Purpose:
			Executes the ``repair spacing`` operation for the ``PdfParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
		
		Returns:
			str: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def rejoin_hyphenation( self, text: str, repair_embedded: bool = True ) -> str:
		"""Rejoin hyphenation.
		
		Purpose:
			Executes the ``rejoin hyphenation`` operation for the ``PdfParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			text: Text value used by the processing operation. Expected type: ``str``.
			repair_embedded: Repair embedded value used by the processing operation. Expected type: ``bool``. Defaults to ``True``.
		
		Returns:
			str: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def rebuild_pages( self, pages: List[ dict ], preserve_page_breaks: bool = False ) -> str:
		"""Rebuild pages.
		
		Purpose:
			Executes the ``rebuild pages`` operation for the ``PdfParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			pages: Pages value used by the processing operation. Expected type: ``List[ dict ]``.
			preserve_page_breaks: Preserve page breaks value used by the processing operation. Expected type: ``bool``. Defaults to ``False``.
		
		Returns:
			str: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def extract_lines( self, path: str, count: Optional[ int ] = None ) -> List[ str ] | None:
		"""Extract lines.
		
		Purpose:
			Executes the ``extract lines`` operation for the ``PdfParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			path: Path value used by the processing operation. Expected type: ``str``.
			count: Count value used by the processing operation. Expected type: ``Optional[ int ]``. Defaults to ``None``.
		
		Returns:
			List[ str ] | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def extract_text( self, path: str, count: Optional[ int ] = None ) -> str | None:
		"""Extract text.
		
		Purpose:
			Executes the ``extract text`` operation for the ``PdfParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			path: Path value used by the processing operation. Expected type: ``str``.
			count: Count value used by the processing operation. Expected type: ``Optional[ int ]``. Defaults to ``None``.
		
		Returns:
			str | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
		"""
		try:
			throw_if( 'path', path )
			return self.geometric_extract( path=path, count=count )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processors'
			exception.cause = 'PdfParser'
			exception.method = 'extract_text( self, path: str, count: Optional[ int ]=None )'
			Logger( ).write( exception )
			raise exception
	
	def extract_tables( self, path: str, count: Optional[ int ] = None ) -> List[
		                                                                        pd.DataFrame ] | None:
		"""Extract tables.
		
		Purpose:
			Executes the ``extract tables`` operation for the ``PdfParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			path: Path value used by the processing operation. Expected type: ``str``.
			count: Count value used by the processing operation. Expected type: ``Optional[ int ]``. Defaults to ``None``.
		
		Returns:
			List[
				                                                                        pd.DataFrame ] | None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def export_csv( self, tables: List[ pd.DataFrame ], filename: str ) -> None:
		"""Export csv.
		
		Purpose:
			Executes the ``export csv`` operation for the ``PdfParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			tables: Tables value used by the processing operation. Expected type: ``List[ pd.DataFrame ]``.
			filename: Filename value used by the processing operation. Expected type: ``str``.
		
		Returns:
			None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def export_text( self, lines: List[ str ], path: str ) -> None:
		"""Export text.
		
		Purpose:
			Executes the ``export text`` operation for the ``PdfParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			lines: Lines value used by the processing operation. Expected type: ``List[ str ]``.
			path: Path value used by the processing operation. Expected type: ``str``.
		
		Returns:
			None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
	
	def export_excel( self, tables: List[ pd.DataFrame ], path: str ) -> None:
		"""Export excel.
		
		Purpose:
			Executes the ``export excel`` operation for the ``PdfParser`` workflow, updating instance state where required and returning the processed result used by downstream Chonky loading, processing, analysis, tokenization, or embedding steps.
		
		Args:
			tables: Tables value used by the processing operation. Expected type: ``List[ pd.DataFrame ]``.
			path: Path value used by the processing operation. Expected type: ``str``.
		
		Returns:
			None: Result produced by the processing operation.
		
		Raises:
			Error: Raised when validation or processing fails.
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
			Logger( ).write( exception )
			raise exception
			

