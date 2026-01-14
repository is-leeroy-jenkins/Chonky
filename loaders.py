'''
  ******************************************************************************************
      Assembly:                Name
      Filename:                loaders.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="loaders.py" company="Terry D. Eppler">

	     loaders.py
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
    loaders.py
  </summary>
  ******************************************************************************************
'''
import arxiv
import docx2txt

from boogr import Error, ErrorDialog
import config as cfg
import glob
from langchain_community.chat_models import ChatOpenAI
from langchain_core.tools import Tool
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    JSONLoader,
    GithubFileLoader,
    UnstructuredExcelLoader,
    RecursiveUrlLoader,
    WebBaseLoader,
    YoutubeLoader,
	ArxivLoader,
    WikipediaLoader,
	UnstructuredEmailLoader,
    SharePointLoader,
    GoogleDriveLoader,
    UnstructuredPowerPointLoader,
    OutlookMessageLoader,
    OneDriveLoader
)

from langchain_community.document_loaders.parsers import PyPDFParser
from langchain_core.document_loaders.base import BaseLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
import os
from pathlib import Path
import re
from typing import Optional, List, Dict, Any
import wikipedia

def throw_if( name: str, value: Any ) -> None:
	'''

		Purpose:
		-----------
		Simple guard which raises ValueError when `value` is falsy (None, empty).

		Parameters:
		-----------
		name (str): Variable name used in the raised message.
		value (Any): Value to validate.

		Returns:
		-----------
		None: Raises ValueError when `value` is falsy.

	'''
	if value is None:
		raise ValueError( f"Argument '{name}' cannot be empty!" )

class Loader( ):
	'''

		Purpose:
		--------
		Base class providing shared utilities for concrete loader wrappers.
		Encapsulates file validation, path resolution, and document splitting.

		Attributes:
		----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int

		Methods:
		-------
		_ensure_existing_file( self, path: str ) -> str
		_resolve_paths( self, pattern: str ) -> List[ str ]
		_split_documents( self, docs: List[ Document ], chunk: int=1000, overlap: int=200 ) ->
		List[ Document ]

	'''
	documents: Optional[ List[ Document ] ]
	file_path: Optional[ str ]
	pattern: Optional[ str ]
	expanded: Optional[ List[ str ] ]
	candidates: Optional[ List[ str ] ]
	resolved: Optional[ List[ str ] ]
	loader: Optional[ BaseLoader ]
	splitter: Optional[ RecursiveCharacterTextSplitter | CharacterTextSplitter ]
	chunk_size: Optional[ int ]
	overlap_amount: Optional[ int ]
	
	def __init__( self ) -> None:
		self.documents = [ ]
		self.candidates = [ ]
		self.resolved = [ ]
		self.expanded = [ ]
		self.file_path = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def verify_exists( self, path: str ) -> str | None:
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
			exception.cause = 'Loader'
			exception.method = '_ensure_existing_file( self, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def resolve_paths( self, pattern: str ) -> List[ str ] | None:
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
			exception.cause = 'Loader'
			exception.method = '_resolve_paths( self, pattern: str ) -> List[ str ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def load_documents( self, path: str, encoding: Optional[ str ], csv_args: Optional[Dict[ str, Any ] ],
			source_column: Optional[ str ] ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load files into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the CSV file.
			encoding (Optional[str]): File encoding (e.g., 'utf-8') if known.
			source_column (Optional[str]): Column name used for source attribution.

			Returns:
			--------
			List[Document]: List of LangChain Document objects parsed from the CSV.

		'''
		try:
			self.file_path = self.verify_exists( path )
			self.encoding = encoding
			self.csv_args = csv_args
			self.source_column = source_column
			self.loader = BaseLoader( file_path=self.file_path, encoding=self.encoding,
				csv_args=self.csv_args, source_column=self.source_column )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'CSV'
			exception.method = 'loader( )'
			error = ErrorDialog( exception )
			error.show( )
	
	def split_documents( self, docs: List[ Document ], chunk: int=1000, overlap: int=200 ) -> \
	List[ Document ] | None:
		'''

			Purpose:
			--------
			Split long Document objects into smaller chunks for better token management.

			Parameters:
			-----------
			docs (List[Document]): Input LangChain Document objects.
			chunk_size (int): Max characters in each chunk.
			chunk_overlap (int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Re-chunked list of Document objects.

		'''
		try:
			throw_if( 'docs', docs )
			self.documents = docs
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder( model_name='gpt-4o',
				chunk_size=self.chunk_size, overlap=self.overlap_amount )
			return self.splitter.split_documents( documents=self.documents )
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'Loader'
			exception.method = ('split_documents( self, docs: List[ Document ], chunk: int=1000, '
			                    'overlap: int=200 ) -> List[ Document ]')
			error = ErrorDialog( exception )
			error.show( )

class TextLoader( Loader ):
	'''
		
		Purpose:
		-------
		Class for loading text documents
		
	'''
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	splitter: Optional[ RecursiveCharacterTextSplitter | CharacterTextSplitter ]
	raw_text: Optional[ str ]
	separator: Optional[ str ]
	length_function: Optional[ object ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.raw_text = None
		self.documents = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.separator = "\n\n"
		self.length_function = len

	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
		return [ 'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split',]
	
	def load( self, filepath: str, size: int=1000, amount: int=200, seps: str="\n\n"  ) -> List[ Document ] | None:
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
				self.chunk_size = size
				self.overlap_amount = amount
				self.separator = seps
				self.raw_text = open( self.file_path, mode='r', encoding='utf-8', errors='ignore' ).read()
				self.splitter = CharacterTextSplitter( separator=self.separator,
					chunk_size=self.chunk_size, chunk_overlap=self.overlap_amount )
				self.documents = self.splitter.create_documents( texts=self.raw_text )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Loader'
			exception.cause = 'TextLoader'
			exception.method = 'load( self, file_path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def split_tokens( self, size: int=1000, amount: int=200 ) -> List[ Document ] | None:
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
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			self.chunk_size = size
			self.overlap_amount = amount
			self.splitter = CharacterTextSplitter.from_tiktoken_encoder( encoding_name='cl100k_base',
				chunk_size=self.chunk_size, chunk_overlap=self.overlap_amount)
			self.documents = self.splitter.create_documents( texts=self.raw_text )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'TextLoader'
			exception.method = 'split_tokens( self, size: int=1000, amount: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split_chars( self, size: int=1000, amount: int=200, seps: str="\n\n" ) -> List[ Document ] | None:
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
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			self.chunk_size = size
			self.overlap_amount = amount
			self.separator = seps
			self.splitter = CharacterTextSplitter.from_tiktoken_encoder( encoding_name='cl100k_base',
				chunk_size=self.chunk_size, chunk_overlap=self.overlap_amount)
			self.documents = self.splitter.create_documents( texts=self.raw_text )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'TextLoader'
			exception.method = 'split_chars( self, size: int=1000, amount: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
		
class CsvLoader( Loader ):
	'''

		Purpose:
		--------
		Provides CSVLoader functionality to parse CSV files into Document objects.


		Attributes:
		----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int
		csv_args: Dict[ str, Any ]
		columns - List[ str ]

		Methods:
		-------
		verify_exists( self, path: str ) -> str,
		resolve_paths( self, pattern: str ) -> List[ str ],
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ]
		load( self, path: str, encoding: Optional[ str ]=None,

	'''
	loader: Optional[ CSVLoader ]
	documents: Optional[ List[ Document ] ]
	splitter: Optional[ RecursiveCharacterTextSplitter ]
	file_path: Optional[ str ]
	quote_char: Optional[ str ]
	csv_args: Optional[ Dict[ str, Any ] ]
	columns: Optional[ List[ str ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.columns = None
		self.csv_args = None
		self.documents = None
		self.quote_char = '"'
		self.pattern = ","
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'delimiter'
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split',
		         'csv_args',
		         'columns' ]
	
	def load( self, filepath: str, columns: Optional[ List[ str ] ]=None,
			delimiter: str="\n\n", quotechar: str='"'  ) -> List[ Document ] | None:
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
			self.file_path = self.verify_exists( filepath )
			self.columns = columns
			self.pattern = delimiter
			self.quote_char = quotechar
			self.csv_args = { 'delimiter': self.pattern, 'fieldnames': self.columns, 'quotechar': self.quote_char }
			self.loader = CSVLoader( file_path=self.file_path, csv_args=self.csv_args,
				content_columns=self.columns )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'CsvLoader'
			exception.method = 'loader( )'
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
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			self.chunk_size = size
			self.overlap_amount = amount
			_documents = self.split_documents( docs=self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return _documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'CsvLoader'
			exception.method = 'split( self, size: int=1000, amount: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class WebLoader( Loader ):
	'''

		Purpose:
		--------
		Functionality to load all text from HTML webpages into
		a document format that we can use downstream.
		To bypass SSL verification errors during fetching, you can set the “verify” option:
		
		You can also pass in a list of pages to load from.
			loader_multiple_pages = WebBaseLoader(
			    ["https://www.example.com/", "https://google.com"] )

		Attributes:
		----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int
		url - str
		loader - WebBaseLoader

		Methods:
		--------
		verify_exists( self, path: str ) -> str,
		resolve_paths( self, pattern: str ) -> List[ str ],
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ]
		load( urls: List[ str ] ) -> List[ Documents ]
		split( ) -> List[ Document ]

	'''
	loader: Optional[ RecursiveUrlLoader | WebBaseLoader ]
	url: Optional[ str  ]
	web_paths: Optional[ str | List[ str ] ]
	documents: Optional[ List[ Document ] ]
	file_path: Optional[ str ]
	max_depth: Optional[ int ]
	tiemout: Optional[ int ]
	ignore: Optional[ bool ]
	with_progress: Optional[ bool ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.max_depth = None
		self.tiemout = None
		self.url = None
		self.documents = None
		self.file_path = None
		self.web_paths = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'max_depth',
		         'timeout',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split',
		         'urls', ]
	
	def load_recursive( self, url: str, depth: int=2, max_time: int=10, ignore: bool=True ) -> List[ Document ] | None:
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
			throw_if( 'url', url )
			self.url= url
			self.max_depth = depth
			self.tiemout = max_time
			self.ignore = ignore
			self.loader = RecursiveUrlLoader( self.url, max_depth=self.max_depth,
				timeout=self.tiemout, continue_on_failure=self.ignore )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WebLoader'
			exception.method = 'load_recursive( self, url: str, depth: int=2, max_time: int=10, ignore: bool=True ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def load_pages( self, urls: List[ str ], depth: int=2, timeout: int=10,
			ignore: bool=True, progress: bool=True ) -> List[ Document ] | None:
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
			self.web_paths = urls
			self.max_depth = depth
			self.tiemout = timeout
			self.ignore = ignore
			self.with_progress = progress
			self.loader = WebBaseLoader( web_paths=self.web_paths, show_progress=self.with_progress,
				continue_on_failure=self.ignore )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WebLoader'
			exception.method = 'load( self, urls: List[ str ] ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded web documents into smaller chunks for better LLM processing.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlap between chunks in characters.

			Returns:
			--------
			List[Document]: Chunked Document objects.

		'''
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			_documents = self.split_documents( docs=self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return _documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WebLoader'
			exception.method = 'split( self, chunk: int=1000 , overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class PdfLoader( Loader ):
	"""

		Purpose:
		-------
		Public, SDK-oriented PDF loader with:
			- Page-aware metadata
			- Two-stage chunking
			- Configurable chunk profiles
			- Table isolation
			- Optional OCR fallback

	"""
	loader: Optional[ PyPDFLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	mode: Optional[ str ]
	extraction: Optional[ str ]
	include_images: Optional[ bool ]
	image_format: Optional[ str ]
	custom_delimiter: Optional[ str ]
	image_parser: Optional[ RapidOCRBlobParser ]
	
	def __init__( self, size: int=1000, overlap: int=150,
			has_tables: bool=True, include: bool=True ) -> None:
		"""

			Purpose:
			---------
			Initialize the PdfLoader.

			Parameters:
				path:
					Path to the PDF file.
				size:
					Target chunk size (characters).
				overlap:
					Overlap between chunks.
				has_tables:
					Enable table detection and isolation.
				use_ocr:
					Enable OCR fallback for image-only PDFs.
		"""
		super( ).__init__( )
		self.enable_tables = has_tables
		self.include_images = include
		self.file_path = None
		self.documents = [ ]
		self.pattern = None
		self.chunk_size = size
		self.overlap_amount = overlap
		self.loader = None
		self.mode = None
		self.image_format = None
		self.custom_delimiter = None
	
	@property
	def mode_options( self ):
		'''

			Returns:
			--------
			A List[ str ] of mode options

		'''
		return [ 'page',
		         'single' ]
	
	@property
	def extraction_options( self ):
		'''

			Returns:
			--------
			A List[ str ] of mode options

		'''
		return [ 'plain',
		         'layout' ]
	
	@property
	def image_options( self ):
		'''

			Returns:
			--------
			A List[ str ] of mode options

		'''
		return [ 'html-img',
		         'markdown-img',
		         'text-img' ]
	
	def load( self, filepath: str, mode: str='single', extract: str='plain',
			include: bool=True, format: str='markdown-img' ) -> List[ Document ]:
		"""

			Purpose:
			---------
			Loads PDF document into Langchain documnet

			Returns:
				List[Document]
		"""
		try:
			throw_if( 'path', filepath )
			self.file_path = self.verify_exists( filepath )
			self.mode = mode
			self.extraction = extract
			self.include_images = include
			self.image_format = format
			if self.include_images:
				self.image_parser = RapidOCRBlobParser( )
				self.loader = PyPDFLoader( file_path=self.file_path, mode=self.mode,
					extraction_mode=self.extraction, extract_images=self.inlude_images,
					images_inner_format=self.image_format, images_parser=self.image_parser )
				self.documents = self.loader.load( )
				return self.documents
			else:
				self.loader = PyPDFLoader( file_path=self.file_path, mode=self.mode,
					extraction_mode=self.extraction )
				self.documents = self.loader.load( )
				return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'PdfLoader'
			exception.method = 'load( self, path: str, mode: str=single, extract: str=plain ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class ExcelLoader( Loader ):
	'''


		Purpose:
		--------
		Provides LangChain's UnstructuredExcelLoader functionality
		to parse Excel spreadsheets into documents.

		Attibutes:
		----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int

		Methods:
		--------
		verify_exists( self, path: str ) -> str,
		resolve_paths( self, pattern: str ) -> List[ str ],
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ]
		load( path: str, mode: str ) -> List[ Document ]
		split( ) -> List[ Document ]


	'''
	loader: Optional[ UnstructuredExcelLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	mode: Optional[ str ]
	has_headers: Optional[ bool ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = [ ]
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	@property
	def mode_options( self ):
		'''
			
			Returns:
			-------
			List[ str ] of loading mode options
			
		'''
		return [ 'single', 'page' ]
	
	def load( self, path: str, mode: str='elements', headers: bool=True ) -> List[
		                                                                             Document ] | None:
		'''


			Purpose:
			--------
			Load and convert Excel data into LangChain Document objects.

			Parameters:
			-----------
			path (str): File path to the Excel spreadsheet.
			mode (str): Extraction mode, either 'elements' or 'paged'.
			headers (bool): Whether to include column headers in parsing.

			Returns:
			--------
			List[Document]: List of parsed Document objects from Excel content.


		'''
		try:
			throw_if( 'path', path )
			self.mode = mode
			self.file_path = self.verify_exists( path )
			self.loader = UnstructuredExcelLoader( file_path=self.file_path, mode=self.mode )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'ExcelLoader'
			exception.method = 'load( self, path: str, mode: str=elements, include_headers: bool=True ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''


			Purpose:
			--------
			Split loaded Excel documents into manageable chunks.

			Parameters:
			-----------
			chunk_size (int): Maximum characters per chunk.
			chunk_overlap (int): Characters overlapping between chunks.

			Returns:
			--------
			List[Document]: Chunked and cleaned list of Document objects.


		'''
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=overlap )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'ExcelLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class WordLoader( Loader ):
	'''


		Purpose:
		--------
		Provides LangChain's Docx2txtLoader functionality to
		convert docx files into Document objects.

		Attributes:
		----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int

		Methods:
		--------
		verify_exists( self, path: str ) -> str,
		resolve_paths( self, pattern: str ) -> List[ str ],
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ]
		load( path: str, mode: str ) -> List[ Document ]
		split( ) -> List[ Document ]


	'''
	loader: Optional[ Docx2txtLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.documents = None
		self.file_path = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
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
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = Docx2txtLoader( self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WordLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''


			Purpose:
			--------
			Split Word documents into text chunks suitable for LLM processing.

			Parameters:
			-----------
			chunk_size (int): Maximum characters per chunk.
			chunk_overlap (int): Overlap between chunks in characters.

			Returns:
			--------
			List[Document]: Chunked list of Document objects.


		'''
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			_splits = self.split_documents( docs=self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return _splits
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WordLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class MarkdownLoader( Loader ):
	'''


		Purpose:
		--------
		Wrap LangChain's UnstructuredMarkdownLoader to parse Markdown files into Document objects.

		Attributes:
		-----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int

		Methods:
		--------
		verify_exists( self, path: str ) -> str,
		resolve_paths( self, pattern: str ) -> List[ str ],
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ]
		load( path: str, mode: str ) -> List[ Document ]
		split( ) -> List[ Document ]


	'''
	loader: Optional[ UnstructuredMarkdownLoader ]
	file_path: str | None
	documents: List[ Document ] | None
	mode: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = [ ]
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	@property
	def mode_options( self ):
		'''

			Returns:
			--------
			A List[ str ] of mode options

		'''
		return [ 'page',
		         'single' ]
	
	def load( self, path: str, mode: str='single' ) -> List[ Document ] | None:
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
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.mode = mode
			self.loader = UnstructuredMarkdownLoader( file_path=self.file_path, mode=self.mode )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'MarkdownLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ] '
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded Markdown content into text chunks for LLM consumption.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Number of characters that overlap between chunks.

			Returns:
			--------
			List[Document]: Split Document chunks from the original Markdown content.

		'''
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			_documents = self.split_documents( docs=self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return _documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'MarkdownLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class HtmlLoader( Loader ):
	'''

		Purpose:
		--------
		Provides the UnstructuredHTMLLoader's functionality to parse HTML files into Document objects.

		Attributes:
		-----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int

		Methods:
		--------
		verify_exists( self, path: str ) -> str,
		resolve_paths( self, pattern: str ) -> List[ str ],
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ]
		load( path: str, mode: str ) -> List[ Document ]
		split( ) -> List[ Document ]

	'''
	loader: Optional[ UnstructuredHTMLLoader ]
	file_path: str | None
	documents: List[ Document ] | None
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
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
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = UnstructuredHTMLLoader( file_path=self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'HTML'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded HTML documents into manageable text chunks.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Chunked list of LangChain Document objects.

		'''
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'HtmlLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class JsonLoader( Loader ):
	'''

		Purpose:
		--------
		Provides the UnstructuredHTMLLoader's functionality to parse HTML files into Document objects.

		Attributes:
		-----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int

		Methods:
		--------
		verify_exists( self, path: str ) -> str,
		resolve_paths( self, pattern: str ) -> List[ str ],
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ]
		load( path: str, mode: str ) -> List[ Document ]
		split( ) -> List[ Document ]

	'''
	loader: Optional[ JSONLoader ]
	file_path: str | None
	jq: Optional[ str ]
	is_text: Optional[ bool ]
	is_lines: Optional[ bool ]
	documents: List[ Document ] | None
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.is_text = None
		self.is_lines = None
		self.jq = '.messages[].content'
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	def load( self, filepath: str, is_text: bool=True, is_lines: bool=False ) -> List[ Document ] | None:
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
			throw_if( 'filepath', filepath )
			self.file_path = self.verify_exists( filepath )
			self.is_text = is_text
			self.is_lines = is_lines
			self.loader = JSONLoader( file_path=self.file_path, jq_schema=self.jq,
				text_content=self.is_text, json_lines=self.is_lines  )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'JsonLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded HTML documents into manageable text chunks.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Chunked list of LangChain Document objects.

		'''
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'JsonLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
			
class YouTubeLoader( Loader ):
	'''

		Purpose:
		--------
		Provides functionality to parse youttube video transcripts into Document objects.

		Attributes:
		-----------
		documents - List[ Document ];
		file_path -  str;
		pattern -  str;
		expanded - List[ str ];
		candidates - List[ str ];
		resolved - List[ str ];
		splitter - RecursiveCharacterTextSplitter;
		chunk_size - int;
		overlap_amount - int;

		Methods:
		--------
		verify_exists( self, path: str ) -> str;
		resolve_paths( self, pattern: str ) -> List[ str ];
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ];
		load( path: str, mode: str ) -> List[ Document ];
		split( ) -> List[ Document ];

	'''
	loader: Optional[ YoutubeLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	include_info: Optional[ bool ]
	llm: Optional[ ChatOpenAI ]
	language: Optional[ str ]
	translation: Optional[ str ]
	temperature: Optional[ int ]
	api_key: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.include_info = None
		self.temperature = 0
		self.api_key = cfg.OPENAI_API_KEY
		self.llm = ChatOpenAI( temperature=self.temperature, api_key=self.api_key )
		self.file_path = None
		self.documents = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'include_info',
		         'llm',
		         'language',
		         'temperature',
		         'api_key',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	@property
	def language_options( self ):
		'''
			
			Returns:
			--------
			A List[ str ] of languages in order of decreasing preference
			
		'''
		return [ 'en-US', 'ceb', 'es', 'zh-CN', 'fil', 'de', 'ja', 'ru' ]
	
	@property
	def translation_options( self ):
		'''

		Returns:
		--------
		A List[ str ] of languages in order of decreasing preference
		'''
		return [ 'en-US',
		         'ceb',
		         'es',
		         'zh-CN',
		         'fil',
		         'de',
		         'ja',
		         'ru' ]
	
	def load( self, youtube_url: str, lang: str='en',
			trans: str='en', add_info: bool=True ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load an video file and convert its contents into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'youtube_url', youtube_url )
			self.file_path = self.verify_exists( youtube_url )
			self.include_info = add_info
			self.language = lang
			self.translation = trans
			self.loader = YoutubeLoader.from_youtube_url( self.file_path,
				add_video_info=self.include_info, language=[ self.language ],
				translation=self.translation   )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'YoutubeLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded Youtube Transcript documents into manageable text chunks.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Chunked list of LangChain Document objects.

		'''
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( docs=self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'YoutubeLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class ArXivLoader( Loader ):
	'''

		Purpose:
		--------
		alaods documents from an open-access archive for 2 million scholarly articles in the
		fields of physics,  mathematics, computer science, quantitative biology,
		quantitative finance, statistics,  electrical engineering and systems science, and economics.

		Attributes:
		-----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int

		Methods:
		--------
		verify_exists( self, path: str ) -> str;
		resolve_paths( self, pattern: str ) -> List[ str ];
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ];
		load( path: str, mode: str ) -> List[ Document ];
		split( ) -> List[ Document ];

	'''
	loader: Optional[ ArxivLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	max_documents: Optional[ int ]
	max_characters: Optional[ int ]
	include_metadata: Optional[ bool ]
	query: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.max_documents = None
		self.max_characters = None
		self.include_metadata = False
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'max_documents',
		         'max_characters',
		         'include_metadata',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	def load( self, query: str, max_chars: int=1000 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load an video file and convert its contents into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'query', query )
			self.query = query
			self.max_characters = max_chars
			self.loader = ArxivLoader( query=self.query, doc_content_chars_max=self.max_characters )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'ArxivLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded Youtube Transcript documents into manageable text chunks.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Chunked list of LangChain Document objects.

		'''
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'ArxivLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class WikiLoader( Loader ):
	'''

		Purpose:
		--------
		Provides the Arxiv loading functionality
		to parse video research papers into Document objects.

		Attributes:
		-----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int

		Methods:
		--------
		verify_exists( self, path: str ) -> str;
		resolve_paths( self, pattern: str ) -> List[ str ];
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ];
		load( path: str, mode: str ) -> List[ Document ];
		split( ) -> List[ Document ];

	'''
	loader: Optional[ WikipediaLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	query: Optional[ str ]
	max_documents: Optional[ int ]
	max_characters: Optional[ int ]
	include_all: Optional[ bool ]
	query: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.max_documents = None
		self.max_characters = None
		self.include_all
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'max_documents',
		         'max_characters',
		         'include_all',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	def load( self, query: str, max_docs: int=25, max_chars: int=4000 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load an wikipedia and convert its contents into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'query', query)
			self.query = query
			self.max_documents = max_docs
			self.max_characters = max_chars
			self.loader = WikipediaLoader( query=self.query, max_documents=self.max_documents,
				load_all_available_meta=self.include_all, doc_content_chars_max=self.max_characters )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WikiLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded Wikipedia search documents into manageable text chunks.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Chunked list of LangChain Document objects.

		'''
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WikiLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class GithubLoader( Loader ):
	'''

		Purpose:
		--------
		Provides the functionality to laod github files in to langchain documents

		Attributes:
		-----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int

		Methods:
		--------
		verify_exists( self, path: str ) -> str;
		resolve_paths( self, pattern: str ) -> List[ str ];
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ];
		load( path: str, mode: str ) -> List[ Document ];
		split( ) -> List[ Document ];

	'''
	loader: Optional[ GithubFileLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	query: Optional[ str ]
	max_documents: Optional[ int ]
	max_characters: Optional[ int ]
	include_all: Optional[ bool ]
	query: Optional[ str ]
	repo: Optional[ str ]
	branch: Optional[ str ]
	access_token: Optional[ str ]
	github_url: Optional[ str ]
	file_filter: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.max_documents = None
		self.max_characters = None
		self.include_all = None
		self.github_url = None
		self.repo = None
		self.branch = None
		self.file_filter = None
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'max_documents',
		         'max_characters',
		         'include_all',
		         'repo',
		         'branch',
		         'file_filter',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	def load( self, url: str, repo: str, branch: str, filetype: str='.md' ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load filtered contents of Github repo/branch into LangChain Document objects.

			Parameters:
			-----------
			url (str):
			repo (str):
			branch (str):
			filetype (str):

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'url', url )
			self.github_url = url
			self.repo = repo
			self.branch = branch
			self.pattern = filetype
			self.file_filter = lambda file_path: file_path.endswith( self.pattern )
			self.loader = GithubFileLoader( repo=self.repo, branch=self.branch,
				github_api_url=self.github_url, file_filter=self.file_filter )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'GithubLoader'
			exception.method = 'load( self, url: str, repo: str, branch: str, filetype: str=md ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded Wikipedia search documents into manageable text chunks.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Chunked list of LangChain Document objects.

		'''
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'GithubLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class PowerPointLoader( Loader ):
	'''

		Purpose:
		--------
		Provides PowerPoint loading functionality
		to parse ppt files into Document objects.

		Attributes:
		-----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int

		Methods:
		--------
		verify_exists( self, path: str ) -> str;
		resolve_paths( self, pattern: str ) -> List[ str ];
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ];
		load( path: str, mode: str ) -> List[ Document ];
		split( ) -> List[ Document ];

	'''
	loader: Optional[ UnstructuredPowerPointLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	mode: Optional[ str ]
	query: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'query',
		         'mode',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	def load( self, path: str, mode: str='single' ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load PowerPoint slides and convert their content into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.mode = mode
			self.loader = UnstructuredPowerPointLoader( file_path=self.file_path, mode=self.mode )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'PowerPointLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def load_multiple( self, path: str ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load PowerPoint slides and convert their content into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.mode = 'multiple'
			self.loader = UnstructuredPowerPointLoader( file_path=self.file_path, mode=self.mode )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'PowerPointLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded Wikipedia search documents into manageable text chunks.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Chunked list of LangChain Document objects.

		'''
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'PowerPointLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

			
class OutlookLoader( Loader ):
	'''

		Purpose:
		--------
		Provides the Arxiv loading functionality
		to parse video research papers into Document objects.

		Attributes:
		-----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int

		Methods:
		--------
		verify_exists( self, path: str ) -> str;
		resolve_paths( self, pattern: str ) -> List[ str ];
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ];
		load( path: str, mode: str ) -> List[ Document ];
		split( ) -> List[ Document ];

	'''
	loader: Optional[ OutlookMessageLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	query: Optional[ str ]
	max_documents: Optional[ int ]
	max_characters: Optional[ int ]
	query: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.max_documents = 2
		self.max_characters = 1000
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'max_charactes',
		         'max_documents',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	def load( self, path: str ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load Outlook Message from a path converting contents into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = OutlookMessageLoader( file_path=self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'OutlookLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded Wikipedia search documents into manageable text chunks.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Chunked list of LangChain Document objects.

		'''
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'OutlookLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class SpfxLoader( Loader ):
	'''

		Purpose:
		--------
		Provides the Sharepoint loading functionality
		to parse video research papers into Document objects.

		Attributes:
		-----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int
		loader - SharePointLoader
		library_id - str
		subsite_id - str
		folder_id - str
		object_ids - List[ str ]
		query - str
		with_token - bool
		is_recursive - bool

		Methods:
		--------
		verify_exists( self, path: str ) -> str;
		resolve_paths( self, pattern: str ) -> List[ str ];
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ];
		load( path: str, mode: str ) -> List[ Document ];
		split( ) -> List[ Document ];

	'''
	loader: Optional[ SharePointLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	library_id: Optional[ str ]
	subsite_id: Optional[ str ]
	folder_id: Optional[ str ]
	object_ids: Optional[ List[ str ] ]
	query: Optional[ str ]
	with_token: Optional[ bool ]
	is_recursive: Optional[ bool ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.folder_id = None
		self.library_id = None
		self.subsite_id = None
		self.object_ids = [ ]
		self.with_token = None
		self.is_recursive = None
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'folder_id',
		         'library_id',
		         'subsite_id',
		         'object_id',
		         'with_token',
		         'is_recursive',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	def load( self, library_id: str ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load Sharepoint files and convert their contents into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'library_id', library_id )
			self.library_id = library_id
			self.is_recursive = True
			self.with_token = True
			self.loader = SharePointLoader( document_library_id=self.library_id,
				recursive=self.is_recursive, auth_with_token=self.with_token )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'SpfxLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def load_folder( self, library_id: str, folder_id: str ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load Sharepoint files and convert their contents into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'library_id', library_id )
			throw_if( 'folder_id', folder_id )
			self.library_id = library_id
			self.folder_id = folder_id
			self.loader = SharePointLoader( document_library_id=self.library_id, folder_id=self.folder_id )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'SpfxLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded Sharepoint file documents into manageable text chunks.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Chunked list of LangChain Document objects.

		'''
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'SpfxLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
			
class OneDriveLoader( Loader ):
	'''

		Purpose:
		--------
		Provides OneDrvie loading functionality
		to parse contents into Document objects.

		Attributes:
		-----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int

		Methods:
		--------
		verify_exists( self, path: str ) -> str;
		resolve_paths( self, pattern: str ) -> List[ str ];
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ];
		load( path: str, mode: str ) -> List[ Document ];
		split( ) -> List[ Document ];

	'''
	loader: Optional[ OneDriveLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	client_id: Optional[ str ]
	drive_id: Optional[ str ]
	client_secret: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.drive_id = None
		self.client_id = None
		self.client_secret = None
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'query',
		         'drive_id',
		         'client_id',
		         'client_secret',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'load_folder',
		         'split', ]
	
	@property
	def file_options( self ):
		'''

			Returns:
			-------
			List[ str ] of file options

		'''
		return [ 'pdf', 'doc', 'docx', 'txt']
	
	def load( self, id: str ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load an onedrive file and convert its contents into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'id', id )
			self.drive_id = id
			self.loader = OneDriveLoader( drive_id=self.drive_id )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WikiLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def load_folder( self, id: str, path: str ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load an onedrive file and convert its contents into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'id', id )
			self.drive_id = id
			self.file_path = path
			self.loader = OneDriveLoader( drive_id=self.drive_id, folder_path=self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WikiLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded Wikipedia search documents into manageable text chunks.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Chunked list of LangChain Document objects.

		'''
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WikiLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class GoogleLoader( Loader ):
	'''

		Purpose:
		--------
		Provides Google Drive loading functionality
		to parse contents into Document objects.

		Attributes:
		-----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int

		Methods:
		--------
		verify_exists( self, path: str ) -> str;
		resolve_paths( self, pattern: str ) -> List[ str ];
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ];
		load( path: str, mode: str ) -> List[ Document ];
		split( ) -> List[ Document ];

	'''
	loader: Optional[ GoogleDriveLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	query: Optional[ str ]
	file_id: Optional[ str ]
	folder_id: Optional[ str ]
	query: Optional[ str ]
	is_recursive: Optional[ bool ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.file_id = None
		self.folder_id = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.is_recursive = None
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'query',
		         'folder_id',
		         'file_id',
		         'is_recursive',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'load_folder',
		         'split', ]
	
	@property
	def file_options( self ):
		'''

			Returns:
			-------
			List[ str ] of file options

		'''
		return [ 'document',
		         'sheet',
		         'pdf' ]
	
	def load_file( self, file_id: str, recursive: bool=False ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load an google drive file by id and convert its contents into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'file_id', file_id )
			throw_if( 'recursive', recursive )
			self.file_id = file_id
			self.is_recursive = recursive
			self.loader = GoogleDriveLoader( file_ids=[ self.file_id ],
				recursive=self.is_recursive )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'GoogleDriveLoader'
			exception.method = 'load_File( self, file_id: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def load_folder( self, folder_id: str, recursive: bool=False ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load an google drive file and convert its contents into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'folder_id', folder_id )
			self.folder_id = folder_id
			self.is_recursive = recursive
			self.loader = GoogleDriveLoader( folder_id=self.folder_id, recursive=self.is_recursive )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'GoogleDriveLoader'
			exception.method = 'load_folder( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded google drive documents into manageable text chunks.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Chunked list of LangChain Document objects.

		'''
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'GoogleDriveLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class EmailLoader( Loader ):
	'''


		Purpose:
		--------
		Provides LangChain's UnstructuredEmailLoader functionality
		to parse email documents (*.eml) into documents.

		Attibutes:
		----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int

		Methods:
		--------
		verify_exists( self, path: str ) -> str,
		resolve_paths( self, pattern: str ) -> List[ str ],
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ]
		load( path: str, mode: str ) -> List[ Document ]
		split( ) -> List[ Document ]


	'''
	loader: Optional[ UnstructuredEmailLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	has_attachments: Optional[ bool ]
	mode: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = [ ]
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'has_attachments',
		         'mode',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	def load( self, path: str, mode: str='single', attachments: bool=True ) -> List[ Document ]:
		'''


			Purpose:
			--------
			Load and convert Email data (*.eml) into LangChain Document objects.

			Parameters:
			-----------
			path (str): File path to the Excel spreadsheet.
			mode (str): Extraction mode, either 'elements' or 'paged'.
			include_headers (bool): Whether to include column headers in parsing.

			Returns:
			--------
			List[Document]: List of parsed Document objects from Email content.


		'''
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.mode = mode
			self.has_attachments = attachments
			self.loader = UnstructuredEmailLoader( file_path=self.file_path, mode=self.mode,
				process_attachments=self.has_attachments )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'EmailLoader'
			exception.method = ('load( self, path: str, mode: str=elements, '
			                    'include_headers: bool=True ) -> List[ Document ]')
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''


			Purpose:
			--------
			Split loaded Email documents into manageable chunks.

			Parameters:
			-----------
			chunk_size (int): Maximum characters per chunk.
			chunk_overlap (int): Characters overlapping between chunks.

			Returns:
			--------
			List[Document]: Chunked and cleaned list of Document objects.


		'''
		try:
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'EmailLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )