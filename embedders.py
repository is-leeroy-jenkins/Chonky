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

	     name.py
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
    name.py
  </summary>
  ******************************************************************************************
'''
from groq import Groq
from requests.models import Response

from openai import OpenAI
from google import genai
from google.genai import types
from google.genai.types import (EmbedContentConfig, HttpOptions)
import config as cfg
from typing import List, Optional, Any, Union, Dict
import tiktoken
import os
from boogr import Error
import requests
from llama_cpp import Llama

def throw_if( name: str, value: object ):
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class GPT( ):
	"""

	    Purpose
	    ___________
	    Class used for creating vectors using OpenAI's embedding models

	    Parameters
	    ------------
	    None

	    Attributes
	    -----------
	    api_key
	    client
	    model
	    embedding
	    response

	    Methods
	    ------------
	    create( self, text: str ) -> get_list[ float ]


    """
	
	web_options: Optional[ List[ str ] ]
	input: Optional[ List[ str ] ]
	client: Optional[ OpenAI ]
	prompt: Optional[ str ]
	response_format: Optional[ str ]
	input: Optional[ List[ str ] ]
	response: Optional[ Response ]
	embedding: Optional[ List[ float ] ]
	encoding_format: Optional[ str ]
	dimensions: Optional[ int ]
	
	def __init__( self ):
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
		self.encoding_format = None
		self.input = None
		self.model = None
		self.embedding = None
		self.response = None
	
	@property
	def model_options( self ) -> List[ str ]:
		'''
	
			Returns:
			--------
			List[ str ] of embedding models

		'''
		return [ 'text-embedding-3-small',
		         'text-embedding-3-large',
		         'text-embedding-ada-002' ]
	
	@property
	def encoding_options( self ) -> List[ str ]:
		'''

			Returns:
			--------
			List[ str ] of available format options

		'''
		return [ 'float',
		         'base64' ]
	
	def create( self, text: str, model: str = 'text-embedding-3-small', format: str = 'float' ) -> \
			List[ float ]:
		"""

	        Purpose
	        _______
	        Creates an embedding ginve a text


	        Parameters
	        ----------
	        text: str


	        Returns
	        -------
	        get_list[ float

        """
		try:
			throw_if( 'text', text )
			self.input = text
			self.model = model
			self.encoding_format = format
			self.response = self.client.embeddings.create( input=self.input, model=self.model,
				encoding_format=self.encoding_format )
			self.embedding = self.response.data[ 0 ].embedding
			return self.embedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'embedders'
			exception.cause = 'GPT'
			exception.method = 'create( self, text: str, model: str ) -> List[ float ]'
			raise exception
	
	def embed( self, texts: List[ str ], model: str = 'text-embedding-3-small' ) -> List[
		List[ float ] ]:
		"""
		
			Purpose:
			--------
			Generate embeddings for a batch of text inputs.
			
			Parameters:
			-----------
			texts : list[str]
				Text inputs to embed.
			model : str
				OpenAI embedding model.
			
			Returns:
			--------
			list[list[float]]
				Embedding vectors.
				
		"""
		try:
			throw_if( 'texts', texts )
			
			if not isinstance( texts, list ):
				raise TypeError( 'Argument "texts" must be a list of strings.' )
			
			self.model = model
			
			cleaned: List[ str ] = [ ]
			for text in texts:
				if isinstance( text, str ):
					value = text.strip( )
					if value:
						cleaned.append( value )
			
			if not cleaned:
				return [ ]
			
			max_batch_size: int = 2048
			all_vectors: List[ List[ float ] ] = [ ]
			
			for i in range( 0, len( cleaned ), max_batch_size ):
				batch: List[ str ] = cleaned[ i:i + max_batch_size ]
				self.response = self.client.embeddings.create(
					model=self.model,
					input=batch
				)
				all_vectors.extend( [ item.embedding for item in self.response.data ] )
			
			return all_vectors
		except Exception as e:
			exception = Error( e )
			exception.module = 'embedders'
			exception.cause = 'GPT'
			exception.method = 'embed( self, text: str, model: str ) -> List[ List[ float ] ]'
			raise exception
	
	def count_tokens( self, text: str, coding: str ) -> int | None:
		'''

	        Purpose:
	        -------
	        Returns the num of words in a documents path.

	        Parameters:
	        -----------
	        text: str - The string that is tokenized
	        coding: str - The encoding to use for tokenizing

	        Returns:
	        --------
	        int - The number of words

        '''
		try:
			throw_if( 'text', text )
			throw_if( 'coding', coding )
			_encoding = tiktoken.get_encoding( coding )
			_tokens = len( _encoding.encode( text ) )
			return _tokens
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embedding'
			exception.method = 'count_tokens( self, text: str, coding: str ) -> int'
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		        Purpose:
		        --------
                Method returns a list of strings representing members

		        Parameters:
		        ----------
                self

		        Returns:
		        ---------
                List[ str ] | None

        '''
		return [ 'create',
		         'api_key',
		         'client',
		         'model',
		         'count_tokens',
		         'path',
		         'model_options', ]

class Gemini( ):
	'''

		Purpose:
		--------
		Class handling text embedding generation with the Google GenAI SDK.

		Attributes:
		-----------
		client              : Client - Initialized GenAI client
		response            : any - raw API response
		embedding           : list - Generated vector of floats
		encoding_format     : str - Format of the embedding response
		dimensions          : int - Size of the embedding vector
		use_vertex          : bool - Cloud integration flag
		task_type           : str - Type of task (RETRIEVAL, etc)
		http_options        : HttpOptions - Client networking settings
		embedding_config    : EmbedContentConfig - Configuration for embeddings
		contents            : list - Input strings
		input_text          : str - Current text being processed
		file_path           : str - Path to source text
		response_modalities : str - Modality configuration

		Methods:
		--------
		generate( text, model ) : Creates an embedding vector for input text

	'''
	number: Optional[ int ]
	api_key: Optional[ str ]
	project: Optional[ str ]
	location: Optional[ str ]
	prompt: Optional[ str ]
	credentials: Optional[ str ]
	model: Optional[ str ]
	response_format: Optional[ str ]
	dimensions: Optional[ int ]
	client: Optional[ genai.Client ]
	response: Optional[ Any ]
	embedding: Optional[ List[ float ] ]
	embeddings: Optional[ List[ List[ float ] ] ]
	encoding_format: Optional[ str ]
	dimensions: Optional[ int ]
	use_vertex: Optional[ bool ]
	task_type: Optional[ str ]
	http_options: Optional[ HttpOptions ]
	embedding_config: Optional[ types.EmbedContentConfig ]
	contents: Optional[ List[ str ] ]
	input_text: Optional[ List[ str ] ]
	file_path: Optional[ str ]
	response_modalities: Optional[ str ]
	
	def __init__( self, version: str = 'v1alpha', use_ai: bool = False, dimensions: int = 768 ):
		super( ).__init__( )
		self.api_key = cfg.VERTEX_API_KEY
		self.project = cfg.GOOGLE_CLOUD_PROJECT_ID
		self.location = cfg.GOOGLE_CLOUD_LOCATION
		self.credentials = cfg.GOOGLE_APPLICATION_CREDENTIALS
		self.model = None
		self.api_version = version
		self.use_vertex = use_ai
		self.dimensions = dimensions
		self.credentials = None
		self.http_options = HttpOptions( api_version=self.api_version )
		self.client = genai.Client( vertexai=self.use_vertex, api_key=self.api_key,
			http_options=self.http_options )
		self.embedding = None
		self.embeddings = None
		self.response = None
		self.encoding_format = None
		self.input_text = None
		self.file_path = None
		self.dimensions = None
		self.task_type = None
		self.response_modalities = None
		self.embedding_config = None
		self.content_config = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Returns list of embedding models."""
		return [ 'gemini-embedding-001', 'text-multilingual-embedding-002' ]
	
	@property
	def task_options( self ) -> List[ str ] | None:
		"""Returns list of task type options."""
		return [ 'RETRIEVAL_QUERY', 'RETRIEVAL_DOCUMENT', 'SEMANTIC_SIMILARITY' ]
	
	def generate( self, text: str, model: str = 'gemini-embedding-001', dimensions: int = 768 ) -> \
	Optional[ List[ float ] ]:
		"""
			
			Purpose:
			---------
			Generates a vector representation of the provided text.
			
			Parameters:
			-----------
			text: str - Input text string.
			model: str - Embedding model identifier.
			
			Returns:
			--------
			Optional[ List[ float ] ] - List of embedding values or None on failure.
		
		"""
		try:
			throw_if( 'text', text )
			self.input_text = text
			self.model = model
			self.dimensions = dimensions
			self.embedding_config = EmbedContentConfig( task_type=self.task_type )
			self.response = self.client.models.embed_content( model=self.model,
				contents=self.input_text, config=self.embedding_config )
			self.embedding = self.response.embeddings[ 0 ].values
			return self.embedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'embedders'
			exception.cause = 'Gemini'
			exception.method = 'generate( self, text, model ) -> List[ float ]'
			raise exception
	
	def embed( self, texts: List[ str ], task: str = 'RETRIEVAL_DOCUMENT',
			model: str = 'gemini-embedding-001', dimensions: int = 768 ) -> List[ List[ float ] ]:
		"""
			
			Purpose:
			---------
			Generates embeddings for a batch of strings.
		
			Parameters:
			-----------
			texts: List[str] - List of input text strings.
			task: str - Gemini embedding task type.
			model: str - Embedding model identifier.
			dimensions: int - Requested output dimensionality.
		
			Returns:
			--------
			List[List[float]] - A list of embedding vectors.
		
		"""
		try:
			throw_if( 'texts', texts )
			
			if not isinstance( texts, list ):
				raise TypeError( 'Argument "texts" must be a list of strings.' )
			
			self.model = model
			self.dimensions = dimensions
			self.task_type = task
			
			cleaned: List[ str ] = [ ]
			for text in texts:
				if isinstance( text, str ):
					value = text.strip( )
					if value:
						cleaned.append( value )
			
			if not cleaned:
				return [ ]
			
			all_vectors: List[ List[ float ] ] = [ ]
			max_batch_size: int = 250
			
			for i in range( 0, len( cleaned ), max_batch_size ):
				batch: List[ str ] = cleaned[ i:i + max_batch_size ]
				
				self.embedding_config = EmbedContentConfig(
					task_type=self.task_type,
					output_dimensionality=self.dimensions
				)
				
				self.response = self.client.models.embed_content(
					model=self.model,
					contents=batch,
					config=self.embedding_config
				)
				
				all_vectors.extend(
					[ embedding.values for embedding in self.response.embeddings ]
				)
			
			return all_vectors
		except Exception as e:
			exception = Error( e )
			exception.module = 'embedders'
			exception.cause = 'Gemini'
			exception.method = 'embed( self, texts, task, model, dimensions ) -> List[ List[ float ] ]'
			raise exception

class Grok( ):
	'''

		Purpose:
		-------
		Base configuration class for Groq AI services and shared hyper-parameters.

		Attributes:
		-----------
		api_key           : str - Groq API Key
		prompt            : str - Current request prompt
		model             : str - Current model ID
		response_format   : dict - Schema control

	'''
	content_type: Optional[ str ]
	api_key: Optional[ str ]
	authorization: Optional[ str ]
	prompt: Optional[ str ]
	model: Optional[ str ]
	response_format: Optional[ Union[ str, Dict[ str, str ] ] ]
	contents: Optional[ List[ str ] ]
	http_options: Optional[ Dict[ str, Any ] ]
	embedding: Optional[ float ]
	endpoint: Optional[ str ]
	
	def __init__( self, model: str = 'text-embedding-3-small' ):
		self.api_key = cfg.GROQ_API_KEY
		self.model = model
		self.content_type = 'application/json'
		self.authorization = f'Bearer {self.api_key}'
		self.headers = { 'Content-Type': self.content_type, 'Authorization': self.authorization }
		self.endpoint = f'https://api.openai.com/v1/embeddings'
		self.client = Groq( api_key=self.api_key )
		self.contents = [ ]
		self.payload = Dict[ str, Any ] = None
		self.encoding_format = 'float'
		self.input_text = None
		self.embedding = None
		self.response = None
	
	@property
	def model_options( self ) -> List[ str ]:
		return [ 'nomic-embed-text-v1.5',
		         'text-embedding-3-small',
		         'text-embedding-3-large',
		         'text-embedding-ada-002' ]
	
	@property
	def encoding_options( self ) -> List[ str ]:
		return [ 'float', 'base64' ]
	
	def create( self, text: str, model: str = 'text-embedding-3-small', format: str = 'float' ) -> \
	List[ float ] | None:
		"""Purpose: Generates text embeddings via Hybrid POST."""
		try:
			throw_if( 'text', text )
			self.input_text = text;
			self.model = model;
			self.encoding_format = format
			self.payload = \
				{
						'input': self.input_text,
						'model': self.model,
						'encoding_format': self.encoding_format
				}
			
			_response = requests.post( url=self.endpoint, headers=self.headers, json=self.payload )
			if _response.status_code == 200:
				self.response = _response.json( )
				self.embedding = self.response[ 'data' ][ 0 ][ 'embedding' ]
				return self.embedding
			return None
		except Exception as e:
			exception = Error( e );
			exception.module = 'embeddings'
			exception.cause = 'Grok'
			exception.method = 'create( self, text, model, format )'
			raise exception
	
	def embed( self, texts: List[ str ], model: str = 'text-embedding-3-small' ) -> List[
		List[ float ] ]:
		"""
		
			Purpose:
			--------
			Generate embeddings using xAI Grok.
		
			Parameters:
			-----------
			texts : List[str]
				Text inputs to embed.
			model : str
				Grok embedding model.
		
			Returns:
			--------
			List[List[float]]
				Embedding vectors.
				
		"""
		try:
			throw_if( 'texts', texts )
			
			if not isinstance( texts, list ):
				raise TypeError( 'Argument "texts" must be a list of strings.' )
			
			self.model = model
			self.headers = \
				{
						'Authorization': f'Bearer {self.api_key}',
						'Content-Type': 'application/json',
				}
			
			cleaned: List[ str ] = [ ]
			for text in texts:
				if isinstance( text, str ):
					value = text.strip( )
					if value:
						cleaned.append( value )
			
			if not cleaned:
				return [ ]
			
			max_batch_size: int = 2048
			all_vectors: List[ List[ float ] ] = [ ]
			
			for i in range( 0, len( cleaned ), max_batch_size ):
				batch: List[ str ] = cleaned[ i:i + max_batch_size ]
				self.payload = { 'model': model, 'input': batch }
				
				response = requests.post(
					url=self.endpoint,
					headers=self.headers,
					json=self.payload,
					timeout=30,
				)
				response.raise_for_status( )
				data = response.json( )
				all_vectors.extend( [ item[ 'embedding' ] for item in data[ 'data' ] ] )
			
			return all_vectors
		except Exception as e:
			exception = Error( e )
			exception.module = 'embeddings'
			exception.cause = 'Grok'
			exception.method = 'embed( self, texts, model )'
			raise exception
	
	def count_tokens( self, text: str, coding: str = 'cl100k_base' ) -> Optional[ int ]:
		"""Purpose: Simple word-count estimation for token limits."""
		try:
			throw_if( 'text', text )
			self.input_text = text
			self.encoding_format = coding
			return len( self.input_text.split( ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'embeddings'
			exception.cause = 'Grok'
			exception.method = 'count_tokens( self, text, coding )'
			raise exception;

class Loca( ):
	'''
	
		Purpose:
		--------
		Base class for local GGUF embedding models loaded through llama-cpp-python.
	
		Attributes:
		-----------
		model             : str - Logical model name
		model_path        : str - GGUF file path
		client            : Llama - llama.cpp wrapper
		response          : Any - Raw embedding response
		embedding         : List[ float ] - Single embedding vector
		embeddings        : List[ List[ float ] ] - Batch embedding vectors
	
	'''
	model: Optional[ str ]
	model_path: Optional[ str ]
	client: Optional[ Llama ]
	response: Optional[ Any ]
	embedding: Optional[ List[ float ] ]
	embeddings: Optional[ List[ List[ float ] ] ]
	
	def __init__( self, model: Optional[ str ] = None, model_path: Optional[ str ] = None ):
		super( ).__init__( )
		self.model = model
		self.model_path = model_path
		self.client = None
		self.response = None
		self.embedding = None
		self.embeddings = None
	
	def load( self ) -> None:
		"""
		
			Purpose:
			--------
			Load the configured local GGUF embedding model.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		try:
			throw_if( 'model', self.model )
			throw_if( 'model_path', self.model_path )
			
			if self.client is not None:
				return
			
			if not os.path.exists( self.model_path ):
				raise FileNotFoundError( f'Local GGUF model not found: {self.model_path}' )
			
			self.client = Llama(
				model_path=self.model_path,
				embedding=True,
				verbose=False
			)
		except Exception as e:
			exception = Error( e )
			exception.module = 'embedders'
			exception.cause = 'Loca'
			exception.method = 'load( self ) -> None'
			raise exception
	
	def create( self, text: str ) -> List[ float ]:
		"""
		
			Purpose:
			--------
			Create a single embedding using the configured local GGUF model.
		
			Parameters:
			-----------
			text: str - Input text.
		
			Returns:
			--------
			List[ float ] - Embedding vector.
			
		"""
		try:
			throw_if( 'text', text )
			
			if not isinstance( text, str ):
				raise TypeError( 'Argument "text" must be a string.' )
			
			value: str = text.strip( )
			if not value:
				return [ ]
			
			self.load( )
			self.response = self.client.create_embedding( value )
			
			if isinstance( self.response, dict ) and 'data' in self.response:
				self.embedding = self.response[ 'data' ][ 0 ][ 'embedding' ]
			else:
				self.embedding = self.response
			
			return self.embedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'embedders'
			exception.cause = 'Loca'
			exception.method = 'create( self, text ) -> List[ float ]'
			raise exception
	
	def embed( self, texts: List[ str ] ) -> List[ List[ float ] ]:
		"""
		
			Purpose:
			--------
			Create embeddings for a batch of strings using the configured local GGUF model.
		
			Parameters:
			-----------
			texts: List[str] - Input texts.
		
			Returns:
			--------
			List[List[float]] - Embedding vectors.
			
		"""
		try:
			throw_if( 'texts', texts )
			
			if not isinstance( texts, list ):
				raise TypeError( 'Argument "texts" must be a list of strings.' )
			
			cleaned: List[ str ] = [ ]
			for text in texts:
				if isinstance( text, str ):
					value = text.strip( )
					if value:
						cleaned.append( value )
			
			if not cleaned:
				return [ ]
			
			self.load( )
			self.response = self.client.create_embedding( cleaned )
			
			if isinstance( self.response, dict ) and 'data' in self.response:
				self.embeddings = [ item[ 'embedding' ] for item in self.response[ 'data' ] ]
			else:
				self.embeddings = self.response
			
			return self.embeddings
		except Exception as e:
			exception = Error( e )
			exception.module = 'embedders'
			exception.cause = 'Loca'
			exception.method = 'embed( self, texts ) -> List[ List[ float ] ]'
			raise exception

class Booger( Loca ):
	'''
	
		Purpose:
		--------
		Local BGE small English GGUF embedder.
	
	'''
	
	def __init__( self ):
		super( ).__init__(
			model='boogr-small-en-v1.5',
			model_path=os.path.abspath(
				os.path.join( 'models', 'boogr', 'boogr-small-en-v1.5-q8_0.gguf' )
			)
		)

class Nomnom( Loca ):
	'''
	
		Purpose:
		--------
		Local Nomic GGUF embedder.
	
	'''
	
	def __init__( self ):
		super( ).__init__(
			model='nomnom-embed-text-v1.5',
			model_path=os.path.abspath(
				os.path.join( 'models', 'nomi', 'nomnom-embed-text-v1.5.Q4_K_M.gguf' )
			)
		)

class Bobo( Loca ):
	'''
	
		Purpose:
		--------
		Local Mixedbread GGUF embedder.
	
	'''
	
	def __init__( self ):
		super( ).__init__(
			model='bobo-embed-large-v1',
			model_path=os.path.abspath(
				os.path.join( 'models', 'bobo', 'bobo-embed-large-v1-f16.gguf' )
			)
		)

