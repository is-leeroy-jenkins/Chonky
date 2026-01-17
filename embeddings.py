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
import config as cfg
from typing import List, Optional, Any, Union, Dict
import tiktoken
from groq import Groq
from openai.types import CreateEmbeddingResponse
from google import genai
from google.genai import types
from google.genai.types import ( EmbedContentConfig, ContentEmbedding, HttpOptions )
from openai import OpenAI
from boogr import ErrorDialog, Error
import requests


def throw_if( name: str, value: object ):
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )


class OpenAi( ):
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
	client: Optional[ OpenAI ]
	prompt: Optional[ str ]
	response_format: Optional[ str ]
	response: Optional[ CreateEmbeddingResponse ]
	embedding: Optional[ List[ float ] ]
	encoding_format: Optional[ str ]
	dimensions: Optional[ int ]
	
	def __init__( self  ):
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
		self.encoding_format = None
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
	
	def create( self, text: str, model: str='text-embedding-3-small', format: str='float' ) -> \
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
			exception.module = 'gpt'
			exception.cause = 'Embedding'
			exception.method = 'create( self, text: str, model: str ) -> List[ float ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def count_tokens( self, text: str, coding: str ) -> int:
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
			error = ErrorDialog( exception )
			error.show( )
	
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
	prompt: Optional[ str ]
	model: Optional[ str ]
	response_format: Optional[ str ]
	client: Optional[ genai.Client ]
	response: Optional[ Any ]
	embedding: Optional[ List[ float ] ]
	encoding_format: Optional[ str ]
	dimensions: Optional[ int ]
	use_vertex: Optional[ bool ]
	task_type: Optional[ str ]
	http_options: Optional[ HttpOptions ]
	embedding_config: Optional[ types.EmbedContentConfig ]
	contents: Optional[ List[ str ] ]
	input_text: Optional[ str ]
	file_path: Optional[ str ]
	response_modalities: Optional[ str ]
	
	def __init__( self, model: str='text-embedding-004', version: str='v1alpha', use_ai: bool=False ):
		super( ).__init__( )
		self.model = model
		self.api_version = version
		self.use_vertex = use_ai
		self.http_options = HttpOptions( api_version=self.api_version )
		self.client = genai.Client( vertexai=self.use_vertex, api_key=self.api_key,
			http_options=self.http_options )
		self.embedding = None;
		self.response = None;
		self.encoding_format = None
		self.input_text = None;
		self.file_path = None;
		self.dimensions = None
		self.task_type = None;
		self.response_modalities = None
		self.embedding_config = None;
		self.content_config = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Returns list of embedding models."""
		return [ 'text-embedding-004',
		         'text-multilingual-embedding-002' ]
	
	def generate( self, text: str, model: str='text-embedding-004' ) -> Optional[ List[ float ] ]:
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
			self.input_text = text;
			self.model = model
			self.embedding_config = EmbedContentConfig( task_type=self.task_type )
			self.response = self.client.models.embed_content( model=self.model,
				contents=self.input_text, config=self.embedding_config )
			self.embedding = self.response.embeddings[ 0 ].values
			return self.embedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'embeddings'
			exception.cause = 'Gemini'
			exception.method = 'generate( self, text, model ) -> List[ float ]'
			error = ErrorDialog( exception )
			error.show( )

class Grok:
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
	endpoint: Optional[ str ]
	
	def __init__( self, model: str='text-embedding-3-small'):
		self.api_key = cfg.GROQ_API_KEY
		self.model = model
		self.content_type = 'application/json'
		self.authorization = f'Bearer {self.api_key}'
		self.headers = { 'Content-Type': self.content_type, 'Authorization': self.authorization }
		self.endpoint = f'https://api.openai.com/v1/embeddings'
		self.client = Groq( api_key=self.api_key )
		self.contents = [ ]
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
	
	def create( self, text: str, model: str='text-embedding-3-small', format: str='float' ) -> List[ float ] | None:
		"""Purpose: Generates text embeddings via Hybrid POST."""
		try:
			throw_if( 'text', text )
			self.input_text = text;
			self.model = model;
			self.encoding_format = format
			payload = \
			{
				'input': self.input_text,
				'model': self.model,
				'encoding_format': self.encoding_format
			}
			
			resp = requests.post( url=self.embeddings, headers=self.headers, json=payload )
			if resp.status_code == 200:
				self.response = resp.json( )
				self.embedding = self.response[ 'data' ][ 0 ][ 'embedding' ]
				return self.embedding
			return None
		except Exception as e:
			exception = Error( e );
			exception.module = 'groq'
			exception.cause = 'Embedding'
			exception.method = 'create( self, text, model, format )'
			error = ErrorDialog( exception )
			error.show( )
	
	def count_tokens( self, text: str, coding: str='cl100k_base' ) -> Optional[ int ]:
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
			error = ErrorDialog( exception );
			error.show( )

