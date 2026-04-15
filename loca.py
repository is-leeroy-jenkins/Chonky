'''
  ******************************************************************************************
      Assembly:                Chonky
      Filename:                loca.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="loca.py" company="Terry D. Eppler">

	     loca.py
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
    loca.py
  </summary>
  ******************************************************************************************
'''
import os
from typing import List, Optional, Any
from llama_cpp import Llama
from boogr import Error

def throw_if( name: str, value: object ):
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

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
		super( ).__init__( model='boogr-small-en-v1.5', model_path=os.path.abspath(
			os.path.join( 'models', 'boogr', 'boogr-small-en-v1.5-q8_0.gguf' ) ) )

class Nomnom( Loca ):
	'''
	
		Purpose:
		--------
		Local Nomic GGUF embedder.
	
	'''
	
	def __init__( self ):
		super( ).__init__( model='nomnom-embed-text-v1.5',
			model_path=os.path.abspath(
				os.path.join( 'models', 'nomi', 'nomnom-embed-text-v1.5.Q4_K_M.gguf' ) ) )

class Bobo( Loca ):
	'''
	
		Purpose:
		--------
		Local Mixedbread GGUF embedder.
	
	'''
	
	def __init__( self ):
		super( ).__init__( model='bobo-embed-large-v1', model_path=os.path.abspath(
			os.path.join( 'models', 'bobo', 'bobo-embed-large-v1-f16.gguf' ) ) )
