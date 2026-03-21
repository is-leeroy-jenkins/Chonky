'''
  ******************************************************************************************
      Assembly:                Chonky
      Filename:                config.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="config.py" company="Terry D. Eppler">

	     config.py
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
    config.py
  </summary>
  ******************************************************************************************
'''
import  os

#------ PATHS ----------------------------------
ICON = r'resources/images/favicon.ico'
LOGO = r'resources/images/chonky.png'
DB = r'stores/sqlite/data.db'

#------ CREDENTIALS ----------------------------------
GOOGLE_APPLICATION_CREDENTIALS = os.getenv( 'GOOGLE_APPLICATION_CREDENTIALS' )

#------ API KEYS ----------------------------------
OPENAI_API_KEY = os.getenv( 'OPENAI_API_KEY' )
GEMINI_API_KEY = os.getenv( 'GEMINI_API_KEY' )
GROQ_API_KEY = os.getenv( 'GROQ_API_KEY' )
GOOGLE_API_KEY = os.getenv( 'GOOGLE_API_KEY' )
GOOGLE_CSE_ID = os.getenv( 'GOOGLE_CSE_ID' )
GOOGLE_CLOUD_LOCATION = os.getenv( 'GOOGLE_CLOUD_LOCATION' )
GOOGLE_CLOUD_PROJECT_ID = os.getenv( 'GOOGLE_CLOUD_PROJECT_ID' )
GOOGLE_APPLICATION_CREDENTIALS = os.getenv( 'GOOGLE_APPLICATION_CREDENTIALS' )
GOOGLEMAPS_API_KEYS = os.getenv( 'GOOGLEMAPS_API_KEYS' )
LANGSMITH_API_KEYS = os.getenv( 'LANGSMITH_API_KEYS' )
PINECONE_API_KEY = os.getenv( 'PINECONE_API_KEY' )
VERTEX_API_KEY = os.getenv( 'VERTEX_API_KEY' )


# ------- CONSTANTS ------------------------------
BLUE_DIVIDER = "<div style='height:2px;align:left;background:#0078FC;margin:6px 0 10px 0;'></div>"

TABS = [ 'Loading', 'Processing', 'Analysis', 'Tokenization',
         'Embeddings', 'Database' ]

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

PROVIDERS = [ 'OpenAI', 'Gemini', 'Groq' ]

GPT_MODELS = [ 'text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002' ]

GEMINI_MODELS = [ 'text-embedding-004', 'text-multilingual-embedding-002' ]

GROK_MODELS = [ 'nomic-embed-text-v1.5', 'text-embedding-3-small',
                'text-embedding-3-large', 'text-embedding-ada-002' ]

SESSION_STATE_DEFAULTS = {
		# -----------------------------
		# Ingestion
		# -----------------------------
		'documents': None,
		'raw_documents': None,
		'active_loader': None,
		# -----------------------------
		# Input
		# -----------------------------
		'raw_text': None,
		'raw_tokens': None,
		'raw_text_view': None,
		# -----------------------------
		# Processing
		# -----------------------------
		'parser': None,
		'processed_text': None,
		'processed_text_view': None,
		# -----------------------------
		# Performance
		# -----------------------------
		'start_time': None,
		'end_time': None,
		'total_time': None,
		# -----------------------------
		# Tokenization / Vocabulary
		# -----------------------------
		'tokens': None,
		'vocabulary': None,
		'token_counts': None,
		'df_synsets': None,
		# -----------------------------
		# SQLite / Excel
		# -----------------------------
		'active_table': None,
		# -----------------------------
		# Chunking
		# -----------------------------
		'lines': None,
		'chunks': None,
		'chunk_modes': None,
		'chunked_documents': None,
		# -----------------------------
		# Embeddings
		# -----------------------------
		'embedder': None,
		'embeddings': None,
		'embedding_provider': None,
		'embedding_model': None,
		'embedding_source': None,
		'embedding_documents': None,
		'df_embedding_input': None,
		'df_embedding_output': None,
		# -----------------------------
		# Retrieval / Search
		# -----------------------------
		'search_results': None,
		# -----------------------------
		# DataFrames
		# -----------------------------
		'df_frequency': None,
		'df_tables': None,
		'df_schema': None,
		'df_preview': None,
		'df_count': None,
		'df_chunks': None,
		# -----------------------------
		# Data
		# -----------------------------
		'data_connection': None,
		# -----------------------------
		# Sidebar / API Keys
		# -----------------------------
		'api_keys': {
				'openai': None,
				'groq': None,
				'google': None,
				'pinecone': None,
				'google_credentials_path': None,
		},
		# -----------------------------
		# XML Loader (explicit contract)
		# -----------------------------
		'xml_loader': None,
		'xml_documents': None,
		'xml_split_documents': None,
		'xml_tree_loaded': None,
		'xml_namespaces': None,
		'xml_xpath_results': None,
		# -----------------------------
		# WordNet Caches
		# -----------------------------
		'wordnet_synsets_sig': None,
		'df_wordnet_synsets': None,
		'df_wordnet_lemmas': None,
}

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