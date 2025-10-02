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
from boogr import Error, ErrorDialog
from enums import Source, Provider, SQL, ParamStyle
import json
import sqlite3 as sqlite
from sqlite3 import Cursor, Row
import numpy as np
from pinecone import Pinecone
from pandas import DataFrame
from pandas import read_sql as sqlreader
import pyodbc as db
import os
from typing import Optional, List, Tuple


def throw_if( name: str, value: object ):
	if value is None:
		raise Exception( f'Argument "{name}" cannot be empty!' )


class Casing( ):
	'''

		Purpose:
		---------
		Class splits string 'input' argument into Pascal Casing

	'''

	def __init__( self, input: str=None ):
		self.input = input
		self.output = input if input.istitle( ) else self.join( )

	def __str__( self ) -> str:
		if self.output is not None:
			return self.output

	def __dir__( self ) -> list[ str ]:
		'''

			Purpose:
			--------
			Retunes a list[ str ] of member names.

		'''
		return [ 'input', 'split', 'join' ]

	def pascalize( self ) -> str:
		'''

			Purpose:
			--------

			Parameters:
			-----------

			Returns:
			--------

		'''

		try:
			_buffer = [ str( c ) for c in self.output ]
			_output = [ ]
			_retval = ''
			for char in _buffer:
				if char.islower( ):
					_output.append( char )
				elif char.isupper( ) and _buffer.index( char ) == 0:
					_output.append( char )
				elif char.isupper( ) and _buffer.index( char ) > 0:
					_output.append( ' ' )
					_output.append( char )
			for o in _output:
				_retval += f'{o}'
			return _retval
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'data'
			_exc.cause = 'Casing'
			_exc.method = 'pascalize( self )'
			_err = ErrorDialog( _exc )
			_err.show( )

	def unpascalize( self ) -> str:
		'''

			Purpose:

			Parameters:

			Returns:

		'''
		try:
			if self.input.count( ' ' ) > 0:
				_buffer = [ str( c ) for c in self.input ]
				_output = [ ]
				_retval = ''
				for char in _buffer:
					if char != ' ':
						_output.append( char )
					elif char == ' ':
						_index = _buffer.index( char )
						_next = str( _buffer[ _index + 1 ] )
						if _next.islower( ):
							_cap = _next.upper( )
							_buffer.remove( _next )
							_buffer.insert( _index + 1, _cap )
						_buffer.remove( char )
						_output.append( _next.upper( ) )

			for o in _output:
				_retval += f'{o}'

			return _retval
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'data'
			_exc.cause = 'Casing'
			_exc.method = 'unpascalize( self )'
			_err = ErrorDialog( _exc )
			_err.show( )

class SqlPath( ):
	'''

	Constructor:
	SqlPath( )

	Purpose:
	Class providing relative_path paths to the
	folders containing sqlstatement files and driverinfo
	paths used in the application

	'''

	def __init__( self ):
		self.sqlite_driver = 'sqlite3'
		self.sqlite_path = r'../data/sqlite/datamodels/sql'
		self.access_driver = r'DRIVER={Microsoft ACCDB Driver (*.mdb, *.accdb)};DBQ='
		self.access_path = r'../data/access/datamodels/sql'
		self.sqlserver_driver = r'DRIVER={ODBC Driver 17 for SQL Server};SERVER=.\SQLExpress;'
		self.sqlserver_database = r'data\mssql\datamodels\sql'

	def __dir__( self ) -> list[ str ]:
		'''
		Retunes a list[ str ] of member names.
		'''
		return [ 'sqlite_driver', 'sqlite_database',
		         'access_driver', 'access_database',
		         'sqlserver_driver', 'sqlserver_database' ]

class SqlFile( ):
	'''

		Purpose:
		-------

			Class providing access to sqlstatement sub-folders in the application provided
			optional arguments source, provider, and command.

	'''

	def __init__( self, source: Source=None, provider: Provider=Provider.SQLite,
	              commandtype: SQL = SQL.SELECTALL ):
		self.command_type = commandtype
		self.source = source
		self.provider = provider
		self.data = [ 'Apportionments',
		              'Files',
		              'Partitions',
		              'AgencyAccounts',
		              'Appropriations',
		              'FiscalYears',
		              'OMB Circular A-11 Preparation Submission And Execution Of The Budget',
		              'OMB Circular A-11 Section 120 Apportionment Process',
		              'OMB Circular A-11 SF-132',
		              'Principles Of Federal Appropriations Law',
		              'Title 31 Code Of Federal Regulations',
		              'Prompts',
		              'Search',
		              'Locations' ]

	def __dir__( self ) -> list[ str ]:
		'''
		Retunes a list[ str ] of member names.
		'''
		return [ 'source', 'provider', 'command_type', 'get_file_path',
		         'get_folder_path', 'get_command_text' ]

	def get_file_path( self ) -> str:
		'''

			Purpose:
			-------


			Parameters:
			----------


			Returns:
			--------

		'''

		try:
			_sqlpath = SqlPath( )
			_data = self.data
			_provider = self.provider.name
			_tablename = self.source.name
			_command = self.command_type.name
			_current = os.getcwd( )
			_filepath = ''
			if _provider == 'SQLite' and _tablename in _data:
				_filepath = f'{_sqlpath.sqlite_database}\\{_command}\\{_tablename}.sql'
				return os.path.join( _current, _filepath )
			elif _provider == 'ACCDB' and _tablename in _data:
				_filepath = f'{_sqlpath.access_database}\\{_command}\\{_tablename}.sql'
				return os.path.join( _current, _filepath )
			elif _provider == 'SqlServer' and _tablename in _data:
				_filepath = f'{_sqlpath.sqlserver_database}\\{_command}\\{_tablename}.sql'
				return os.path.join( _current, _filepath )
			else:
				_filepath = f'{_sqlpath.sqlite_database}\\{_command}\\{_tablename}.sql'
				return os.path.join( _current, _filepath )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Data'
			_exc.cause = 'SqlFile'
			_exc.method = 'get_file_path( self )'
			_err = ErrorDialog( _exc )
			_err.show( )

	def get_folder_path( self ) -> str:
		'''

			Purpose:
			-------


			Parameters:
			----------


			Returns:
			--------

		'''

		try:
			_sqlpath = SqlPath( )
			_data = self.data
			_source = self.source.name
			_provider = self.provider.name
			_command = self.command_type.name
			_current = os.getcwd( )
			_folder = ''
			if _provider == 'SQLite' and _source in _data:
				_folder = f'{_sqlpath.sqlite_database}\\{_command}'
				return os.path.join( _current, _folder )
			elif _provider == 'ACCDB' and _source in _data:
				_folder = f'{_sqlpath.access_database}\\{_command}'
				return os.path.join( _current, _folder )
			elif _provider == 'SqlServer' and _source in _data:
				_folder = f'{_sqlpath.sqlserver_database}\\{_command}'
				return os.path.join( _current, _folder )
			else:
				_folder = f'{_sqlpath.sqlite_database}\\{_command}'
				return os.path.join( _current, _folder )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Data'
			_exc.cause = 'SqlFile'
			_exc.method = 'get_folder_path( self )'
			_err = ErrorDialog( _exc )
			_err.show( )

	def get_command_text( self ) -> str:
		'''
		Purpose:

		Parameters:

		Returns:
		'''

		try:
			_source = self.source.name
			_paths = self.get_file_path( )
			_folder = self.get_folder_path( )
			_sql = ''
			for name in os.listdir( _folder ):
				if name.endswith( '.sql' ) and os.path.splitext( name )[ 0 ] == _source:
					_path = os.path.join( _folder, name )
					_query = open( _path )
					_sql = _query.read( )
				if _sql is None:
					_msg = 'INVALID INPUT!'
					raise ValueError( _msg )
				else:
					return _sql
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Data'
			_exc.cause = 'SqlFile'
			_exc.method = 'get_command_text( self, other )'
			_err = ErrorDialog( _exc )
			_err.show( )

class DbConfig( ):
	'''

		Purpose:
		--------
			Class provides list of Budget Execution tables across two databases

	'''

	def __init__( self, src: Source, pro: Provider=Provider.SQLite ):
		self.provider = pro
		self.source = src
		self.table_name = src.name
		self.sqlite_path = os.getcwd( ) + r'\data\sqlite\datamodels\Data.data'
		self.access_driver = r'DRIVER={ Microsoft Access Driver (*.mdb, *.accdb) };DBQ='
		self.access_path = os.getcwd( ) + r'\db\access\datamodels\sql\Data.accdb'
		self.sqlserver_driver = r'DRIVER={ ODBC Driver 17 for SQL Server };SERVER=.\SQLExpress;'
		self.sqlserver_path = os.getcwd( ) + r'\db\mssql\datamodels\Data.mdf'
		self.data = [ 'Apportionments',
		              'Files',
		              'Partitions',
		              'AgencyAccounts',
		              'Appropriations',
		              'FiscalYears',
		              'OMB Circular A-11 Preparation Submission And Execution Of The Budget',
		              'OMB Circular A-11 Section 120 Apportionment Process',
		              'OMB Circular A-11 SF-132',
		              'Principles Of Federal Appropriations Law',
		              'Title 31 Code Of Federal Regulations',
		              'Prompts',
		              'Search',
		              'Locations' ]

	def __str__( self ) -> str:
		if self.table_name is not None:
			return self.table_name

	def __dir__( self ) -> list[ str ]:
		'''
		Retunes a list[ str ] of member names.
		'''
		return [ 'source', 'provider', 'table_name', 'get_driver_info',
		         'sqlite_path', 'access_driver', 'access_path',
		         'sqlserver_driver', 'sqlserver_path',
		         'get_data_path', 'get_connection_string' ]

	def get_driver_info( self ) -> str:
		'''

		'''
		try:
			if self.provider.name == 'SQLite':
				return self.sqlite_path
			elif self.provider.name == 'SqlServer':
				return self.sqlserver_driver
			else:
				return self.sqlite_driver
		except Exception as e:
			_exc = Error( e )
			_exc.cause = 'DbConfig Class'
			_exc.method = 'getdriver_info( self )'
			_error = ErrorDialog( _exc )
			_error.show( )

	def get_data_path( self ) -> str:
		'''
	
			Purpose:
			--------
	
			Parameters:
			----------
	
			Returns:
			--------

		'''

		try:
			if self.provider.name == 'SQLite':
				return self.sqlite_path
			elif self.provider.name == 'Access':
				return self.access_path
			elif self.provider.name == 'SqlServer':
				return self.sqlserver_path
			else:
				return self.sqlite_path
		except Exception as e:
			_exc = Error( e )
			_exc.cause = 'DbConfig Class'
			_exc.method = 'get_data_path( self )'
			_error = ErrorDialog( _exc )
			_error.show( )

	def get_connection_string( self ) -> str:
		'''

			Purpose:
			--------
	
			Parameters:
			----------
	
			Returns:
			--------

		'''

		try:
			_path = self.get_data_path( )
			if self.provider.name == Provider.Access.name:
				return self.get_driver_info( ) + _path
			elif self.provider.name == Provider.SqlServer.name:
				return r'DRIVER={ ODBC Driver 17 for SQL Server };Server=.\SQLExpress;' \
					+ f'AttachDBFileName={_path}' \
					+ f'DATABASE={_path}Trusted_Connection=yes;'
			else:
				return f'{_path} '
		except Exception as e:
			_exc = Error( e )
			_exc.cause = 'DbConfig Class'
			_exc.method = 'get_connection_string( self )'
			_error = ErrorDialog( _exc )
			_error.show( )

class Connection( DbConfig ):
	'''

		Purpose:
		--------
			Class providing object used to connect to the databases

	'''

	def __init__( self, src: Source, pro: Provider=Provider.SQLite ):
		super( ).__init__( src, pro )
		self.source = super( ).source
		self.provider = super( ).provider
		self.data_path = super( ).get_data_path( )
		self.driver = super( ).get_driver_info( )
		self.dsn = super( ).table_name + ';'
		self.connection_string = super( ).get_connection_string( )

	def __dir__( self ) -> list[ str ]:
		'''
		
			Purpose:
			---------
			Retunes a list[ str ] of member names.
			
		'''
		return [ 'source', 'provider', 'table_name', 'getdriver_info',
		         'get_data_path', 'get_connection_string',
		         'driver_info', 'data_path',
		         'connection_string', 'connect' ]

	def connect( self ):
		'''
		
			Purpose:
			--------
				Establishes a data connections using the connecdtion
				string.
	
			Parameters:
			----------
				self
	
			Returns:
			--------
				None
				
		'''

		try:
			if self.provider.name == Provider.Access.name:
				return db.connect( self.connection_string )
			elif self.provider.name == Provider.SqlServer.name:
				return db.connect( self.connection_string )
			else:
				return sqlite.connect( self.connection_string )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Data'
			_exc.cause = 'Connection'
			_exc.method = 'connect( self )'
			_err = ErrorDialog( _exc )
			_err.show( )

class SqlConfig( ):
	'''
		
		
		Purpose:
		--------
		
	'''
	def __init__( self, cmd: SQL=SQL.SELECTALL, names: list[ str ]=None,
	              values: tuple=None, paramstyle: ParamStyle=None ):
		self.command_type = cmd
		self.column_names = names
		self.column_values = values
		self.parameter_style = paramstyle
		self.criteria = dict( zip( names, list( values ) ) ) \
			if names is not None and values is not None else None

	def __dir__( self ) -> list[ str ]:
		'''
			
			
			Purpose:
			--------
	
			Parameters:
			---------
	
			Returns:
			--------
			
		'''
		return [ 'command_type', 'column_names', 'column_values',
		         'parameter_style', 'pair_dump', 'where_dump',
		         'set_dump', 'column_dump', 'value_dump' ]

	def pair_dump( self ) -> str:
		'''
			
			
			Purpose:
			--------
	
			Parameters:
			---------
	
			Returns:
			--------
			
		'''
		try:
			_criteria = None
			if self.column_names is not None and self.column_values is not None:
				_pairs = ''
				_kvp = zip( self.column_names, self.column_values )
				for k, v in _kvp:
					_pairs += f'{k} = \'{v}\' AND '
				_criteria = _pairs.rstrip( ' AND ' )
				if _criteria is None:
					_msg = 'INVALID INPUT!'
					raise ValueError( _msg )
			else:
				return _criteria
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Data'
			_exc.cause = 'SqlConfig'
			_exc.method = 'pair_dump( self )'
			_err = ErrorDialog( _exc )
			_err.show( )

	def where_dump( self ) -> str:
		'''
			
			
			Purpose:
			--------
	
			Parameters:
			---------
	
			Returns:
			--------
			
		'''
		try:
			_criteria = None
			if (isinstance( self.column_names, list ) and
					isinstance( self.column_values, tuple )):
				pairs = ''
				for k, v in zip( self.column_names, self.column_values ):
					pairs += f'{k} = \'{v}\' AND '
				_criteria = 'WHERE ' + pairs.rstrip( ' AND ' )
				if _criteria is None:
					_msg = 'INVALID INPUT!'
					raise ValueError( _msg )
			else:
				return _criteria
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Data'
			_exc.cause = 'SqlConfig'
			_exc.method = 'where_dump( self )'
			_err = ErrorDialog( _exc )
			_err.show( )

	def set_dump( self ) -> str:
		'''
			
			
			Purpose:
			--------
	
			Parameters:
			---------
	
			Returns:
			--------
			
		'''
		try:
			_criteria = None
			if self.column_names is not None and self.column_values is not None:
				_pairs = ''
				_criteria = ''
				for k, v in zip( self.column_names, self.column_values ):
					_pairs += f'{k} = \'{v}\', '
				_criteria = 'SET ' + _pairs.rstrip( ', ' )
				if _criteria is None:
					_msg = 'INVALID INPUT!'
					raise ValueError( _msg )
			else:
				return _criteria
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Data'
			_exc.cause = 'SqlConfig'
			_exc.method = 'set_dump( self )'
			_err = ErrorDialog( _exc )
			_err.show( )

	def column_dump( self ) -> str:
		'''
			
			
			Purpose:
			--------
	
			Parameters:
			---------
	
			Returns:
			--------
			
		'''
		try:
			_columns = None
			if self.column_names is not None:
				_colnames = ''
				for n in self.column_names:
					_colnames += f'{n}, '
				_columns = '(' + _colnames.rstrip( ', ' ) + ')'
				if _columns is None:
					_msg = 'INVALID INPUT!'
					raise ValueError( _msg )
			else:
				return _columns
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Data'
			_exc.cause = 'SqlConfig'
			_exc.method = 'column_dump( self )'
			_err = ErrorDialog( _exc )
			_err.show( )

	def value_dump( self ) -> str:
		'''
			
			
			Purpose:
			--------
	
	
			Returns:
			--------
			
		'''
		try:
			_values = None
			if self.column_values is not None:
				_vals = ''
				for v in self.column_values:
					_vals += f'{v}, '
					_values = 'VALUES (' + _vals.rstrip( ', ' ) + ')'
					if _values is None:
						_msg = 'INVALID INPUT!'
						raise ValueError( _msg )
			else:
				return _values
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Data'
			_exc.cause = 'SqlConfig'
			_exc.method = 'value_dump( self )'
			_err = ErrorDialog( _exc )
			_err.show( )

class SqlStatement( ):
	'''
	
		Purpose:
		-------
	
			Class represents the values models used in the SQLite database
			

	'''

	def __init__( self, dbcfg: DbConfig, sqcfg: SqlConfig ):
		self.command_type = sqcfg.command_type
		self.provider = dbcfg.provider
		self.source = dbcfg.source
		self.table_name = dbcfg.table_name
		self.column_names = sqcfg.column_dump( )
		self.column_values = sqcfg.value_dump( )
		self.updates = sqcfg.set_dump( )
		self.criteria = dict( zip( self.column_names, list( self.column_values ) ) ) \
			if self.column_names is not None and self.column_values is not None else None
		self.command_text = self.__getquerytext( )

	def __str__( self ) -> str:
		if self.command_text is not None:
			return self.command_text

	def __dir__( self ) -> list[ str ]:
		'''
			
			Purpose:
			--------
			Returns a list[ str ] of member names.

		'''
		return [ 'provider', 'table_name',
		         'command_type', 'column_names', 'values',
		         'updates', 'command_text' ]

	def __getquerytext( self ) -> str:
		'''
			
			Purpose:
			--------
	
			Parameters:
			---------
	
			Returns:
			--------
			
		'''
		try:
			_table = self.table_name
			_cols = self.column_names
			_vals = self.column_values
			_where = self.criteria
			_cmd = self.command_type
			_updates = self.updates
			if _cmd == SQL.SELECTALL and _cols is None and _vals is None and _where is None:
				return f'SELECT * FROM {_table}'
			elif _cmd == SQL.SELECT and _cols is not None and _vals is None and _where is not None:
				return f'SELECT ' + _cols + f'FROM {_table}' + f' {_where}'
			elif _cmd == SQL.INSERT and len( _where.items( ) ) > 0:
				return f'SELECT ' + _cols + f' FROM {_table}' + f' {_where}'
			elif _cmd == SQL.INSERT and _cols is not None and _vals is not None:
				return f'INSERT INTO {_table} ' + f'{_cols} ' + f'{_vals}'
			elif _cmd == SQL.UPDATE and _cols is not None and _vals is None and _where is not None:
				_set = self.updates
				return f'UPDATE {_table} ' + f'{_set}' + f'{_vals}' + f'{_where}'
			elif _cmd == SQL.DELETE and _cols is None and _vals is None and _where is not None:
				return f'DELETE FROM {_table} ' + f'{_where}'
			elif _cmd == SQL.SELECT and _cols is not None and _vals is None and _where is None:
				cols = _cols.lstrip( '(' ).rstrip( ')' )
				return f'SELECT {cols} FROM {_table}'
			elif _cmd == SQL.SELECTALL and _cols is None and _vals is None and _where is not None:
				return f'SELECT * FROM {_table}' + f'{_where}'
			elif _cmd == SQL.DELETE and _cols is None and _vals is None and _where is not None:
				return f'DELETE FROM {_table}' + f'{_where}'
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Data'
			_exc.cause = 'SqlStatement'
			_exc.method = '__getquerytext( self )'
			_err = ErrorDialog( _exc )
			_err.show( )

class Query( ):
	'''

	
		Purpose:
		--------
	
		Base class for database interaction
		

	'''

	def __init__( self, conn: Connection, sql: SqlStatement ):
		self.connection = conn
		self.sql_statement = sql
		self.sql_config = SqlConfig( conn.source, conn.provider )
		self.source = conn.source
		self.table_name = self.source.name
		self.provider = conn.provider
		self.command_type = sql.command_type
		self.data_path = conn.data_path
		self.connection_string = conn.connection_string
		self.column_names = self.sql_config.column_names
		self.column_values = tuple( self.sql_config.criteria.values( ) ) \
			if self.column_values is not None else None
		self.command_text = sql.command_text

	def __str__( self ) -> str:
		if self.command_text is not None:
			return self.command_text

	def __dir__( self ) -> list[ str ]:
		return [ 'source', 'provider', 'data_path', 'connection', 'sql_statement',
		         'command_type', 'table_name', 'column_names', 'values',
		         'command_text', 'connection_string' ]

class SQLiteData( Query ):
	'''

		Purpose:
		--------
	
			Class represents the SQLite data factory

	'''

	def __init__( self, conn: Connection, sql: SqlStatement ):
		super( ).__init__( conn, sql )
		self.provider = Provider.SQLite
		self.connection = super( ).connection
		self.sql_statement = super( ).sqlstatement
		self.source = super( ).source
		self.table_name = super( ).source.name
		self.command_type = super( ).command_type
		self.data_path = super( ).data_path
		self.driver_info = super( ).connection.driver_info
		self.connection_string = super( ).connection_string
		self.column_names = super( ).column_names
		self.column_values = super( ).column_values
		self.command_text = super( ).command_text

	def __str__( self ) -> str:
		if self.__query is not None:
			return self.__query

	def __dir__( self ) -> list[ str ]:
		'''

			Purpose:
			--------
			Returns a list[ str ] of member names

		'''
		return [ 'source', 'provider', 'data_path', 'connection', 'sql_statement',
		         'command_type', 'table_name', 'column_names', 'column_values', 'driver_info',
		         'command_text', 'connection_string', 'create_table', 'create_frame' ]

	def create_frame( self ) -> DataFrame:
		'''
	
			Purpose:
			--------
	
			Parameters:
			---------
	
			Returns:
			--------

		'''

		try:
			_path = self.data_path
			_source = self.source
			_table = self.source.name
			_connection = sqlite.connect( _path )
			_sql = f'SELECT * FROM {_table};'
			_frame = sqlreader( _sql, _connection )
			if _frame is None:
				_msg = "INVALID INPUT!"
				raise ValueError( _msg )
			else:
				return _frame
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Data'
			_exc.cause = 'SQLiteData'
			_exc.method = 'create_frame( self )'
			_err = ErrorDialog( _exc )
			_err.show( )

	def create_tuples( self ) -> list[ tuple ]:
		'''

			Purpose:
	
			Parameters:
	
			Returns:

		'''

		try:
			_path = self.data_path
			_source = self.source
			_table = self.source.name
			_connection = sqlite.connect( _path )
			_sql = f'SELECT * FROM {_table};'
			_frame = sqlreader( _sql, _connection )
			_data = [ tuple( i ) for i in _frame.iterrows( ) ]
			if _data is None:
				_msg = "INVALID INPUT!"
				raise ValueError( _msg )
			else:
				return _data
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Data'
			_exc.cause = 'SQLiteData'
			_exc.method = 'create_tuples( self )'
			_err = ErrorDialog( _exc )
			_err.show( )

class AccessData( Query ):
	'''
	
		Purpose:
		--------
		Class represents the main execution
		values model classes in the MS ACCDB database

	'''

	def __init__( self, conn: Connection, sql: SqlStatement ):
		super( ).__init__( conn, sql )
		self.source = super( ).source
		self.provider = Provider.Access
		self.connection = super( ).connection
		self.sql_statement = super( ).sqlstatement
		self.command_text = super( ).command_text
		self.driver_info = r'DRIVER={ Microsoft ACCDB Driver( *.mdb, *.accdb ) };'
		self.data = [ ]

	def __str__( self ) -> str:
		if self.command_text is not None:
			return self.command_text

	def __dir__( self ) -> list[ str ]:
		'''

			Purpose:
			--------
			Returns a list[ str ] of member names

		'''
		return [ 'source', 'provider', 'data_path', 'connection', 'sql_statement',
		         'command_type', 'table_name', 'column_names', 'column_values',
		         'command_text', 'connection_string',
		         'create_table', 'create_frame' ]

	def create_frame( self ) -> DataFrame:
		'''
	
	
			Purpose:
			--------
	
			Parameters:
			----------
	
			Returns:
			--------

		'''
		try:
			_path = self.data_path
			_source = self.source
			_table = self.source.name
			_connection = sqlite.connect( _path )
			_sql = f'SELECT * FROM {_table};'
			_frame = sqlreader( _sql, _connection )
			if _frame is None:
				_msg = "INVALID INPUT!"
				raise ValueError( _msg )
			else:
				return _frame
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Data'
			_exc.cause = 'AccessData'
			_exc.method = 'create_frame( self )'
			_err = ErrorDialog( _exc )
			_err.show( )

	def create_tuples( self ) -> list[ tuple ]:
		'''


			Purpose:
			--------
	
			Parameters:
			----------
	
			Returns:
			-------

		'''
		try:
			_path = self.data_path
			_source = self.source
			_table = self.source.name
			_connection = sqlite.connect( _path )
			_sql = f'SELECT * FROM {_table};'
			_frame = sqlreader( _sql, _connection )
			_data = [ tuple( i ) for i in _frame.iterrows( ) ]
			if _data is None:
				_msg = "INVALID INPUT!"
				raise ValueError( _msg )
			else:
				return _data
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Data'
			_exc.cause = 'AccessData'
			_exc.method = 'create_tuples( self )'
			_err = ErrorDialog( _exc )
			_err.show( )

class SqlServerData( Query ):
	'''
	
		 
		 Purpose:
		 --------
		 Class providing object represents the value models in the MS SQL Server database

	 '''
	def __init__( self, conn: Connection, sql: SqlStatement ):
		super( ).__init__( conn, sql )
		self.provider = Provider.SqlServer
		self.connection = super( ).connection
		self.source = super( ).source
		self.command_text = super( ).command_text
		self.table_name = super( ).table_name
		self.sqlserver_path = r'(LocalDB)\MSSQLLocalDB;'
		self.driver_info = r'{ SQL Server Native Client 11.0 };'

	def __str__( self ) -> str:
		if self.source is not None:
			return self.source.name

	def __dir__( self ) -> list[ str ]:
		'''

			Purpose:
			--------
			Returns a list[ str ] of member names

		'''
		return [ 'source', 'provider', 'sqlserver_path', 'connection',
		         'table_name', 'driver_info',
		         'command_text', 'create_table', 'create_frame' ]

	def create_frame( self ) -> DataFrame:
		'''

				Purpose:
				---------
		
				Parameters:
		
				Returns:

		'''
		try:
			_path = self.data_path
			_source = self.source
			_table = self.source.name
			_connection = sqlite.connect( _path )
			_sql = f'SELECT * FROM {_table};'
			_frame = sqlreader( _sql, _connection )
			if _frame is None:
				_msg = "INVALID INPUT!"
				raise ValueError( _msg )
			else:
				return _frame
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Data'
			_exc.cause = 'SqlServerData'
			_exc.method = 'create_frame( self )'
			_err = ErrorDialog( _exc )
			_err.show( )

	def create_tuples( self ) -> list[ tuple ]:
		'''
	
			Purpose:
	
			Parameters:
	
			Returns:

		'''
		try:
			_path = self.data_path
			_source = self.source
			_table = self.source.name
			_connection = sqlite.connect( _path )
			_sql = f'SELECT * FROM {_table};'
			_frame = sqlreader( _sql, _connection )
			_data = [ tuple( i ) for i in _frame.iterrows( ) ]
			if _data is None:
				_msg = "INVALID INPUT!"
				raise ValueError( _msg )
			else:
				return _data
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Data'
			_exc.cause = 'SqlServerData'
			_exc.method = 'create_tuples( self )'
			_err = ErrorDialog( _exc )
			_err.show( )

class BudgetData( ):
	'''


		Purpose:
		--------
		Class containing factory method for providing
		pandas dataframes.

	'''
	def __init__( self, src: Source ):
		self.source = src
		self.table_name = src.name
		self.data_path = DbConfig( src ).get_data_path( )
		self.command_text = f'SELECT * FROM {src.name};'

	def __dir__( self ) -> list[ str ]:
		'''
		
			Returns a list[ str ] of member names
			
		'''
		return [ 'source', 'data_path', 'table_name',
		         'command_text', 'create_frame', 'create_tuples' ]

	def create_frame( self ) -> DataFrame:
		'''
		
	
			Purpose:
			--------
	
			Parameters:
			---------
	
			Returns:
			--------

		'''
		try:
			_path = self.data_path
			_source = self.source
			_table = self.source.name
			_connection = sqlite.connect( _path )
			_sql = f'SELECT * FROM {_table};'
			_frame = sqlreader( _sql, _connection )
			if _frame is None:
				_msg = "INVALID INPUT!"
				raise ValueError( _msg )
			else:
				return _frame
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Data'
			_exc.cause = 'BudgetData'
			_exc.method = 'create_frame( self )'
			_err = ErrorDialog( _exc )
			_err.show( )

	def create_tuples( self ) -> list[ tuple ]:
		'''

		Purpose:

		Parameters:

		Returns:

		'''

		try:
			_path = self.data_path
			_source = self.source
			_table = self.source.name
			_connection = sqlite.connect( _path )
			_sql = f'SELECT * FROM {_table};'
			_frame = sqlreader( _sql, _connection )
			_data = [ tuple( i ) for i in _frame.iterrows( ) ]
			if _data is None:
				_msg = "INVALID INPUT!"
				raise ValueError( _msg )
			else:
				return _data
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Data'
			_exc.cause = 'BudgetData'
			_exc.method = 'create_tuples( self )'
			_err = ErrorDialog( _exc )
			_err.show( )

class DataBuilder( BudgetData ):
	'''
	Constructor:

		DataBuilder( source: Source, provider = Provider.SQLite,
					  commandtype = SQL.SELECTALL, names: list[ str ]=None,
					  values: tuple=None ).

	Purpose:

		Class provides functionality to access application data.

	'''

	# Fields
	source: Source=None
	table_name: str=None
	data_path: str=None
	command_text: str=None

	def __init__( self, src: Source ):
		super( ).__init__( src )
		self.source = super( ).source
		self.table_name = super( ).table_name
		self.data_path = super( ).data_path
		self.command_text = super( ).command_text

	def create_tuples( self ) -> list[ tuple ]:
		try:
			_data = super( ).create_tuples( )
			if _data is None:
				_msg = 'INVALID INPUT!'
				raise ValueError( _msg )
			else:
				return _data
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Data'
			_exc.cause = 'DataBuilder'
			_exc.method = 'create_tuples( self )'
			_error = ErrorDialog( _exc )
			_error.show( )

	def create_frame( self ) -> DataFrame:
		try:
			_frame = super( ).create_frame( )
			if _frame is None:
				_msg = 'INVALID INPUT!'
				raise ValueError( _msg )
			else:
				return _frame
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Data'
			_exc.cause = 'DataBuilder'
			_exc.method = 'create_frame( self )'
			_error = ErrorDialog( _exc )
			_error.show( )

class DataColumn( ):
	'''

	Constructor:

		DataColumn( name: str = '', dtype: type = None, value: object = None )

	Purpose:

		Defines the class providing schema information.

	 '''

	# Fields
	name: str=None
	label: str=None
	caption: str=None
	type: type = None
	value: object = None

	def __init__( self, name: str = '', dtype: type = None, value: object = None ):
		self.name = name
		self.label = name
		self.caption = name
		self.type = dtype
		self.value = value

	def __str__( self ) -> str:
		if self.name is not None:
			return self.name

	def is_numeric( self ) -> bool:
		try:
			if self.value is not None:
				return True
			else:
				return False
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Data'
			_exc.cause = 'DataColumn'
			_exc.method = 'is_numeric( self )'
			_err = ErrorDialog( _exc )
			_err.show( )

	def is_text( self ) -> bool:
		try:
			if self.value is not None:
				return True
			else:
				return False
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Data'
			_exc.cause = 'DataColumn'
			_exc.method = 'is_text( self )'
			_err = ErrorDialog( _exc )
			_err.show( )

class DataRow( ):
	'''

	Constructor:

	DataRow( names: list[ str ]=None, values: tuple = ( ), source: Source=None)

	Purpose:

	Defines the class representing rows of data

	'''

	def __init__( self, names: list[ str ]=None, values: tuple=( ),
	              source: Source=None ):
		self.source = source
		self.names = names
		self.column_values = values
		self.items = zip( names, list( values ) )
		self.key = str( self.names[ 0 ] )
		self.index = int( self.column_values[ 0 ] )

	def __str__( self ) -> str:
		if self.index is not None:
			return 'Row ID: ' + str( self.index )

class DataTable( ):
	'''
	Constructor:

	DataTable( columns: list[ str ]=None, rows: list = None,
		source: Source=None, dataframe: DataFrame = None  )

	Purpose:

	Defines the class representing table of data

	'''

	def __init__( self, columns: list[ str ]=None, rows: list=None,
	              source: Source=None, dataframe: DataFrame=None ):
		self.frame = dataframe
		self.name = source.name
		self.rows = [ tuple( r ) for r in dataframe.iterrows( ) ]
		self.data = self.rows
		self.columns = [ str( c ) for c in columns ]
		self.schema = [ DataColumn( c ) for c in columns ]

	def __str__( self ) -> str:
		if self.name is not None:
			return self.name

# noinspection SqlResolve,SqlWithoutWhere
class SQLite( ):
	"""

		Purpose:
		-------
			Manages storage and retrieval of text chunks and their vector embeddings in a SQLite
			database. Supports operations for inserting, retrieving, and deleting chunk-level
			information along with their associated embedding vectors.

		Methods:
		--------
			create(): Initializes the embeddings table.
			insert(): Inserts a single embedding with metadata.
			insert_many(): Batch insert of multiple chunks and embeddings.
			fetch_all(): Retrieves all chunks and vectors from the database.
			fetch_by_file(): Retrieves chunks and vectors filtered by source file.
			delete_by_file(): Deletes all records associated with a given source file.
			close(): Closes the database connection.

	"""
	db_path: Optional[ str ]
	connection: Optional[ Connection ]
	cursor: Optional[ Cursor ]
	source_file: Optional[ str ]
	vector: Optional[ str ]
	vectors: Optional[ List[ str ] ]
	embedding: Optional[ np.ndarray ]
	text: Optional[ str ]
	index: Optional[ int ]
	rows: Optional[ List[ Row ] ]
	texts: Optional[ List[ str ] ]
	records: Optional[ List[ Tuple[ str, int, str, str ] ] ]
	
	def __init__( self, db_path: str = './embeddings.db' ) -> None:
		"""

				Purpose:
				_______
				Initializes a new SQLite connection and ensures the embeddings table is created.

				Parameters:
				----------
				db_path (str): File path to the SQLite database.

				Returns:
					None

		"""
		self.db_path = db_path
		self.connection = sqlite3.connect( self.db_path )
		self.cursor = self.conn.cursor( )
		self.embedding = None
		self.text = None
		self.index = 0
		self.rows = [ ]
		self.texts = [ ]
		self.records = [ ]
		self.vectors = [ ]
		self.create( )
	
	def create( self ) -> None:
		"""

			Purpose:
			--------
			Creates the 'embeddings' table if it does not already exist. Ensures schema integrity
			for text chunk storage with associated vector embeddings.

			Parameters:
			----------
			None

			Returns:
			-----
			None

		"""
		try:
			sql = \
				"""
                CREATE TABLE IF NOT EXISTS embeddings
                (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_file TEXT    NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text  TEXT    NOT NULL,
                    embedding   TEXT    NOT NULL,
                    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
                ); \
				"""
			self.cursor.execute( sql )
			self.connection.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'SQLite'
			exception.method = 'create( self ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def insert( self, source: str, index: int, text: str, embd: np.ndarray ) -> None:
		"""

			Purpose:
			________
			Inserts a single cleaned text chunk and its associated vector embedding.

			Parameters:
			__________
			source (str): Name or path of the source document.
			index (int): Position of the chunk within the document.
			text (str): Cleaned sentence or paragraph text.
			embedding (np.ndarray): Numpy array containing the embedding vector.

			Returns:
			_______
			None

		"""
		try:
			self.source_file = source
			self.index = index
			self.vector = json.dumps( embd.tolist( ) )
			sql = \
				'''
                INSERT INTO embeddings (source_file, chunk_index, chunk_text, embedding)
                VALUES (?, ?, ?, ?)
				'''
			self.cursor.execute( sql, (source, index, text, vector_str) )
			self.connection.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'SQLite'
			exception.method = 'insert( self, src: str, indx: int, txt: str, emb: np.ndarray )->None'
			error = ErrorDialog( exception )
			error.show( )
	
	def insert_many( self, file: str, chunks: List[ str ], vectors: np.ndarray ) -> None:
		"""

			Purpose:
			-------
			Batch inserts multiple cleaned text chunks and their associated embeddings.

			Parameters:
			----------
			source_file (str): Name or path of the source document.
			chunks (List[str]): List of cleaned text chunks.
			vectors (np.ndarray): 2D numpy array of shape (n_chunks, vector_dim).

			Returns:
			None

		"""
		try:
			self.records = [ (file, i, chunks[ i ],
			                  json.dumps( vectors[ i ].tolist( ) )) for i in
			                 range( len( chunks ) ) ]
			sql = \
				"""
                INSERT INTO embeddings (source_file, chunk_index, chunk_text, embedding)
                VALUES (?, ?, ?, ?)
				"""
			self.cursor.executemany( sql, records )
			self.connection.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'SQLite'
			exception.method = 'insert_many( self, file: str, chks: List[ str ], vect: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_all( self ) -> Tuple[ List[ str ], np.ndarray ]:
		"""

			Purpose:
			--------
			Retrieves all stored chunk texts and their corresponding embedding vectors.

			Parameters:
			----------
			None

			Returns:
			--------
			Tuple[List[str], np.ndarray]:
			A list of chunk texts and a numpy matrix of embeddings.

		"""
		try:
			sql = 'SELECT chunk_text, embedding FROM embeddings'
			self.cursor.execute( sql )
			self.rows = self.cursor.fetchall( )
			self.texts, self.vectors = [ ], [ ]
			for text, emb in rows:
				self.texts.append( text )
				self.vectors.append( np.array( json.loads( emb ) ) )
			return self.texts, np.array( self.vectors )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'SQLite'
			exception.method = 'fetch_all( self ) -> Tuple[ List[ str ], np.ndarray ] '
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_by_file( self, file: str ) -> Tuple[ List[ str ], np.ndarray ]:
		"""

			Purpose:
			-------
			Retrieves all chunk texts and embeddings for a given source file.

			Parameters:
			---------
			file (str): Name of the file to filter results by.

			Returns:
			-------
			Tuple[List[str], np.ndarray]:
			Filtered chunk texts and corresponding embedding matrix.


		"""
		try:
			sql = \
				"""
                SELECT chunk_text, embedding \
                FROM embeddings
                WHERE source_file = ?
				"""
			self.cursor.execute( sql, (file,) )
			self.rows = self.cursor.fetchall( )
			self.texts, self.vectors = [ ], [ ]
			for text, emb in self.rows:
				self.texts.append( text )
				self.vectors.append( np.array( json.loads( emb ) ) )
			return self.texts, np.array( self.vectors )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'SQLite'
			exception.method = 'fetch_by_file( self, file: str ) -> Tuple[ List[ str ], np.ndarray ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def delete_by_file( self, file: str ) -> None:
		"""

			Purpose:
				Deletes all entries associated with a specific source file.

			Parameters:
				file (str): File identifier to target deletion.

			Returns:
				None

		"""
		try:
			sql = 'DELETE FROM embeddings WHERE source_file = ?'
			self.cursor.execute( sql, (file,) )
			self.connection.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'SQLite'
			exception.method = 'delete_by_file( self, file: str ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def close( self ) -> None:
		"""

			Purpose:
			-------
			Closes the database connection cleanly.

			Parameters:
			None

			Returns:
			None

		"""
		try:
			self.connection.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'SQLite'
			exception.method = 'close( self ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def purge_all( self ) -> None:
		"""
			Purpose:
			Deletes all records from the embeddings table without removing the schema.

			Parameters:
			None

			Returns:
			None

		"""
		try:
			sql = 'DELETE FROM embeddings'
			self.cursor.execute( sql )
			self.connection.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'SQLite'
			exception.method = 'purge_all( self ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def count_rows( self ) -> int | None:
		"""

			Purpose:
			Returns the total number of rows in the embeddings table.

			Parameters:
			None

			Returns:
			int: Number of entries in the table.

		"""
		try:
			sql = 'SELECT COUNT(*) FROM embeddings'
			self.cursor.execute( sql )
			return self.cursor.fetchone( )[ 0 ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'SQLite'
			exception.method = ('count_rows( self ) -> int ')
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_metadata( self ) -> List[ Tuple[ int, str, int ] ] | None:
		"""

			Purpose:
			Retrieves all row metadata: ID, source file, and chunk index.

			Parameters:
			None

			Returns:
			list[tuple[int, str, int]]: Metadata for all rows.

		"""
		try:
			sql = 'SELECT id, source_file, chunk_index FROM embeddings'
			self.cursor.execute( sql )
			return self.cursor.fetchall( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'SQLite'
			exception.method = 'fetch_metadata( self ) -> List[ Tuple[ int, str, int ] ]'
			error = ErrorDialog( exception )
			error.show( )

class Chroma( ):
	"""

		Purpose:
		---------
		Provides persistent storage and retrieval of sentence-level embeddings using ChromaDB.
		Supports adding documents, metadata tagging, semantic querying, and deletion by ID.

		Methods:
		--------
		add(): Adds documents with embeddings and optional metadata.
		query(): Performs semantic search given a list of query texts.
		delete(): Deletes embeddings by document ID.
		count(): Returns the number of stored embeddings.
		clear(): Removes all documents from the collection.
		persist(): Commits changes to disk.

	"""
	client: Optional[ Client ]
	db_path: Optional[ str ]
	ids: Optional[ List[ str ] ]
	texts: Optional[ List[ str ] ]
	embeddings: Optional[ List[ List[ float ] ] ]
	metadata: Optional[ List[ Dict ] ]
	collection: Optional[ Collection ]
	collection_name: Optional[ str ]
	telemetry: Optional[ bool ]
	where: Optional[ Dict ]
	n_results: Optional[ int ]
	
	def __init__( self, path: str = './chroma', colname: str = 'embeddings' ) -> None:
		"""

			Purpose:
				Initializes the Chroma client and retrieves or creates the specified collection.

			Parameters:
				path (str): Directory path for Chroma persistence.
				colname (str): Name of the Chroma collection to use.

			Returns:
				None

		"""
		self.telemetry = False
		self.db_path = path
		self.collection_name = colname
		self.client = chromadb.Client( Settings( persist_directory=self.db_path,
			anonymized_telemetry=self.telemetry ) )
		self.collection = self.client.get_or_create_collection( name=self.collection_name )
		self.ids = [ ]
		self.texts = [ ]
		self.embeddings = [ ]
		self.metadata = [ ]
		self.collection = None
		self.telemetry = False
		self.where = { }
		self.n_results = None
	
	def add( self, ids: List[ str ], texts: List[ str ], embd: List[ List[ float ] ],
			metadatas: Optional[ List[ Dict ] ] = None ) -> None:
		"""

			Purpose:
			---------
			Adds a list of documents, their embeddings, and optional metadata to the collection.

			Parameters:
			-----------
			ids (List[str]): Unique identifiers for each document.
			texts (List[str]): Raw sentence or paragraph texts.
			embeddings (List[List[float]]): Corresponding embedding vectors.
			metadatas (Optional[List[dict]]): Optional metadata dictionaries for filtering.

		"""
		try:
			self.ids = ids
			self.texts = texts
			self.embeddings = embd
			self.metadata = metadatas
			self.collection.add( documents=self.texts, embeddings=self.embeddings, ids=self.ids,
				metadatas=self.metadata )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Chroma'
			exception.method = 'add( self, ids, texts, embeddings, ids, metadatas )'
			error = ErrorDialog( exception )
			error.show( )
	
	def query( self, texts: List[ str ], n_results: int = 5, where: Optional[ Dict ] = None ) -> \
			List[ str ] | None:
		"""

			Purpose:
				Performs semantic similarity search over stored embeddings.

			Parameters:
				texts (List[str]): List of query strings.
				n_results (int): Number of top matches to return.
				where (Optional[dict]): Optional metadata filter.

			Returns:
				List[str]: List of matched document texts.

		"""
		try:
			self.texts = texts
			self.n_results = n_results
			self.where = where
			result = self.collection.query( self.texts,
				n_results=self.n_results, where=self.where )
			return result.get( 'documents', [ ] )[ 0 ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Chroma'
			exception.method = 'query( self, text: List[ str ], n_results: int, where: Dict )'
			error = ErrorDialog( exception )
			error.show( )
	
	def delete( self, ids: List[ str ] ) -> None:
		"""

			Purpose:
				Deletes one or more documents from the collection by ID.

			Parameters:
				ids (List[str]): Unique identifiers of documents to delete.

			Returns:
				None

		"""
		try:
			self.ids = ids
			self.collection.delete( ids=self.ids )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Chroma'
			exception.method = ('delete(self, ids: List[ str ] ) -> None')
			error = ErrorDialog( exception )
			error.show( )
	
	def count( self ) -> int | None:
		"""

			Purpose:
				Returns the number of stored embeddings in the collection.

			Returns:
				int: Total count of stored documents.

		"""
		try:
			return self.collection.count( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Chroma'
			exception.method = 'count( self ) -> int'
			error = ErrorDialog( exception )
			error.show( )
	
	def clear( self ) -> None:
		"""

			Purpose:
				Clears all embeddings and metadata from the collection.

			Parameters:
				None

			Returns:
				None

		"""
		try:
			self.collection.delete( where=self.where )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Chroma'
			exception.method = 'clear( self )'
			error = ErrorDialog( exception )
			error.show( )
	
	def persist( self ) -> None:
		"""

			Purpose:
				Persists the current collection state to disk.

			Returns:
				None

		"""
		try:
			self.client.persist( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'processing'
			exception.cause = 'Chroma'
			exception.method = 'persist( self )'
			error = ErrorDialog( exception )
			error.show( )

class Cone( ):
	"""

		Purpose:
		Encapsulates a Pinecone vector database interface that allows embedding
		storage, search, and index management for semantic applications.

		Parameters:
		api_key (str):
			Pinecone API key required for authentication.
		environment (str):
			Pinecone environment (e.g., "gcp-starter").
		index_name (str):
			The name of the vector index to operate on.
		dimension (int):
			Dimensionality of the vector embeddings.

		Attributes:
		client (Pinecone): Pinecone client instance.
		index_name (str): Name of the active vector index.
		dimension (int): Vector size expected by the index.

	"""
	
	client: Pinecone
	index_name: str
	dimension: int
	
	def __init__( self, api_key: str, environment: str, index_name: str, dimension: int ):
		"""

			Purpose:
			---------
			Initializes the Pinecone client and ensures the target index exists.

			Parameters:
			---------
			api_key (str): API key for authentication.
			environment (str): Pinecone environment string.
			index_name (str): Vector index name.
			dimension (int): Vector dimensionality.

			Returns:
			None

		"""
		try:
			self.index_name = index_name
			self.dimension = dimension
			self.client = Pinecone( api_key=api_key )
			if self.index_name not in self.client.list_indexes( ).names( ):
				self.client.create_index( name=self.index_name, dimension=self.dimension,
					metric='cosine', spec=ServerlessSpec( cloud='aws', region=environment ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = "Cone"
			exception.method = "__init__(...)"
			error = ErrorDialog( exception )
			error.show( )
	
	def upsert( self, ids: List[ str ], vectors: List[ List[ float ] ],
			metadata: List[ dict ] = None ) -> None:
		"""

			Purpose:
			-------
			Adds or updates vector embeddings in the Pinecone index.

			Parameters:
			--------
			ids (List[str]): List of unique vector identifiers.
			vectors (List[List[float]]): List of vector embeddings.
			metadata (List[dict], optional): Optional metadata to associate with vectors.

			Returns:
			None

		"""
		try:
			throw_if( "ids", ids )
			throw_if( "vectors", vectors )
			records = [ (id_, vec, metadata[ i ] if metadata else None)
			            for i, (id_, vec) in enumerate( zip( ids, vectors ) ) ]
			index = self.client.Index( self.index_name )
			index.upsert( vectors=records )
		except Exception as e:
			exception = Error( e )
			exception.module = "cone"
			exception.cause = "Cone"
			exception.method = "upsert(...)"
			ErrorDialog( exception ).show( )
	
	def query( self, vector: List[ float ], top_k: int = 5 ) -> list:
		"""

			Purpose:
			Performs similarity search on the Pinecone index using a query vector.

			Parameters:
			vector (List[float]): Query vector.
			top_k (int): Number of top results to return.

			Returns:
			list: List of matching vector records with scores and metadata.

		"""
		try:
			throw_if( "vector", vector )
			index = self.client.Index( self.index_name )
			response = index.query( vector=vector, top_k=top_k, include_metadata=True )
			return response.matches
		except Exception as e:
			exception = Error( e )
			exception.module = "cone"
			exception.cause = "Cone"
			exception.method = "query(...)"
			ErrorDialog( exception ).show( )
	
	def delete( self, ids: List[ str ] ) -> None:
		"""

			Purpose:
			Deletes specific vector records by ID from the Pinecone index.

			Parameters:
			ids (List[str]): List of vector identifiers to remove.

			Returns:
			None

		"""
		try:
			throw_if( "ids", ids )
			index = self.client.Index( self.index_name )
			index.delete( ids=ids )
		except Exception as e:
			exception = Error( e )
			exception.module = "cone"
			exception.cause = "Cone"
			exception.method = "delete(...)"
			ErrorDialog( exception ).show( )
	
	def clear_index( self ) -> None:
		"""

			Purpose:
			Removes all vectors from the index without deleting the index itself.

			Parameters:
			None

			Returns:
			None

		"""
		try:
			index = self.client.Index( self.index_name )
			index.delete( delete_all=True )
		except Exception as e:
			exception = Error( e )
			exception.module = "cone"
			exception.cause = "Cone"
			exception.method = "clear_index()"
			ErrorDialog( exception ).show( )
	
	def drop_index( self ) -> None:
		"""

			Purpose:
			Permanently deletes the entire Pinecone index.

			Parameters:
			None

			Returns:
			None

		"""
		try:
			self.client.delete_index( self.index_name )
		except Exception as e:
			exception = Error( e )
			exception.module = "cone"
			exception.cause = "Cone"
			exception.method = "drop_index()"
			ErrorDialog( exception ).show( )
