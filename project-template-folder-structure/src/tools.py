
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool

from src.agentwrap.terminal_logger import log,log_panel

from langchain_core.tools import tool
from sqlalchemy import create_engine, text, Engine
import pandas as pd
from langgraph.types import Command
from langchain_core.tools.base import InjectedToolCallId
from typing import Annotated
from langchain_core.messages import ToolMessage
from src.agentwrap import env




class ServerSession:
    """A session for server-side state management and operations. 
    
    In practice, this would be a separate service from where the agent is running and the agent would communicate with it using a REST API. 
    In this simplified example, we use it to persist the db engine and data returned from the query_db tool.
    """
    def __init__(self):
        self.engine: Engine = None
        self.df: pd.DataFrame = None

        self.engine = self._get_engine()

    def _get_engine(self):
        if self.engine is None:
            # Get the latest SUPABASE_URL from environment
            service_apis = env.set_service_api_values()
            supabase_url = service_apis["SUPABASE_URL"]
            # Configure SQLAlchemy for session pooling
            _engine = create_engine(
                supabase_url,
                pool_size=5,                # Smaller pool size since the pooler manages connections
                max_overflow=5,             # Fewer overflow connections needed
                pool_timeout=10,            # Shorter timeout for getting connections
                pool_recycle=1800,          # Recycle connections more frequently
                pool_pre_ping=True,         # Keep this to verify connections
                pool_use_lifo=True,         # Keep LIFO to reduce number of open connections
                connect_args={
                    "application_name": "onlyvans_agent",
                    "options": "-c statement_timeout=30000",
                    # Keepalives less important with transaction pooler but still good practice
                    "keepalives": 1,
                    "keepalives_idle": 60,
                    "keepalives_interval": 30,
                    "keepalives_count": 3
                }
            )
            return _engine
        return self.engine


# Create a global instance of the ServerSession
session = ServerSession()


@contextmanager
def with_sql_cursor(readonly: bool = True):
    """
    Context manager for a SQLAlchemy cursor using the ServerSession's engine.
    Uses the global 'session' object to provide a connection and cursor.
    """
    conn = session.engine.connect()
    trans = conn.begin()
    try:
        yield conn
        if readonly:
            trans.commit()
    except Exception as e:
        trans.rollback()
        raise e
    finally:
        pass



def get_available_tools()-> List[BaseTool]:
    """Get a list of available tools."""
    return [list_tables,sample_table,describe_table,execute_sql]


def call_tool(tool_call:ToolCall)->Any:
    """Call a tool with the given tool call."""
    tool_by_name = {tool.name: tool for tool in get_available_tools()}
    tool = tool_by_name.get(tool_call["name"])
    response  = tool.invoke(tool_call["args"])
    return ToolMessage(
        content=response,
        tool_call_id=tool_call["id"],
    )


@tool(parse_docstring=False)
def list_tables(reasoning: str) -> str:
    """List all the user created tables in the database.(Excludes system tables)
    Args:
        reasoning (str): Detailed explaination of why you need to see all tables.(relate to the user query)
    Returns:
        str: A string listing all available tables in the database.    

    """
    log_panel(title= "List Tables Tool",
              content=f"Reasoning: {reasoning}",)
    try:
        with with_sql_cursor(readonly=True) as conn:
            result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'onlysql' AND table_type = 'BASE TABLE';"))
            tables = [row[0] for row in result.fetchall()]
        return f"Available tables: {', '.join(tables)}"
    except Exception as e:
        log(f"Error listing tables: {str(e)}")
        return f"Error listing tables: {str(e)}"
    


@tool(parse_docstring=False)
def sample_table(table_name: str, reasoning: str, row_sample_size: int) -> str:
    """Retrieve a sample of rows from a specified table in the database to understand its structure and content.
    Args:
        table_name (str): The name of the table to sample from.
        reasoning (str): Detailed explaination of why you need to sample this table.(relate to the user query)
        row_sample_size (int): The number of rows to sample from the table.(default is 5)
    Returns:
        str: A string containing a sample of rows from the table.
    """
    log_panel(title= "Sample Table Tool",
              content=f"Table Name: {table_name}\nReasoning: {reasoning}\nRows:{row_sample_size}",)
    try:
        with with_sql_cursor(readonly=True) as conn:
            result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT :limit;"), {"limit": row_sample_size})
            rows = result.fetchall()
        return f"Sampled rows from {table_name}: {rows}"
    except Exception as e:
        log(f"Error sampling table {table_name}: {str(e)}")
        return f"Error sampling table {table_name}: {str(e)}"
    

@tool(parse_docstring=False)
def describe_table(table_name: str, reasoning: str) -> str:
    """Return the structure of a specified table in the database, including column names and types.
    Args:
        table_name (str): The name of the table to describe.
        reasoning (str): Detailed explaination of why you need to describe this table.(relate to the user query)
    Returns:
        str: A string describing the structure of the table, including column names and types.
    """
    log_panel(title= "Describe Table Tool",
              content=f"Table Name: {table_name}\nReasoning: {reasoning}",)
    try:
        with with_sql_cursor(readonly=True) as conn:
            result = conn.execute(text(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = :table_name;"), {"table_name": table_name})
            columns = result.fetchall()
        return f"Table {table_name} structure: {columns}"
    except Exception as e:
        log(f"Error describing table {table_name}: {str(e)}")
        return f"Error describing table {table_name}: {str(e)}"
    

@tool(parse_docstring=False)
def execute_sql(query: str, reasoning: str) -> str:
    """Execute a SQL query on the database and return the results.
    Args:
        query (str): The SQL query to execute.
        reasoning (str): Detailed explaination of why you need to execute this query.(relate to the user query)
    Returns:
        str: The result of the executed SQL query.
    """
    log_panel(title= "Execute SQL Tool",
              content=f"Query: {query}\nReasoning: {reasoning}",)
    try:
        with with_sql_cursor(readonly=True) as conn:
            result = conn.execute(text(query))
            if result.returns_rows:
                rows = result.fetchall()
                return f"Query executed successfully. Results: {rows}"
            else:
                return "Query executed successfully. No results returned."
    except Exception as e:
        log(f"Error executing query '{query}': {str(e)}")
        return f"Error executing query '{query}': {str(e)}"
    


@tool
def query_db(query: str) -> str:
    """Query the database using Postgres SQL.

    Args:
        query: The SQL query to execute. Must be a valid postgres SQL string that can be executed directly.

    Returns:
        str: The query result as a markdown table.
    """
    try:
        # Use the global engine in the server session to connect to Supabase
        with session.engine.connect().execution_options(
            isolation_level="READ COMMITTED"
        ) as conn:
            result = conn.execute(text(query))

            columns = list(result.keys())
            rows = result.fetchall()
            df = pd.DataFrame(rows, columns=columns)

            # Store the DataFrame in the server session
            session.df = df

            conn.close()  # Explicitly close the connection
        return df.to_markdown(index=False)
    except Exception as e:
        return f"Error executing query: {str(e)}"
