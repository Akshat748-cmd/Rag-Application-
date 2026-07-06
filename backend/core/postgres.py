"""
PostgreSQL Storage Module
Handles connection tests, table creation, and bulk inserts for document chunks.
"""
import psycopg2
from psycopg2.extras import execute_values
from typing import List, Dict, Tuple, Optional
import urllib.parse as urlparse

def parse_connection_uri(uri: str) -> Dict[str, str]:
    """Parse connection URI into dictionary parameters if needed."""
    urlparse.uses_netloc.append("postgres")
    url = urlparse.urlparse(uri)
    return {
        "database": url.path[1:],
        "user": url.username,
        "password": url.password,
        "host": url.hostname,
        "port": url.port
    }

def get_connection(
    connection_uri: Optional[str] = None,
    host: str = "localhost",
    port: int = 2004,
    database: str = "postgres",
    user: str = "postgres",
    password: str = "postgres"
):
    """Establish a connection to the PostgreSQL database."""
    if connection_uri and connection_uri.strip():
        return psycopg2.connect(connection_uri)
    else:
        return psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )

def test_db_connection(
    connection_uri: Optional[str] = None,
    host: str = "localhost",
    port: int = 2004,
    database: str = "postgres",
    user: str = "postgres",
    password: str = "postgres"
) -> Tuple[bool, str]:
    """Test connection to PostgreSQL database."""
    conn = None
    try:
        conn = get_connection(
            connection_uri=connection_uri,
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        return True, "Successfully connected to PostgreSQL!"
    except Exception as e:
        return False, str(e)
    finally:
        if conn:
            conn.close()

def store_chunks_in_postgres(
    chunks: List[Dict],
    table_name: str = "document_chunks",
    connection_uri: Optional[str] = None,
    host: str = "localhost",
    port: int = 2004,
    database: str = "postgres",
    user: str = "postgres",
    password: str = "postgres",
    document_name: str = "unknown"
) -> Tuple[bool, str, int]:
    """
    Creates table if not exists and bulk inserts chunks.
    Returns: (success_bool, message_str, inserted_count)
    """
    if not chunks:
        return False, "No chunks to store.", 0

    # Sanitize table name to prevent SQL injection on table name
    # Allow only letters, numbers, and underscores
    # Also cap at 63 chars (PostgreSQL max identifier length) to avoid silent truncation
    sanitized_table = "".join(c for c in table_name if c.isalnum() or c == "_")
    sanitized_table = sanitized_table[:63].rstrip("_")  # truncate + clean trailing _
    if not sanitized_table:
        return False, "Invalid table name.", 0

    conn = None
    try:
        conn = get_connection(
            connection_uri=connection_uri,
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        cur = conn.cursor()

        # Create table query
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {sanitized_table} (
            id SERIAL PRIMARY KEY,
            document_name VARCHAR(255) DEFAULT 'unknown',
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            word_count INTEGER,
            strategy VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cur.execute(create_table_query)

        # Add document_name column if table already exists without it (migration)
        cur.execute(f"""
            ALTER TABLE {sanitized_table}
            ADD COLUMN IF NOT EXISTS document_name VARCHAR(255) DEFAULT 'unknown';
        """)
        conn.commit()

        # Prep values
        # Expect chunks to be objects or dictionaries
        # Each chunk has: "id" (or index), "text" (or content), "word_count", "strategy"
        values = []
        for idx, c in enumerate(chunks):
            # Check if chunk is dict or string
            if isinstance(c, dict):
                chunk_index = c.get("id", idx + 1)
                content = c.get("text", c.get("content", ""))
                strategy = c.get("strategy", "fixed")
            else:
                chunk_index = idx + 1
                content = str(c)
                strategy = "fixed"

            word_count = len(content.split())
            values.append((document_name, chunk_index, content, word_count, strategy))

        # Bulk insert
        insert_query = f"""
        INSERT INTO {sanitized_table} (document_name, chunk_index, content, word_count, strategy)
        VALUES %s;
        """
        
        execute_values(cur, insert_query, values)
        conn.commit()

        # Get database info for message
        db_info = "URI Connection"
        if not connection_uri or not connection_uri.strip():
            db_info = f"database '{database}' on host '{host}'"
        
        return True, f"Successfully stored {len(values)} chunks in table '{sanitized_table}' ({db_info})!", len(values)

    except Exception as e:
        if conn:
            conn.rollback()
        return False, f"Failed to store chunks: {str(e)}", 0
    finally:
        if conn:
            conn.close()
