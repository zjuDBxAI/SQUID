import psycopg2
import sys
import os
from psycopg2 import sql

from controller.clear_database import clear_tables

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from services.embedding_service import generate_embedding
from services.config import get_db_connection, config

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def create_database_if_not_exists():
    conn = psycopg2.connect(
        dbname=os.environ.get("DB_MAINTENANCE_DB", "postgres"),
        user=config["user"],
        password=config["password"],
        host=config["host"],
        port=config["port"],
    )
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (config["dbname"],))
    exists = cur.fetchone()
    if not exists:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(config["dbname"])))
    cur.close()
    conn.close()

def create_pgvector_extension():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    cur.close()
    conn.close()



def clear_db():
    create_database_if_not_exists()

    create_pgvector_extension()

    clear_tables()

