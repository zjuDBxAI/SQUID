import json
import psycopg2
import sys
import os

from controller.clear_database import clear_tables

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from services.embedding_service import generate_embedding
from services.config import get_db_connection, config

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def create_database_if_not_exists():
    conn = get_db_connection()
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{config['dbname']}'")
    exists = cur.fetchone()
    if not exists:
        cur.execute(f'CREATE DATABASE {config["dbname"]}')
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


