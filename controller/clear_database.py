import psycopg2
import sys
import os

from services.config import get_db_connection

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


def clear_tables(batch_size=10):
    conn = get_db_connection()
    cur = conn.cursor()

    # Retrieve all tables in the public schema
    cur.execute("""
        SELECT tablename 
        FROM pg_tables 
        WHERE schemaname = 'public';
    """)
    tables = cur.fetchall()

    # Drop tables in batches to avoid running out of shared memory
    for i in range(0, len(tables), batch_size):
        batch = tables[i:i + batch_size]
        for table in batch:
            cur.execute(f"DROP TABLE IF EXISTS {table[0]} CASCADE;")
        conn.commit()  # Commit after each batch

    cur.close()
    conn.close()
    print("All tables have been cleared from the database.")


if __name__ == '__main__':
    clear_tables()
