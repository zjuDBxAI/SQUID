import psycopg2
from psycopg2 import sql
import sys
import os

from services.config import get_db_connection

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


def _drop_batch(cur, table_names):
    statements = [
        sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(sql.Identifier(table_name))
        for table_name in table_names
    ]
    cur.execute(sql.SQL("\n").join(statements))


def clear_tables(batch_size=32):
    conn = get_db_connection()
    cur = conn.cursor()

    # Retrieve all tables in the public schema. We drop in adaptive batches:
    # large batches are fast, and lock-heavy batches are retried smaller to
    # avoid PostgreSQL max_locks_per_transaction failures.
    cur.execute("""
        SELECT c.relname, count(d.objid) AS dep_count
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        LEFT JOIN pg_depend d ON d.refobjid = c.oid
        WHERE n.nspname = 'public'
          AND c.relkind IN ('r', 'p')
        GROUP BY c.oid, c.relname
        ORDER BY dep_count ASC, c.relname;
    """)
    tables = [str(row[0]) for row in cur.fetchall()]

    dropped = 0
    index = 0
    current_batch_size = max(1, int(batch_size))
    while index < len(tables):
        batch = tables[index:index + current_batch_size]
        try:
            _drop_batch(cur, batch)
            conn.commit()
            dropped += len(batch)
            index += len(batch)
            if current_batch_size < int(batch_size):
                current_batch_size = min(int(batch_size), current_batch_size * 2)
        except psycopg2.errors.OutOfMemory:
            conn.rollback()
            if current_batch_size <= 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            print(
                "DROP TABLE batch exceeded PostgreSQL lock memory; "
                f"retrying with batch_size={current_batch_size}."
            )
        except psycopg2.Error:
            conn.rollback()
            raise

    cur.close()
    conn.close()
    print(f"All tables have been cleared from the database. Dropped {dropped} tables.")


if __name__ == '__main__':
    clear_tables()
