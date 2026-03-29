import os
import sys

from psycopg2 import sql

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(project_root)

from services.config import get_db_connection


CURATOR_TABLE_NAME = "documentblocks_curator"


def drop_curator_table() -> None:
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(sql.Identifier(CURATOR_TABLE_NAME)))
        conn.commit()
    finally:
        cur.close()
        conn.close()


def initialize_curator_table(drop_existing: bool = True, analyze: bool = True) -> None:
    """
    Materialize a Curator-specific table that stores each block once together with
    the list of user IDs that may access it.

    The Curator access method is expected to read access_tenants during index
    build and to use the vector column for ANN search.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        if drop_existing:
            cur.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(
                    sql.Identifier(CURATOR_TABLE_NAME)
                )
            )

        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {} AS
                SELECT
                    db.block_id,
                    db.document_id,
                    db.block_content,
                    db.vector,
                    ARRAY_AGG(DISTINCT ur.user_id ORDER BY ur.user_id)::integer[] AS access_tenants
                FROM documentblocks db
                JOIN PermissionAssignment pa ON pa.document_id = db.document_id
                JOIN UserRoles ur ON ur.role_id = pa.role_id
                GROUP BY db.block_id, db.document_id, db.block_content, db.vector;
                """
            ).format(sql.Identifier(CURATOR_TABLE_NAME))
        )

        cur.execute(
            sql.SQL(
                """
                ALTER TABLE {}
                ALTER COLUMN access_tenants SET NOT NULL;
                """
            ).format(sql.Identifier(CURATOR_TABLE_NAME))
        )

        cur.execute(
            sql.SQL(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS {}
                ON {} (block_id, document_id);
                """
            ).format(
                sql.Identifier(f"{CURATOR_TABLE_NAME}_block_doc_idx"),
                sql.Identifier(CURATOR_TABLE_NAME),
            )
        )

        cur.execute(
            sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {}
                ON {} (document_id);
                """
            ).format(
                sql.Identifier(f"{CURATOR_TABLE_NAME}_document_idx"),
                sql.Identifier(CURATOR_TABLE_NAME),
            )
        )

        cur.execute(
            sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {}
                ON {} USING gin (access_tenants);
                """
            ).format(
                sql.Identifier(f"{CURATOR_TABLE_NAME}_access_tenants_gin_idx"),
                sql.Identifier(CURATOR_TABLE_NAME),
            )
        )

        if analyze:
            cur.execute(sql.SQL("ANALYZE {};").format(sql.Identifier(CURATOR_TABLE_NAME)))

        conn.commit()
    finally:
        cur.close()
        conn.close()
