import os
import sys

from psycopg2 import sql

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(project_root)

from services.config import config, get_db_connection, get_maintenance_settings


CURATOR_TABLE_NAME = "documentblocks_curator"
CURATOR_INDEX_NAME = "documentblocks_curator_vector_idx"


def _curator_config() -> dict:
    curator_config = config.get("curator", {})
    return {
        "nlist": int(curator_config.get("nlist", 32)),
        "bf_capacity": int(curator_config.get("bf_capacity", 1000)),
        "bf_false_pos": float(curator_config.get("bf_false_pos", curator_config.get("bf_error_rate", 0.01))),
        "max_sl_size": int(curator_config.get("max_sl_size", 256)),
        "tenant_column": str(curator_config.get("tenant_column", "access_tenants")),
    }


def _configure_build_session(cur, disable_sync_commit: bool = True) -> None:
    maintenance_settings = get_maintenance_settings()
    cur.execute(
        f"SET maintenance_work_mem = '{maintenance_settings['maintenance_work_mem_gb']}GB';"
    )
    cur.execute(
        f"SET max_parallel_maintenance_workers = {maintenance_settings['max_parallel_maintenance_workers']};"
    )
    if disable_sync_commit:
        cur.execute("SET synchronous_commit = OFF;")


def drop_curator_index() -> None:
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            sql.SQL("DROP INDEX IF EXISTS {} CASCADE;").format(
                sql.Identifier(CURATOR_INDEX_NAME)
            )
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()


def build_curator_index(
    nlist: int | None = None,
    bf_capacity: int | None = None,
    bf_false_pos: float | None = None,
    max_sl_size: int | None = None,
    tenant_column: str | None = None,
    disable_sync_commit: bool = True,
    drop_existing: bool = True,
    analyze: bool = True,
) -> None:
    """
    Build the Curator ANN index on documentblocks_curator.

    This assumes pgvector has been extended with:
    - access method curator
    - reloption tenant_column
    - build options nlist, bf_capacity, bf_false_pos, max_sl_size
    """
    defaults = _curator_config()
    nlist = defaults["nlist"] if nlist is None else int(nlist)
    bf_capacity = defaults["bf_capacity"] if bf_capacity is None else int(bf_capacity)
    bf_false_pos = defaults["bf_false_pos"] if bf_false_pos is None else float(bf_false_pos)
    max_sl_size = defaults["max_sl_size"] if max_sl_size is None else int(max_sl_size)
    tenant_column = defaults["tenant_column"] if tenant_column is None else str(tenant_column)

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        _configure_build_session(cur, disable_sync_commit=disable_sync_commit)

        if drop_existing:
            cur.execute(
                sql.SQL("DROP INDEX IF EXISTS {} CASCADE;").format(
                    sql.Identifier(CURATOR_INDEX_NAME)
                )
            )

        cur.execute(
            sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {table_name}
                USING curator (vector vector_l2_ops)
                WITH (
                    tenant_column = {tenant_column},
                    nlist = {nlist},
                    bf_capacity = {bf_capacity},
                    bf_false_pos = {bf_false_pos},
                    max_sl_size = {max_sl_size}
                );
                """
            ).format(
                index_name=sql.Identifier(CURATOR_INDEX_NAME),
                table_name=sql.Identifier(CURATOR_TABLE_NAME),
                tenant_column=sql.Literal(tenant_column),
                nlist=sql.Literal(nlist),
                bf_capacity=sql.Literal(bf_capacity),
                bf_false_pos=sql.Literal(bf_false_pos),
                max_sl_size=sql.Literal(max_sl_size),
            )
        )

        if analyze:
            cur.execute(sql.SQL("ANALYZE {};").format(sql.Identifier(CURATOR_TABLE_NAME)))

        conn.commit()
    finally:
        cur.close()
        conn.close()
