import os
import sys
import time

from psycopg2 import sql

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(project_root)

from services.config import config, get_db_connection


CURATOR_TABLE_NAME = "documentblocks_curator"


def _curator_search_config() -> dict:
    curator_config = config.get("curator", {})
    return {
        "gamma1": float(curator_config.get("gamma1", 2.0)),
        "gamma2": float(curator_config.get("gamma2", 128.0)),
        "enable_seqscan": bool(curator_config.get("enable_seqscan", False)),
    }


def _apply_curator_runtime_settings(cur, user_id: int, gamma1: float, gamma2: float, enable_seqscan: bool) -> None:
    cur.execute("SET LOCAL jit = off;")
    if not enable_seqscan:
        cur.execute("SET LOCAL enable_seqscan = off;")
    cur.execute("SELECT set_config('curator.tenant_id', %s, true);", [str(int(user_id))])
    cur.execute("SELECT set_config('curator.gamma1', %s, true);", [str(float(gamma1))])
    cur.execute("SELECT set_config('curator.gamma2', %s, true);", [str(float(gamma2))])


def curator_search(user_id, query_vector, topk=5, statistics_type="sql"):
    if statistics_type == "sql":
        return curator_search_statistics_sql(user_id, query_vector, topk)
    if statistics_type == "system":
        return curator_search_statistics_system(user_id, query_vector, topk)
    raise ValueError(f"Unknown statistics_type: {statistics_type}")


def curator_search_statistics_system(user_id, query_vector, topk=5):
    runtime = _curator_search_config()
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        _apply_curator_runtime_settings(
            cur,
            user_id=int(user_id),
            gamma1=runtime["gamma1"],
            gamma2=runtime["gamma2"],
            enable_seqscan=runtime["enable_seqscan"],
        )

        start_time = time.time()
        cur.execute(
            sql.SQL(
                """
                SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
                FROM {}
                ORDER BY vector <-> %s::vector
                LIMIT %s;
                """
            ).format(sql.Identifier(CURATOR_TABLE_NAME)),
            [query_vector, query_vector, topk],
        )
        results = cur.fetchall()
        total_time = time.time() - start_time
        return results, total_time
    finally:
        cur.close()
        conn.close()


def curator_search_statistics_sql(user_id, query_vector, topk=5):
    runtime = _curator_search_config()
    conn = get_db_connection()
    cur = conn.cursor()
    total_query_time = 0.0
    try:
        _apply_curator_runtime_settings(
            cur,
            user_id=int(user_id),
            gamma1=runtime["gamma1"],
            gamma2=runtime["gamma2"],
            enable_seqscan=runtime["enable_seqscan"],
        )

        cur.execute(
            sql.SQL(
                """
                EXPLAIN ANALYZE
                SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
                FROM {}
                ORDER BY vector <-> %s::vector
                LIMIT %s;
                """
            ).format(sql.Identifier(CURATOR_TABLE_NAME)),
            [query_vector, query_vector, topk],
        )
        explain_plan = cur.fetchall()
        for row in explain_plan:
            line = row[0]
            if "Execution Time" in line:
                total_query_time += float(line.split()[-2]) / 1000

        cur.execute(
            sql.SQL(
                """
                SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
                FROM {}
                ORDER BY vector <-> %s::vector
                LIMIT %s;
                """
            ).format(sql.Identifier(CURATOR_TABLE_NAME)),
            [query_vector, query_vector, topk],
        )
        results = cur.fetchall()
        return results, total_query_time
    finally:
        cur.close()
        conn.close()
