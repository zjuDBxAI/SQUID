import json
import re
import time

import pandas as pd
import psycopg2
from psycopg2 import sql
import sys
import os

from basic_benchmark.common_function import save_query_plan
from controller.baseline.prefilter.initialize_partitions import initialize_user_partitions, initialize_role_partitions
from controller.clear_database import clear_tables
from controller.prepare_database import create_database_if_not_exists, create_pgvector_extension, read_file

from services.config import get_db_connection

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)


def search_documents_combination_partition(user_id, query_vector, topk=5, statistics_type="sql"):
    """
    Search for documents in user-role combination partitions with specified statistics type.

    Args:
        user_id (int): User ID for the query.
        query_vector (str): Query vector as a string.
        topk (int): Number of top results to retrieve.
        statistics_type (str): Type of statistics to collect ('sql' or 'system').

    Returns:
        tuple: List of query results and total query time.
    """
    if statistics_type == "sql":
        return search_documents_combination_partition_statistics_sql(user_id, query_vector, topk)
    elif statistics_type == "system":
        return search_documents_combination_partition_statistics_system(user_id, query_vector, topk)
    else:
        raise ValueError(f"Unknown statistics_type: {statistics_type}")

def search_documents_combination_partition_statistics_system(user_id, query_vector, topk=5):
    """
    Search for documents in user-role combination partitions using system time statistics.

    Args:
        user_id (int): User ID for the query.
        query_vector (str): Query vector as a string.
        topk (int): Number of top results to retrieve.

    Returns:
        tuple: List of query results and total system query time.
    """
    import time
    start_time = time.time()  # Start time tracking

    conn = get_db_connection()
    cur = conn.cursor()

    # Retrieve the role combination for the user
    cur.execute("""
        SELECT array_agg(role_id ORDER BY role_id) AS roles
        FROM userroles
        WHERE user_id = %s
        GROUP BY user_id;
    """, [user_id])
    roles = cur.fetchone()

    if not roles:
        return [], 0  # Return empty results if user has no roles

    table_name = f"documentblocks_combination_{'_'.join(map(str, roles[0]))}"

    # Execute the query on the role combination table
    query = sql.SQL("""
        SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
        FROM {}
        ORDER BY distance
        LIMIT %s;
    """).format(sql.Identifier(table_name))

    cur.execute(query, [query_vector, topk])
    results = cur.fetchall()

    cur.close()
    conn.close()

    total_time = time.time() - start_time  # Calculate elapsed time
    return results, total_time


def search_documents_combination_partition_statistics_sql(user_id, query_vector, topk=5):
    """
    Search for documents in user-role combination partitions using SQL statistics (EXPLAIN ANALYZE).

    Args:
        user_id (int): User ID for the query.
        query_vector (str): Query vector as a string.
        topk (int): Number of top results to retrieve.

    Returns:
        tuple: List of query results and total SQL query time.
    """
    # Fetch ef_search and nprobe configurations
    from basic_benchmark.common_function import get_nprobe_value
    import efconfig

    # Fetch ef_search and nprobe configurations
    probes = get_nprobe_value()
    ef_search = efconfig.ef_search

    conn = get_db_connection()
    cur = conn.cursor()

    # Set ef_search and ivfflat.probes configurations
    cur.execute(f"SET hnsw.ef_search = {ef_search};")
    cur.execute(f"SET ivfflat.probes = {probes};")
    cur.execute(f"SET max_parallel_workers_per_gather = 0;")
    total_query_time = 0  # Accumulate total query time

    # Retrieve the role combination for the user
    cur.execute("""
        SELECT array_agg(role_id ORDER BY role_id) AS roles
        FROM userroles
        WHERE user_id = %s
        GROUP BY user_id;
    """, [user_id])
    roles = cur.fetchone()

    if not roles:
        return [], 0  # Return empty results if user has no roles

    table_name = f"documentblocks_combination_{'_'.join(map(str, roles[0]))}"

    # EXPLAIN ANALYZE the query on the role combination table
    explain_query = sql.SQL("""
        EXPLAIN ANALYZE
        SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
        FROM {}
        ORDER BY distance
        LIMIT %s;
    """).format(sql.Identifier(table_name))

    cur.execute(explain_query, [query_vector, topk])
    explain_plan = cur.fetchall()

    # Parse the execution time from EXPLAIN ANALYZE
    for row in explain_plan:
        if "Execution Time" in row[0]:
            total_query_time += float(row[0].split()[-2]) / 1000  # Convert ms to seconds

    # Execute the actual query to fetch results
    query = sql.SQL("""
        SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
        FROM {}
        ORDER BY distance
        LIMIT %s;
    """).format(sql.Identifier(table_name))

    cur.execute(query, [query_vector, topk])
    results = cur.fetchall()

    cur.close()
    conn.close()

    return results, total_query_time
