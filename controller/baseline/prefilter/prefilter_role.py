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


def search_documents_role_partition(user_id, query_vector, topk=5, statistics_type="sql"):
    if statistics_type == "sql":
        return search_documents_role_partition_statistics_sql_run_time(user_id, query_vector, topk)
    elif statistics_type == "system": \
            return search_documents_role_partition_statistics_system(user_id, query_vector, topk)


def search_documents_role_partition_statistics_system(user_id, query_vector, topk=5):
    from basic_benchmark.common_function import get_nprobe_value
    probes = get_nprobe_value()

    import time
    start_time = time.time()  # Start system time tracking
    conn = get_db_connection()  # Get the database connection
    cur = conn.cursor()
    cur.execute(f"SET ivfflat.probes = {probes};")
    vector_str = query_vector

    cur.execute("SELECT role_id FROM userroles WHERE user_id = %s", [user_id])
    role_ids = cur.fetchall()

    all_results = []
    for role_id in role_ids:
        table_name = sql.Identifier(f"documentblocks_role_{role_id[0]}")

        query = sql.SQL(
            """
            SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
            FROM {}
            ORDER BY distance
            LIMIT %s
            """
        ).format(table_name)

        cur.execute(query, [vector_str, topk])
        all_results.extend(cur.fetchall())

    unique_results = []
    if len(role_ids) == 1:
        unique_results = all_results
    else:
        # Use a set to efficiently track seen (document_id, block_id) combinations
        seen = set()

        # Sort by distance and keep only unique (document_id, block_id) combinations
        all_results.sort(key=lambda x: x[3])  # x[2] is the distance
        for result in all_results:
            document_block_id = (result[1], result[0])  # (document_id, block_id) combination
            if document_block_id not in seen:
                seen.add(document_block_id)
                unique_results.append(result)
            if len(unique_results) == topk:
                break

    cur.close()
    conn.close()
    return unique_results, time.time() - start_time


def search_documents_role_partition_statistics_sql_run_time(user_id, query_vector, topk=5):
    from basic_benchmark.common_function import get_nprobe_value
    import efconfig
    probes = get_nprobe_value()

    conn = get_db_connection()  # Get the database connection
    cur = conn.cursor()
    cur.execute(f"SET jit = off;")
    cur.execute(f"SET ivfflat.probes = {probes};")
    cur.execute(f"SET max_parallel_workers_per_gather = 0;")
    cur.execute(f"SET hnsw.ef_search = {efconfig.ef_search};")
    total_query_time = 0  # Variable to accumulate the total SQL query time
    total_blocks_accessed = 0

    # Query 1: Get the role IDs for the user, using EXPLAIN ANALYZE
    explain_query = """
        EXPLAIN ANALYZE
        SELECT role_id FROM userroles WHERE user_id = %s
    """
    cur.execute(explain_query, [user_id])
    explain_plan = cur.fetchall()  # Fetch the EXPLAIN ANALYZE plan

    # ----------------------save query plan-----------start
    import inspect
    current_function_name = inspect.currentframe().f_code.co_name

    # save_query_plan(explain_plan, current_function_name)
    # ----------------------save query plan-------------end

    # Parse the time from the EXPLAIN ANALYZE output (but do not print the plan)
    for row in explain_plan:
        if "Execution Time" in row[0]:
            query_time = float(row[0].split()[-2]) / 1000  # Convert ms to seconds
            total_query_time += query_time  # Add to total time

    # Query 2: For each role, search document blocks using EXPLAIN ANALYZE
    all_results = []

    # Fetch role IDs without EXPLAIN
    cur.execute("SELECT role_id FROM userroles WHERE user_id = %s", [user_id])
    role_ids = cur.fetchall()

    i = 0
    for role_id in role_ids:
        table_name = sql.Identifier(f"documentblocks_role_{role_id[0]}")

        # EXPLAIN ANALYZE for the actual document block search
        explain_query = sql.SQL(
            """
            EXPLAIN (ANALYZE, VERBOSE)
            SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
            FROM {}
            ORDER BY distance
            LIMIT %s
            """
        ).format(table_name)

        cur.execute(explain_query, [query_vector, topk])
        explain_plan = cur.fetchall()
        # if role_id[0] == 11:
        #     # ----------------------save query plan-----------start
        #     import inspect
        #     current_function_name = inspect.currentframe().f_code.co_name
        #
        #     save_query_plan(explain_plan, current_function_name)
        # ----------------------save query plan-------------end

        # import inspect
        # current_function_name = inspect.currentframe().f_code.co_name
        #
        # save_query_plan(explain_plan, current_function_name)
        # Parse the time from the EXPLAIN ANALYZE output (but do not print the plan)
        for row in explain_plan:
            if "Execution Time" in row[0]:
                query_time = float(row[0].split()[-2]) / 1000  # Convert ms to seconds
                total_query_time += query_time  # Add to total time
            elif "rows=" in row[0]:
                blocks_accessed = int(row[0].split("rows=")[1].split(" ")[0])
                total_blocks_accessed += blocks_accessed

        # Execute the actual query without EXPLAIN ANALYZE to get the results
        query = sql.SQL(
            """
            SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
            FROM {}
            ORDER BY distance
            LIMIT %s
            """
        ).format(table_name)

        cur.execute(query, [query_vector, topk])
        all_results.extend(cur.fetchall())

    unique_results = []
    if len(role_ids) == 1:
        unique_results = all_results
    else:
        # Use a set to efficiently track seen (document_id, block_id) combinations
        seen = set()

        # Sort by distance and keep only unique (document_id, block_id) combinations
        all_results.sort(key=lambda x: x[3])  # x[2] is the distance
        for result in all_results:
            document_block_id = (result[1], result[0])  # (document_id, block_id) combination
            if document_block_id not in seen:
                seen.add(document_block_id)
                unique_results.append(result)
            if len(unique_results) == topk:
                break

    # Close the cursor and connection
    cur.close()
    conn.close()

    return unique_results, total_query_time


def search_documents_role_partition_get_parameter(user_id, query_vector, topk=5):
    conn = get_db_connection()  # Get the database connection
    cur = conn.cursor()
    cur.execute(f"SET max_parallel_workers_per_gather = 0;")

    total_query_time = 0  # Variable to accumulate the total SQL query time
    total_blocks_accessed = 0

    # Query 1: Get the role IDs for the user, using EXPLAIN ANALYZE
    explain_query = """
        EXPLAIN ANALYZE
        SELECT role_id FROM userroles WHERE user_id = %s
    """
    cur.execute(explain_query, [user_id])
    explain_plan = cur.fetchall()  # Fetch the EXPLAIN ANALYZE plan

    # Parse the time from the EXPLAIN ANALYZE output (but do not print the plan)
    for row in explain_plan:
        if "Execution Time" in row[0]:
            query_time = float(row[0].split()[-2]) / 1000  # Convert ms to seconds
            total_query_time += query_time  # Add to total time

    # Query 2: For each role, search document blocks using EXPLAIN ANALYZE
    all_results = []

    # Fetch role IDs without EXPLAIN
    cur.execute("SELECT role_id FROM userroles WHERE user_id = %s", [user_id])
    role_ids = cur.fetchall()

    # Regular expression to match Seq Scan on documentblocks_role and extract actual time and rows
    seq_scan_pattern = re.compile(
        r"->\s+(?:Parallel\s+)?Seq Scan on public\.documentblocks_role_\d+.*\(actual time=\d+\.\d+\.\.(\d+\.\d+) rows=(\d+)"
    )

    total_seqscan_time = 0
    for role_id in role_ids:
        table_name = sql.Identifier(f"documentblocks_role_{role_id[0]}")

        # EXPLAIN ANALYZE for the actual document block search
        explain_query = sql.SQL(
            """
            EXPLAIN (ANALYZE, VERBOSE)
            SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
            FROM {}
            ORDER BY distance
            LIMIT %s
            """
        ).format(table_name)

        cur.execute(explain_query, [query_vector, topk])
        explain_plan = cur.fetchall()

        # Parse the time from the EXPLAIN ANALYZE output (but do not print the plan)
        for row in explain_plan:
            if "Execution Time" in row[0]:
                query_time = float(row[0].split()[-2]) / 1000  # Convert ms to seconds
                total_query_time += query_time  # Add to total time
                total_seqscan_time +=  float(row[0].split()[-2])
            else:
                match = seq_scan_pattern.search(row[0])
                if match:
                    actual_time = float(match.group(1))
                    rows_accessed = int(match.group(2))
                    total_blocks_accessed += rows_accessed
        # Execute the actual query without EXPLAIN ANALYZE to get the results
        query = sql.SQL(
            """
            SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
            FROM {}
            ORDER BY distance
            LIMIT %s
            """
        ).format(table_name)

        cur.execute(query, [query_vector, topk])
        all_results.extend(cur.fetchall())

    unique_results = []
    if len(role_ids) == 1:
        unique_results = all_results
    else:
        # Use a set to efficiently track seen (document_id, block_id) combinations
        seen = set()

        # Sort by distance and keep only unique (document_id, block_id) combinations
        all_results.sort(key=lambda x: x[3])  # x[2] is the distance
        for result in all_results:
            document_block_id = (result[1], result[0])  # (document_id, block_id) combination
            if document_block_id not in seen:
                seen.add(document_block_id)
                unique_results.append(result)
            if len(unique_results) == topk:
                break

    # Close the cursor and connection
    cur.close()
    conn.close()

    return unique_results, total_seqscan_time / total_blocks_accessed


def search_documents_role_partition_union(user_id, query_vector, topk=5):
    conn = get_db_connection()
    cur = conn.cursor()

    vector_str = query_vector

    # Step 1: Get the user's roles
    cur.execute("SELECT role_id FROM userroles WHERE user_id = %s", [user_id])
    role_ids = cur.fetchall()

    all_results = []

    # Group role_ids into smaller batches for partial UNION queries
    batch_size = 10  # Adjust this based on your performance testing
    for i in range(0, len(role_ids), batch_size):
        batch_role_ids = role_ids[i:i + batch_size]

        # Build the UNION query for the current batch of roles
        union_queries = []
        for role_id in batch_role_ids:
            table_name = sql.Identifier(f"documentblocks_role_{role_id[0]}")
            union_queries.append(sql.SQL(
                """
                SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
                FROM {}
                """
            ).format(table_name))

        # Combine the queries with UNION ALL and order the results by distance
        full_query = sql.SQL(" UNION ALL ").join(union_queries) + sql.SQL(" ORDER BY distance LIMIT %s")

        # Execute the query with appropriate parameters
        query_params = [vector_str] * len(batch_role_ids) + [topk]
        cur.execute(full_query, query_params)
        all_results.extend(cur.fetchall())

    # Step 2: Use a set to track seen (document_id, block_id) combinations and remove duplicates
    seen = set()
    unique_results = []

    # Sort by distance and keep only unique (document_id, block_id) combinations
    all_results.sort(key=lambda x: x[3])  # x[3] is the distance
    for result in all_results:
        document_block_id = (result[1], result[0])  # (document_id, block_id) combination
        if document_block_id not in seen:
            seen.add(document_block_id)
            unique_results.append(result)
        if len(unique_results) == topk:
            break

    cur.close()
    conn.close()

    return unique_results
