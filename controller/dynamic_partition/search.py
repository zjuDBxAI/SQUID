import os
import statistics
import sys
import re
import psycopg2
from psycopg2 import sql
import time



project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from basic_benchmark import efconfig
from controller.baseline.pg_row_security.row_level_security import get_db_connection_for_many_users
from services.config import get_db_connection

def dynamic_partition_search(user_id, query_vector, topk=5, statistics_type="sql"):
    """
    Entry point for dynamic partition search with SQL or system time measurement.
    :param user_id: User ID to identify associated roles.
    :param query_vector: Query vector for similarity search.
    :param topk: Number of top results to return.
    :param statistics_type: Either 'sql' or 'system' for performance tracking.
    """
    if statistics_type == "sql":
        return dynamic_partition_search_statistics_sql(user_id, query_vector, topk)
    elif statistics_type == "system":
        return dynamic_partition_search_statistics_system(user_id, query_vector, topk)


def dynamic_partition_search_statistics_sql(user_id, query_vector, topk=5, ef_search=40):
    """
    Search using SQL query execution time for performance statistics with RLS.
    """
    # conn = get_db_connection()
    import efconfig
    conn = get_db_connection_for_many_users(user_id)
    cur = conn.cursor()
    cur.execute(f"SET max_parallel_workers_per_gather = 0;")
    cur.execute(f"SET jit = off;")
    cur.execute(f"SET hnsw.ef_search = {efconfig.ef_search};")
    total_query_time = 0
    query_plan_output = []  # Store query plans for validation
    cur.execute("""
            SELECT role_id
            FROM UserRoles
            WHERE user_id = %s;
        """, [user_id])
    user_roles = {row[0] for row in cur.fetchall()}

    # Step 2: Sort the roles to create a canonical representation
    sorted_roles = sorted(user_roles)

    cur.execute("""
        SELECT partition_id
        FROM CombRolePartitions
        WHERE comb_role = %s::integer[];
    """, [sorted_roles])

    accessible_partitions = {row[0] for row in cur.fetchall()}
    # Step 2: Search each partition with EXPLAIN ANALYZE
    all_results = []
    for partition_id in accessible_partitions:
        partition_table = sql.Identifier(f"documentblocks_partition_{partition_id}")

        explain_query = sql.SQL(
            """
            EXPLAIN ANALYZE
            SELECT block_id, document_id, block_content,
                   vector <-> %s::vector AS distance
            FROM {}
            ORDER BY distance
            LIMIT %s;
            """
        ).format(partition_table)

        # Capture query plan and execution time
        cur.execute(explain_query, [query_vector, topk])
        explain_plan = cur.fetchall()
        query_plan_output.append(f"Partition {partition_id}:\n")
        query_plan_output.extend([row[0] for row in explain_plan])
        query_plan_output.append("\n")

        for row in explain_plan:
            line = row[0].strip()  # Clean each row

            # Check for the overall execution time
            if "Execution Time" in line:
                execution_time = float(line.split()[-2]) / 1000  # Convert ms to seconds
                total_query_time += execution_time

        # Execute actual search
        query = sql.SQL(
            """
            SELECT block_id, document_id, block_content,
                   vector <-> %s::vector AS distance
            FROM {}
            ORDER BY distance
            LIMIT %s;
            """
        ).format(partition_table)

        cur.execute(query, [query_vector, topk])
        all_results.extend(cur.fetchall())

    cur.close()
    conn.close()

    result = merge_results(all_results, topk)

    return result, total_query_time


def merge_results_with_filter(all_results, accessible_documents, topk):
    """
    Merge, deduplicate, and filter search results based on (document_id, block_id).
    """
    seen = set()
    unique_results = []

    # Sort all results by distance
    all_results.sort(key=lambda x: x[3])  # Sort by distance

    # Deduplicate and filter by accessible documents
    for result in all_results:
        doc_block_id = (result[1], result[0])  # (document_id, block_id)
        if doc_block_id not in seen and result[1] in accessible_documents:
            seen.add(doc_block_id)
            unique_results.append(result)
        if len(unique_results) == topk:
            break

    return unique_results


def dynamic_partition_search_statistics_system(user_id, query_vector, topk=5):
    """
    Search using system time measurement for performance statistics with RLS.
    """
    import time
    # conn = get_db_connection()
    conn = get_db_connection_for_many_users(user_id)
    cur = conn.cursor()

    start_time = time.time()  # Track overall system time

    # Step 1: Fetch roles and partitions for the user
    cur.execute("""
        SELECT partition_id FROM RolePartitions rp
        JOIN UserRoles ur ON rp.role_id = ur.role_id
        WHERE ur.user_id = %s;
    """, [user_id])
    accessible_partitions = {row[0] for row in cur.fetchall()}

    # Step 2: Search across partitions
    all_results = []
    for partition_id in accessible_partitions:
        partition_table = sql.Identifier(f"documentblocks_partition_{partition_id}")

        query = sql.SQL(
            """
            SELECT block_id, document_id, block_content, 
                   vector <-> %s::vector AS distance
            FROM {}
            ORDER BY distance
            LIMIT %s;
            """
        ).format(partition_table)

        cur.execute(query, [query_vector, topk])
        all_results.extend(cur.fetchall())

    cur.close()
    conn.close()

    # Measure total time taken
    total_time = time.time() - start_time

    result = merge_results(all_results, topk)

    return result, total_time


def dynamic_partition_search_stats_parameter(user_id, query_vector, topk=5):
    """
    Search using SQL query execution time for performance statistics with RLS.
    """
    conn = get_db_connection_for_many_users(user_id)
    cur = conn.cursor()
    cur.execute("SET jit = off;")
    total_query_time = 0
    hashjoin_times = []
    proj_times = []
    adjusted_times = []
    partition_count = 0  # Count partitions for averaging

    # Step 1: Check if the user has access to role_id = ?
    cur.execute("""
        SELECT ur.role_id 
        FROM UserRoles ur 
        WHERE ur.user_id = %s AND ur.role_id = 11;
    """, [user_id])

    # Step 1: Fetch accessible partitions using EXPLAIN ANALYZE
    explain_query = """
        EXPLAIN (ANALYZE, VERBOSE)
        SELECT partition_id FROM RolePartitions rp
        JOIN UserRoles ur ON rp.role_id = ur.role_id
        WHERE ur.user_id = %s;
    """
    cur.execute(explain_query, [user_id])

    # Regular expressions to parse times
    seq_scan_pattern = re.compile(
        r"->\s+(?:Parallel\s+)?Seq Scan on public\.documentblocks_partition_\d+.*actual time=(\d+\.\d+)\.\.(\d+\.\d+).*fetch time=(\d+\.\d+) rows=(\d+) qual time=(\d+\.\d+) rows=(\d+) proj time=(\d+\.\d+) rows=(\d+)"
    )
    subplan_pattern = re.compile(r"^SubPlan 2$")
    actual_time_pattern = re.compile(r"actual time=\d+\.\d+\.\.(\d+\.\d+)")

    # Fetch partition IDs
    cur.execute("""
        SELECT partition_id FROM RolePartitions rp
        JOIN UserRoles ur ON rp.role_id = ur.role_id
        WHERE ur.user_id = %s;
    """, [user_id])
    accessible_partitions = {row[0] for row in cur.fetchall()}

    # Step 2: Search each partition with EXPLAIN ANALYZE
    all_results = []
    for partition_id in accessible_partitions:
        partition_table = sql.Identifier(f"documentblocks_partition_{partition_id}")

        explain_query = sql.SQL(
            """
            EXPLAIN (ANALYZE,VERBOSE,BUFFERS)
            SELECT block_id, document_id, block_content, 
                   vector <-> %s::vector AS distance
            FROM {}
            ORDER BY distance
            LIMIT %s;
            """
        ).format(partition_table)

        # Capture query plan and execution time
        cur.execute(explain_query, [query_vector, topk])
        explain_plan = cur.fetchall()

        # Initialize partition-specific accumulators
        partition_fetch_qual_sum = 0.0
        partition_proj_time = 0.0
        adjusted_time = 0.0
        fetch_rows = qual_rows = proj_rows = 0
        in_subplan = False
        first_hashjoin_time_recorded = False

        # Parse the explain plan for this partition
        for row in explain_plan:
            line = row[0].strip()

            # Check if we enter SubPlan 2 section for hash join
            if subplan_pattern.match(line):
                in_subplan = True

            # Match Seq Scan only on documentblocks_partition_xx and extract times
            seq_scan_match = seq_scan_pattern.search(line)
            if seq_scan_match:
                actual_start_time = float(seq_scan_match.group(1))
                fetch_time = float(seq_scan_match.group(3))
                qual_time = float(seq_scan_match.group(5))
                proj_time = float(seq_scan_match.group(7))

                partition_fetch_qual_sum += fetch_time + qual_time
                partition_proj_time += proj_time
                fetch_rows += int(seq_scan_match.group(4))
                qual_rows += int(seq_scan_match.group(6))
                proj_rows += int(seq_scan_match.group(8))

                # Calculate adjusted time by subtracting the first actual start time from fetch + qual time
                adjusted_time += partition_fetch_qual_sum - actual_start_time

            # If inside SubPlan 2, capture only the first actual time for hash join
            if in_subplan and not first_hashjoin_time_recorded:
                hashjoin_match = actual_time_pattern.search(line)
                if hashjoin_match:
                    hashjoin_times.append(float(hashjoin_match.group(1)))
                    first_hashjoin_time_recorded = True

            # Check for the overall execution time
            if "Execution Time" in line:
                execution_time = float(line.split()[-2]) / 1000  # Convert ms to seconds
                total_query_time += execution_time

        # Calculate per-partition adjusted and proj times, accounting for row counts
        if fetch_rows > 0:
            adjusted_time /= fetch_rows
        if proj_rows > 0:
            partition_proj_time /= proj_rows

        # Store per-partition results for calculating medians
        adjusted_times.append(adjusted_time)
        proj_times.append(partition_proj_time)
        partition_count += 1  # Increment partition count for averaging

        # Execute actual search
        query = sql.SQL(
            """
            SELECT block_id, document_id, block_content, 
                   vector <-> %s::vector AS distance
            FROM {}
            ORDER BY distance
            LIMIT %s;
            """
        ).format(partition_table)

        cur.execute(query, [query_vector, topk])
        all_results.extend(cur.fetchall())

    # Calculate medians across partitions
    median_adjusted_time = statistics.median(adjusted_times) if adjusted_times else 0
    median_proj_time = statistics.median(proj_times) if proj_times else 0
    median_hashjoin_time = statistics.median(hashjoin_times) if hashjoin_times else 0

    cur.close()
    conn.close()

    result = merge_results(all_results, topk)

    # Return results along with timing statistics
    return result, median_adjusted_time, median_hashjoin_time, median_proj_time


def get_user_roles_and_partitions(cur, user_id):
    """
    Get roles and their accessible partitions for a user.
    """
    cur.execute("SELECT role_id FROM UserRoles WHERE user_id = %s", [user_id])
    role_ids = [row[0] for row in cur.fetchall()]

    partitions = []
    for role_id in role_ids:
        cur.execute("SELECT partition_id FROM RolePartitions WHERE role_id = %s", [role_id])
        partitions.extend([row[0] for row in cur.fetchall()])

    return role_ids, set(partitions)  # Return unique partitions


def merge_results(all_results, topk):
    """
    Merge and deduplicate search results based on (document_id, block_id).
    """
    seen = set()
    unique_results = []

    all_results.sort(key=lambda x: x[3])  # Sort by distance

    for result in all_results:
        doc_block_id = (result[1], result[0])  # (document_id, block_id)
        if doc_block_id not in seen:
            seen.add(doc_block_id)
            unique_results.append(result)
        if len(unique_results) == topk:
            break

    return unique_results
