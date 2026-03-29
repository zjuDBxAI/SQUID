import psycopg2
from psycopg2 import sql

from services.config import get_db_connection, config


def create_database_users():
    conn = get_db_connection()
    cur = conn.cursor()

    # Retrieve all user_ids from the Users table
    cur.execute("SELECT user_id FROM Users;")
    user_ids = cur.fetchall()

    # Create a database role for each user_id
    for user_id in user_ids:
        cur.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD '123';").format(sql.Identifier(str(user_id[0]))))

    conn.commit()
    cur.close()
    conn.close()


def drop_database_users():
    conn = get_db_connection()
    cur = conn.cursor()

    # Retrieve all user_ids from the Users table
    cur.execute("SELECT user_id FROM Users;")
    user_ids = cur.fetchall()

    # Drop the database role for each user_id
    for user_id in user_ids:
        cur.execute(sql.SQL("DROP ROLE IF EXISTS {};").format(sql.Identifier(str(user_id[0]))))

    conn.commit()
    cur.close()
    conn.close()


def enable_row_level_security():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("GRANT SELECT ON PermissionAssignment TO PUBLIC;")
    cur.execute("GRANT SELECT ON UserRoles TO PUBLIC;")
    cur.execute("GRANT SELECT ON DocumentBlocks TO PUBLIC;")

    # Enable row-level security on the DocumentBlocks table
    cur.execute("ALTER TABLE DocumentBlocks ENABLE ROW LEVEL SECURITY;")
    cur.execute("ALTER TABLE DocumentBlocks FORCE ROW LEVEL SECURITY;")

    # Create a row-level security policy that restricts access based on user roles
    cur.execute("""
        CREATE POLICY block_access_policy ON DocumentBlocks FOR SELECT
        USING (
            EXISTS (
                SELECT 1
                FROM PermissionAssignment pa
                JOIN UserRoles ur ON pa.role_id = ur.role_id
                WHERE pa.document_id = DocumentBlocks.document_id
                  AND ur.user_id = current_user::int
            )
        );
    """)

    conn.commit()
    cur.close()
    conn.close()


def disable_row_level_security():
    conn = get_db_connection()
    cur = conn.cursor()

    # Disable row-level security on the DocumentBlocks table
    cur.execute("ALTER TABLE DocumentBlocks DISABLE ROW LEVEL SECURITY;")

    # Drop the row-level security policy if it exists
    cur.execute("DROP POLICY IF EXISTS block_access_policy ON DocumentBlocks;")

    conn.commit()
    cur.close()
    conn.close()


def get_db_connection_for_many_users(user_id=None):
    user = str(user_id) if user_id else config["user"]
    return psycopg2.connect(
        dbname=config["dbname"],
        user=user,
        password=config["password"],
        host=config["host"],
        port=config["port"]
    )


def search_documents_rls_statistics_sql(user_id, query_vector, topk=5):
    """
    Searches document blocks based on vector similarity using EXPLAIN ANALYZE to track query execution time.
    """
    from basic_benchmark.common_function import get_nprobe_value
    import efconfig
    probes = get_nprobe_value()
    conn = get_db_connection_for_many_users(user_id)
    cur = conn.cursor()
    cur.execute(f"SET max_parallel_workers_per_gather = 0;")
    cur.execute(f"SET jit = off;")
    cur.execute(f"SET ivfflat.probes = {probes};")
    cur.execute(f"SET hnsw.ef_search = {efconfig.ef_search};")

    total_query_time = 0  # Variable to accumulate the total SQL query time
    total_blocks_accessed = 0  # Variable to accumulate the total blocks accessed

    # Step 1: Perform the vector search with EXPLAIN ANALYZE
    explain_query = """
        EXPLAIN (ANALYZE, verbose)
        SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
        FROM DocumentBlocks
        ORDER BY distance
        LIMIT %s
    """
    cur.execute(explain_query, [query_vector, topk])
    explain_plan = cur.fetchall()

    # ----------------------save query plan-----------start
    # import inspect
    # current_function_name = inspect.currentframe().f_code.co_name
    #
    # from benchmark.common_function import save_query_plan
    # save_query_plan(explain_plan, current_function_name)
    # ----------------------save query plan-------------end

    # Parse the execution time from EXPLAIN ANALYZE
    for row in explain_plan:
        line = row[0]
        if "Execution Time" in line:
            # Extract the execution time in milliseconds and convert to seconds
            try:
                query_time = float(line.split()[-2]) / 1000
                total_query_time += query_time
            except (IndexError, ValueError):
                pass  # Handle unexpected format gracefully
        elif "rows=" in row[0]:
            blocks_accessed = int(row[0].split("rows=")[1].split(" ")[0])
            total_blocks_accessed += blocks_accessed

    # Step 2: Perform the actual query without EXPLAIN ANALYZE to fetch the results
    query = """
        SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
        FROM DocumentBlocks
        ORDER BY distance
        LIMIT %s
    """
    cur.execute(query, [query_vector, topk])
    results = cur.fetchall()

    cur.close()
    conn.close()

    return results, total_query_time


def search_documents_rls_statistics_system(user_id, query_vector, topk=5):
    """
    Searches document blocks based on vector similarity using system time to track query execution time.
    """
    from basic_benchmark.common_function import get_nprobe_value
    probes = get_nprobe_value()
    import time
    start_time = time.time()  # Start system time tracking
    conn = get_db_connection_for_many_users(user_id)
    cur = conn.cursor()
    cur.execute(f"SET ivfflat.probes = {probes};")
    # Perform the vector search query
    query = """
        SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
        FROM DocumentBlocks
        ORDER BY distance
        LIMIT %s
    """
    cur.execute(query, [query_vector, topk])
    results = cur.fetchall()

    cur.close()
    conn.close()

    total_query_time = time.time() - start_time  # Calculate total query time

    return results, total_query_time


def search_documents_rls(user_id, query_vector, topk=5, statistics_type="sql"):
    """
    Main function to search documents using either SQL-based time statistics or system-based time statistics.
    """
    if statistics_type == "sql":
        return search_documents_rls_statistics_sql(user_id, query_vector, topk)
    elif statistics_type == "system":
        return search_documents_rls_statistics_system(user_id, query_vector, topk)
    else:
        raise ValueError(f"Unknown statistics type: {statistics_type}")
