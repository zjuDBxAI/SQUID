import os
import sys
import json
import time
from psycopg2 import sql
import importlib
from typing import Callable
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(project_root,"basic_benchmark", "result")
os.makedirs(RESULT_DIR,exist_ok=True)

sys.path.append(project_root)

from basic_benchmark.condition_config import CONDITION_CONFIG
from controller.baseline.prefilter.initialize_partitions import initialize_user_partitions, initialize_role_partitions, \
    initialize_combination_partitions, drop_prefilter_partition_tables
from services.config import get_db_connection
from services.read_dataset_function import generate_query_dataset, load_queries_from_dataset

# Global cache for FAISS GPU ground truth
_faiss_user_data_cache = None
_faiss_role_data_cache = None  # Cache by role instead of user
_faiss_query_counter = {'total': 0, 'completed': 0}


def get_nprobe_value(config_file="config_params.json"):
    # Open and read the configuration file
    benchmark_folder = os.path.join(project_root, "basic_benchmark")
    file = os.path.join(benchmark_folder, config_file)
    with open(file, "r") as file:
        config = json.load(file)

    # Return only the nprobe value, defaulting to 1 if not found
    return config.get("nprobe", 1)


def get_index_type(table_name):
    # Connect to your PostgreSQL database
    conn = get_db_connection()
    cursor = conn.cursor()

    # SQL query to fetch index information for the table
    query = f"""
    SELECT indexdef FROM pg_indexes WHERE tablename = '{table_name}';
    """
    cursor.execute(query)
    indexes = cursor.fetchall()

    # Example of finding the index type based on the index definition
    for index in indexes:
        index_definition = index[0].lower()
        if 'ivfflat' in index_definition:
            return 'ivfflat'
        elif 'squidhnsw' in index_definition:
            return 'squidhnsw'
        elif 'vedahnsw' in index_definition:
            return 'vedahnsw'
        elif 'hnsw' in index_definition:
            return 'hnsw'

    # If no matching index type is found, return None or a default value
    return None


def drop_extra_tables():
    drop_prefilter_partition_tables()


def predicate_prefilter(user_id, query_vector, topk=5, statistics_type="sql"):
    if statistics_type == "sql":
        return predicate_prefilter_statistics_sql(user_id, query_vector, topk)
    elif statistics_type == "system": \
            return predicate_prefilter_statistics_system(user_id, query_vector, topk)


def predicate_postfilter(user_id, query_vector, topk=5, statistics_type="sql"):
    if statistics_type == "sql":
        return predicate_postfilter_statistics_sql(user_id, query_vector, topk)
    elif statistics_type == "system": \
            return predicate_postfilter_statistics_system(user_id, query_vector, topk)


def predicate_prefilter_statistics_sql(user_id, query_vector, topk=5):
    probes = get_nprobe_value()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"SET ivfflat.probes = {probes};")

    total_query_time = 0  # Variable to accumulate the total SQL query time
    total_blocks_accessed = 0  # Variable to accumulate the total blocks accessed

    vector_str = query_vector

    # Step 1: Retrieve all document_ids the user has permission to access using EXPLAIN ANALYZE
    explain_query = """
        EXPLAIN ANALYZE
        SELECT DISTINCT pa.document_id 
        FROM PermissionAssignment pa
        JOIN UserRoles ur ON pa.role_id = ur.role_id
        WHERE ur.user_id = %s
    """
    cur.execute(explain_query, [user_id])
    explain_plan = cur.fetchall()

    # ----------------------save query plan-----------start
    import inspect
    current_function_name = inspect.currentframe().f_code.co_name

    # save_query_plan(explain_plan, current_function_name)
    # ----------------------save query plan-------------end

    # Parse the execution time from EXPLAIN ANALYZE
    for row in explain_plan:
        if "Execution Time" in row[0]:
            query_time = float(row[0].split()[-2]) / 1000  # Convert ms to seconds
            total_query_time += query_time

    # Fetch accessible document_ids without EXPLAIN ANALYZE
    cur.execute(
        """
        SELECT DISTINCT pa.document_id 
        FROM PermissionAssignment pa
        JOIN UserRoles ur ON pa.role_id = ur.role_id
        WHERE ur.user_id = %s
        """,
        [user_id]
    )
    accessible_document_ids = cur.fetchall()
    accessible_document_ids = [doc_id for (doc_id,) in accessible_document_ids]

    if not accessible_document_ids:
        cur.close()
        conn.close()
        print(f"Total SQL query time: {total_query_time} seconds")
        return []  # No accessible documents for this user

    # Step 2: Query the closest vectors among the accessible document blocks using EXPLAIN ANALYZE
    explain_query = sql.SQL(
        """
        EXPLAIN ANALYZE
        SELECT db.block_id, db.block_content, db.vector <-> %s AS distance
        FROM documentblocks db
        WHERE db.document_id = ANY(%s)
        ORDER BY distance
        LIMIT %s
        """
    )
    cur.execute(explain_query, [vector_str, accessible_document_ids, topk])
    explain_plan = cur.fetchall()

    # ----------------------save query plan-----------start
    import inspect
    current_function_name = inspect.currentframe().f_code.co_name

    # save_query_plan(explain_plan, current_function_name)
    # ----------------------save query plan-------------end

    # Parse the execution time from EXPLAIN ANALYZE
    for row in explain_plan:
        if "Execution Time" in row[0]:
            query_time = float(row[0].split()[-2]) / 1000  # Convert ms to seconds
            total_query_time += query_time
        elif "rows=" in row[0]:
            # This line typically appears in the EXPLAIN output indicating rows processed
            # Depending on the exact format, you might need to adjust the parsing
            try:
                blocks_accessed = int(row[0].split("rows=")[1].split(" ")[0])
                total_blocks_accessed += blocks_accessed
            except (IndexError, ValueError):
                pass  # Handle cases where parsing fails

    # Perform the actual query without EXPLAIN ANALYZE to fetch results
    query = sql.SQL(
        """
        SELECT db.block_id, db.block_content, db.vector <-> %s AS distance
        FROM documentblocks db
        WHERE db.document_id = ANY(%s)
        ORDER BY distance
        LIMIT %s
        """
    )
    cur.execute(query, [vector_str, accessible_document_ids, topk])

    results = cur.fetchall()

    cur.execute("SELECT COUNT(block_id) FROM documentblocks;")
    total_blocks = cur.fetchone()[0]

    # Calculate selectivity
    block_selectivity = total_blocks_accessed / total_blocks if total_blocks else 0

    cur.close()
    conn.close()

    return results, total_query_time, block_selectivity


def predicate_prefilter_statistics_system(user_id, query_vector, topk=5):
    """
    Perform pre-filtered vector similarity search based on user permissions.
    """

    probes = get_nprobe_value()
    start_time = time.time()  # Start system time tracking

    # Establish database connection
    conn = get_db_connection()
    cur = conn.cursor()

    # Set the IVF flat probes parameter
    cur.execute(f"SET ivfflat.probes = {probes};")

    # Step 1: Retrieve all accessible document block IDs for the user
    cur.execute(
        """
        SELECT DISTINCT db.block_id
        FROM documentblocks db
        JOIN PermissionAssignment pa ON db.document_id = pa.document_id
        JOIN UserRoles ur ON pa.role_id = ur.role_id
        WHERE ur.user_id = %s
        """,
        [user_id]
    )
    accessible_block_ids = cur.fetchall()
    accessible_block_ids = [block_id for (block_id,) in accessible_block_ids]

    if not accessible_block_ids:
        return [], time.time() - start_time  # No accessible blocks

    # Step 2: Perform vector similarity search within the filtered blocks
    query = sql.SQL(
        """
        SELECT block_id, block_content, vector <-> %s AS distance
        FROM (
            SELECT block_id, block_content, vector 
            FROM documentblocks 
            WHERE block_id = ANY(%s)
        ) AS filtered_blocks
        ORDER BY distance ASC
        LIMIT %s
        """
    )

    cur.execute(query, [query_vector, accessible_block_ids, topk])

    results = cur.fetchall()

    # Close the database connection
    cur.close()
    conn.close()

    return results, time.time() - start_time


#
# def predicate_prefilter_statistics_system(user_id, query_vector, topk=5):
#     probes = get_nprobe_value()
#     import time
#     start_time = time.time()  # Start system time tracking
#     conn = get_db_connection()
#     cur = conn.cursor()
#     cur.execute(f"SET ivfflat.probes = {probes};")
#     # Convert query vector to string for the SQL query
#     vector_str = query_vector
#
#     # Step 1: Retrieve all document_ids that the user has permission to access
#     cur.execute(
#         """
#         SELECT DISTINCT pa.document_id
#         FROM PermissionAssignment pa
#         JOIN UserRoles ur ON pa.role_id = ur.role_id
#         WHERE ur.user_id = %s
#         """,
#         [user_id]
#     )
#     accessible_document_ids = cur.fetchall()
#     accessible_document_ids = [doc_id for (doc_id,) in accessible_document_ids]
#
#     if not accessible_document_ids:
#         return []  # No accessible documents for this user
#
#     # Step 2: Query the closest vectors among the accessible document blocks
#     query = sql.SQL(
#         """
#         SELECT db.block_id, db.block_content, db.vector <-> %s AS distance
#         FROM documentblocks db
#         WHERE db.document_id = ANY(%s)
#         ORDER BY distance
#         LIMIT %s
#         """
#     )
#
#     cur.execute(query, [vector_str, accessible_document_ids, topk])
#
#     results = cur.fetchall()
#     cur.close()
#     conn.close()
#
#     return results, time.time() - start_time


def predicate_postfilter_statistics_sql(user_id, query_vector, topk=5):
    probes = get_nprobe_value()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"SET ivfflat.probes = {probes};")

    total_query_time = 0  # Variable to accumulate the total SQL query time
    total_blocks_accessed = 0

    vector_str = query_vector

    # Combined query using EXPLAIN ANALYZE to retrieve closest vectors among accessible document blocks
    explain_query = """
        EXPLAIN ANALYZE
        SELECT db.block_id, db.block_content, db.vector <-> %s AS distance
        FROM documentblocks db
        WHERE db.document_id IN (
            SELECT DISTINCT pa.document_id 
            FROM PermissionAssignment pa
            JOIN UserRoles ur ON pa.role_id = ur.role_id
            WHERE ur.user_id = %s
        )
        ORDER BY distance
        LIMIT %s
    """
    cur.execute(explain_query, [vector_str, user_id, topk])
    explain_plan = cur.fetchall()

    # ----------------------save query plan-----------start
    import inspect
    current_function_name = inspect.currentframe().f_code.co_name

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
        elif "rows=" in line:
            # Extract the number of rows (blocks) accessed
            try:
                blocks_accessed = int(line.split("rows=")[1].split(" ")[0])
                total_blocks_accessed += blocks_accessed
            except (IndexError, ValueError):
                pass  # Handle unexpected format gracefully

    # Perform the actual query without EXPLAIN ANALYZE to fetch results
    query = """
        SELECT db.block_id, db.block_content, db.vector <-> %s AS distance
        FROM documentblocks db
        WHERE db.document_id IN (
            SELECT DISTINCT pa.document_id 
            FROM PermissionAssignment pa
            JOIN UserRoles ur ON pa.role_id = ur.role_id
            WHERE ur.user_id = %s
        )
        ORDER BY distance
        LIMIT %s
    """
    cur.execute(query, [vector_str, user_id, topk])

    results = cur.fetchall()

    # Step 3: Calculate the total number of blocks to compute selectivity
    total_blocks_query = "SELECT COUNT(block_id) FROM documentblocks;"
    cur.execute(total_blocks_query)
    total_blocks = cur.fetchone()[0]

    # Step 4: Calculate selectivity
    block_selectivity = (total_blocks_accessed / total_blocks) if total_blocks else 0

    cur.close()
    conn.close()

    return results, total_query_time, block_selectivity


def predicate_postfilter_statistics_system(user_id, query_vector, topk=5):
    probes = get_nprobe_value()
    import time
    start_time = time.time()  # Start system time tracking
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"SET ivfflat.probes = {probes};")
    # Convert query vector to string for the SQL query
    vector_str = query_vector

    # Combined query to retrieve closest vectors among accessible document blocks
    query = """
        SELECT db.block_id, db.block_content, db.vector <-> %s AS distance
        FROM documentblocks db
        WHERE db.document_id IN (
            SELECT DISTINCT pa.document_id 
            FROM PermissionAssignment pa
            JOIN UserRoles ur ON pa.role_id = ur.role_id
            WHERE ur.user_id = %s
        )
        ORDER BY distance
        LIMIT %s
    """

    cur.execute(query, [vector_str, user_id, topk])

    results = cur.fetchall()
    cur.close()
    conn.close()

    return results, time.time() - start_time


def _load_role_vectors_for_faiss(role_id):
    """
    Load all vectors and metadata for a specific role (for FAISS GPU caching).

    Args:
        role_id: Role ID to load vectors for

    Returns:
        (vectors, metadata) tuple where metadata is list of (block_id, document_id, block_content)
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        query = f"""
            SELECT block_id, document_id, block_content, vector
            FROM documentblocks_role_{role_id}
        """
        cur.execute(query)
        results = cur.fetchall()
    except Exception as e:
        cur.close()
        conn.close()
        return None, []

    cur.close()
    conn.close()

    if not results:
        return None, []

    # Remove duplicates by (block_id, document_id)
    unique_results = {}
    for row in results:
        block_id = row[0]
        document_id = row[1]
        key = (block_id, document_id)
        if key not in unique_results:
            unique_results[key] = row

    results = list(unique_results.values())

    # Extract vectors and metadata
    metadata = [(row[0], row[1], row[2]) for row in results]

    # Parse vector strings to numpy arrays
    vectors = []
    for row in results:
        vector_str = row[3]
        if isinstance(vector_str, str):
            vector_str = vector_str.strip('[]')
            vector = np.array([float(x) for x in vector_str.split(',')])
        else:
            vector = np.array(vector_str)
        vectors.append(vector)

    vectors = np.array(vectors, dtype=np.float32)
    return vectors, metadata


def _load_user_vectors_for_faiss(user_id, use_role_partition=True):
    """
    Load all vectors and metadata for a specific user (for FAISS GPU).

    Args:
        user_id: User ID to load vectors for
        use_role_partition: If True, use role partition tables (faster, avoids JOIN)
    """
    conn = get_db_connection()
    cur = conn.cursor()

    if use_role_partition:
        # Get user's roles
        cur.execute("SELECT role_id FROM userroles WHERE user_id = %s", [user_id])
        role_ids = [row[0] for row in cur.fetchall()]

        if not role_ids:
            cur.close()
            conn.close()
            return None, []

        # Query each role partition table (no JOIN needed!)
        all_results = []
        for role_id in role_ids:
            try:
                query = f"""
                    SELECT block_id, document_id, block_content, vector
                    FROM documentblocks_role_{role_id}
                """
                cur.execute(query)
                all_results.extend(cur.fetchall())
            except Exception:
                # Role partition table might not exist, fall back to full table
                use_role_partition = False
                break

        if use_role_partition:
            results = all_results
        else:
            # Fall back to JOIN approach
            query = """
                SELECT db.block_id, db.document_id, db.block_content, db.vector
                FROM documentblocks db
                JOIN PermissionAssignment pa ON db.document_id = pa.document_id
                JOIN UserRoles ur ON pa.role_id = ur.role_id
                WHERE ur.user_id = %s
            """
            cur.execute(query, [user_id])
            results = cur.fetchall()
    else:
        # Original JOIN approach
        query = """
            SELECT db.block_id, db.document_id, db.block_content, db.vector
            FROM documentblocks db
            JOIN PermissionAssignment pa ON db.document_id = pa.document_id
            JOIN UserRoles ur ON pa.role_id = ur.role_id
            WHERE ur.user_id = %s
        """
        cur.execute(query, [user_id])
        results = cur.fetchall()

    cur.close()
    conn.close()

    if not results:
        return None, []

    # Remove duplicates (user might have multiple roles accessing same block)
    # Use (block_id, document_id) as key since same block_id can have different document_ids
    unique_results = {}
    for row in results:
        block_id = row[0]
        document_id = row[1]
        key = (block_id, document_id)
        if key not in unique_results:
            unique_results[key] = row

    results = list(unique_results.values())

    # Extract vectors and metadata
    metadata = [(row[0], row[1], row[2]) for row in results]  # (block_id, document_id, block_content)

    # Parse vector strings to numpy arrays
    vectors = []
    for row in results:
        vector_str = row[3]
        # Convert PostgreSQL vector format "[1,2,3]" to numpy array
        if isinstance(vector_str, str):
            vector_str = vector_str.strip('[]')
            vector = np.array([float(x) for x in vector_str.split(',')])
        else:
            vector = np.array(vector_str)
        vectors.append(vector)

    vectors = np.array(vectors, dtype=np.float32)
    return vectors, metadata


def _ground_truth_func_faiss_gpu(user_id, query_vector, topk=5):
    """GPU-accelerated ground truth using FAISS."""
    try:
        import faiss
    except ImportError:
        print("Warning: faiss not installed, falling back to PostgreSQL ground truth")
        return _ground_truth_func_postgres(user_id, query_vector, topk)

    global _faiss_user_data_cache, _faiss_query_counter

    # Build cache if needed
    if _faiss_user_data_cache is None:
        _faiss_user_data_cache = {}

    # Load user data if not cached
    if user_id not in _faiss_user_data_cache:
        vectors, metadata = _load_user_vectors_for_faiss(user_id)
        if vectors is None:
            return []

        # Build FAISS index
        dimension = vectors.shape[1]
        index_flat = faiss.IndexFlatL2(dimension)

        # Try GPU first, fall back to CPU if it fails
        try:
            res = faiss.StandardGpuResources()
            res.setTempMemory(128 * 1024 * 1024)  # Reduce temp memory to 128MB
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
            gpu_index.add(vectors)
            print(f"FAISS GPU index created for user {user_id} with {len(vectors)} vectors")

            _faiss_user_data_cache[user_id] = {
                'index': gpu_index,
                'metadata': metadata,
                'res': res,
                'is_gpu': True
            }
        except RuntimeError as e:
            print(f"GPU allocation failed ({e}), using CPU FAISS instead")
            index_flat.add(vectors)
            _faiss_user_data_cache[user_id] = {
                'index': index_flat,
                'metadata': metadata,
                'is_gpu': False
            }

    cache_entry = _faiss_user_data_cache[user_id]
    index = cache_entry['index']
    metadata = cache_entry['metadata']

    # Parse query vector
    if isinstance(query_vector, str):
        query_vector = query_vector.strip('[]')
        query_vec = np.array([[float(x) for x in query_vector.split(',')]], dtype=np.float32)
    else:
        query_vec = np.array([query_vector], dtype=np.float32)

    # Search using FAISS (GPU or CPU)
    distances, indices = index.search(query_vec, topk)

    # Format results to match PostgreSQL output
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        block_id, document_id, block_content = metadata[idx]
        results.append((block_id, document_id, block_content, float(dist)))

    # Log completion with progress
    _faiss_query_counter['completed'] += 1
    gpu_status = "GPU" if cache_entry.get('is_gpu', False) else "CPU"
    total = _faiss_query_counter['total']
    completed = _faiss_query_counter['completed']
    if total > 0:
        print(f"✓ FAISS {gpu_status} query done (user {user_id}) [{completed}/{total}]")
    else:
        print(f"✓ FAISS {gpu_status} query done (user {user_id})")

    return results


def _format_ground_truth_results(rows):
    """Normalize ground-truth rows so FAISS/Postgres paths return identical shapes."""
    formatted = []
    for row in rows:
        if row is None or len(row) < 4:
            continue
        block_id = row[0]
        document_id = row[1]
        block_content = row[2]
        distance = row[3]
        try:
            distance = float(distance)
        except (TypeError, ValueError):
            pass
        formatted.append((block_id, document_id, block_content, distance))
    return formatted


def _ground_truth_func_postgres(user_id, query_vector, topk=5, use_role_partition=False):
    """PostgreSQL-based ground truth (brute force), optionally using role partitions."""
    global _faiss_query_counter

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SET enable_indexscan = off;")
    cur.execute("SET enable_bitmapscan = off;")
    cur.execute("SET enable_indexonlyscan = off;")

    # Convert query vector to string for the SQL query
    vector_str = query_vector

    if use_role_partition:
        # Get user's roles
        cur.execute("SELECT role_id FROM userroles WHERE user_id = %s", [user_id])
        role_ids = [row[0] for row in cur.fetchall()]

        if not role_ids:
            cur.close()
            conn.close()
            return []

        # Query each role partition table and collect results
        all_results = []
        for role_id in role_ids:
            try:
                query = f"""
                    SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
                    FROM documentblocks_role_{role_id}
                    ORDER BY distance
                """
                cur.execute(query, [vector_str])
                all_results.extend(cur.fetchall())
            except Exception:
                # Role partition doesn't exist, fall back to JOIN
                conn.rollback()
                cur.execute("SET enable_indexscan = off;")
                cur.execute("SET enable_bitmapscan = off;")
                cur.execute("SET enable_indexonlyscan = off;")
                use_role_partition = False
                break

        if use_role_partition:
            # Sort all results by distance and take top-k
            all_results.sort(key=lambda x: x[3])  # Sort by distance
            results = _format_ground_truth_results(all_results[:topk])
        else:
            # Fall back to JOIN approach
            query = """
                SELECT db.block_id, db.document_id, db.block_content, db.vector <-> %s AS distance
                FROM documentblocks db
                JOIN PermissionAssignment pa ON db.document_id = pa.document_id
                JOIN UserRoles ur ON pa.role_id = ur.role_id
                WHERE ur.user_id = %s
                ORDER BY distance
                LIMIT %s
            """
            cur.execute(query, [vector_str, user_id, topk])
            results = _format_ground_truth_results(cur.fetchall())
    else:
        # Original JOIN approach
        query = """
            SELECT db.block_id, db.document_id, db.block_content, db.vector <-> %s AS distance
            FROM documentblocks db
            JOIN PermissionAssignment pa ON db.document_id = pa.document_id
            JOIN UserRoles ur ON pa.role_id = ur.role_id
            WHERE ur.user_id = %s
            ORDER BY distance
            LIMIT %s
        """
        cur.execute(query, [vector_str, user_id, topk])
        results = _format_ground_truth_results(cur.fetchall())

    cur.execute("RESET enable_indexscan;")
    cur.execute("RESET enable_bitmapscan;")
    cur.execute("RESET enable_indexonlyscan;")

    cur.close()
    conn.close()

    # Log completion with progress
    _faiss_query_counter['completed'] += 1
    total = _faiss_query_counter['total']
    completed = _faiss_query_counter['completed']
    partition_info = "partition" if use_role_partition else "JOIN"
    if total > 0:
        print(f"✓ PostgreSQL query done ({partition_info}, user {user_id}) [{completed}/{total}]")
    else:
        print(f"✓ PostgreSQL query done ({partition_info}, user {user_id})")

    return results


def ground_truth_func(user_id, query_vector, topk=5):
    """
    Ground truth function with optional GPU acceleration.

    Checks config.json for 'use_gpu_groundtruth' flag.
    If True and FAISS is available, uses GPU acceleration.
    Otherwise falls back to PostgreSQL brute force.
    """
    # Read config
    config_path = os.path.join(project_root, "config.json")
    use_gpu = False

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            use_gpu = config.get('use_gpu_groundtruth', False)

    if use_gpu:
        return _ground_truth_func_faiss_gpu(user_id, query_vector, topk)
    else:
        return _ground_truth_func_postgres(user_id, query_vector, topk)


def clear_faiss_cache():
    """Clear FAISS GPU cache to free memory."""
    global _faiss_user_data_cache, _faiss_role_data_cache
    _faiss_user_data_cache = None
    _faiss_role_data_cache = None
    print("FAISS cache cleared")


def clear_ground_truth_cache():
    """Clear ground truth cache file."""
    cache_file = os.path.join(project_root, "basic_benchmark", "ground_truth_cache.json")
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print("Ground truth cache cleared")
    else:
        print("No ground truth cache to clear")


def _to_json_serializable(obj):
    """Recursively convert objects (e.g. memoryview, numpy) into JSON-safe types."""
    if isinstance(obj, dict):
        return {key: _to_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_json_serializable(value) for value in obj]
    if isinstance(obj, tuple):
        return [_to_json_serializable(value) for value in obj]
    if isinstance(obj, memoryview):
        data = obj.tobytes()
        try:
            return data.decode('utf-8')
        except UnicodeDecodeError:
            return list(data)
    if isinstance(obj, bytes):
        try:
            return obj.decode('utf-8')
        except UnicodeDecodeError:
            return list(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    return obj


def _save_ground_truth_cache(queries, results, cache_file):
    """Save ground truth results to cache file."""
    try:
        cache_data = []
        for i, query in enumerate(queries):
            cache_data.append({
                'query': _to_json_serializable({
                    'user_id': query['user_id'],
                    'query_vector': query['query_vector'],
                    'topk': query.get('topk', 5)
                }),
                'ground_truth': _to_json_serializable(results[i])
            })

        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        print(f"✓ Ground truth cached to {cache_file}")
    except Exception as e:
        print(f"Warning: Failed to save ground truth cache: {e}")


def set_ground_truth_total_queries(total):
    """Set total number of queries for progress tracking."""
    global _faiss_query_counter
    _faiss_query_counter['total'] = total
    _faiss_query_counter['completed'] = 0


def _ground_truth_workers(workers=None):
    if workers is not None:
        try:
            return max(1, int(workers))
        except (TypeError, ValueError):
            return 1

    raw_workers = os.environ.get("GROUND_TRUTH_WORKERS")
    if not raw_workers:
        return 8
    try:
        return max(1, int(raw_workers))
    except ValueError:
        print(f"Invalid GROUND_TRUTH_WORKERS={raw_workers!r}; using 8")
        return 8


def _ground_truth_func_postgres_batch(queries, workers=None):
    workers = min(_ground_truth_workers(workers), max(1, len(queries)))
    if workers <= 1 or len(queries) <= 1:
        return [ground_truth_func(q['user_id'], q['query_vector'], q.get('topk', 5)) for q in queries]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(f"PostgreSQL ground truth batch: {len(queries)} queries, workers={workers}")
    results = [None] * len(queries)
    report_every = max(1, min(50, len(queries) // 20 or 1))

    def run_one(index, query):
        return index, ground_truth_func(query['user_id'], query['query_vector'], query.get('topk', 5))

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_one, index, query) for index, query in enumerate(queries)]
        for future in as_completed(futures):
            index, query_result = future.result()
            results[index] = query_result
            completed += 1
            if completed % report_every == 0 or completed == len(queries):
                print(f"✓ PostgreSQL ground truth batch progress: [{completed}/{len(queries)}]")

    return results


def ground_truth_func_batch(queries, use_faiss=None, use_cache=True, workers=None):
    """
    Batch ground truth computation for multiple queries (much faster).

    Args:
        queries: List of query dicts with 'user_id', 'query_vector', 'topk'
        use_faiss: If True, use FAISS GPU batch search. If None, read from config.json
        use_cache: If True, cache ground truth results to disk
        workers: PostgreSQL fallback worker count. Defaults to GROUND_TRUTH_WORKERS or 8.

    Returns:
        List of results corresponding to each query
    """
    # Check cache first
    cache_file = os.path.join(project_root, "basic_benchmark", "ground_truth_cache.json")

    if use_cache and os.path.exists(cache_file):
        print("Loading ground truth from cache...")
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)

            if len(cached_data) >= len(queries):
                subset = cached_data[:len(queries)]
                if len(cached_data) > len(queries):
                    print(
                        f"✓ Ground truth cache larger than needed ({len(cached_data)}); "
                        f"using first {len(queries)} entries"
                    )
                else:
                    print(f"✓ Ground truth cache valid! Loaded {len(subset)} results")

                cached_results = []
                for entry in subset:
                    if isinstance(entry, dict):
                        cached_results.append(entry.get("ground_truth", []))
                    else:
                        cached_results.append(entry)
                return cached_results
            else:
                print(f"Cache size mismatch ({len(cached_data)} vs {len(queries)}), recomputing...")
        except Exception as e:
            print(f"Error loading cache: {e}, recomputing...")

    # Read config if not specified
    if use_faiss is None:
        config_path = os.path.join(project_root, "config.json")
        use_faiss = False
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                use_faiss = config.get('use_gpu_groundtruth', False)

    if not use_faiss:
        results = _ground_truth_func_postgres_batch(queries, workers=workers)
        if use_cache:
            _save_ground_truth_cache(queries, results, cache_file)
        return results
    else:
        # Use FAISS for ground truth
        try:
            import faiss
        except ImportError:
            print("Warning: faiss not installed, falling back to single query processing")
            results = _ground_truth_func_postgres_batch(queries, workers=workers)
            # Save cache and return
            if use_cache:
                _save_ground_truth_cache(queries, results, cache_file)
            return results

        global _faiss_user_data_cache, _faiss_role_data_cache, _faiss_query_counter

        # Group queries by user_id for batch processing
        from collections import defaultdict
        queries_by_user = defaultdict(list)
        for i, query in enumerate(queries):
            user_id = query['user_id']
            queries_by_user[user_id].append((i, query))

        # Initialize results array
        all_results = [None] * len(queries)

        # Build cache if needed
        if _faiss_role_data_cache is None:
            _faiss_role_data_cache = {}

    print(f"Batch processing {len(queries)} queries for {len(queries_by_user)} users...")

    # Step 1: Pre-load all role indexes (much faster - shared across users!)
    conn = get_db_connection()
    cur = conn.cursor()

    # Get all unique roles for all users
    all_role_ids = set()
    for user_id in queries_by_user.keys():
        cur.execute("SELECT role_id FROM userroles WHERE user_id = %s", [user_id])
        role_ids = [row[0] for row in cur.fetchall()]
        all_role_ids.update(role_ids)

    cur.close()
    conn.close()

    roles_to_load = [rid for rid in all_role_ids if rid not in _faiss_role_data_cache]
    if roles_to_load:
        print(f"Pre-loading {len(roles_to_load)} role indexes in parallel (will be shared across users)...")

        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        cache_lock = threading.Lock()
        loaded_count = [0]

        def load_role_index(role_id):
            vectors, metadata = _load_role_vectors_for_faiss(role_id)
            if vectors is None:
                return role_id, None

            # Build FAISS index for this role
            dimension = vectors.shape[1]
            index_flat = faiss.IndexFlatL2(dimension)

            # Note: GPU transfer is not thread-safe, so we'll do CPU index in parallel
            # and only transfer to GPU in the main thread if needed
            index_flat.add(vectors)

            result = {
                'index': index_flat,
                'metadata': metadata,
                'vectors': vectors,  # Keep vectors for potential GPU transfer
                'is_gpu': False
            }

            with cache_lock:
                loaded_count[0] += 1
                if loaded_count[0] % 10 == 0 or loaded_count[0] == len(roles_to_load):
                    print(f"  [{loaded_count[0]}/{len(roles_to_load)}] Loaded role {role_id} ({len(vectors)} vectors)")

            return role_id, result

        # Load all role data in parallel (CPU only)
        MAX_WORKERS = 8
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(load_role_index, rid): rid for rid in roles_to_load}

            for future in as_completed(futures):
                role_id, result = future.result()
                if result is not None:
                    _faiss_role_data_cache[role_id] = result

        print(f"✓ All {len(roles_to_load)} role indexes loaded! Now transferring to GPU...")

        # Now transfer to GPU sequentially (GPU operations are not thread-safe)
        gpu_count = 0
        for role_id in roles_to_load:
            if role_id not in _faiss_role_data_cache:
                continue

            cache_entry = _faiss_role_data_cache[role_id]
            if cache_entry['is_gpu']:
                continue

            try:
                vectors = cache_entry['vectors']
                dimension = vectors.shape[1]

                res = faiss.StandardGpuResources()
                res.setTempMemory(128 * 1024 * 1024)
                index_flat = faiss.IndexFlatL2(dimension)
                gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
                gpu_index.add(vectors)

                # Update cache with GPU index
                _faiss_role_data_cache[role_id] = {
                    'index': gpu_index,
                    'metadata': cache_entry['metadata'],
                    'res': res,
                    'is_gpu': True
                }
                gpu_count += 1

                if gpu_count % 20 == 0:
                    print(f"  GPU transfer: {gpu_count}/{len(roles_to_load)}")
            except RuntimeError as e:
                # Keep CPU version
                del cache_entry['vectors']  # Free memory
                pass

        print(f"✓ GPU transfer complete: {gpu_count} on GPU, {len(roles_to_load) - gpu_count} on CPU")

    # Process users in batches to show better progress
    total_users = len(queries_by_user)
    processed_users = 0
    USER_BATCH_SIZE = 20  # Report progress every 20 users (more frequent updates)

    # Process each user's queries in batch
    for user_id, user_queries in queries_by_user.items():
        # Get user's roles
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT role_id FROM userroles WHERE user_id = %s", [user_id])
        user_role_ids = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()

        if not user_role_ids:
            for idx, query in user_queries:
                all_results[idx] = []
            continue

        # Prepare batch query vectors
        query_vectors = []
        topks = []
        for idx, query in user_queries:
            query_vector = query['query_vector']
            topk = query.get('topk', 5)

            # Parse query vector
            if isinstance(query_vector, str):
                query_vector = query_vector.strip('[]')
                query_vec = np.array([float(x) for x in query_vector.split(',')], dtype=np.float32)
            else:
                query_vec = np.array(query_vector, dtype=np.float32)

            query_vectors.append(query_vec)
            topks.append(topk)

        query_matrix = np.array(query_vectors, dtype=np.float32)

        # Search each role's index and merge results
        for i, (idx, query) in enumerate(user_queries):
            topk = topks[i]
            query_vec = query_matrix[i:i+1]  # Single query vector

            # Collect results from all roles
            all_role_results = []
            for role_id in user_role_ids:
                if role_id not in _faiss_role_data_cache:
                    continue

                role_cache = _faiss_role_data_cache[role_id]
                role_index = role_cache['index']
                role_metadata = role_cache['metadata']

                # Search this role's index (get more than topk to account for duplicates)
                distances, indices_result = role_index.search(query_vec, min(len(role_metadata), topk * 3))

                # Format results
                for j in range(len(indices_result[0])):
                    result_idx = indices_result[0][j]
                    if result_idx < 0 or result_idx >= len(role_metadata):
                        continue
                    dist = distances[0][j]
                    block_id, document_id, block_content = role_metadata[result_idx]
                    all_role_results.append((block_id, document_id, block_content, float(dist)))

            # Remove duplicates and sort by distance
            unique_results = {}
            for result in all_role_results:
                block_id, doc_id, content, dist = result
                key = (block_id, doc_id)
                if key not in unique_results or dist < unique_results[key][3]:
                    unique_results[key] = result

            # Sort by distance and take topk
            sorted_results = sorted(unique_results.values(), key=lambda x: x[3])[:topk]
            all_results[idx] = sorted_results

        # Update progress
        _faiss_query_counter['completed'] += len(user_queries)
        processed_users += 1

        # Only print every USER_BATCH_SIZE users or at the end
        if processed_users % USER_BATCH_SIZE == 0 or processed_users == total_users:
            total = _faiss_query_counter['total']
            completed = _faiss_query_counter['completed']
            avg_queries_per_user = completed / processed_users
            print(f"✓ FAISS role-cache batch: [{processed_users}/{total_users}] users, [{completed}/{total}] queries, avg {avg_queries_per_user:.1f} q/user, {len(_faiss_role_data_cache)} roles cached")

    if use_cache:
        _save_ground_truth_cache(queries, all_results, cache_file)
    return all_results


def prepare_query_dataset(regenerate=True, num_queries=1000, query_dataset_path=None):
    query_dataset_file = query_dataset_path or os.path.join(project_root, "basic_benchmark", "query_dataset.json")
    # Check if the dataset file exists
    if not os.path.exists(query_dataset_file) or regenerate:
        generate_query_dataset(num_queries=num_queries, topk=10, output_file=query_dataset_file)

    # Load queries from the dataset
    queries = load_queries_from_dataset(query_dataset_file)
    if len(queries) < num_queries:
        generate_query_dataset(num_queries=num_queries, topk=10, output_file=query_dataset_file)
        queries = load_queries_from_dataset(query_dataset_file)

    return queries[:num_queries]


def compute_recall(true_results, predicted_results):
    true_set = set(true_results)
    predicted_set = set(predicted_results)
    intersection_size = len(true_set & predicted_set)

    recall = intersection_size / len(true_set)
    return recall


def load_function(import_path: str) -> Callable:
    """
    Dynamically loads a function from a given import path.

    Args:
        import_path (str): The full import path of the function (e.g., 'module.submodule.function').

    Returns:
        Callable: The imported function.
    """
    module_path, function_name = import_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, function_name)


def save_query_plan(explain_plan, current_function_name):
    # ----------------------save query plan-----------start
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = os.path.join(os.getcwd(), f"{current_function_name}.txt")

    with open(file_path, 'a') as f:
        f.write(f"\n--- Query Plan at {current_time} ---\n")
        for row in explain_plan:
            f.write(row[0] + "\n")

    print(f"EXPLAIN ANALYZE output saved to {file_path}")

    # ----------------------save query plan-------------end


def run_test(
        queries,
        condition,
        iterations=1,
        output_file=None,
        enable_index=False,
        index_type="ivfflat",
        statistics_type="sql",
        generator_type="tree-based",
        record_recall=True,
        warm_up=True,
        use_ground_truth_cache=True,
        queries_num=None,
):
    """
    Runs an experiment with the provided search function and computes recall, query time, and optionally block selectivity.

    Args:
        queries (list): List of queries to execute. Each query should be a dictionary with keys:
                       - "user_id": The ID of the user.
                       - "query_vector": The vector to search against.
                       - "topk" (optional): The number of top results to return (default is 5).
        condition (str): The condition under which to run the search. Possible values:
                         - "user", "role", "combination", "alg1", "alg2",
                         - "postfilter", "prefilter", "rls", "partition_proposal".
        iterations (int): Number of iterations to run (default is 1).
        output_file (str, optional): Path to the output JSON file (default is based on condition).
        enable_index (bool): Whether to enable indexing (default is False).
        statistics_type (str): Type of statistics to collect ("sql" or "system").
        strategy (optional): Partitioning strategy to use for the search function, if applicable.

    Returns:
        None: Results are saved to the specified output_file.
    """

    # Validate condition
    if condition not in CONDITION_CONFIG:
        raise ValueError(f"Invalid condition specified: {condition}")

    # Retrieve configuration based on condition
    config = CONDITION_CONFIG[condition]
    search_func = load_function(config["search_func_path"])
    space_calc_func = load_function(config["space_calc_func_path"])
    extra_params = config["extra_params"].copy()

    # Set default output_file if not provided
    if output_file is None:
        import efconfig
        ef_value = getattr(efconfig, 'ef_search', 'adaptive')
        if enable_index:
            filename = f"{condition}_{index_type}_{statistics_type}_{generator_type}_{ef_value}_avg_results.json"
        else:
            filename = f"{condition}_{index_type}_{statistics_type}_{generator_type}_avg_results.json"
        output_file = os.path.join(RESULT_DIR, filename)

    # If queries_num is specified in config and not overridden, set it
    if 'queries_num' in config["extra_params"] and 'queries_num' not in extra_params:
        extra_params["queries_num"] = config["extra_params"]["queries_num"]

    # Initialize aggregation variables
    all_results = []
    total_recall = 0
    total_time = 0
    total_selectivity = 0  # Only relevant for 'sql' statistics_type

    # Run the experiment
    experiment_result = run_search_experiment(
        queries=queries,
        search_func=search_func,
        statistics_type=statistics_type,
        queries_num=queries_num if queries_num is not None else extra_params.get("queries_num"),
        generator_type=generator_type,
        enable_index=enable_index,
        iterations=iterations,
        index_type=index_type,
        record_recall=record_recall,
        warm_up=warm_up,
        use_ground_truth_cache=use_ground_truth_cache,
    )

    # Extract average recall and query time from run_search_experiment
    # Assuming run_search_experiment returns a dict with these keys
    avg_recall = experiment_result.get("avg_recall", 0)
    avg_query_time = experiment_result.get("avg_query_time", 0)
    avg_block_selectivity = experiment_result.get("avg_block_selectivity", 0) if statistics_type == "sql" else None

    # Aggregate the results
    all_results.extend(experiment_result.get("all_results", []))
    total_recall += avg_recall
    total_time += avg_query_time
    if statistics_type == "sql":
        total_selectivity += avg_block_selectivity

    # Calculate final averages
    avg_recall_final = total_recall
    avg_query_time_final = total_time
    avg_block_selectivity_final = total_selectivity

    print("The time unit is seconds")
    print(f"Average Recall: {avg_recall_final:.4f}")
    print(f"Average Query Time: {avg_query_time_final:.4f} seconds")

    # Calculate space used based on condition
    space_used_mb = space_calc_func(condition, enable_index=enable_index)

    print(f"Space used: {space_used_mb:.2f} MB")

    # Aggregate results for JSON output
    results_data = {
        "condition": condition,
        "iterations": iterations,
        "enable_index": enable_index,
        "average_results": {
            "avg_recall": avg_recall_final,
            "avg_query_time": avg_query_time_final,
        },
        "space_used_mb": space_used_mb,
        "index_type": index_type,
        "statistics_type": statistics_type,
        "generator_type": generator_type,
    }

    if statistics_type == "sql":
        results_data["average_results"]["avg_block_selectivity"] = avg_block_selectivity_final

    # Write results to JSON file
    os.makedirs(os.path.dirname(output_file),exist_ok=True)
    with open(output_file, "w") as json_file:
        json.dump(results_data, json_file, indent=4)

    print(f"Results saved to {output_file}")


def run_search_experiment(queries, search_func, queries_num=None, statistics_type="sql",
                          generator_type="tree-based", enable_index=False, index_type="ivfflat", iterations=1,
                          record_recall=True, plot=False, warm_up=True, use_ground_truth_cache=True):
    """
    Runs the search experiment and returns aggregated results.

    Args:
        queries (list): List of query dictionaries.
        search_func (function): The search function to execute.
        queries_num (int, optional): Number of queries to process.
        statistics_type (str): Type of statistics to collect ("sql" or "system").

    Returns:
        dict: Contains 'all_results', 'avg_recall', 'avg_query_time', and optionally 'avg_block_selectivity'.
    """
    all_results = []
    total_recall = 0
    total_query_time = 0
    processed_queries = 0

    actual_queries = queries_num if queries_num else len(queries)
    queries_to_process = queries[:actual_queries]

    # Batch compute all ground truths first (much faster)
    ground_truth_results_map = {}
    if record_recall:
        set_ground_truth_total_queries(len(queries_to_process))
        print(f"Computing ground truth for {len(queries_to_process)} queries in batch mode...")
        ground_truth_results_list = ground_truth_func_batch(queries_to_process, use_cache=use_ground_truth_cache)
        ground_truth_results_map = {i: gt for i, gt in enumerate(ground_truth_results_list)}
        print(f"Ground truth computation complete!")

    recalls = []
    for i, query in enumerate(queries_to_process):
        user_id = query["user_id"]
        query_vector = query["query_vector"]
        topk = query.get("topk", 5)

        # Progress logging every 100 queries
        if (i + 1) % 100 == 0:
            print(f"Progress: {i + 1}/{len(queries_to_process)} queries processed")

        query_total_recall = 0
        query_total_time = 0
        ground_truth_results = ground_truth_results_map.get(i, None) if record_recall else None
        for _ in range(iterations):
            # Run search function
            if warm_up:
                for _ in range(2):
                    search_results = search_func(
                        user_id=user_id,
                        query_vector=query_vector,
                        topk=topk,
                        statistics_type=statistics_type
                    )

            search_results = search_func(
                user_id=user_id,
                query_vector=query_vector,
                topk=topk,
                statistics_type=statistics_type
            )

            # Unpack search results based on statistics_type
            results, query_time = search_results

            if record_recall and len(queries_to_process) <= 20:
                print(f"[QDTree-Debug] Query {i} user {user_id} search results: {results}")
                if ground_truth_results is not None:
                    print(f"[QDTree-Debug] Query {i} user {user_id} ground truth: {ground_truth_results}")

            if record_recall:
                # Compute recall
                predicted_results = set((res[1], res[0]) for res in results)  # (document_id, block_id)
                true_results = set((gt[1], gt[0]) for gt in ground_truth_results)  # (document_id, block_id)
                recall = compute_recall(true_results, predicted_results)
                if recall == 0:
                    debug = 1
                recalls.append(recall)
                query_total_recall += recall
                # print(f"{user_id}:{recall}")
            query_total_time += query_time

        # Calculate averages for this query after iterations
        avg_recall = query_total_recall / iterations if iterations else 0
        avg_query_time = query_total_time / iterations if iterations else 0

        # Aggregate results
        all_results.append({
            "user_id": user_id,
            "query_vector": query_vector,
            "recall": avg_recall,
            "query_time": avg_query_time,
            "qps": 1 / avg_query_time if avg_query_time > 0 else 0,
        })

        total_recall += avg_recall
        total_query_time += avg_query_time

        processed_queries += 1


    # Calculate averages
    avg_recall = total_recall / processed_queries if processed_queries else 0
    avg_query_time = total_query_time / processed_queries if processed_queries else 0

    output_file = os.path.join(
        RESULT_DIR,
        f"{search_func.__name__}_{generator_type}_{enable_index}_{index_type}_results.json"
    )
    #
    # output_file = f"{search_func.__name__}_{generator_type}_{enable_index}_{index_type}_results.json"
    with open(output_file, 'w') as json_file:
        json.dump(all_results, json_file, indent=4)

    return {
        "avg_recall": avg_recall,
        "avg_query_time": avg_query_time,
    }
