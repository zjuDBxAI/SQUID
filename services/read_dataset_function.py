import hashlib
import json
import re
import sys
import os
import random
import tarfile
import shutil
from concurrent.futures.process import ProcessPoolExecutor
from datetime import datetime
from psycopg2 import Binary
import psycopg2
from datasets import load_dataset
import numpy as np
from collections import Counter

from basic_benchmark.generate_queries import calculate_block_selectivity, add_query_block_selectivity_to_json
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from services.rbac_generator.tree_based_rbac_data_generator import TreeBasedRBACDataGenerator
from services.embedding_service import generate_embedding
from services.config import get_db_connection, get_dataset_path
from concurrent.futures import ThreadPoolExecutor, as_completed


SIFT_DOCUMENT_VECTOR_COUNT = 100  # Group 100 vectors into a single synthetic document
SIFT10M_FEATURES_MEMBER = "SIFT10M/SIFT10Mfeatures.mat"


def store_document(document_id, document_name):
    conn = get_db_connection()
    cur = conn.cursor()

    created_at = datetime.now()
    updated_at = created_at

    cur.execute(
        "INSERT INTO documents (document_id, document_name, created_at, updated_at) VALUES (%s, %s, %s, %s) ON CONFLICT (document_id) DO NOTHING",
        (document_id, document_name, created_at, updated_at)
    )

    conn.commit()
    cur.close()
    conn.close()
    return {"message": "Document stored successfully"}


def store_document_block_duplication(block_id, document_id, block_content, vector):
    conn = get_db_connection()
    cur = conn.cursor()

    # hash_value = hashlib.sha1(block_content.encode('utf-8')).hexdigest()
    hash_value = "dbbbf35c7b7d74ece496cbb7511503a1c7cd724f"
    #
    # # reduplication
    # cur.execute(
    #     "SELECT block_id FROM documentblocks WHERE hash_value = %s",
    #     (hash_value,)
    # )
    # result = cur.fetchone()

    # if result is None:
    cur.execute(
        "INSERT INTO documentblocks (block_id, document_id, block_content, hash_value, vector) VALUES (%s, %s, %s, %s, %s)",
        (block_id, document_id, block_content, hash_value, vector)
    )

    conn.commit()
    cur.close()
    conn.close()
    return {"message": "Document block stored successfully"}


def store_document_block_duplication_bulk(document_blocks):
    conn = get_db_connection()
    cur = conn.cursor()

    # Prepare insert queries
    insert_documents_query = """
        INSERT INTO documents (document_id, document_name, created_at, updated_at)
        VALUES (%s, %s, %s, %s) ON CONFLICT (document_id) DO NOTHING
    """

    insert_documentblocks_query = """
        INSERT INTO documentblocks (block_id, document_id, block_content, hash_value, vector)
        VALUES (%s, %s, %s, %s, %s)
    """

    # Prepare data for documents table
    created_at = datetime.now()
    documents_data = [(document_id, f"Document {document_id}", created_at, created_at) for _, document_id, _, _, _ in
                      document_blocks]


    try:
        # Bulk insert into documents table
        cur.executemany(insert_documents_query, documents_data)

        # Bulk insert into documentblocks table
        cur.executemany(insert_documentblocks_query, document_blocks)

        # Commit all changes
        conn.commit()
        print(
            f"Inserted {len(document_blocks)} document blocks and {len(set(doc[0] for doc in documents_data))} documents successfully.")

    except Exception as e:
        print(f"Failed to insert document blocks or documents: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def store_document_block(block_id, document_id, block_content, vector):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO documentblocks (block_id, document_id, block_content, vector) VALUES (%s, %s, %s, %s) ON CONFLICT (block_id) DO NOTHING",
        (block_id, document_id, block_content, vector)
    )

    conn.commit()
    cur.close()
    conn.close()
    return {"message": "Document block stored successfully"}


def insert_permission_assignments(permission_assignment_data):
    conn = get_db_connection()
    cur = conn.cursor()

    for role_id, document_id in permission_assignment_data:
        cur.execute(
            "INSERT INTO PermissionAssignment (role_id, document_id) VALUES (%s, %s)",
            (role_id, document_id)
        )

    conn.commit()
    cur.close()
    conn.close()
    return {"message": "Permission assignments inserted successfully"}


def insert_user_roles(user_role_data):
    conn = get_db_connection()
    cur = conn.cursor()

    for user_id, role_id in user_role_data:
        cur.execute(
            "INSERT INTO user_roles (user_id, role_id) VALUES (%s, %s)",
            (user_id, role_id)
        )

    conn.commit()
    cur.close()
    conn.close()
    return {"message": "User roles inserted successfully"}


def store_rbac_data(users, roles, user_roles, permission_assignments):
    conn = get_db_connection()
    cur = conn.cursor()

    # Store Users
    for user in users:
        cur.execute(
            "INSERT INTO Users (user_id, user_name) VALUES (%s, %s)",
            (user['user_id'], user['user_name'])
        )

    # Store Roles
    for role in roles:
        cur.execute(
            "INSERT INTO Roles (role_id, role_name) VALUES (%s, %s)",
            (role.role_id, role.role_name)
        )

    # Store UserRoles
    for user_id, role_id in user_roles:
        cur.execute(
            "INSERT INTO UserRoles (user_id, role_id) VALUES (%s, %s)",
            (user_id, role_id)
        )

    # Store PermissionAssignments
    for role_id, document_id in permission_assignments:
        cur.execute(
            "INSERT INTO PermissionAssignment (role_id, document_id) VALUES (%s, %s)",
            (role_id, document_id)
        )

    conn.commit()
    cur.close()
    conn.close()


def clean_block_content(block_content):
    """
    Clean and prepare the block_content for database insertion.
    :param block_content: The abstract string from JSON.
    :return: Cleaned and prepared string.
    """
    if not block_content:
        return None

    # Step 1: Remove leading/trailing whitespace
    block_content = block_content.strip()

    # Step 2: Replace newline characters with space
    block_content = block_content.replace("\n", " ")

    # Step 3: Decode LaTeX-style escaped characters (optional)
    # Example: Replace `\\` with `\`
    block_content = re.sub(r"\\\\", r"\\", block_content)

    # Step 4: Ensure the content is UTF-8 encoded
    try:
        block_content = block_content.encode('utf-8').decode('utf-8')
    except UnicodeDecodeError:
        return None  # Skip invalid content

    # Step 5: Convert to binary format for PostgreSQL bytea field
    return Binary(block_content.encode('utf-8'))


BATCH_SIZE = 1000  # Set the batch size for bulk insert


def process_subset(data_subset, start_index, dataset_type):
    from psycopg2 import Binary

    batch = []  # Local cache for storing this thread's batch data
    document_counter = start_index  # Start counter from the given index

    for row in data_subset:
        block_id = None
        block_content = None

        if dataset_type == "arxiv":
            block_id = document_counter  # Block ID matches the document ID
            document_id = block_id  # Document ID matches Block ID for arxiv
            document_counter += 1

            # Extract the abstract as block content
            block_content = row.get('abstract')
            if block_content:
                # Clean block_content for arxiv
                try:
                    block_content_str = block_content.strip().replace("\n", " ")
                    block_content_str = block_content_str.encode('utf-8').decode('utf-8')
                except UnicodeDecodeError:
                    print(f"Skipping row due to invalid UTF-8 encoding: {block_content}")
                    continue

                # Ensure non-empty block content
                if not block_content_str.strip():
                    print(f"Skipping row due to empty block content: {row}")
                    continue

                # Generate embedding using the cleaned string
                block_vector = generate_embedding(block_content_str)

                # Convert to binary format for PostgreSQL
                block_content_binary = Binary(block_content_str.encode('utf-8'))

                # Generate hash for deduplication
                hash_value = hashlib.sha1(block_content_str.encode('utf-8')).hexdigest()

                # Add to batch
                batch.append((block_id, document_id, block_content_binary, hash_value, block_vector))

        elif dataset_type == "wikipedia-22-12":
            # Original logic for Wikipedia
            document_id = row.get('wiki_id')
            block_id = row.get('paragraph_id')
            block_content = row.get('text')

            if document_id and block_id and block_content:
                # Generate embedding for the block content
                block_vector = generate_embedding(block_content)

                # Generate hash for deduplication
                hash_value = hashlib.sha1(block_content.encode('utf-8')).hexdigest()

                # Add to batch
                batch.append((block_id, document_id, block_content, hash_value, block_vector))
            else:
                print("Skipping row due to missing fields:", row)
                continue

        else:
            print("Unknown dataset type. Skipping subset.")
            continue

        # Perform bulk insert when batch reaches BATCH_SIZE
        if len(batch) == BATCH_SIZE:
            store_document_block_duplication_bulk(batch)
            batch = []  # Clear the batch cache

    # Insert remaining data if batch is not empty
    if batch:
        store_document_block_duplication_bulk(batch)

    print(f"Processed subset starting at index {start_index}")

def _ingest_numeric_vector_dataset(total_vectors, vector_dim, fetch_chunk, load_number, start_row,
                                   dataset_label):
    if start_row >= total_vectors:
        print(f"Start row ({start_row}) is beyond {dataset_label} size ({total_vectors}); nothing to load.")
        return

    if load_number is None or load_number <= 0:
        end_row = total_vectors
    else:
        end_row = min(total_vectors, start_row + load_number)

    batch_start = start_row
    vectors_processed = 0
    log_threshold = max(BATCH_SIZE * 50, BATCH_SIZE)
    next_log = log_threshold

    while batch_start < end_row:
        batch_end = min(batch_start + BATCH_SIZE, end_row)
        vectors = fetch_chunk(batch_start, batch_end)

        if vectors.shape != (batch_end - batch_start, vector_dim):
            raise ValueError(
                f"{dataset_label}: expected chunk shape {(batch_end - batch_start, vector_dim)} "
                f"but received {vectors.shape}"
            )

        document_blocks = []
        for offset, vector in enumerate(vectors):
            global_index = batch_start + offset
            document_id = (global_index // SIFT_DOCUMENT_VECTOR_COUNT) + 1
            block_id = global_index + 1
            vector_bytes = vector.tobytes()
            block_content = Binary(b"")
            hash_value = Binary(hashlib.sha1(vector_bytes).digest())
            document_blocks.append(
                (
                    block_id,
                    document_id,
                    block_content,
                    hash_value,
                    vector.tolist(),
                )
            )

        if document_blocks:
            store_document_block_duplication_bulk(document_blocks)

        vectors_processed += len(vectors)
        if vectors_processed >= next_log or batch_end == end_row:
            print(f"{dataset_label} loading progress: {vectors_processed} / {end_row - start_row} vectors processed.")
            next_log += log_threshold

        batch_start = batch_end

    print(f"Finished loading {dataset_label} dataset.")


def read_and_store_sift_dataset(load_number=1000, start_row=0):
    """
    Read vectors from the SIFT HDF5 dataset, group every 100 vectors into a document,
    and persist both documents and document blocks.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to load the sift-128-euclidean dataset") from exc

    dataset_path = get_dataset_path()
    dataset_file = os.path.join(dataset_path, "sift-128-euclidean.hdf5")

    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"SIFT dataset not found at {dataset_file}. Please download it first.")

    with h5py.File(dataset_file, "r") as h5_file:
        if "train" in h5_file:
            vector_dataset = h5_file["train"]
        elif "base" in h5_file:
            vector_dataset = h5_file["base"]
        else:
            raise KeyError("Unable to locate 'train' or 'base' dataset inside sift-128-euclidean.hdf5")

        if len(vector_dataset.shape) != 2:
            raise ValueError("Expected a 2D array for SIFT vectors.")

        rows, cols = vector_dataset.shape
        if cols <= rows:
            total_vectors = rows
            vector_dim = cols

            def fetch_chunk(start, end):
                chunk = vector_dataset[start:end, :]
                return np.asarray(chunk, dtype=np.float32)

        else:
            total_vectors = cols
            vector_dim = rows

            def fetch_chunk(start, end):
                chunk = vector_dataset[:, start:end]
                return np.asarray(chunk, dtype=np.float32).T

        print(f"Loading sift-128-euclidean vectors from row {start_row} to "
              f"{'end' if load_number is None or load_number <= 0 else min(total_vectors, start_row + load_number)} "
              f"(exclusive). Detected dimension: {vector_dim}.")

        _ingest_numeric_vector_dataset(
            total_vectors=total_vectors,
            vector_dim=vector_dim,
            fetch_chunk=fetch_chunk,
            load_number=load_number,
            start_row=start_row,
            dataset_label="sift-128-euclidean"
        )


def _ensure_sift10m_features_file(dataset_path):
    """
    Ensure the SIFT10M features file is available on disk.
    If only the tarball is present, extract the features matrix.
    """
    candidate_paths = [
        os.path.join(dataset_path, "SIFT10M", "SIFT10Mfeatures.mat"),
        os.path.join(dataset_path, "SIFT10Mfeatures.mat"),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            return path

    tar_path = os.path.join(dataset_path, "SIFT10M.tar.gz")
    if not os.path.exists(tar_path):
        raise FileNotFoundError(
            f"Unable to locate SIFT10Mfeatures.mat. Expected at one of {candidate_paths}, "
            f"and no SIFT10M.tar.gz found at {tar_path}. Please download the dataset first."
        )

    print("SIFT10M features matrix not found, extracting from SIFT10M.tar.gz (this may take a while)...")
    os.makedirs(os.path.join(dataset_path, "SIFT10M"), exist_ok=True)
    target_path = os.path.join(dataset_path, "SIFT10M", "SIFT10Mfeatures.mat")

    with tarfile.open(tar_path, "r:gz") as tar:
        try:
            member = tar.getmember(SIFT10M_FEATURES_MEMBER)
        except KeyError as exc:
            raise FileNotFoundError(
                f"{SIFT10M_FEATURES_MEMBER} not found inside SIFT10M.tar.gz."
            ) from exc

        # Stream-copy the large MAT file to avoid loading into memory.
        with tar.extractfile(member) as src, open(target_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

    print(f"SIFT10M features extracted to {target_path}")
    return target_path


def read_and_store_sift10m_dataset(load_number=1000, start_row=0):
    """
    Read vectors from the SIFT10M MATLAB dataset, group every 100 vectors into a document,
    and persist both documents and document blocks.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to load the SIFT10M dataset") from exc

    dataset_path = get_dataset_path()
    features_path = _ensure_sift10m_features_file(dataset_path)

    with h5py.File(features_path, "r") as mat_file:
        if "fea" not in mat_file:
            raise KeyError("Unable to locate dataset 'fea' inside SIFT10Mfeatures.mat")

        feature_dataset = mat_file["fea"]
        if len(feature_dataset.shape) != 2:
            raise ValueError("Expected a 2D array for SIFT10M features.")

        rows, cols = feature_dataset.shape
        if cols <= rows:
            total_vectors = rows
            vector_dim = cols

            def fetch_chunk(start, end):
                chunk = feature_dataset[start:end, :]
                return np.asarray(chunk, dtype=np.float32)
        else:
            total_vectors = cols
            vector_dim = rows

            def fetch_chunk(start, end):
                chunk = feature_dataset[:, start:end]
                return np.asarray(chunk, dtype=np.float32).T

        print(f"Loading SIFT10M vectors from row {start_row} to "
              f"{'end' if load_number is None or load_number <= 0 else min(total_vectors, start_row + load_number)} "
              f"(exclusive). Detected dimension: {vector_dim}.")

        _ingest_numeric_vector_dataset(
            total_vectors=total_vectors,
            vector_dim=vector_dim,
            fetch_chunk=fetch_chunk,
            load_number=load_number,
            start_row=start_row,
            dataset_label="SIFT10M"
        )


def read_and_store_dataset_parallel(load_number=1000, start_row=0, num_threads=4, dataset="wikipedia-22-12"):
    # Load the dataset
    dataset_path = get_dataset_path()
    if dataset == "wikipedia-22-12":
        data = load_dataset("json", data_files=f"{dataset_path}/wikipedia-22-12/en/*.jsonl.gz")["train"]
    elif dataset == "sift10m":
        read_and_store_sift10m_dataset(load_number=load_number, start_row=start_row)
        return
    elif dataset == "sift-128-euclidean":
        read_and_store_sift_dataset(load_number=load_number, start_row=start_row)
        return
    elif dataset == "arxiv":
        arxiv_data_file = os.path.join(dataset_path, "arxiv/arxiv-metadata-oai-snapshot.json")
        data = load_dataset("json", data_files=arxiv_data_file)["train"]
    else:
        raise ValueError("Unsupported dataset specified")

    print("data loaded....")
    # Limit load_number to the actual dataset size
    max_size = len(data)
    effective_load_number = min(load_number, max_size - start_row)

    # Pre-split the data into fixed-size chunks to avoid calling select repeatedly
    subsets = []
    for start in range(start_row, start_row + effective_load_number, BATCH_SIZE * num_threads):
        end = min(start + BATCH_SIZE * num_threads, start_row + effective_load_number)
        data_subset = data.select(range(start, end))

        # Further split each large chunk into smaller chunks for each thread
        for i in range(0, len(data_subset), BATCH_SIZE):
            chunk_start = start + i
            chunk_end = min(chunk_start + BATCH_SIZE, end)
            subsets.append((data_subset.select(range(i, min(i + BATCH_SIZE, len(data_subset)))), chunk_start))
    print("start build processpool......")
    # Process each subset in parallel
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_subset, subset, start_index, dataset) for subset, start_index in
                   subsets]

        # Wait for all futures to complete
        for future in futures:
            future.result()

    print("Finished processing dataset.")


def generate_query_cache_batch(subset, document_blocks, block_indices, users, topk, repetitions, query_id_start):
    """
    Generate a batch of queries for a given subset of indices, including block selectivity.
    """
    queries = []

    # Create a connection for this process (thread-safe)
    conn = get_db_connection()
    cur = conn.cursor()

    for i, idx in enumerate(subset):
        user_id = users[idx]
        block_index = block_indices[idx]
        block_id, query_vector = document_blocks[block_index]

        # Step 1: Calculate block selectivity for the user
        sql_query_accessed = """
            SELECT COUNT(db.block_id)
            FROM PermissionAssignment pa
            JOIN UserRoles ur ON pa.role_id = ur.role_id
            JOIN documentblocks db ON db.document_id = pa.document_id
            WHERE ur.user_id = %s;
        """
        cur.execute(sql_query_accessed, (user_id,))
        accessed_blocks = cur.fetchone()[0] if cur.rowcount > 0 else 0

        sql_query_total = "SELECT COUNT(block_id) FROM documentblocks;"
        cur.execute(sql_query_total)
        total_blocks = cur.fetchone()[0] if cur.rowcount > 0 else 1  # Avoid division by zero

        block_selectivity = accessed_blocks / total_blocks

        # Step 2: Generate queries with repetitions
        for repetition in range(repetitions):
            query = {
                "query_id": query_id_start + i * repetitions + repetition,
                "user_id": user_id,
                "query_vector": query_vector,
                "topk": topk,
                "repetition": repetition + 1,  # Mark repetition (1, 2, 3)
                "query_block_selectivity": block_selectivity
            }
            queries.append(query)

    cur.close()
    conn.close()
    return queries


def generate_query_dataset_for_cache(num_queries=1000, topk=5, output_file="query_dataset.json",
                                     zipf_param=3, repetitions=3, num_threads=os.cpu_count()):
    """
    Generate a query dataset with three repetitions for each query using parallel processing.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # Step 1: Get all users and document blocks
    cur.execute("SELECT user_id FROM Users;")
    all_users = [row[0] for row in cur.fetchall()]

    cur.execute("SELECT block_id, vector FROM documentblocks;")
    document_blocks = cur.fetchall()

    block_ids = [block[0] for block in document_blocks]
    total_blocks = len(block_ids)

    # Step 2: Generate block indices and user IDs
    if zipf_param == 0:
        block_indices = np.random.choice(total_blocks, size=num_queries, replace=True)
    else:
        zipf_distribution = np.random.zipf(zipf_param, size=num_queries)
        block_indices = zipf_distribution % total_blocks  # Ensure indices are within range

    user_indices = np.random.choice(len(all_users), size=num_queries, replace=True)
    users = [all_users[i] for i in user_indices]

    # Step 3: Split queries into subsets for parallel processing
    subset_size = num_queries // num_threads
    subsets = [range(i * subset_size, (i + 1) * subset_size) for i in range(num_threads)]
    if num_queries % num_threads != 0:
        subsets.append(range(num_threads * subset_size, num_queries))

    # Step 4: Process subsets in parallel
    query_id_start = 1
    queries = []
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                generate_query_cache_batch,
                subset,
                document_blocks,
                block_indices,
                users,
                topk,
                repetitions,
                query_id_start + i * subset_size
            )
            for i, subset in enumerate(subsets)
        ]

        for future in futures:
            queries.extend(future.result())

    # Step 5: Save the queries to a JSON file
    with open(output_file, "w") as outfile:
        json.dump(queries, outfile, indent=2)

    cur.close()
    conn.close()

    return block_ids


def generate_query_batch(subset, document_blocks, block_indices, users, topk, total_blocks):
    """
    Generate a batch of queries with block selectivity for the given subset of indices.
    """
    conn = get_db_connection()  # Create a connection for this process
    cur = conn.cursor()
    queries = []

    for i in subset:
        user_id = users[i]

        # Calculate block selectivity for the user
        sql_query = """
            SELECT COUNT(db.block_id)
            FROM PermissionAssignment pa
            JOIN UserRoles ur ON pa.role_id = ur.role_id
            JOIN documentblocks db ON db.document_id = pa.document_id
            WHERE ur.user_id = %s;
        """
        cur.execute(sql_query, (user_id,))
        accessed_blocks_result = cur.fetchone()
        accessed_blocks = accessed_blocks_result[0] if accessed_blocks_result and accessed_blocks_result[0] else 0
        block_selectivity = accessed_blocks / total_blocks if total_blocks > 0 else 0

        # Select a document block based on the precomputed indices
        block_index = block_indices[i]
        block_id, query_vector = document_blocks[block_index]

        # Create the query
        query = {
            "user_id": user_id,
            "query_vector": query_vector,
            "topk": topk,
            "query_block_selectivity": block_selectivity,
        }
        queries.append(query)

    cur.close()
    conn.close()
    return queries


def generate_query_dataset(num_queries=1000, topk=5, output_file="query_dataset.json", zipf_param=3, num_threads=os.cpu_count()):
    """
    Generate a query dataset with query block selectivity using parallel processing.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # Step 1: Get all users and document blocks
    cur.execute("SELECT user_id FROM Users;")
    all_users = [row[0] for row in cur.fetchall()]

    cur.execute("SELECT block_id, vector FROM documentblocks;")
    document_blocks = cur.fetchall()

    block_ids = [block[0] for block in document_blocks]
    total_blocks = len(block_ids)

    # Step 2: Generate block indices and user IDs
    if zipf_param == 0:
        block_indices = np.random.choice(total_blocks, size=num_queries, replace=True)
    else:
        zipf_distribution = np.random.zipf(zipf_param, size=num_queries)
        block_indices = zipf_distribution % total_blocks  # Ensure indices are within range

    user_indices = np.random.choice(len(all_users), size=num_queries, replace=True)
    users = [all_users[i] for i in user_indices]

    # Step 3: Split work into subsets for parallel processing
    subset_size = num_queries // num_threads
    subsets = [range(i * subset_size, (i + 1) * subset_size) for i in range(num_threads)]
    if num_queries % num_threads != 0:
        subsets.append(range(num_threads * subset_size, num_queries))

    # Step 4: Process subsets in parallel
    queries = []
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(generate_query_batch, subset, document_blocks, block_indices, users, topk, total_blocks)
            for subset in subsets
        ]

        # Collect results from all processes
        for future in futures:
            queries.extend(future.result())

    # Step 5: Save the queries to a JSON file
    with open(output_file, "w") as outfile:
        json.dump(queries, outfile, indent=2)

    cur.close()
    conn.close()

    return block_ids

def generate_query_for_role_with_sel(role, user, k, topk, role_to_documents, document_blocks):
    """
    Generate queries for a single role and its assigned user, including block selectivity.
    """
    queries = []

    # Create a database connection for this process
    conn = get_db_connection()
    cur = conn.cursor()

    # Calculate block selectivity for the user
    sql_query = """
        SELECT COUNT(db.block_id)
        FROM PermissionAssignment pa
        JOIN UserRoles ur ON pa.role_id = ur.role_id
        JOIN documentblocks db ON db.document_id = pa.document_id
        WHERE ur.user_id = %s;
    """
    cur.execute(sql_query, (user,))
    accessed_blocks_result = cur.fetchone()
    accessed_blocks = accessed_blocks_result[0] if accessed_blocks_result and accessed_blocks_result[0] else 0

    cur.execute("SELECT COUNT(block_id) FROM documentblocks;")
    total_blocks = cur.fetchone()[0]
    block_selectivity = accessed_blocks / total_blocks if total_blocks > 0 else 0

    # Generate k queries for the role
    for _ in range(k):
        relevant_docs = role_to_documents[role]
        doc_id = random.choice(relevant_docs)
        query_vector = document_blocks.get(doc_id)

        if query_vector is None:
            raise ValueError(f"Document ID {doc_id} not found in document blocks.")

        query = {
            "user_id": user,
            "role": role,
            "query_vector": query_vector,
            "topk": topk,
            "query_block_selectivity": block_selectivity,
        }
        queries.append(query)

    cur.close()
    conn.close()
    return queries


def generate_query_dataset_for_roles(k=5, topk=5, output_file="query_dataset.json", num_threads=os.cpu_count()):
    """
    Generate a query dataset with k * m queries, each user corresponds to one role.
    Block selectivity is calculated inline for each query.

    :param k: Number of queries per role.
    :param topk: Number of top results to retrieve.
    :param output_file: Output JSON file for queries.
    :param num_threads: Number of threads for parallel processing.
    :return: List of generated queries.
    """
    # Step 1: Connect to the database and fetch necessary data
    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch all roles
    cur.execute("SELECT role_id FROM Roles;")
    roles = [row[0] for row in cur.fetchall()]

    # Fetch valid document IDs from permissions
    cur.execute("SELECT DISTINCT document_id FROM PermissionAssignment;")
    valid_document_ids = {row[0] for row in cur.fetchall()}

    # Fetch document blocks (ensure consistent document ordering)
    cur.execute("SELECT DISTINCT document_id, vector FROM documentblocks ORDER BY document_id;")
    document_blocks = {row[0]: row[1] for row in cur.fetchall()}

    # Fetch role-to-document permissions
    cur.execute("SELECT role_id, document_id FROM PermissionAssignment;")
    permissions = cur.fetchall()

    # Build a mapping from roles to their allowed documents
    role_to_documents = {role: [] for role in roles}
    for role_id, doc_id in permissions:
        if doc_id in valid_document_ids:
            role_to_documents[role_id].append(doc_id)

    # Fetch users with their roles
    cur.execute("SELECT user_id, role_id FROM UserRoles;")
    user_roles = cur.fetchall()
    user_to_roles = {}
    for user_id, role_id in user_roles:
        if role_id in role_to_documents:
            if user_id not in user_to_roles:
                user_to_roles[user_id] = []
            user_to_roles[user_id].append(role_id)

    # Step 2: Select unique roles and their users
    role_to_user = {}
    for role in roles:
        candidates = [user for user, assigned_roles in user_to_roles.items() if role in assigned_roles]
        if candidates:
            role_to_user[role] = random.choice(candidates)  # Pick a user for each role

    # Step 3: Generate queries in parallel
    queries = []
    role_user_pairs = [(role, user) for role, user in role_to_user.items()]

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                generate_query_for_role_with_sel, role, user, k, topk, role_to_documents, document_blocks
            )
            for role, user in role_user_pairs
        ]

        for future in futures:
            queries.extend(future.result())

    # Save queries to JSON
    with open(output_file, "w") as outfile:
        json.dump(queries, outfile, indent=2)

    cur.close()
    conn.close()

    return queries


def generate_query_for_role_with_repetitions(role, user, k, topk, repetitions, role_to_documents, document_blocks):
    """
    Generate queries for a single role and its assigned user, including block selectivity and repetitions.

    :param role: Role ID for which queries are generated.
    :param user: User ID associated with the role.
    :param k: Number of unique queries to generate per role.
    :param topk: Number of top results to retrieve.
    :param repetitions: Number of repetitions per query.
    :param role_to_documents: Mapping of roles to their allowed document IDs.
    :param document_blocks: Mapping of document IDs to their vector representations.
    :return: List of generated queries.
    """
    queries = []

    # Create a database connection for this process
    conn = get_db_connection()
    cur = conn.cursor()

    # Calculate block selectivity for the user
    sql_query = """
        SELECT COUNT(db.block_id)
        FROM PermissionAssignment pa
        JOIN UserRoles ur ON pa.role_id = ur.role_id
        JOIN documentblocks db ON db.document_id = pa.document_id
        WHERE ur.user_id = %s;
    """
    cur.execute(sql_query, (user,))
    accessed_blocks_result = cur.fetchone()
    accessed_blocks = accessed_blocks_result[0] if accessed_blocks_result and accessed_blocks_result[0] else 0

    cur.execute("SELECT COUNT(block_id) FROM documentblocks;")
    total_blocks = cur.fetchone()[0]
    block_selectivity = accessed_blocks / total_blocks if total_blocks > 0 else 0

    # Generate k unique queries for the role
    for query_index in range(k):
        relevant_docs = role_to_documents[role]
        doc_id = random.choice(relevant_docs)
        query_vector = document_blocks.get(doc_id)

        if query_vector is None:
            raise ValueError(f"Document ID {doc_id} not found in document blocks.")

        # Add repetitions for the same query
        for repetition in range(repetitions):
            query = {
                "user_id": user,
                "role": role,
                "query_vector": query_vector,
                "topk": topk,
                "repetition": repetition + 1,  # Mark repetition (1, 2, 3)
                "query_block_selectivity": block_selectivity,
            }
            queries.append(query)

    cur.close()
    conn.close()
    return queries


def generate_query_dataset_with_roles_and_repetitions(
    k=1000, topk=5, output_file="query_dataset.json", num_threads=os.cpu_count()
):
    """
    Generate a query dataset with k * m queries, each user corresponds to one role, and repetitions are added.
    Block selectivity is calculated inline for each query.

    :param k: Number of unique queries per role.
    :param topk: Number of top results to retrieve.
    :param repetitions: Number of repetitions per query.
    :param output_file: Output JSON file for queries.
    :param num_threads: Number of threads for parallel processing.
    :return: List of generated queries.
    """
    # Step 1: Connect to the database and fetch necessary data
    conn = get_db_connection()
    cur = conn.cursor()
    repetitions = 3
    # Fetch all roles
    cur.execute("SELECT role_id FROM Roles;")
    roles = [row[0] for row in cur.fetchall()]

    # Fetch valid document IDs from permissions
    cur.execute("SELECT DISTINCT document_id FROM PermissionAssignment;")
    valid_document_ids = {row[0] for row in cur.fetchall()}

    # Fetch document blocks (ensure consistent document ordering)
    cur.execute("SELECT DISTINCT document_id, vector FROM documentblocks ORDER BY document_id;")
    document_blocks = {row[0]: row[1] for row in cur.fetchall()}

    # Fetch role-to-document permissions
    cur.execute("SELECT role_id, document_id FROM PermissionAssignment;")
    permissions = cur.fetchall()

    # Build a mapping from roles to their allowed documents
    role_to_documents = {role: [] for role in roles}
    for role_id, doc_id in permissions:
        if doc_id in valid_document_ids:
            role_to_documents[role_id].append(doc_id)

    # Fetch users with their roles
    cur.execute("SELECT user_id, role_id FROM UserRoles;")
    user_roles = cur.fetchall()
    user_to_roles = {}
    for user_id, role_id in user_roles:
        if role_id in role_to_documents:
            if user_id not in user_to_roles:
                user_to_roles[user_id] = []
            user_to_roles[user_id].append(role_id)

    # Step 2: Select unique roles and their users
    role_to_user = {}
    for role in roles:
        candidates = [user for user, assigned_roles in user_to_roles.items() if role in assigned_roles]
        if candidates:
            role_to_user[role] = random.choice(candidates)  # Pick a user for each role

    # Step 3: Generate queries in parallel
    queries = []
    role_user_pairs = [(role, user) for role, user in role_to_user.items()]

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                generate_query_for_role_with_repetitions,
                role,
                user,
                k,
                topk,
                repetitions,
                role_to_documents,
                document_blocks,
            )
            for role, user in role_user_pairs
        ]

        for future in futures:
            queries.extend(future.result())

    # Save queries to JSON
    with open(output_file, "w") as outfile:
        json.dump(queries, outfile, indent=2)

    cur.close()
    conn.close()

    return queries

def load_queries_from_dataset(query_file):
    """
    Load the queries from the specified JSON file.

    Args:
        query_file (str): Path to the query dataset JSON file.

    Returns:
        list: A list of queries, where each query is a dictionary containing 'user_id', 'query_vector', and 'topk'.
    """
    with open(query_file, "r") as infile:
        queries = json.load(infile)

    return queries
