from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import psycopg2
from psycopg2 import sql

import os
import sys
import hashlib

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from controller.dynamic_partition.hnsw.helper import fetch_initial_data, prepare_background_data

from services.config import get_db_connection, get_maintenance_settings, get_document_vector_dimension

_POSTGRES_IDENTIFIER_LIMIT = 63


def _safe_identifier(value: str) -> str:
    candidate = "".join(char if char.isalnum() or char == "_" else "_" for char in str(value).lower()).strip("_")
    if not candidate:
        candidate = "partition"
    if len(candidate) <= _POSTGRES_IDENTIFIER_LIMIT:
        return candidate
    digest = hashlib.blake2b(candidate.encode("utf-8"), digest_size=6).hexdigest()
    return f"{candidate[:_POSTGRES_IDENTIFIER_LIMIT - len(digest) - 1]}_{digest}"


def partition_table_name(partition_id, table_prefix: str | None = None) -> str:
    if table_prefix:
        return _safe_identifier(f"{table_prefix}_partition_{partition_id}")
    return f"documentblocks_partition_{partition_id}"


def comb_role_mapping_table_name(table_prefix: str | None = None) -> str:
    if table_prefix:
        return _safe_identifier(f"{table_prefix}_comb_role_partitions")
    return "CombRolePartitions"


def _configure_index_session(cur, disable_sync_commit: bool = True, hnsw_threads: Optional[int] = None) -> None:
    """
    Apply session-level settings that significantly speed up bulk index creation.
    Mirrors the role-partition pipeline defaults so both flows behave similarly.
    """
    maintenance_settings = get_maintenance_settings()
    maintenance_work_mem_gb = maintenance_settings["maintenance_work_mem_gb"]
    max_parallel_maintenance_workers = maintenance_settings["max_parallel_maintenance_workers"]

    cur.execute(f"SET maintenance_work_mem = '{maintenance_work_mem_gb}GB';")
    cur.execute(f"SET max_parallel_maintenance_workers = {max_parallel_maintenance_workers};")
    if disable_sync_commit:
        cur.execute("SET synchronous_commit = OFF;")
    if hnsw_threads:
        cur.execute(f"SET hnsw.threads = {int(hnsw_threads)};")


def validate_partition_coverage(cur, accessible_partitions, document_to_index):
    """
    Validate if the union of all documents in the accessible partitions
    covers all documents that should be assigned to partitions.
    """
    # Fetch all documents in accessible partitions
    partition_documents = set()
    for partition_id in accessible_partitions:
        partition_table = sql.Identifier(f"documentblocks_partition_{partition_id}")
        query = sql.SQL("SELECT DISTINCT document_id FROM {};").format(partition_table)
        cur.execute(query)
        partition_documents.update([row[0] for row in cur.fetchall()])

    # Fetch all documents from the provided document-to-index mapping
    expected_documents = set(document_to_index.keys())

    # Validate if partition documents cover all expected documents
    missing_docs = expected_documents - partition_documents
    if missing_docs:
        raise ValueError(f"Missing documents in partitions: {missing_docs}")
    print("All documents are correctly assigned across partitions.")


def delete_partitions_and_role_mappings(comb_role_tracker=None, increment_update=False, table_prefix: str | None = None):
    """
    Deletes all partition tables and clears the RolePartitions mapping table.

    Args:
        comb_role_tracker (dict): Mapping of role combinations to the partitions they are assigned to.
        increment_update (bool): Whether to incrementally update partitions.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Step 1: Retrieve all existing partition tables
        like_pattern = f"{_safe_identifier(table_prefix)}_partition_%" if table_prefix else "documentblocks_partition_%"
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name LIKE %s;
        """, [like_pattern])
        partition_tables = {row[0] for row in cur.fetchall()}

        # Step 2: Identify valid partitions based on comb_role_tracker
        if increment_update and comb_role_tracker:
            valid_partitions = {
                partition_table_name(partition_id, table_prefix)
                for partitions in comb_role_tracker.values()
                for partition_id in partitions
            }
        else:
            valid_partitions = set()  # If not increment update, no partition is valid

        # Step 3: Determine partitions to delete (existing - valid)
        partitions_to_delete = partition_tables - valid_partitions

        # Step 4: Drop invalid partitions
        for table_name in partitions_to_delete:
            try:
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(sql.Identifier(table_name)))
                print(f"Partition table {table_name} deleted.")
                conn.commit()  # Commit immediately after each delete
            except psycopg2.Error as e:
                print(f"Error deleting partition {table_name}: {e}")
                conn.rollback()  # Rollback only the current delete

        # Step 5: Clear the RolePartitions table
        mapping_table = comb_role_mapping_table_name(table_prefix)
        cur.execute(sql.SQL("CREATE TABLE IF NOT EXISTS {} (comb_role INT[] NOT NULL, partition_id INT NOT NULL, PRIMARY KEY (comb_role, partition_id));").format(sql.Identifier(mapping_table)))
        cur.execute(sql.SQL("DELETE FROM {};").format(sql.Identifier(mapping_table)))
        conn.commit()
        print("RolePartitions table cleared.")

    except psycopg2.Error as e:
        print(f"Database error during deletion: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def create_and_populate_partition_table_increment(partition_id, partition_assignment, document_to_index, table_prefix: str | None = None):
    conn = get_db_connection()
    cur = conn.cursor()
    partition_table = partition_table_name(partition_id, table_prefix)
    vector_dimension = get_document_vector_dimension()

    try:
        # Step 1: Check if the table already exists
        cur.execute("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_name = %s
            );
            """, (partition_table,))
        table_exists = cur.fetchone()[0]

        # Step 2: Collect current data if table exists
        current_docs = set()
        if table_exists:
            cur.execute(sql.SQL("SELECT document_id FROM {};").format(sql.Identifier(partition_table)))
            current_docs = {row[0] for row in cur.fetchall()}

        # Step 3: Get new documents to be inserted
        assigned_docs = partition_assignment.get(partition_id, set())
        index_to_document = {v: k for k, v in document_to_index.items()}
        new_docs = {index_to_document[idx] for idx in assigned_docs if idx in index_to_document}

        # Step 4: Compare current and new data
        if table_exists and current_docs == new_docs:
            print(f"Partition table {partition_table} is up-to-date. Skipping.")
            return partition_id

        # If not up-to-date, recreate the table
        print(f"Updating partition table {partition_table}...")
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(sql.Identifier(partition_table)))

        # Create the partition table
        cur.execute(
            sql.SQL("""
                CREATE TABLE {table} (
                    block_id BIGINT NOT NULL,
                    document_id INT NOT NULL REFERENCES Documents(document_id),
                    block_content BYTEA NOT NULL,
                    vector VECTOR({dimension}),
                    PRIMARY KEY (block_id, document_id)
                );
            """).format(
                table=sql.Identifier(partition_table),
                dimension=sql.SQL(str(vector_dimension))
            )
        )
        conn.commit()

        # Populate the table with new data
        for document_id in new_docs:
            cur.execute(
                sql.SQL("""
                    INSERT INTO {} (block_id, document_id, block_content, vector)
                    SELECT block_id, document_id, block_content, vector
                    FROM documentblocks
                    WHERE document_id = %s;
                """).format(sql.Identifier(partition_table)),
                [document_id]
            )

        conn.commit()
        print(f"Partition table {partition_table} updated with new data.")
        return partition_id

    except psycopg2.Error as e:
        print(f"Database error while updating {partition_table}: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def create_and_populate_partition_table(partition_id, partition_assignment, document_to_index, table_prefix: str | None = None):
    """
    Create or update a partition table based on `partition_assignment`.

    Args:
        partition_id (int): The ID of the partition to update.
        partition_assignment (dict): Mapping of partition IDs to sets of document indices.
        document_to_index (dict): Mapping of document IDs to indices (used for reverse lookup).
    """
    conn = get_db_connection()
    cur = conn.cursor()
    partition_table = partition_table_name(partition_id, table_prefix)
    vector_dimension = get_document_vector_dimension()

    try:
        # Step 1: Create the partition table
        cur.execute(
            sql.SQL("""
                CREATE TABLE IF NOT EXISTS {table} (
                    block_id BIGINT NOT NULL,
                    document_id INT NOT NULL REFERENCES Documents(document_id),
                    block_content BYTEA NOT NULL,
                    vector VECTOR({dimension}),
                    PRIMARY KEY (block_id, document_id)
                );
            """).format(
                table=sql.Identifier(partition_table),
                dimension=sql.SQL(str(vector_dimension))
            )
        )
        conn.commit()
        print(f"Partition table {partition_table} created.")

        # Step 2: Populate the partition table based on `partition_assignment`
        # Use `partition_assignment` to get the document indices, then map to document IDs
        assigned_docs = partition_assignment.get(partition_id, set())
        index_to_document = {v: k for k, v in document_to_index.items()}
        new_docs = {index_to_document[idx] for idx in assigned_docs if idx in index_to_document}

        # Step 3: Insert documents into the partition table
        for document_id in new_docs:
            cur.execute(
                sql.SQL("""
                    INSERT INTO {} (block_id, document_id, block_content, vector)
                    SELECT block_id, document_id, block_content, vector
                    FROM documentblocks
                    WHERE document_id = %s;
                """).format(sql.Identifier(partition_table)),
                [document_id]
            )

        conn.commit()
        print(f"Partition table {partition_table} populated with document blocks.")
        return partition_id

    except psycopg2.Error as e:
        print(f"Database error while creating or populating {partition_table}: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def insert_comb_role_partition_mapping(role_comb, partition_ids, mapping_table: str | None = None):
    """
    Insert role combination to partition mappings into the CombRolePartitions table.

    Args:
        role_comb (tuple): A role combination (tuple of role IDs).
        partition_ids (set): The partitions accessed by this role combination.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    target_mapping_table = mapping_table or comb_role_mapping_table_name()

    try:
        # Convert role_comb into a sorted list for consistent storage
        comb_roles_array = list(sorted(role_comb))

        for partition_id in partition_ids:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (comb_role, partition_id)
                    VALUES (%s, %s)
                    ON CONFLICT (comb_role, partition_id) DO NOTHING;
                    """
                ).format(sql.Identifier(target_mapping_table)),
                (comb_roles_array, partition_id),
            )

        conn.commit()
        print(f"Role combination {comb_roles_array} mapped to partitions {partition_ids}.")

    except psycopg2.Error as e:
        print(f"Database error while mapping role_comb {role_comb}: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def initialize_partitions_and_role_mappings(
    partition_assignment,
    comb_role_tracker,
    document_to_index,
    num_threads=os.cpu_count(),
    increment_update=False,
    table_prefix: str | None = None,
):
    num_threads=min(4, os.cpu_count())
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Step 1: Create CombRolePartitions Mapping table once
        mapping_table = comb_role_mapping_table_name(table_prefix)
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {} (
                    comb_role INT[] NOT NULL,
                    partition_id INT NOT NULL,
                    PRIMARY KEY (comb_role, partition_id)
                );
                """
            ).format(sql.Identifier(mapping_table))
        )
        conn.commit()
        print(f"{mapping_table} table created successfully.")
    except psycopg2.Error as e:
        print(f"Error creating CombRolePartitions table: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

    # Step 2: Create partition tables first, and fail fast if any worker fails.
    partition_ids = list(partition_assignment.keys())
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        if increment_update:
            partition_futures = [
                executor.submit(create_and_populate_partition_table_increment, partition_id, partition_assignment,
                                document_to_index, table_prefix)
                for partition_id in partition_ids
            ]
        else:
            partition_futures = [
                executor.submit(create_and_populate_partition_table, partition_id, partition_assignment,
                                document_to_index, table_prefix)
                for partition_id in partition_ids
            ]

        created_partition_ids = {future.result() for future in partition_futures}

    missing_partition_ids = sorted(set(partition_ids) - created_partition_ids)
    if missing_partition_ids:
        raise RuntimeError(
            f"Failed to materialize partition tables for partition ids: {missing_partition_ids[:10]}"
        )

    # Step 3: Only insert combination mappings after all referenced partition tables exist.
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        comb_role_futures = [
            executor.submit(insert_comb_role_partition_mapping, comb_role, partitions, mapping_table)
            for comb_role, partitions in comb_role_tracker.items()
        ]
        for future in comb_role_futures:
            future.result()

    print("All partitions and role mappings have been processed.")


def create_index_for_partition(
    table_name,
    index_type="hnsw",
    *,
    hnsw_m: int = 16,
    hnsw_ef_construction: int = 64,
    hnsw_threads: Optional[int] = None,
    disable_sync_commit: bool = True,
):
    """
    Create an index for a specific partition table.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        _configure_index_session(
            cur,
            disable_sync_commit=disable_sync_commit,
            hnsw_threads=hnsw_threads,
        )
        if index_type.lower() == "hnsw":
            cur.execute(
                sql.SQL("""
                    CREATE INDEX IF NOT EXISTS {} 
                    ON {} USING hnsw (vector vector_l2_ops)
                    WITH (m = {m}, ef_construction = {ef});
                """).format(
                    sql.Identifier(f"{table_name}_vector_idx"),
                    sql.Identifier(table_name),
                    m=sql.Literal(int(hnsw_m)),
                    ef=sql.Literal(int(hnsw_ef_construction)),
                )
            )
            print(f"HNSW index created for {table_name}.")
        elif index_type.lower() == "ivfflat":
            cur.execute(
                sql.SQL("""
                    CREATE INDEX IF NOT EXISTS {} 
                    ON {} USING ivfflat (vector vector_l2_ops);
                """).format(
                    sql.Identifier(f"{table_name}_vector_idx"),
                    sql.Identifier(table_name)
                )
            )
            print(f"IVFFlat index created for {table_name}.")
        conn.commit()
    except psycopg2.Error as e:
        print(f"Error creating index for {table_name}: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def create_indexes_for_all_partitions(
    index_type="hnsw",
    *,
    table_prefix: str | None = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    hnsw_m: int = 16,
    hnsw_ef_construction: int = 64,
    hnsw_threads: Optional[int] = None,
    disable_sync_commit: bool = True,
):
    """
    Create indexes for all partition tables in parallel.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    maintenance_settings = get_maintenance_settings()
    maintenance_work_mem_gb = maintenance_settings["maintenance_work_mem_gb"]
    max_parallel_maintenance_workers = maintenance_settings["max_parallel_maintenance_workers"]

    _configure_index_session(
        cur,
        disable_sync_commit=disable_sync_commit,
        hnsw_threads=hnsw_threads,
    )
    print(
        "PostgreSQL parameters set: maintenance_work_mem = "
        f"{maintenance_work_mem_gb}GB, max_parallel_maintenance_workers = {max_parallel_maintenance_workers}"
    )
    try:
        # Retrieve all partition tables
        like_pattern = f"{_safe_identifier(table_prefix)}_partition_%" if table_prefix else "documentblocks_partition_%"
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name LIKE %s;
        """, [like_pattern])
        partition_tables = [row[0] for row in cur.fetchall()]
    except psycopg2.Error as e:
        print(f"Error retrieving partition tables: {e}")
        conn.rollback()
        cur.close()
        conn.close()
        return

    cur.close()
    conn.close()

    if not partition_tables:
        print("No dynamic partition tables found. Skipping index creation.")
        return

    env_max_workers = os.environ.get("DYNAMIC_INDEX_MAX_WORKERS")
    if max_workers is None and env_max_workers:
        max_workers = max(1, int(env_max_workers))
    env_parallel = os.environ.get("DYNAMIC_INDEX_PARALLEL")
    if env_parallel is not None and env_parallel.strip().lower() in {"0", "false", "off", "no"}:
        parallel = False
    if hnsw_threads is None and os.environ.get("HNSW_INDEX_THREADS"):
        hnsw_threads = max(1, int(os.environ["HNSW_INDEX_THREADS"]))

    worker_count = max_workers or max(1, os.cpu_count() // 2)

    if parallel and worker_count > 1:
        # Process each partition table in parallel
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    create_index_for_partition,
                    table_name,
                    index_type,
                    hnsw_m=hnsw_m,
                    hnsw_ef_construction=hnsw_ef_construction,
                    hnsw_threads=hnsw_threads,
                    disable_sync_commit=disable_sync_commit,
                )
                for table_name in partition_tables
            ]

            # Wait for all futures to complete
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in parallel execution: {e}")
    else:
        for table_name in partition_tables:
            create_index_for_partition(
                table_name,
                index_type,
                hnsw_m=hnsw_m,
                hnsw_ef_construction=hnsw_ef_construction,
                hnsw_threads=hnsw_threads,
                disable_sync_commit=disable_sync_commit,
            )

    print("Finished creating indexes for all partition tables.")


def drop_indexes_for_all_partitions():
    """
    Drop all indexes from partition tables.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Retrieve all partition tables
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name LIKE 'documentblocks_partition_%';
        """)
        partition_tables = [row[0] for row in cur.fetchall()]

        for table_name in partition_tables:
            cur.execute(
                sql.SQL("DROP INDEX IF EXISTS {} CASCADE;").format(
                    sql.Identifier(f"{table_name}_vector_idx")
                )
            )
            print(f"Index dropped for {table_name}.")

        conn.commit()
    except psycopg2.Error as e:
        print(f"Error dropping indexes: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def initialize_rls_for_partitions():
    """
    Enable Row Level Security (RLS) and create policies for each partition table.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    roles, documents, permissions, avg_blocks_per_document, _ = fetch_initial_data()
    # Prepare background data
    role_to_documents, document_to_index = prepare_background_data(roles, documents, permissions)

    cur.execute("GRANT SELECT ON PermissionAssignment TO PUBLIC;")
    cur.execute("GRANT SELECT ON UserRoles TO PUBLIC;")
    cur.execute("GRANT SELECT ON DocumentBlocks TO PUBLIC;")
    cur.execute("GRANT SELECT ON CombRolePartitions TO PUBLIC;")
    # Step 2: Dynamically grant SELECT permission to all partition tables
    cur.execute("""
        SELECT tablename 
        FROM pg_tables 
        WHERE tablename LIKE 'documentblocks_partition_%';
    """)
    partition_tables = [row[0] for row in cur.fetchall()]

    for table_name in partition_tables:
        cur.execute(sql.SQL("GRANT SELECT ON {} TO PUBLIC;").format(sql.Identifier(table_name)))

    try:
        cur.execute("DROP MATERIALIZED VIEW IF EXISTS user_accessible_documents;")
        cur.execute("""
            CREATE MATERIALIZED VIEW user_accessible_documents AS
            SELECT ur.user_id, pa.document_id
            FROM UserRoles ur
            JOIN PermissionAssignment pa ON ur.role_id = pa.role_id
            GROUP BY ur.user_id, pa.document_id;
        """)
        cur.execute("CREATE INDEX user_accessible_documents_idx ON user_accessible_documents (user_id, document_id);")
        cur.execute("GRANT SELECT ON user_accessible_documents TO PUBLIC;")
        conn.commit()
        print("Materialized view 'user_accessible_documents' created and indexed.")

    except psycopg2.Error as e:
        print(f"Error creating materialized view: {e}")
        conn.rollback()
        return

    try:
        # Retrieve all partition tables
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name LIKE 'documentblocks_partition_%';
        """)
        partition_tables = [row[0] for row in cur.fetchall()]

        for table_name in partition_tables:
            # Fetch all document IDs associated with the current partition
            cur.execute(sql.SQL("SELECT DISTINCT document_id FROM {};").format(sql.Identifier(table_name)))
            partition_document_ids = {row[0] for row in cur.fetchall()}

            # Precompute role document sets for subset checks
            role_document_sets = {role: set(docs) for role, docs in role_to_documents.items()}
            # Extract partition ID from the table name
            partition_id = table_name.split('_')[-1]

            # Retrieve all comb_roles linked to this partition
            cur.execute("""
                SELECT comb_role
                FROM CombRolePartitions
                WHERE partition_id = %s;
            """, [partition_id])

            # Ensure comb_role is stored as a sorted tuple for consistency
            associated_comb_roles = {tuple(sorted(row[0])) for row in cur.fetchall()}

            # Determine if RLS can be skipped based on comb_role's document access
            skip_rls = True
            for comb_role in associated_comb_roles:
                comb_docs = set()  # Collect all required documents for this comb_role
                for role in comb_role:
                    comb_docs.update(role_document_sets.get(role, set()))

                    # If partition contains documents outside the required comb_role scope, enable RLS
                if not partition_document_ids.issubset(comb_docs):
                    skip_rls = False
                    break

            if skip_rls:
                print(f"Skipping RLS for {table_name}, all associated comb_roles have matching document access.")
                continue

            # Enable Row Level Security (RLS)
            cur.execute(sql.SQL("ALTER TABLE {} ENABLE ROW LEVEL SECURITY;").format(sql.Identifier(table_name)))
            cur.execute(sql.SQL("ALTER TABLE {} FORCE ROW LEVEL SECURITY;").format(sql.Identifier(table_name)))

            # Create an RLS policy to restrict document access per user
            policy_query = sql.SQL("""
                CREATE POLICY partition_access_policy ON {} 
                FOR SELECT
                USING (
                    EXISTS (
                        SELECT 1
                        FROM user_accessible_documents uad
                        WHERE uad.document_id = {}.document_id
                          AND uad.user_id = current_user::int
                    )
                );
            """).format(sql.Identifier(table_name), sql.Identifier(table_name))

            cur.execute(policy_query)

        conn.commit()
        print("RLS initialization completed for all partitions.")

    except psycopg2.Error as e:
        print(f"Error initializing RLS: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def disable_rls_for_partitions():
    """
    Disable RLS and drop policies for each partition table.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Retrieve all partition tables
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name LIKE 'documentblocks_partition_%';
        """)
        partition_tables = [row[0] for row in cur.fetchall()]

        for table_name in partition_tables:
            # Disable RLS on the partition table
            cur.execute(sql.SQL("ALTER TABLE {} DISABLE ROW LEVEL SECURITY;").format(sql.Identifier(table_name)))

            # Drop the RLS policy if it exists
            cur.execute(
                sql.SQL("DROP POLICY IF EXISTS partition_access_policy ON {};").format(sql.Identifier(table_name)))

        conn.commit()
        print("RLS disabling for all partitions completed.")

    except psycopg2.Error as e:
        print(f"Error disabling RLS: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def load_result_to_database(partition_assignment=None, comb_role_tracker=None, increment_update=False, table_prefix: str | None = None):
    delete_partitions_and_role_mappings(
        comb_role_tracker=comb_role_tracker,
        increment_update=increment_update,
        table_prefix=table_prefix,
    )
    # Step 2: Fetch roles and document-to-index mapping from the database
    roles, documents, permissions, _, _ = fetch_initial_data()
    role_to_documents, document_to_index = prepare_background_data(roles, documents, permissions)

    # Step 3: Initialize partitions and role mappings based on the x values
    initialize_partitions_and_role_mappings(partition_assignment, comb_role_tracker, document_to_index,
                                            increment_update=increment_update, table_prefix=table_prefix)

    print("Partition and role mappings initialization completed.")
