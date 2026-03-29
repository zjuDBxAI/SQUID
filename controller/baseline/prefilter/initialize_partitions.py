import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import psycopg2
from psycopg2 import sql

from controller.clear_database import clear_tables
from services.config import get_db_connection, get_maintenance_settings, get_document_vector_dimension


def _configure_index_session(cur, disable_sync_commit=True, hnsw_threads=None):
    """
    Apply session-level settings that speed up bulk index builds.
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


def drop_prefilter_partition_tables(condition="role"):
    """
    Drops user partition tables, role partition tables, combination partition tables,
    or any related main partitioned tables based on the specified condition.

    Args:
        condition (str): Determines which type of partition tables to drop.
                         Options are "user", "role", "combination", or "all".
                         Default is "role".
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        if condition in {"user", "all"}:
            # Drop all user-specific partition tables
            cur.execute("""
                SELECT tablename 
                FROM pg_tables 
                WHERE tablename LIKE 'documentblocks_user_%';
            """)
            partition_tables = cur.fetchall()

            for table_name in partition_tables:
                try:
                    cur.execute(f"DROP TABLE IF EXISTS \"{table_name[0]}\" CASCADE;")
                    conn.commit()
                    print(f"User partition table {table_name[0]} dropped successfully.")
                except psycopg2.Error as e:
                    print(f"Error dropping user partition table {table_name[0]}: {e}")
                    conn.rollback()

        if condition in {"role", "all"}:
            # Drop all role-specific partition tables
            cur.execute("""
                SELECT tablename 
                FROM pg_tables 
                WHERE tablename LIKE 'documentblocks_role_%';
            """)
            partition_tables = cur.fetchall()

            for table_name in partition_tables:
                try:
                    cur.execute(f"DROP TABLE IF EXISTS \"{table_name[0]}\" CASCADE;")
                    conn.commit()
                    print(f"Role partition table {table_name[0]} dropped successfully.")
                except psycopg2.Error as e:
                    print(f"Error dropping role partition table {table_name[0]}: {e}")
                    conn.rollback()

        if condition in {"combination", "all"}:
            # Drop all combination-specific partition tables
            cur.execute("""
                SELECT tablename 
                FROM pg_tables 
                WHERE tablename LIKE 'documentblocks_comb_%';
            """)
            partition_tables = cur.fetchall()

            for table_name in partition_tables:
                try:
                    cur.execute(f"DROP TABLE IF EXISTS \"{table_name[0]}\" CASCADE;")
                    conn.commit()
                    print(f"Combination partition table {table_name[0]} dropped successfully.")
                except psycopg2.Error as e:
                    print(f"Error dropping combination partition table {table_name[0]}: {e}")
                    conn.rollback()

    finally:
        cur.close()
        conn.close()


def initialize_user_partitions(enable_index=False):
    conn = get_db_connection()
    cur = conn.cursor()
    vector_dimension = get_document_vector_dimension()

    try:
        # Retrieve all user_ids and create an independent table for each user
        cur.execute("SELECT user_id FROM Users;")
        users = cur.fetchall()

        for user_id, in users:
            table_name = f"documentblocks_user_{user_id}"
            try:
                # Create an independent table for each user
                cur.execute(
                    sql.SQL("""
                               CREATE TABLE IF NOT EXISTS {table} (
                                   id SERIAL PRIMARY KEY,
                                   block_id BIGINT NOT NULL,
                                   document_id INT NOT NULL REFERENCES Documents(document_id),
                                   block_content BYTEA NOT NULL,
                                   vector VECTOR({dimension}),
                                   user_id INT NOT NULL REFERENCES Users(user_id),
                                   UNIQUE (block_id, document_id, user_id)
                               );
                           """).format(table=sql.Identifier(table_name),
                                       dimension=sql.SQL(str(vector_dimension)))
                )
                conn.commit()

                if enable_index:
                    # Create an index on the vector column for fast similarity search
                    cur.execute(
                        sql.SQL("""
                            CREATE INDEX IF NOT EXISTS {}_document_id_idx
                            ON {} (document_id);
                        """).format(sql.Identifier(f"{table_name}_document_id"), sql.Identifier(table_name))
                    )
                    cur.execute(
                        sql.SQL("""
                            CREATE INDEX IF NOT EXISTS {}_vector_idx
                            ON {} USING ivfflat (vector);
                        """).format(sql.Identifier(f"{table_name}_vector"), sql.Identifier(table_name))
                    )
                    cur.execute(
                        sql.SQL("""
                            CREATE INDEX IF NOT EXISTS {}_document_id_vector_idx
                            ON {} (document_id, vector);
                        """).format(sql.Identifier(f"{table_name}_document_id_vector"), sql.Identifier(table_name))
                    )
                    conn.commit()

            except psycopg2.Error as e:
                print(f"Error creating table for user_id {user_id}: {e}")
                conn.rollback()

            try:
                # Copy data to the user's table based on user_id, role_id, and document_id
                cur.execute(
                    sql.SQL("""
                               INSERT INTO {} (block_id, document_id, block_content, vector, user_id)
                               SELECT DISTINCT db.block_id, db.document_id, db.block_content, db.vector, ur.user_id
                               FROM documentblocks db
                               JOIN PermissionAssignment pa ON db.document_id = pa.document_id
                               JOIN UserRoles ur ON pa.role_id = ur.role_id
                               WHERE ur.user_id = %s;
                           """).format(sql.Identifier(table_name)),
                    [user_id]
                )
                conn.commit()

                # Verify data consistency after insertion
                # cur.execute(
                #     sql.SQL("""
                #                SELECT DISTINCT pa.document_id
                #                FROM PermissionAssignment pa
                #                JOIN UserRoles ur ON pa.role_id = ur.role_id
                #                WHERE ur.user_id = %s;
                #            """),
                #     [user_id]
                # )
                # expected_document_ids = set(row[0] for row in cur.fetchall())
                #
                # cur.execute(
                #     sql.SQL("SELECT DISTINCT document_id FROM {};").format(sql.Identifier(table_name))
                # )
                # actual_document_ids = set(row[0] for row in cur.fetchall())
                #
                # assert expected_document_ids == actual_document_ids, (
                #     f"Inconsistency found for user_id {user_id}: "
                #     f"Expected {expected_document_ids}, but found {actual_document_ids}"
                # )
                #
                # print(f"Data consistency verified for user_id {user_id}.")

            except psycopg2.Error as e:
                print(f"Error inserting data into table for user_id {user_id}: {e}")
                conn.rollback()

    finally:
        cur.close()
        conn.close()


def verify_documentblocks_consistency():
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Retrieve all role_ids
        cur.execute("SELECT role_id FROM Roles;")
        roles = cur.fetchall()

        inconsistent_roles = []

        # Loop through each role to verify its associated table
        for role_id, in roles:
            table_name = f"documentblocks_role_{role_id}"

            try:
                # Step 1: Fetch distinct document_ids from the role's table
                cur.execute(sql.SQL("SELECT DISTINCT document_id FROM {};").format(sql.Identifier(table_name)))
                role_document_ids = set(row[0] for row in cur.fetchall())

                # Step 2: Fetch distinct document_ids from PermissionAssignment for the role
                cur.execute("""
                    SELECT DISTINCT pa.document_id
                    FROM PermissionAssignment pa
                    WHERE pa.role_id = %s;
                """, [role_id])
                expected_document_ids = set(row[0] for row in cur.fetchall())

                # Step 3: Compare document_ids in the role table and PermissionAssignment
                if role_document_ids != expected_document_ids:
                    inconsistent_roles.append({
                        "role_id": role_id,
                        "missing_in_role_table": expected_document_ids - role_document_ids,
                        "extra_in_role_table": role_document_ids - expected_document_ids
                    })

            except psycopg2.Error as e:
                print(f"Error verifying data for role_id {role_id}: {e}")

        # Print out any inconsistencies found
        if inconsistent_roles:
            print(f"Inconsistencies found for {len(inconsistent_roles)} role(s):")
            for role in inconsistent_roles:
                print(f"Role ID: {role['role_id']}")
                print(f"  Missing in role table: {role['missing_in_role_table']}")
                print(f"  Extra in role table: {role['extra_in_role_table']}")
        else:
            print("All role tables are consistent with PermissionAssignment.")

    finally:
        cur.close()
        conn.close()


def process_role_partition(role_id, enable_index=False, index_type="ivfflat", vector_dimension=300):
    conn = get_db_connection()
    cur = conn.cursor()
    table_name = f"documentblocks_role_{role_id}"

    # Load maintenance settings from config for repeatable behaviour
    maintenance_settings = get_maintenance_settings()
    maintenance_work_mem_gb = maintenance_settings["maintenance_work_mem_gb"]
    max_parallel_maintenance_workers = maintenance_settings["max_parallel_maintenance_workers"]

    # Set PostgreSQL parameters for optimal index creation
    cur.execute(f"SET maintenance_work_mem = '{maintenance_work_mem_gb}GB';")
    cur.execute(f"SET max_parallel_maintenance_workers = {max_parallel_maintenance_workers};")
    print(f"PostgreSQL parameters set: maintenance_work_mem = {maintenance_work_mem_gb}GB, "
          f"max_parallel_maintenance_workers = {max_parallel_maintenance_workers}")
    data_inserted = False

    try:
        try:
            # Create an independent table for each role
            cur.execute(
                sql.SQL("""
                    CREATE TABLE IF NOT EXISTS {table} (
                        block_id BIGINT NOT NULL,
                        document_id INT NOT NULL,
                        block_content BYTEA NOT NULL,
                        vector VECTOR({dimension}),
                        role_id INT NOT NULL,
                        UNIQUE (block_id, document_id, role_id)
                    );
                """).format(table=sql.Identifier(table_name),
                            dimension=sql.SQL(str(vector_dimension)))
            )
            conn.commit()
        except psycopg2.Error as e:
            print(f"Error creating table for role_id {role_id}: {e}")
            conn.rollback()
            return

        try:
            # Insert data based on role_id and document_id
            cur.execute(
                sql.SQL("""
                    INSERT INTO {} (block_id, document_id, block_content, vector, role_id)
                    SELECT DISTINCT db.block_id, db.document_id, db.block_content, db.vector, pa.role_id
                    FROM documentblocks db
                    JOIN PermissionAssignment pa ON db.document_id = pa.document_id
                    WHERE pa.role_id = %s;
                """).format(sql.Identifier(table_name)),
                [role_id]
            )
            conn.commit()
            data_inserted = True
        except psycopg2.Error as e:
            print(f"Error inserting data into table for role_id {role_id}: {e}")
            conn.rollback()

        if enable_index and data_inserted:
            try:
                # Dynamically create vector index based on the index type (HNSW or IVFFlat)
                if index_type.lower() == "hnsw":
                    cur.execute(
                        sql.SQL("""
                            CREATE INDEX IF NOT EXISTS {} 
                            ON {} USING hnsw (vector vector_l2_ops)
                            WITH (m = 16, ef_construction = 64);
                        """).format(sql.Identifier(f"{table_name}_vector_idx"), sql.Identifier(table_name))
                    )
                    print(f"HNSW index created for {table_name}.")
                elif index_type.lower() == "ivfflat":
                    cur.execute(
                        sql.SQL("""
                            CREATE INDEX IF NOT EXISTS {} 
                            ON {} USING ivfflat (vector vector_l2_ops);
                        """).format(sql.Identifier(f"{table_name}_vector_idx"), sql.Identifier(table_name))
                    )
                    print(f"IVFFlat index created for {table_name}.")
                else:
                    print(f"Unknown index type: {index_type}. Please use 'hnsw' or 'ivfflat'.")

                conn.commit()
            except psycopg2.Error as e:
                print(f"Error creating index for role_id {role_id}: {e}")
                conn.rollback()
    finally:
        cur.close()
        conn.close()


def initialize_role_partitions(enable_index=False, index_type="ivfflat"):
    conn = get_db_connection()
    cur = conn.cursor()

    # Retrieve all role_ids
    cur.execute("SELECT role_id FROM Roles;")
    roles = cur.fetchall()
    cur.close()
    conn.close()

    vector_dimension = get_document_vector_dimension()

    # Process each role in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(process_role_partition, role_id[0], enable_index, index_type, vector_dimension)
            for role_id in roles
        ]

        # Wait for all futures to complete
        for future in futures:
            future.result()

    print("Finished processing all role partitions.")



def create_index_for_role(role_id, index_type="ivfflat", hnsw_m=16, hnsw_ef_construction=64,
                          hnsw_threads=None, disable_sync_commit=True):
    conn = get_db_connection()
    cur = conn.cursor()
    table_name = f"documentblocks_role_{role_id}"

    try:
        _configure_index_session(cur, disable_sync_commit=disable_sync_commit, hnsw_threads=hnsw_threads)

        # Dynamically create vector index based on the index type (HNSW or IVFFlat)
        if index_type.lower() == "hnsw":
            with_clause = sql.SQL("WITH (m = {m}, ef_construction = {ef})").format(
                m=sql.Literal(int(hnsw_m)),
                ef=sql.Literal(int(hnsw_ef_construction))
            )
            cur.execute(
                sql.SQL("""
                    CREATE INDEX IF NOT EXISTS {} 
                    ON {} USING hnsw (vector vector_l2_ops)
                    {with_clause};
                """).format(
                    sql.Identifier(f"{table_name}_vector_idx"),
                    sql.Identifier(table_name),
                    with_clause=with_clause
                )
            )
            print(f"HNSW index created for {table_name}.")
        elif index_type.lower() == "ivfflat":
            cur.execute(
                sql.SQL("""
                    CREATE INDEX IF NOT EXISTS {} 
                    ON {} USING ivfflat (vector vector_l2_ops);
                """).format(sql.Identifier(f"{table_name}_vector_idx"), sql.Identifier(table_name))
            )
            print(f"IVFFlat index created for {table_name}.")
        else:
            print(f"Unknown index type: {index_type}. Please use 'hnsw' or 'ivfflat'.")

        conn.commit()

    except psycopg2.Error as e:
        print(f"Error creating index for role_id {role_id}: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def create_indexes_for_all_role_tables(index_type="ivfflat", parallel=True, max_workers=None,
                                       hnsw_m=16, hnsw_ef_construction=64, hnsw_threads=None,
                                       disable_sync_commit=True):
    conn = get_db_connection()
    cur = conn.cursor()

    # Retrieve all role_ids
    cur.execute("SELECT role_id FROM Roles;")
    roles = cur.fetchall()

    cur.close()
    conn.close()

    if parallel:
        # Process each role in parallel using ProcessPoolExecutor
        worker_count = max_workers or max(1, os.cpu_count() // 2)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    create_index_for_role,
                    role_id[0],
                    index_type,
                    hnsw_m,
                    hnsw_ef_construction,
                    hnsw_threads,
                    disable_sync_commit
                )
                for role_id in roles
            ]

            # Wait for all futures to complete
            for future in futures:
                future.result()

        print("Finished creating indexes for all role tables.")
    else:
        for role_id in roles:
            create_index_for_role(
                role_id[0],
                index_type,
                hnsw_m,
                hnsw_ef_construction,
                hnsw_threads,
                disable_sync_commit
            )


def drop_indexes_for_all_role_tables():
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Retrieve all role_ids
        cur.execute("SELECT role_id FROM Roles;")
        roles = cur.fetchall()

        for role_id, in roles:
            table_name = f"documentblocks_role_{role_id}"

            # Retrieve all constraint names related to the current table
            cur.execute(
                sql.SQL("""
                    SELECT conname
                    FROM pg_constraint
                    WHERE conrelid = %s::regclass;
                """),
                [table_name]
            )
            constraints = cur.fetchall()

            # Drop each constraint in the table
            for (constraint_name,) in constraints:
                cur.execute(
                    sql.SQL("""
                        ALTER TABLE {} DROP CONSTRAINT IF EXISTS {};
                    """).format(sql.Identifier(table_name), sql.Identifier(constraint_name))
                )
                print(f"Constraint {constraint_name} dropped for {table_name}.")

            # Retrieve all index names for the current table
            cur.execute(
                sql.SQL("""
                    SELECT indexname
                    FROM pg_indexes
                    WHERE tablename = %s;
                """),
                [table_name]
            )
            indexes = cur.fetchall()

            # Drop each index in the table with CASCADE
            for (index_name,) in indexes:
                cur.execute(
                    sql.SQL("""
                        DROP INDEX IF EXISTS {} CASCADE;
                    """).format(sql.Identifier(index_name))
                )
                print(f"Index {index_name} dropped for {table_name} with CASCADE.")

        conn.commit()
    except psycopg2.Error as e:
        print(f"Error dropping indexes: {e}")
        conn.rollback()
        sys.exit(1)  # Exit the program with an error code
    finally:
        cur.close()
        conn.close()



def initialize_combination_partitions(enable_index=False, index_type="ivfflat"):
    """
    Initialize combination-based partitions for all unique user role combinations.

    Args:
        enable_index (bool): Whether to create indexes on the partitions.
        index_type (str): Type of index to create ('hnsw' or 'ivfflat').

    Returns:
        None
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # Retrieve unique role combinations
    cur.execute("""
        SELECT DISTINCT array_agg(role_id ORDER BY role_id) AS roles
        FROM userroles
        GROUP BY user_id
        ORDER BY roles;
    """)
    role_combinations = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()

    # Determine the number of workers and split combinations
    num_workers = os.cpu_count()
    chunk_size = math.ceil(len(role_combinations) / num_workers)
    role_combination_chunks = [
        role_combinations[i:i + chunk_size]
        for i in range(0, len(role_combinations), chunk_size)
    ]

    print(f"Divided {len(role_combinations)} role combinations into {len(role_combination_chunks)} chunks.")

    vector_dimension = get_document_vector_dimension()

    # Process each chunk in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_combination_partition_chunk,
                chunk,
                enable_index,
                index_type,
                vector_dimension
            )
            for chunk in role_combination_chunks
        ]

        # Wait for all tasks to complete
        for future in futures:
            future.result()

    print("Finished initializing all combination partitions.")

def process_combination_partition_chunk(role_combination_chunk, enable_index, index_type, vector_dimension):
    """
    Process a chunk of role combinations to create combination partitions.

    Args:
        role_combination_chunk (list): A chunk of role combinations to process.
        enable_index (bool): Whether to create indexes on the partitions.
        index_type (str): Type of index to create ('hnsw' or 'ivfflat').

    Returns:
        None
    """
    for role_combination in role_combination_chunk:
        try:
            process_combination_partition(role_combination, enable_index, index_type, vector_dimension)
            print(f"Successfully processed combination partition for roles: {role_combination}")
        except Exception as e:
            print(f"Error processing combination partition for roles {role_combination}: {e}")


def process_combination_partition(roles, enable_index=False, index_type="ivfflat", vector_dimension=300):
    """
    Create and populate a table for a specific combination of roles.

    Args:
        roles (list): List of role IDs in the combination.
        enable_index (bool): Whether to create indexes on the partition.
        index_type (str): Type of index to create ('hnsw' or 'ivfflat').

    Returns:
        None
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # Generate table name based on the role combination
    table_name = f"documentblocks_combination_{'_'.join(map(str, roles))}"

    try:
        # Create a table for the role combination
        cur.execute(
            sql.SQL("""
                CREATE TABLE IF NOT EXISTS {table} (
                    block_id BIGINT NOT NULL,
                    document_id INT NOT NULL,
                    block_content BYTEA NOT NULL,
                    vector VECTOR({dimension}),
                    UNIQUE (block_id, document_id)
                );
            """).format(
                table=sql.Identifier(table_name),
                dimension=sql.SQL(str(vector_dimension))
            )
        )
        conn.commit()

        # Create indexes if enabled
        if enable_index:
            if index_type.lower() == "hnsw":
                cur.execute(
                    sql.SQL("""
                        CREATE INDEX IF NOT EXISTS {} 
                        ON {} USING hnsw (vector vector_l2_ops)
                        WITH (m = 16, ef_construction = 64);
                    """).format(
                        sql.Identifier(f"{table_name}_vector_idx"),
                        sql.Identifier(table_name)
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
            else:
                print(f"Unknown index type: {index_type}. Please use 'hnsw' or 'ivfflat'.")

        conn.commit()

    except psycopg2.Error as e:
        print(f"Error creating table for roles {roles}: {e}")
        conn.rollback()

    try:
        # Populate the table with data
        cur.execute(
            sql.SQL("""
                INSERT INTO {} (block_id, document_id, block_content, vector)
                SELECT DISTINCT db.block_id, db.document_id, db.block_content, db.vector
                FROM documentblocks db
                JOIN PermissionAssignment pa ON db.document_id = pa.document_id
                WHERE pa.role_id = ANY(%s);
            """).format(
                sql.Identifier(table_name)
            ),
            [roles]
        )
        conn.commit()
    except psycopg2.Error as e:
        print(f"Error inserting data into table for roles {roles}: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def create_index_for_combination(roles, index_type="ivfflat"):
    """
    Create a vector index for a specific combination-based partition.

    Args:
        roles (list): List of role IDs in the combination.
        index_type (str): Type of index to create ('hnsw' or 'ivfflat').

    Returns:
        None
    """
    conn = get_db_connection()
    cur = conn.cursor()
    table_name = f"documentblocks_combination_{'_'.join(map(str, roles))}"

    try:
        # Dynamically create vector index based on the index type (HNSW or IVFFlat)
        if index_type.lower() == "hnsw":
            cur.execute(
                sql.SQL("""
                    CREATE INDEX IF NOT EXISTS {} 
                    ON {} USING hnsw (vector vector_l2_ops)
                    WITH (m = 16, ef_construction = 64);
                """).format(
                    sql.Identifier(f"{table_name}_vector_idx"),
                    sql.Identifier(table_name)
                )
            )
            print(f"Successfully created HNSW index for {table_name}.")
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
            print(f"Successfully created IVFFlat index for {table_name}.")
        else:
            print(f"Unknown index type: {index_type}. Please use 'hnsw' or 'ivfflat'.")

        conn.commit()

    except psycopg2.Error as e:
        print(f"Error creating index for combination {roles}: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def create_indexes_for_all_combination_tables(index_type="ivfflat"):
    """
    Create vector indexes for all combination-based partitions.

    Args:
        index_type (str): Type of index to create ('hnsw' or 'ivfflat').

    Returns:
        None
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # Retrieve unique role combinations
    cur.execute("""
        SELECT DISTINCT array_agg(role_id ORDER BY role_id) AS roles
        FROM userroles
        GROUP BY user_id
        ORDER BY roles;
    """)
    combinations = cur.fetchall()
    cur.close()
    conn.close()

    # Process each combination in parallel
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for combination in combinations:
            future = executor.submit(create_index_for_combination, combination[0], index_type)
            futures.append(future)
            print(f"Scheduled creation for combination: {combination[0]}")  # Output to console immediately

        # Wait for all tasks to complete
        for future in futures:
            future.result()

    print("Finished creating indexes for all combination tables.")


def drop_indexes_for_all_combination_tables():
    """
    Drop all indexes for combination-based partitions.

    Returns:
        None
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Retrieve unique role combinations
        cur.execute("""
            SELECT array_agg(role_id ORDER BY role_id) AS roles
            FROM userroles
            GROUP BY user_id;
        """)
        combinations = cur.fetchall()

        for combination in combinations:
            table_name = f"documentblocks_combination_{'_'.join(map(str, combination[0]))}"

            # Retrieve all constraint names related to the current table
            cur.execute(
                sql.SQL("""
                    SELECT conname
                    FROM pg_constraint
                    WHERE conrelid = %s::regclass;
                """),
                [table_name]
            )
            constraints = cur.fetchall()

            # Drop each constraint in the table
            for (constraint_name,) in constraints:
                cur.execute(
                    sql.SQL("""
                        ALTER TABLE {} DROP CONSTRAINT IF EXISTS {};
                    """).format(sql.Identifier(table_name), sql.Identifier(constraint_name))
                )
                print(f"Constraint {constraint_name} dropped for {table_name}.")

            # Retrieve all index names for the current table
            cur.execute(
                sql.SQL("""
                    SELECT indexname
                    FROM pg_indexes
                    WHERE tablename = %s;
                """),
                [table_name]
            )
            indexes = cur.fetchall()

            # Drop each index in the table with CASCADE
            for (index_name,) in indexes:
                cur.execute(
                    sql.SQL("""
                        DROP INDEX IF EXISTS {} CASCADE;
                    """).format(sql.Identifier(index_name))
                )
                print(f"Index {index_name} dropped for {table_name} with CASCADE.")

        conn.commit()
    except psycopg2.Error as e:
        print(f"Error dropping indexes: {e}")
        conn.rollback()
        sys.exit(1)  # Exit the program with an error code
    finally:
        cur.close()
        conn.close()
