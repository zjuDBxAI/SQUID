import os
import re
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)
from controller.dynamic_partition.hnsw.helper import calculate_hnsw_user_avg_qps, calculate_hnsw_recall, \
    save_solution_to_file, compute_role_partition_access, fetch_initial_data, prepare_background_data, \
    delete_faiss_files
from controller.dynamic_partition.get_parameter import get_alpha2, get_alpha_beta_gamma_hashjointime, \
    save_parameter_to_json, get_recall_parameters, get_QPS_parameters

def fetch_partition_assignment():
    """
    Query all partition tables and retrieve document assignments.

    Returns:
        dict: Mapping of partition_id to set of document_ids
    """
    conn = get_db_connection()
    cur = conn.cursor()

    partition_assignment = {}

    try:
        # Step 1: Fetch all partition table names
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name LIKE 'documentblocks_partition_%';
        """)
        partition_tables = [row[0] for row in cur.fetchall()]

        # Step 2: Retrieve document_ids for each partition
        for table_name in partition_tables:
            partition_id = int(table_name.split('_')[-1])  # Extract numeric partition_id
            cur.execute(sql.SQL("SELECT DISTINCT document_id FROM {};").format(sql.Identifier(table_name)))
            document_ids = {row[0] for row in cur.fetchall()}
            partition_assignment[partition_id] = document_ids

        print(f"[INFO] Loaded partition assignment for {len(partition_assignment)} partitions.")
        return partition_assignment

    except psycopg2.Error as e:
        print(f"[ERROR] Failed to fetch partition assignment: {e}")
        return {}

    finally:
        cur.close()
        conn.close()


def fetch_partition_role_mapping():
    """
    Query combrolepartitions to retrieve role assignments for each partition.

    Returns:
        dict: Mapping of partition_id to set of roles assigned to it.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    partition_roles = {}

    try:
        cur.execute("SELECT partition_id, comb_role FROM combrolepartitions;")
        for partition_id, role_array in cur.fetchall():
            role_tuple = tuple(sorted(role_array))  # Convert to tuple for consistency
            if partition_id not in partition_roles:
                partition_roles[partition_id] = set()
            partition_roles[partition_id].update(role_tuple)

        print(f"[INFO] Loaded role mappings for {len(partition_roles)} partitions.")
        return partition_roles

    except psycopg2.Error as e:
        print(f"[ERROR] Failed to fetch role mappings: {e}")
        return {}

    finally:
        cur.close()
        conn.close()


import psycopg2
from psycopg2 import sql
import math

import json
import os
import math
import psycopg2
from psycopg2 import sql
from services.config import get_db_connection

import json
import os
import math
import random
import psycopg2
from psycopg2 import sql
from services.config import get_db_connection

import random
import random
import random


def generate_users_for_role(new_role_id, user_to_roles, num_users, max_roles_per_user, existing_roles):
    """
    Generate new users and assign them roles, including the new role.

    Args:
        new_role_id (int): The new role ID to be assigned.
        user_to_roles (dict): Mapping of users to their existing roles (user_id -> list of role_ids).
        num_users (int): Number of new users to create.
        max_roles_per_user (int): Maximum roles per user.
        existing_roles (set): Set of all existing role IDs in the database.

    Returns:
        list: A list of (user_id, role_id) tuples for insertion into UserRoles.
        list: A list of (user_id, user_name) tuples for insertion into Users.
        list: A list of (role_id, role_name) tuples for insertion into Roles (only if new_role_id is missing).
    """
    # Ensure new_role_id exists in roles table
    new_roles = []
    if new_role_id not in existing_roles:
        new_roles.append((new_role_id, f"role_{new_role_id}"))

    # Extract all unique roles
    all_roles = set(existing_roles)

    existing_users = list(user_to_roles.keys())
    new_user_roles = []
    new_users = []
    new_user_start_id = max(existing_users) + 1 if existing_users else 1

    for i in range(num_users):
        user_id = new_user_start_id + i
        user_name = f"user_{user_id}"
        new_users.append((user_id, user_name))

        num_roles = random.randint(1, max_roles_per_user)
        selected_roles = {new_role_id}
        available_roles = list(all_roles - {new_role_id})

        if len(available_roles) >= num_roles - 1:
            selected_roles.update(random.sample(available_roles, num_roles - 1))
        else:
            selected_roles.update(available_roles)

        for role in selected_roles:
            new_user_roles.append((user_id, role))

    return new_user_roles, new_users, new_roles


import math
import psycopg2
from psycopg2 import sql
from services.config import get_db_connection


def insert_new_role(new_role_id, new_role_documents, partition_assignment, partition_roles,
                    topk, k, beta, a, b):
    """
    Find the best partition for a new role based on ΔQueryTime / ΔStorage,
    considering both existing partitions and a potential new partition.

    Args:
        new_role_id (int): The ID of the new role.
        new_role_documents (set): The set of document IDs associated with the new role.
        partition_assignment (dict): Mapping of partition IDs to their respective document sets.
        partition_roles (dict): Mapping of partition IDs to assigned roles.
        topk (int): Number of top results required.
        k (float): Recall parameter.
        beta (float): Query performance parameter.
        a (float): Query scaling factor.
        b (float): Query base factor.

    Returns:
        int: The best partition ID to insert the new role.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    x = 3
    while (1 + x / 10) - k >= 1:
        x -= 1
    dynamic_value = 1 + x / 10

    try:
        # Step 1: Retrieve all partition tables and their document assignments
        cur.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_name LIKE 'documentblocks_partition_%';
        """)
        partition_tables = {row[0] for row in cur.fetchall()}  # Fetch all partition tables

        # Step 2: Parse partition_assignment to get all documents assigned to each partition
        partition_docs_map = {}
        for partition in partition_tables:
            partition_id = int(partition.split("_")[-1])  # Extract partition ID from table name
            cur.execute(sql.SQL("SELECT DISTINCT document_id FROM {};").format(sql.Identifier(partition)))
            partition_docs_map[partition_id] = {row[0] for row in cur.fetchall()}  # Store partition -> document set

        # Step 3: Compute ΔQueryTime / ΔStorage
        partition_costs = {}  # Store partition_id -> (ΔQueryTime / ΔStorage)
        for partition_id, partition_docs in partition_docs_map.items():
            existing_role_sels = [
                len(partition_docs & role_to_documents.get(role, set())) / len(partition_docs)
                if partition_docs else 0
                for role in partition_roles.get(partition_id, [])
            ]
            new_role_sel = len(new_role_documents & partition_docs) / len(partition_docs) if partition_docs else 0

            # Compute sel_avg before and after insertion
            sel_avg_before = sum(existing_role_sels) / len(existing_role_sels) if existing_role_sels else 0
            sel_avg_after = (sum(existing_role_sels) + new_role_sel) / (
                        len(existing_role_sels) + 1) if existing_role_sels else new_role_sel

            # Compute ef_search before and after insertion
            ef_search_before = math.log(1 / (dynamic_value - k) - 1) / (
                        -4 * beta * sel_avg_before) * topk + k * topk / sel_avg_before
            ef_search_after = math.log(1 / (dynamic_value - k) - 1) / (
                        -4 * beta * sel_avg_after) * topk + k * topk / sel_avg_after

            # Compute query_time before and after insertion
            query_time_before = math.log(len(partition_docs)) * (a * ef_search_before + b)
            query_time_after = math.log(len(partition_docs) + len(new_role_documents)) * (a * ef_search_after + b)

            # Compute ΔQueryTime and ΔStorage
            delta_query_time = query_time_after - query_time_before
            updated_partition_docs = partition_docs | new_role_documents  # New document set after insertion
            delta_storage = len(updated_partition_docs) - len(partition_docs)  # Actual storage change

            # Compute ΔQueryTime / ΔStorage
            if delta_storage > 0:
                partition_costs[partition_id] = delta_query_time / delta_storage
            else:
                partition_costs[partition_id] = float("inf")  # Avoid division by zero

        # Step 4: **Include a new empty partition as an option**
        # 1. Compute additional storage cost if a new partition is created (equal to new_role_documents size)
        # 2. Assume new partition has no other roles, so sel_avg = new_role_sel
        new_partition_id = max(partition_docs_map.keys(), default=0) + 1  # Generate a new partition ID
        new_partition_sel = 1  # Assume this partition is dedicated to the new role, so sel = 1
        ef_search_new_partition = math.log(1 / (dynamic_value - k) - 1) / (
                    -4 * beta * new_partition_sel) * topk + k * topk / new_partition_sel
        query_time_new_partition = math.log(len(new_role_documents)) * (a * ef_search_new_partition + b)

        delta_query_time_new_partition = query_time_new_partition  # Since the partition didn't exist before, ΔQueryTime = query_time itself
        delta_storage_new_partition = len(
            new_role_documents)  # Directly use the number of new documents as storage cost

        # Compute ΔQueryTime / ΔStorage for the new partition
        if delta_storage_new_partition > 0:
            partition_costs[new_partition_id] = delta_query_time_new_partition / delta_storage_new_partition
        else:
            partition_costs[new_partition_id] = float("inf")  # Avoid division by zero

        # Step 5: **Select the partition with the smallest ΔQueryTime / ΔStorage**
        best_partition = min(partition_costs, key=partition_costs.get)
        if best_partition == new_partition_id:
            print(f"Selected NEW partition {best_partition} for new role {new_role_id}")
        else:
            print(f"Selected existing partition {best_partition} for new role {new_role_id}")

        return best_partition

    except psycopg2.Error as e:
        print(f"Database error while processing new role: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


import psycopg2
from psycopg2 import sql
from services.config import get_db_connection

def update_database_for_new_role(new_role_id, best_partition, new_role_documents, new_user_roles, new_users, new_roles):
    """
    Update the database with the new role, including partition assignments, user-role mappings,
    role combinations, and partition index reconstruction.

    Args:
        new_role_id (int): The new role ID.
        best_partition (int): The partition to insert the new role into.
        new_role_documents (set): The documents assigned to the new role.
        new_user_roles (list): List of (user_id, role_id) tuples for UserRoles table.
        new_users (list): List of (user_id, user_name) tuples for Users table.
        new_roles (list): List of (role_id, role_name) tuples for Roles table.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Step 1: Insert new role into roles table
        if new_roles:
            cur.executemany("""
                INSERT INTO Roles (role_id, role_name)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING;
            """, new_roles)

        # Step 2: Insert new users
        cur.executemany("""
            INSERT INTO Users (user_id, user_name)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING;
        """, new_users)

        # Step 3: Insert user-role mappings
        cur.executemany("""
            INSERT INTO UserRoles (user_id, role_id)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING;
        """, new_user_roles)

        # Step 4: Insert new role's document permissions
        for document_id in new_role_documents:
            cur.execute("""
                INSERT INTO PermissionAssignment (role_id, document_id)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING;
            """, (new_role_id, document_id))

        # Step 5: Update combrolepartitions for new role
        cur.execute("""
            INSERT INTO CombRolePartitions (comb_role, partition_id)
            VALUES (ARRAY[%s], %s)
            ON CONFLICT (comb_role, partition_id) DO NOTHING;
        """, (new_role_id, best_partition))

        # Step 6: Insert the new role's documents into the best partition
        partition_table = f"documentblocks_partition_{best_partition}"
        for document_id in new_role_documents:
            cur.execute(
                sql.SQL("""
                    INSERT INTO {} (block_id, document_id, block_content, vector)
                    SELECT block_id, document_id, block_content, vector
                    FROM documentblocks
                    WHERE document_id = %s
                    ON CONFLICT (block_id, document_id) DO NOTHING;
                """).format(sql.Identifier(partition_table)),
                [document_id]
            )


        # Step 6: Update partition assignment
        partition_table = f"documentblocks_partition_{best_partition}"

        # Step 7: Rebuild HNSW Index
        print(f"[INFO] Rebuilding HNSW index for partition {best_partition}...")
        cur.execute(
            sql.SQL("DROP INDEX IF EXISTS {};").format(
                sql.Identifier(f"{partition_table}_vector_idx")
            )
        )
        cur.execute(
            sql.SQL("""
                CREATE INDEX {} 
                ON {} USING hnsw (vector vector_l2_ops)
                WITH (m = 16, ef_construction = 64);
            """).format(
                sql.Identifier(f"{partition_table}_vector_idx"),
                sql.Identifier(partition_table)
            )
        )

        conn.commit()
        print(f"[INFO] Database updated for new role {new_role_id} in partition {best_partition}.")

    except psycopg2.Error as e:
        print(f"[ERROR] Failed to update database for new role {new_role_id}: {e}")
        conn.rollback()

    finally:
        cur.close()
        conn.close()



def update_partition_assignment(partition_id, new_documents):
    """
    Insert new documents into the specified partition.

    Args:
        partition_id (int): The target partition ID
        new_documents (set): The set of new document IDs
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        partition_table = f"documentblocks_partition_{partition_id}"
        for document_id in new_documents:
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
        print(f"[INFO] Inserted {len(new_documents)} documents into partition {partition_id}.")

    except psycopg2.Error as e:
        print(f"[ERROR] Failed to update partition {partition_id}: {e}")
        conn.rollback()

    finally:
        cur.close()
        conn.close()


def update_partition_role_mapping(partition_id, new_role_id):
    """
    Update combrolepartitions table to map the new role to its assigned partition.

    Args:
        partition_id (int): The target partition ID
        new_role_id (int): The new role ID
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            INSERT INTO combrolepartitions (comb_role, partition_id)
            VALUES (ARRAY[%s], %s)
            ON CONFLICT (comb_role, partition_id) DO NOTHING;
        """, (new_role_id, partition_id))

        conn.commit()
        print(f"[INFO] Role {new_role_id} mapped to partition {partition_id}.")

    except psycopg2.Error as e:
        print(f"[ERROR] Failed to update role mapping for {new_role_id}: {e}")
        conn.rollback()

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    partition_assignment = fetch_partition_assignment()
    partition_roles = fetch_partition_role_mapping()
    from basic_benchmark.initialize_dynamic_partition_tables import initialize_dynamic_partition_tables, \
        initialize_dynamic_partition_tables_in_comb

    roles, documents, permissions, avg_blocks_per_document, user_to_roles = fetch_initial_data()
    # Prepare background data
    role_to_documents, document_to_index = prepare_background_data(roles, documents, permissions)

    # Number of new roles to insert
    num_roles_to_insert = 3  # You can adjust this value
    initial_new_role_id = 101  # Starting new_role_id

    # Load parameter configurations
    json_file_path = "parameter_hnsw.json"
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as json_file:
            result_data = json.load(json_file)
            print("Data loaded from parameter_hnsw.json")
    else:
        # Generate parameters if the file does not exist
        params_recall = get_recall_parameters(index_type="hnsw")
        k = params_recall[0]
        beta = params_recall[1]
        params_qps, join_times = get_QPS_parameters(index_type="hnsw")
        a = params_qps[0]
        b = params_qps[1]
        print("Parameters:")
        print(f"  k: {k}")
        print(f"  beta: {beta}")
        print(f"  a: {a}")
        print(f"  b: {b}")
        print(f"  join_times: {join_times}")
        result_data = {
            "k": k,
            "beta": beta,
            "a": a,
            "b": b,
            "join_times": join_times
        }
        # Save parameters to JSON file
        with open(json_file_path, "w") as json_file:
            json.dump(result_data, json_file, indent=4)
        print("Data written to parameter_hnsw.json")

    max_roles_per_user = max(len(roles) for roles in user_to_roles.values())

    # Insert multiple new roles
    for i in range(num_roles_to_insert):
        new_role_id = initial_new_role_id + i  # Increment new_role_id for each insertion

        # Calculate the sampling ratio to prevent excessive document allocation for the new role
        num_roles = len(role_to_documents)
        sample_ratio = 1 / num_roles if num_roles > 0 else 0.05  # Ensure at least a 5% sampling as a fallback

        # Sample a subset of documents from each role
        new_role_documents = set()
        for role, docs in role_to_documents.items():
            sample_size = max(1, int(len(docs) * sample_ratio))  # Ensure at least one document is selected
            new_role_documents.update(random.sample(docs, sample_size) if len(docs) > sample_size else docs)

        # Determine the best partition for the new role
        best_partition = insert_new_role(
            new_role_id, new_role_documents, partition_assignment, partition_roles,
            topk=10, k=result_data["k"], beta=result_data["beta"], a=result_data["a"], b=result_data["b"]
        )

        # Generate users and assign roles
        new_user_roles, new_users, new_roles = generate_users_for_role(
            new_role_id, user_to_roles,
            len(user_to_roles) // len(partition_roles),
            max_roles_per_user, roles
        )

        # Update the database with the new role, assigned partition, and necessary role-user mappings
        update_database_for_new_role(new_role_id, best_partition, new_role_documents, new_user_roles, new_users, new_roles)

    initialize_dynamic_partition_tables_in_comb(index_type="hnsw")
    print(f"Inserted {num_roles_to_insert} new roles successfully.")
