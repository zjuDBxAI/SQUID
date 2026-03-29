import psycopg2
from psycopg2 import sql
from services.config import get_db_connection
from controller.dynamic_partition.hnsw.insertion import fetch_partition_assignment, fetch_partition_role_mapping
from controller.dynamic_partition.hnsw.helper import fetch_initial_data, prepare_background_data


def delete_role_and_related_data(role_id, partition_roles, role_to_documents):
    """
    Delete a role and all related data, including:
    - UserRole mappings
    - Users that only have this role
    - Role's document permissions
    - Role's partition mappings
    - Documents from partitions (keeping necessary ones)

    Args:
        role_id (int): The role ID to be deleted.
        partition_roles (dict): Mapping of partition_id to set of roles assigned to it.
        role_to_documents (dict): Mapping of role_id to its assigned document set.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Step 1: Find partitions that contain this role
        affected_partitions = [p_id for p_id, roles in partition_roles.items() if role_id in roles]

        # Step 2: Get documents assigned to this role (from role_to_documents, no need for SQL)
        role_documents = role_to_documents.get(role_id, set())

        # Step 3: Remove user-role mappings for this role
        cur.execute("DELETE FROM UserRoles WHERE role_id = %s;", (role_id,))

        # Step 4: Identify users that ONLY have this role (should be deleted)
        cur.execute("""
            DELETE FROM Users WHERE user_id IN (
                SELECT user_id FROM Users 
                WHERE user_id NOT IN (SELECT user_id FROM UserRoles)
            );
        """)

        # Step 5: Remove role-document permissions
        cur.execute("DELETE FROM PermissionAssignment WHERE role_id = %s;", (role_id,))

        # Step 6: Remove the role from role table
        cur.execute("DELETE FROM Roles WHERE role_id = %s;", (role_id,))

        # Step 7: Remove role from combrolepartitions
        cur.execute("""
            UPDATE combrolepartitions 
            SET comb_role = array_remove(comb_role, %s)
            WHERE %s = ANY(comb_role);
        """, (role_id, role_id))

        # Step 7.1: Delete any rows where comb_role is now empty
        cur.execute("""
            DELETE FROM combrolepartitions WHERE comb_role = '{}';
        """)

        # Step 8: Remove role's documents from each partition (if not needed by others)
        for partition_id in affected_partitions:
            partition_table = f"documentblocks_partition_{partition_id}"

            # Compute remaining roles in the partition (excluding role_id)
            remaining_roles = partition_roles[partition_id] - {role_id}

            # Compute documents needed by other roles (from role_to_documents, no SQL needed)
            other_role_documents = set()
            for other_role in remaining_roles:
                other_role_documents.update(role_to_documents.get(other_role, set()))

            # Compute documents that should be deleted
            documents_to_delete = role_documents - other_role_documents

            if documents_to_delete:
                cur.execute(
                    sql.SQL("DELETE FROM {} WHERE document_id IN %s;").format(
                        sql.Identifier(partition_table)
                    ),
                    (tuple(documents_to_delete),)
                )
                print(f"[INFO] Deleted {len(documents_to_delete)} documents from partition {partition_id}.")

            # Step 9: Rebuild HNSW Index for the affected partition
            print(f"[INFO] Rebuilding HNSW index for partition {partition_id}...")
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
        print(f"[INFO] Successfully deleted role {role_id} and cleaned up related data.")

    except psycopg2.Error as e:
        print(f"[ERROR] Failed to delete role {role_id}: {e}")
        conn.rollback()

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    from basic_benchmark.initialize_dynamic_partition_tables import initialize_dynamic_partition_tables_in_comb

    # Number of roles to delete (modify as needed)
    num_roles_to_delete = 2
    initial_role_id = 100  # Starting role_id for deletion

    # Fetch necessary data
    roles, documents, permissions, avg_blocks_per_document, user_to_roles = fetch_initial_data()
    role_to_documents, document_to_index = prepare_background_data(roles, documents, permissions)
    partition_roles = fetch_partition_role_mapping()

    for i in range(num_roles_to_delete):
        role_id_to_delete = initial_role_id - i  # Increment role_id for each deletion
        print(f"[INFO] Deleting role {role_id_to_delete}...")

        delete_role_and_related_data(role_id_to_delete, partition_roles, role_to_documents)

    # Reinitialize partitions after all deletions
    initialize_dynamic_partition_tables_in_comb(index_type="hnsw")
    print(f"[INFO] Successfully deleted {num_roles_to_delete} roles, starting from role {initial_role_id}.")
