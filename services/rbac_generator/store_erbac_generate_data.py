import argparse
import json
import sys
import os


project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

    
from controller.dynamic_partition.hnsw.helper import fetch_initial_data
from services.rbac_generator.common import convert_to_role_assignments, compute_average_selectivity, \
    convert_permissions_to_roles

print(sys.path)
from services.rbac_generator.erbac_data_generator import ERBACDataGenerator

from services.read_dataset_function import store_rbac_data
from services.rbac_generator.tree_based_rbac_data_generator import TreeBasedRBACDataGenerator
from services.config import get_db_connection


def delete_gap_documents():
    """Delete document_ids in documentblocks not present in PermissionAssignment."""
    # Establish a database connection
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Fetch all document_ids from PermissionAssignment
        cur.execute("SELECT DISTINCT document_id FROM PermissionAssignment;")
        permission_assignment_documents = set(row[0] for row in cur.fetchall())

        # Fetch all document_ids from documentblocks
        cur.execute("SELECT DISTINCT document_id FROM documentblocks;")
        documentblocks_documents = set(row[0] for row in cur.fetchall())

        # Identify document_ids to delete (in documentblocks but not in PermissionAssignment)
        documents_to_delete = documentblocks_documents - permission_assignment_documents

        if not documents_to_delete:
            print("No document_id gaps found. No deletions required.")
            return

        # Delete rows with document_ids not in PermissionAssignment
        delete_query = "DELETE FROM documentblocks WHERE document_id = ANY(%s);"
        cur.execute(delete_query, (list(documents_to_delete),))

        delete_query = "DELETE FROM documents WHERE document_id = ANY(%s);"
        cur.execute(delete_query, (list(documents_to_delete),))

        # Commit the transaction
        conn.commit()

        print(f"Deleted {cur.rowcount} document(s) from documentblocks.")
    except Exception as e:
        conn.rollback()
        print(f"An error occurred: {e}")
    finally:
        # Close the cursor and connection
        cur.close()
        conn.close()


def check_duplicate_roles(functional_roles_permissions):
    seen = set()
    duplicates = []
    for role_id, permissions in functional_roles_permissions.items():
        permissions_set = frozenset(permissions)
        if permissions_set in seen:
            duplicates.append(role_id)
        else:
            seen.add(permissions_set)
    return duplicates


def main(num_documents):
    conn = get_db_connection()
    cur = conn.cursor()

    # Use TRUNCATE to clear all tables, which is faster than DELETE
    cur.execute("TRUNCATE TABLE userroles, permissionassignment, users, roles RESTART IDENTITY CASCADE;")

    conn.commit()
    cur.close()
    conn.close()

    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch all document_ids from the documents table
    cur.execute("SELECT document_id FROM documents;")
    document_ids = [row[0] for row in cur.fetchall()]

    cur.close()
    conn.close()

    # Calculate m_perms as one-tenth of the number of documents
    m_perms = num_documents // 26

    #used for sharing degree
    erbac_generator = ERBACDataGenerator(
        n_froles=40,      # Number of functional roles
        n_broles=100,      # Number of business roles
        document_ids=document_ids,  # Use the specified number of documents
        m_perms=m_perms,  # Max permissions per functional role (one-tenth of num_documents)
        m_froles=3,       # Max functional roles per business role
        m_broles=3        # Max business roles per user
    )

    # Generate data for 1000 users
    users, roles, user_roles, role_permissions = erbac_generator.generate_rbac_data(num_users=1000)

    role_assignments = convert_to_role_assignments(role_permissions)

    # Extract all permissions from role_permissions
    assigned_permissions = set(permission for _, permission in role_permissions)

    # Convert document_ids to a set for comparison
    document_ids_set = set(erbac_generator.document_ids)

    # Check if all document_ids are covered by role_permissions
    if document_ids_set.issubset(assigned_permissions):
        print("All document_ids are now covered by role_permissions.")
    else:
        missing_permissions = document_ids_set - assigned_permissions
        print("Some document_ids are still not covered by role_permissions.")
        print(f"Missing document_ids: {sorted(missing_permissions)}")

    avg_selectivity = compute_average_selectivity(role_assignments, len(document_ids))
    print(f"📊 Average Role Selectivity: {avg_selectivity:.4f}")

    total_assigned_documents = sum(len(docs) for docs in role_assignments.values())
    storage = total_assigned_documents / len(documents)

    print(f"role partition storage: {storage:.4f}")
    # Store the generated RBAC data (simulating database storage)
    store_rbac_data(users, roles, user_roles, role_permissions)




if __name__ == '__main__':
    # Create argument parser
    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch document IDs from documentblocks
    cur.execute("SELECT DISTINCT document_id FROM documentblocks ORDER BY document_id;")
    documents = [row[0] for row in cur.fetchall()]

    # parser = argparse.ArgumentParser(description="ERBAC Data Generator")
    # parser.add_argument('--num_documents', type=int, required=True, help="Number of documents to be used in generation")
    #
    # # Parse arguments
    # args = parser.parse_args()

    # Call main function with parsed argument
    main(len(documents))

    roles, documents, permissions, avg_blocks_per_document, user_to_roles = fetch_initial_data()
    
    # Check for duplicate role permissions
    unique_permissions = set(permissions)

    if len(unique_permissions) < len(permissions):
        print("Duplicate role permissions detected. Please check the data generation logic.")
        raise ValueError("Duplicate entries found in role permissions.")

    print("No duplicate role permissions found. Proceeding with data storage.")
