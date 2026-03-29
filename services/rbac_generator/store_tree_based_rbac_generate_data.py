import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from services.read_dataset_function import store_rbac_data
from services.rbac_generator.tree_based_rbac_data_generator import TreeBasedRBACDataGenerator
from services.config import get_db_connection

if __name__ == '__main__':
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

    # After documents and document blocks are stored, generate RBAC data
    rbac_generator = TreeBasedRBACDataGenerator(
        num_users=1000,
        num_roles=100,
        document_ids=document_ids,  # Use actual document_ids from the dataset
        h=4,  # Tree height (can adjust based on requirements)
        b0=3,  # Minimum number of children per internal node
        b1=4  # Maximum number of children per internal node
    )

    #eg2: if you want to improve number of roles, you can adjust h, b0, b1
    # rbac_generator = TreeBasedRBACDataGenerator(
    #     num_users=1000,
    #     num_roles=300,
    #     document_ids=document_ids,  # Use actual document_ids from the dataset
    #     h=6,  # Tree height (can adjust based on requirements)
    #     b0=3,  # Minimum number of children per internal node
    #     b1=4  # Maximum number of children per internal node
    # )

    # Generate users, user roles, and permission assignments based on the tree-based RBAC model
    users, user_roles, document_assignments, permission_assignments = rbac_generator.generate_rbac_data()

    # Assert that all document_ids are assigned
    assigned_document_ids = set()
    for docs in document_assignments.values():
        assigned_document_ids.update(docs)

    # Check if all document_ids are covered
    assert set(document_ids) == assigned_document_ids, "Not all document_ids are assigned!"

    # Store the generated RBAC data in the database
    store_rbac_data(users, rbac_generator.original_roles, user_roles, permission_assignments)