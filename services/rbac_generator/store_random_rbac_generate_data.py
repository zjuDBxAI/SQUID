import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from services.rbac_generator.common import convert_to_role_assignments, compute_average_selectivity

from services.rbac_generator.random_rbac_data_generator import RandomRBACDataGenerator

from services.read_dataset_function import store_rbac_data
from services.rbac_generator.tree_based_rbac_data_generator import TreeBasedRBACDataGenerator
from services.config import get_db_connection


def generate_random_data(
        num_users=10000,
        num_roles=100,
        m_roles=3,
        m_perms=2000):
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

    # Generate RBAC data using the random RBAC model
    rbac_generator = RandomRBACDataGenerator(
        num_users=num_users,
        num_roles=num_roles,
        document_ids=document_ids,  # Actual document IDs representing permissions
        m_roles=m_roles,  # Maximum of x roles per user
        m_perms=m_perms,  # Maximum of x permissions per role
    )

    # Generate users, roles, user-role assignments, and permission assignments
    users, roles, user_roles, user_permissions = rbac_generator.generate_rbac_data()

    # Store the generated RBAC data (simulating database storage)
    store_rbac_data(users, roles, user_roles, user_permissions)


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

    num_roles = 100

    #used for sharing degree experiment
    rbac_generator = RandomRBACDataGenerator(
        num_users=1000,
        num_roles=num_roles,
        document_ids=document_ids,  # Actual document IDs representing permissions
        m_roles=1,  # Maximum of x roles per user
        m_perms=int(len(document_ids) / num_roles * 9),  # Maximum of x permissions per role
    )
    # Generate users, roles, user-role assignments, and permission assignments
    users, roles, user_roles, user_permissions = rbac_generator.generate_rbac_data()
    role_assignments = convert_to_role_assignments(user_permissions)  # Convert first
    avg_selectivity = compute_average_selectivity(role_assignments, len(document_ids))
    print(f"📊 Average Role Selectivity: {avg_selectivity:.4f}")

    # Store the generated RBAC data (simulating database storage)
    store_rbac_data(users, roles, user_roles, user_permissions)
