import json
import random
import os
import sys
from collections import defaultdict

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from controller.dynamic_partition.hnsw.helper import prepare_background_data, fetch_initial_data


def convert_to_role_assignments(permission_assignments):
    """
    Convert a list of (role_id, document_id) tuples into a role-to-documents mapping.

    :param permission_assignments: List of (role_id, document_id) tuples.
    :return: Dictionary {role_id: set(document_ids)}
    """
    role_assignments = defaultdict(set)
    for role_id, doc_id in permission_assignments:
        role_assignments[role_id].add(doc_id)
    return role_assignments


def compute_average_selectivity(role_assignments, total_documents):
    """
    Compute the average selectivity of roles.

    :param role_assignments: Dictionary mapping role IDs to their assigned document sets.
    :param total_documents: Total number of documents in the dataset.
    :return: The average selectivity across all roles.
    """
    # Step 1: Compute selectivity for each role
    selectivity_values = [len(docs) / total_documents for docs in role_assignments.values()]

    # Step 2: Compute the average selectivity
    avg_selectivity = sum(selectivity_values) / len(selectivity_values)

    return avg_selectivity


def compute_user_selectivity(user_assignments, role_to_documents, total_documents):
    """
    Compute the average selectivity of users.

    :param user_assignments: Dictionary mapping user IDs to their assigned roles.
    :param role_to_documents: Dictionary mapping role IDs to their assigned document sets.
    :param total_documents: Total number of documents in the dataset.
    :return: The average selectivity across all users.
    """
    # Step 1: For each user, compute the set of documents they have access to
    user_documents = []
    for user_roles in user_assignments.values():
        user_docs = set()
        for role in user_roles:
            user_docs.update(role_to_documents.get(role, []))  # Add documents for this role
        user_documents.append(user_docs)

    # Step 2: Compute the selectivity for each user (documents they can access / total documents)
    selectivity_values = [len(user_docs) / total_documents for user_docs in user_documents]

    # Step 3: Compute the average selectivity across all users
    avg_selectivity = sum(selectivity_values) / len(selectivity_values)

    return avg_selectivity

def convert_permissions_to_roles(functional_roles_permissions):
    """
    Convert functional roles permissions to a dictionary where keys are permissions (documents),
    and values are lists of roles associated with each permission.
    """
    permissions_to_roles = {}

    # Iterate through functional roles and their permissions
    for role_id, permissions in functional_roles_permissions.items():
        for perm in permissions:
            if perm not in permissions_to_roles:
                permissions_to_roles[perm] = set()
            permissions_to_roles[perm].add(role_id)  # Add role_id to the set for this document

    # Convert sets to lists for consistency
    for perm in permissions_to_roles:
        permissions_to_roles[perm] = list(permissions_to_roles[perm])

    return permissions_to_roles



if __name__ == '__main__':
    from basic_benchmark.initialize_dynamic_partition_tables import initialize_dynamic_partition_tables, \
        initialize_dynamic_partition_tables_in_comb

    roles, documents, permissions, avg_blocks_per_document, user_to_roles = fetch_initial_data()
    # Prepare background data
    role_to_documents, document_to_index = prepare_background_data(roles, documents, permissions)

    sel = compute_average_selectivity(role_to_documents, len(documents))
    user_sel = compute_user_selectivity(user_to_roles, role_to_documents, len(documents))

    total_assigned_documents = sum(len(docs) for docs in role_to_documents.values())
    storage = total_assigned_documents / len(documents)
    print(f"Role Selectivity: {sel}")
    print(f"User Selectivity: {user_sel}")
    print(f"Storage: {storage}")
