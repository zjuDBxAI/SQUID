import random
from abc import ABC, abstractmethod
import os
import sys
from collections import defaultdict

import numpy as np
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

class Role:
    def __init__(self, role_id, role_name, hierarchy_level):
        self.role_id = role_id
        self.role_name = role_name
        self.hierarchy_level = hierarchy_level
        self.children = []

    def add_child(self, child_role):
        """Adds a child role to the current role"""
        self.children.append(child_role)


class SharingTreeBasedRBACDataGenerator:
    def __init__(self, num_users=10000, num_roles=100, document_ids=None, h=4, b0=3, b1=4, doc_sharing_distribution=(30, 30, 1, 100)):
        """
        Initialize the RBAC generator with the number of users, roles, document IDs,
        and parameters to control the height of the tree and the number of child roles per node.
        """
        self.num_users = num_users
        self.num_roles = num_roles
        self.document_ids = list(document_ids)  # Ensure the document IDs are a list
        self.h = h  # Height of the tree
        self.b0 = b0  # Minimum number of children for each internal node
        self.b1 = b1  # Maximum number of children for each internal node
        self.doc_sharing_distribution = doc_sharing_distribution

        # Generate roles with random hierarchy levels
        self.original_roles = [Role(i, f'role_{i}', hierarchy_level=random.randint(1, h)) for i in
                               range(1, num_roles + 1)]
        self.roles = self.original_roles.copy()  # Make a copy of the original roles for tree generation

        # Generate users
        self.users = [{'user_id': i, 'user_name': f'user_{i}'} for i in range(1, num_users + 1)]

        # Initialize the root role and generate the role tree
        self.root_role = Role(0, 'root', 0)
        self.role_tree = self.generate_role_tree()

    # Generate the hierarchical role tree structure
    def generate_role_tree(self):
        """Recursively generate a tree structure for roles based on the given parameters"""

        def add_children(parent_role, current_level):
            # Stop if the current level exceeds height or if there are no more roles
            if current_level >= self.h or not self.roles:
                return

            # Adjust the number of children to avoid invalid range when roles are fewer than b0
            num_children = min(random.randint(self.b0, self.b1), len(self.roles))

            # Add child roles to the parent
            for _ in range(num_children):
                if not self.roles:  # Check if roles are exhausted
                    break
                child_role = self.roles.pop(0)  # Get the next available role
                parent_role.add_child(child_role)  # Assign this role as a child of the parent
                add_children(child_role, current_level + 1)  # Recursively add children to this role

        # Begin by adding children to the root role
        add_children(self.root_role, 0)

        # Check if all roles have been added to the tree
        if self.roles:
            print(f"Warning: {len(self.roles)} roles were not assigned to the tree.")

        return self.root_role

    # Calculate the total number of nodes in the tree
    def calculate_total_nodes(self):
        """Recursively calculate the total number of nodes in the tree."""

        def count_nodes(role):
            total_nodes = 1  # Count the current node
            for child_role in role.children:
                total_nodes += count_nodes(child_role)
            return total_nodes

        return count_nodes(self.root_role)

    def split_documents_into_shared_sets(self, document_ids, num_sets):
        """
        Split documents into overlapping sets, based on Poisson distribution-defined sharing degree.

        :param document_ids: List of document IDs.
        :param num_sets: The number of sets to create (number of roles).
        :return: A dictionary mapping role IDs to assigned document sets.
        """
        percent_shared, poisson_mean, min_roles, max_roles = self.doc_sharing_distribution

        total_docs = len(document_ids)
        num_shared_docs = round((percent_shared / 100) * total_docs)  # Determine how many docs should be shared

        # Step 1: Generate Poisson-distributed sharing degrees
        sharing_degrees = np.random.poisson(lam=poisson_mean, size=total_docs)
        sharing_degrees = np.clip(sharing_degrees, min_roles, max_roles)  # Ensure sharing limits

        # Step 2: Create role-document assignments
        document_assignments = defaultdict(set)  # Document -> Assigned roles
        role_assignments = defaultdict(set)  # Role -> Assigned documents
        all_roles = list(range(1, num_sets + 1))  # Role IDs

        for doc_id, share_degree in zip(document_ids[:num_shared_docs], sharing_degrees[:num_shared_docs]):
            assigned_roles = random.sample(all_roles, min(share_degree, len(all_roles)))
            for role_id in assigned_roles:
                document_assignments[doc_id].add(role_id)
                role_assignments[role_id].add(doc_id)

        # Step 3: Assign remaining docs uniquely to roles (ensuring every role has at least one doc)
        remaining_docs = document_ids[num_shared_docs:]
        shuffled_roles = random.sample(all_roles, len(all_roles))  # Shuffle roles for fair distribution
        for doc_id, role_id in zip(remaining_docs, shuffled_roles * (len(remaining_docs) // len(shuffled_roles) + 1)):
            document_assignments[doc_id].add(role_id)
            role_assignments[role_id].add(doc_id)

        # **Step 4: Ensure all documents are assigned**
        assigned_docs = {doc for docs in role_assignments.values() for doc in docs}
        missing_docs = set(document_ids) - assigned_docs
        assert not missing_docs, f"❌ Error: The following documents were not assigned to any role: {missing_docs}"

        # **Final Validation**
        assert assigned_docs == set(document_ids), "❌ Error: Assigned documents do not match total document set!"
        assert all(role_assignments.values()), "❌ Error: Some roles have no documents!"

        print(f"✅ All {len(all_roles)} roles successfully assigned at least one document.")
        print(f"✅ All {len(document_ids)} documents successfully assigned.")

        return role_assignments


        # # Assign documents to each role in the tree
    def assign_sharing_permissions_to_tree(self):
        """
        Assign permissions (document sets) to each node in the tree,
        including permissions inherited from ancestor nodes.
        """
        total_nodes = self.calculate_total_nodes()

        sharing_sets = self.split_documents_into_shared_sets(self.document_ids, total_nodes - 1)


        assigned_document_ids = set()  # Track assigned documents
        doc_index = 1  # Define the internal global variable for tracking document index

        def assign_documents(role, inherited_permissions=None):
            nonlocal doc_index  # Use the enclosing function's variable
            if inherited_permissions is None:
                inherited_permissions = set()

            document_assignments = {}
            if role.role_id != self.root_role.role_id:  # Exclude root role
                current_permissions = sharing_sets[doc_index]
                doc_index += 1  # Increment the internal global index after assigning
                all_permissions = inherited_permissions.union(current_permissions)
                assigned_document_ids.update(current_permissions)  # Track assigned documents
                document_assignments[role.role_id] = list(all_permissions)
            else:
                all_permissions = inherited_permissions

            # Recursively assign documents to child nodes
            for child_role in role.children:
                document_assignments.update(assign_documents(child_role, all_permissions))

            return document_assignments

        document_assignments = assign_documents(self.root_role)

        # Validate if all documents are assigned
        unassigned_docs = set(self.document_ids) - assigned_document_ids
        if unassigned_docs:
            raise ValueError(f"Unassigned documents: {unassigned_docs}")

        return document_assignments


    def exclude_root_role(self, roles, root_role_id=0):
        """
        Exclude the root role or any role with a specified role_id from the list of roles.
        :param roles: List of Role objects
        :param root_role_id: The ID of the root role to exclude (default is 0)
        :return: List of roles excluding the root role
        """
        return [role for role in roles if role.role_id != root_role_id]

    # Assign each user to a random role in the role tree
    def assign_users_to_roles_evenly(self):
        """
        Evenly distribute users across all roles in the tree, excluding the root role.
        """
        user_roles = []
        all_roles = []

        # Collect all roles in the tree, excluding the root role
        def collect_all_roles(role):
            if role.role_id != self.root_role.role_id:  # Exclude the root role
                all_roles.append(role)
            for child_role in role.children:
                collect_all_roles(child_role)

        collect_all_roles(self.root_role)

        # Evenly divide users among all roles
        user_subsets = np.array_split(self.users, len(all_roles))  # Divide users evenly

        for role, subset in zip(all_roles, user_subsets):
            for user in subset:
                user_roles.append((user['user_id'], role.role_id))

        return user_roles

    # Generate complete RBAC data
    def generate_rbac_data(self):
        """
        Generate the complete RBAC data, including user-role assignments,
        document-role assignments, and permission assignments.
        """
        # Generate user-role assignments (simplified: assign each user to a random role)
        user_roles = self.assign_users_to_roles_evenly()

        # Generate document assignments with inheritance
        document_assignments = self.assign_sharing_permissions_to_tree()

        # Step 3: Debug log to ensure all documents are assigned
        assigned_document_ids = set()
        for docs in document_assignments.values():
            assigned_document_ids.update(docs)

        assert set(self.document_ids) == assigned_document_ids, (
            f"Document assignment mismatch in generate_rbac_data! "
            f"Unassigned: {set(self.document_ids) - assigned_document_ids}"
        )

        # Create permission assignments (role-document pairs)
        permission_assignments = []
        for role_id, documents in document_assignments.items():
            for document_id in documents:
                permission_assignments.append((role_id, document_id))

        return self.users, user_roles, document_assignments, permission_assignments



if __name__ == '__main__':
    # Define total number of users, roles, and documents
    num_users = 10000
    num_roles = 100
    document_ids = range(1, 10010)  # Assume there are 10000 documents

    # Generate RBAC data with tree height 4, and between 3 to 4 children per internal node
    generator = SharingTreeBasedRBACDataGenerator(num_users=num_users, num_roles=num_roles, document_ids=document_ids, h=4,
                                                  b0=3, b1=4)
    users, user_roles, document_assignments, permission_assignments = generator.generate_rbac_data()

    # Output the number of user-role and permission assignments
    print(f"Total user-role assignments: {len(user_roles)}")
    print(f"Total permission assignments: {len(permission_assignments)}")

    # Print the first 5 user-role assignments
    print("\nFirst 5 user-role assignments:")
    for assignment in user_roles[:5]:
        print(assignment)

    # Print the first 5 permission assignments
    print("\nFirst 5 permission assignments:")
    for assignment in permission_assignments[:5]:
        print(assignment)
