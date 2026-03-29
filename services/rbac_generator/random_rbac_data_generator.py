import random
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


class OverlapException(Exception):
    """Custom exception for handling permission overlap errors."""
    pass


class Role:
    def __init__(self, role_id):
        self.role_id = role_id
        self.role_name = f'role_{role_id}'  # Add role name for database storage


class User:
    def __init__(self, user_id):
        self.user_id = user_id
        self.user_name = f'user_{user_id}'  # Add user name for database storage


class RandomRBACDataGenerator:
    def __init__(self, num_users, num_roles, document_ids, m_roles, m_perms):
        self.num_users = num_users
        self.num_roles = num_roles
        self.document_ids = document_ids
        self.m_roles = m_roles
        self.m_perms = m_perms

        # Initialize users and roles
        self.roles = [Role(role_id=i) for i in range(1, num_roles + 1)]
        self.users = [{'user_id': i, 'user_name': f'user_{i}'} for i in range(1, num_users + 1)]

    def assign_roles_to_users(self):
        """Assign random roles to each user."""
        user_roles = []
        for user in self.users:
            # Randomly assign between 1 and m_roles roles to each user
            num_roles = random.randint(1, self.m_roles)
            assigned_roles = random.sample(self.roles, num_roles)
            for role in assigned_roles:
                user_roles.append((user['user_id'], role.role_id))  # Storing as (user_id, role_id) tuples
        return user_roles

    def assign_permissions_to_roles(self):
        """Assign random permissions (documents) to each role, ensuring all documents are assigned at least once,
        and preventing duplicate role permissions."""

        role_permissions = []
        unique_permission_sets = set()
        unassigned_docs = set(self.document_ids)  # Track unassigned documents

        # Step 1: Randomly assign unique permissions to roles
        for role in self.roles:
            while True:
                # num_permissions = random.randint(0, self.m_perms)
                num_permissions = random.randint(self.m_perms//2, self.m_perms)
                assigned_permissions = tuple(sorted(random.sample(self.document_ids, num_permissions)))

                # Ensure the permission set is unique
                if assigned_permissions not in unique_permission_sets:
                    unique_permission_sets.add(assigned_permissions)
                    for document_id in assigned_permissions:
                        role_permissions.append((role.role_id, document_id))
                        unassigned_docs.discard(document_id)
                    break  # Move to the next role

        # Step 2: Ensure all documents are assigned at least once
        for doc_id in unassigned_docs:
            # Randomly select a role to assign the unassigned document, ensuring no duplication
            while True:
                role = random.choice(self.roles)
                existing_permissions = {doc for r, doc in role_permissions if r == role.role_id}
                if len(existing_permissions) < self.m_perms:
                    role_permissions.append((role.role_id, doc_id))
                    break

        return role_permissions

    def generate_rbac_data(self):
        """Generate all RBAC data: user-role assignments, role-permission assignments, and user-permission sets."""
        # No need to convert to dictionary, directly use objects for users and roles
        users = self.users
        roles = self.roles

        # Assign roles to users
        user_roles = self.assign_roles_to_users()

        # Assign permissions to roles
        role_permissions = self.assign_permissions_to_roles()

        return users, roles, user_roles, role_permissions
