import json
import random
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

class FunctionalRole:
    def __init__(self, role_id):
        self.role_id = role_id
        self.role_name = f'functional_role_{role_id}'  # Role name
        self.permissions = []  # Permissions will be added later


class BusinessRole:
    def __init__(self, role_id):
        self.role_id = role_id
        self.role_name = f'business_role_{role_id}'  # Role name
        self.functional_roles = []  # Associated functional roles


class User:
    def __init__(self, user_id):
        self.user_id = user_id
        self.user_name = f'user_{user_id}'  # User name
        self.business_roles = []  # Associated business roles


class ERBACDataGenerator:
    def __init__(self, n_froles, n_broles, document_ids, m_perms, m_froles, m_broles):
        self.n_froles = n_froles  # Number of functional roles
        self.n_broles = n_broles  # Number of business roles
        self.document_ids = document_ids  # Permissions (documents)
        self.m_perms = m_perms  # Maximum permissions per functional role
        self.m_froles = m_froles  # Maximum functional roles per business role
        self.m_broles = m_broles  # Maximum business roles per user

        # Initialize functional roles and business roles
        self.functional_roles = [FunctionalRole(role_id=i) for i in range(1, n_froles + 1)]
        self.business_roles = [BusinessRole(role_id=i) for i in range(1, n_broles + 1)]
        self.users = []  # Users will be added later

    # def assign_permissions_to_functional_roles(self):
    #     """Assign random permissions to each functional role."""
    #     for role in self.functional_roles:
    #         num_permissions = random.randint(1, self.m_perms)
    #         role.permissions = random.sample(self.document_ids, num_permissions)

    def get_functional_roles(self):
        """
        Return all functional roles.

        Returns:
            list: A list of functional roles, where each role is an instance of FunctionalRole.
        """
        return self.functional_roles

    def get_functional_roles_with_permissions(self):
        """
        Return all functional roles and their associated documents (permissions).

        Returns:
            dict: A dictionary where the key is the functional role ID and the value
                  is the list of document IDs (permissions) associated with that role.
        """
        functional_roles_permissions = {}
        for role in self.functional_roles:
            functional_roles_permissions[role.role_id] = role.permissions
        return functional_roles_permissions

    def assign_permissions_to_functional_roles(self):
        """Assign random permissions to each functional role and ensure all documents are covered while respecting m_perms."""
        unique_role_set = set()

        # Step 1: Randomly assign permissions to functional roles ensuring uniqueness
        for role in self.functional_roles:
            while True:
                num_permissions = random.randint(1, self.m_perms)
                permissions = tuple(sorted(random.sample(self.document_ids, num_permissions)))
                if permissions not in unique_role_set:
                    unique_role_set.add(permissions)
                    role.permissions = list(permissions)
                    break

        # Step 2: Ensure all documents are covered
        covered_documents = set()
        for role in self.functional_roles:
            covered_documents.update(role.permissions)

        uncovered_documents = set(self.document_ids) - covered_documents

        # Step 3: Distribute uncovered documents while respecting m_perms
        if uncovered_documents:
            functional_roles_iter = iter(self.functional_roles)
            for doc in uncovered_documents:
                while True:
                    try:
                        role = next(functional_roles_iter)
                    except StopIteration:
                        functional_roles_iter = iter(self.functional_roles)
                        role = next(functional_roles_iter)

                    if len(role.permissions) < self.m_perms:
                        role.permissions.append(doc)
                        break  # Move to the next document

        # Step 4: Ensure no role exceeds m_perms
        for role in self.functional_roles:
            if len(role.permissions) > self.m_perms:
                role.permissions = random.sample(role.permissions, self.m_perms)

    def assign_functional_roles_to_business_roles(self):
        """Assign random functional roles to each business role ensuring uniqueness."""
        unique_role_set = set()

        for role in self.business_roles:
            while True:
                num_functional_roles = random.randint(1, self.m_froles)
                assigned_roles = tuple(
                    sorted(random.sample(self.functional_roles, num_functional_roles), key=lambda x: x.role_name)
                )
                if assigned_roles not in unique_role_set:
                    unique_role_set.add(assigned_roles)
                    role.functional_roles = list(assigned_roles)
                    break

    def assign_business_roles_to_users(self, num_users):
        """Assign random business roles to each user."""
        self.users = [{'user_id': i, 'user_name': f'user_{i}'} for i in range(1, num_users + 1)]
        user_roles = []
        for user in self.users:
            num_business_roles = random.randint(1, self.m_broles)
            assigned_roles = random.sample(self.business_roles, num_business_roles)
            for role in assigned_roles:
                user_roles.append((user['user_id'], role.role_id))  # Collect (user_id, role_id) for output
        return user_roles

    def generate_business_role_permissions(self):
        """Generate the final permission set for each business role."""
        role_permissions = []
        for br in self.business_roles:
            # Inherit permissions from associated functional roles
            permissions = set()
            for fr in br.functional_roles:
                permissions.update(fr.permissions)  # Inherit permissions from functional roles
            for perm in permissions:
                role_permissions.append((br.role_id, perm))  # Collect (business_role_id, document_id)
        return role_permissions

    def generate_rbac_data(self, num_users):
        """Generate all RBAC data."""
        # Step 1: Assign permissions to functional roles
        self.assign_permissions_to_functional_roles()

        # Step 2: Assign functional roles to business roles
        self.assign_functional_roles_to_business_roles()

        # Step 3: Assign business roles to users
        user_roles = self.assign_business_roles_to_users(num_users)

        # Step 4: Generate business role-permission assignments (for PermissionAssignment table)
        role_permissions = self.generate_business_role_permissions()

        # Combine functional roles and business roles into a single roles list for database storage
        roles = self.business_roles

        return self.users, roles, user_roles, role_permissions

    def get_functional_roles_with_permissions(self):
        """
        Return all functional roles and their associated documents (permissions).

        Returns:
            dict: A dictionary where the key is the functional role ID and the value
                  is the list of document IDs (permissions) associated with that role.
        """
        functional_roles_permissions = {}
        for role in self.functional_roles:
            functional_roles_permissions[role.role_id] = role.permissions
        return functional_roles_permissions

    def save_functional_roles_to_file(self, file_path):
        """
        Save functional roles and their permissions to a JSON file.

        Args:
            file_path (str): The path to the output file.
        """
        functional_roles_permissions = self.get_functional_roles_with_permissions()
        try:
            with open(file_path, 'w') as file:
                json.dump(functional_roles_permissions, file, indent=4)
            print(f"Functional roles with permissions saved to {file_path}")
        except IOError as e:
            print(f"Error saving functional roles to file: {e}")

if __name__ == '__main__':
    # 定义参数
    n_froles = 5      # 功能角色数量
    n_broles = 3      # 业务角色数量
    document_ids = range(1, 101)  # 假设有100个文档（权限）
    m_perms = 10      # 每个功能角色的最大权限数
    m_froles = 2      # 每个业务角色可以关联的最大功能角色数
    m_broles = 2      # 每个用户可以关联的最大业务角色数
    num_users = 10    # 生成10个用户

    # 初始化 ERBAC 数据生成器
    erbac_generator = ERBACDataGenerator(
        n_froles=n_froles,
        n_broles=n_broles,
        document_ids=document_ids,
        m_perms=m_perms,
        m_froles=m_froles,
        m_broles=m_broles
    )

    # 生成 RBAC 数据
    users, roles, user_roles, role_permissions = erbac_generator.generate_rbac_data(num_users)

    # 输出生成的数据
    print("\nUsers:")
    for user in users:
        print(user)

    print("\nRoles:")
    for role in roles:
        print(f'Role ID: {role.role_id}, Role Name: {role.role_name}')

    print("\nUser-Roles Assignments:")
    for ur in user_roles:
        print(f'User ID: {ur[0]}, Role ID: {ur[1]}')

    print("\nRole-Permissions Assignments:")
    for rp in role_permissions:
        print(f'Role ID: {rp[0]}, Document ID: {rp[1]}')