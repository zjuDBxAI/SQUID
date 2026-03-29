import sys
import os
import json
import random
from collections import defaultdict


project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from services.read_dataset_function import store_rbac_data

from services.config import get_db_connection

class Role:
    def __init__(self, role_id, role_name):
        self.role_id = role_id
        self.role_name = role_name


class ArXivGeneratorWithBusinessRoles:
    def __init__(self, data_file, max_business_roles, max_functional_roles_per_business_role, users_per_business_role, total_users):
        """
        Initialize the generator.
        :param data_file: Path to the arXiv JSON file.
        :param max_business_roles: Maximum number of business roles to generate.
        :param max_functional_roles_per_business_role: Maximum number of functional roles per business role.
        :param total_users: Total number of users to generate.
        """
        self.data_file = data_file
        self.max_business_roles = max_business_roles
        self.max_functional_roles_per_business_role = max_functional_roles_per_business_role
        self.total_users = total_users
        self.users_per_business_role = users_per_business_role  # Calculate users per business role
        self.functional_roles = []  # List of functional roles
        self.business_roles = []  # List of business roles
        self.users = []  # List of users
        self.role_permissions = []  # List of tuples (functional_role_id, document_id)
        self.business_role_to_functional_roles = []  # List of tuples (business_role_id, functional_role_id)
        self.user_roles = []  # List of tuples (user_id, business_role_id)

    def load_data(self):
        """
        Load the data from the JSON file and group by categories.
        """
        print("Loading data...")
        with open(self.data_file, 'r') as f:
            self.data = [json.loads(line) for line in f]
        self.categories_to_docs = defaultdict(list)

        # Group documents by categories
        for idx, doc in enumerate(self.data):
            document_id = idx  # Sequential ID for each document
            categories = doc.get('categories', '').split()
            for category in categories:
                self.categories_to_docs[category].append(document_id)

        print(f"Loaded {len(self.data)} documents and grouped into {len(self.categories_to_docs)} categories.")

    def generate_functional_roles(self):
        """
        Generate functional roles based on categories.
        """
        print("Generating functional roles...")
        for category in self.categories_to_docs.keys():
            role_id = len(self.functional_roles) + 1
            role_name = f"functional_role_{category}"
            self.functional_roles.append(Role(role_id, role_name))

        print(f"Generated {len(self.functional_roles)} functional roles.")

    def generate_business_roles(self):
        """
        Generate business roles by combining functional roles (intermediate step).
        Only business roles and their permissions will be stored.
        """
        print("Generating business roles...")
        functional_role_to_docs = defaultdict(set)

        # Step 1: Group documents by functional roles
        for category, docs in self.categories_to_docs.items():
            functional_role_to_docs[category] = set(docs)

        # Step 2: Generate business roles and inherit documents from functional roles
        functional_role_keys = list(functional_role_to_docs.keys())

        for business_role_id in range(1, self.max_business_roles + 1):
            # Randomly select functional roles for this business role
            num_roles_to_assign = random.randint(1, self.max_functional_roles_per_business_role)
            selected_roles = random.sample(functional_role_keys, k=num_roles_to_assign)

            role_name = f"business_role_{business_role_id}"
            self.business_roles.append(Role(business_role_id, role_name))

            # Inherit documents from selected functional roles
            inherited_docs = set()
            for role_key in selected_roles:
                inherited_docs |= functional_role_to_docs[role_key]

            # Store role permissions for business role
            for doc_id in inherited_docs:
                self.role_permissions.append((business_role_id, doc_id))

        print(f"Generated {len(self.business_roles)} business roles with inherited permissions.")

    def generate_users(self):
        """
        Generate users and assign them to business roles.
        """
        print("Generating users...")
        user_id = 1
        for business_role in self.business_roles:
            for _ in range(self.users_per_business_role):
                if user_id > self.total_users:
                    break  # Ensure we do not exceed the total number of users
                user = {'user_id': user_id, 'user_name': f"user_{user_id}"}
                self.users.append(user)
                self.user_roles.append((user_id, business_role.role_id))
                user_id += 1

        print(f"Generated {len(self.users)} users.")

    def generate(self):
        """
        Generate the entire RBAC dataset.
        """
        self.load_data()
        self.generate_functional_roles()
        self.generate_business_roles()
        self.generate_users()

        print("RBAC dataset generation complete.")
        return {
            "users": self.users,
            "functional_roles": self.functional_roles,
            "business_roles": self.business_roles,
            "role_permissions": self.role_permissions,
            "business_role_to_functional_roles": self.business_role_to_functional_roles,
            "user_roles": self.user_roles,
        }


# Main function to use the generator
if __name__ == '__main__':
    conn = get_db_connection()
    cur = conn.cursor()

    # Use TRUNCATE to clear all tables, which is faster than DELETE
    cur.execute("TRUNCATE TABLE userroles, permissionassignment, users, roles RESTART IDENTITY CASCADE;")

    conn.commit()
    cur.close()
    conn.close()
    # Specify the path to the JSON file
    dataset_folder = os.path.join(project_root, "/data/dataset/arxiv")
    arxiv_data_file = os.path.join(dataset_folder, "arxiv-metadata-oai-snapshot.json")

    # Initialize the generator with parameters
    generator = ArXivGeneratorWithBusinessRoles(
        data_file=arxiv_data_file,
        max_business_roles=100,  # Maximum number of business roles
        max_functional_roles_per_business_role=3,  # Maximum functional roles per business role
        users_per_business_role = 10,
        total_users=1000  # Total number of users
    )

    # Generate RBAC data
    rbac_data = generator.generate()

    # Store the generated RBAC data in the database
    store_rbac_data(
        users=rbac_data["users"],
        roles=rbac_data["business_roles"],
        user_roles=rbac_data["user_roles"],
        permission_assignments=rbac_data["role_permissions"],
    )

    # Output results
    print("\nUsers:")
    for user in rbac_data["users"][:5]:
        print(user)

    print("\nFunctional Roles:")
    for role in rbac_data["functional_roles"][:5]:
        print(role)

    print("\nBusiness Roles:")
    for role in rbac_data["business_roles"][:5]:
        print(role)

    print("\nUser-Business Role Assignments:")
    for ur in rbac_data["user_roles"][:10]:
        print(ur)

    print("\nBusiness Role to Functional Role Assignments:")
    for br_fr in rbac_data["business_role_to_functional_roles"][:10]:
        print(br_fr)

    print("\nRole-Permission Assignments:")
    for rp in rbac_data["role_permissions"][:10]:
        print(rp)