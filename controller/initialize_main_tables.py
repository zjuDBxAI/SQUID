import psycopg2
from psycopg2 import sql
import sys
import os


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from services.config import get_db_connection, get_maintenance_settings

def initialize_database_deduplication(enable_index=False, vector_dimension=300):
    conn = get_db_connection()
    cur = conn.cursor()

    # User table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Users (
            user_id SERIAL PRIMARY KEY,
            user_name VARCHAR(255) NOT NULL
        );
    """)

    # Roles table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Roles (
            role_id SERIAL PRIMARY KEY,
            role_name VARCHAR(255) NOT NULL
        );
    """)

    # User roles table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS UserRoles (
            user_id INT NOT NULL REFERENCES Users(user_id),
            role_id INT NOT NULL REFERENCES Roles(role_id),
            PRIMARY KEY (user_id, role_id)
        );
    """)

    # Documents table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Documents (
            document_id SERIAL PRIMARY KEY,
            document_name VARCHAR(255) NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # documentblocks table
    cur.execute(
        sql.SQL("""
        CREATE TABLE IF NOT EXISTS documentblocks (
            block_id BIGINT NOT NULL,
            document_id INT NOT NULL REFERENCES Documents(document_id),
            block_content BYTEA NOT NULL,
            hash_value BYTEA NOT NULL,
            vector VECTOR({dimension}),
            PRIMARY KEY (block_id, document_id)
        );
        """).format(dimension=sql.SQL(str(vector_dimension)))
    )

    # PermissionAssignment table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS PermissionAssignment (
            permission_id SERIAL PRIMARY KEY,
            role_id INT NOT NULL REFERENCES Roles(role_id),
            document_id INT NOT NULL REFERENCES Documents(document_id)
        );
    """)

    conn.commit()
    cur.close()
    conn.close()

    if enable_index:
        create_indexes()


def create_indexes(index_type="ivfflat", hnsw_m=16, hnsw_ef_construction=64,
                   hnsw_threads=None, disable_sync_commit=True):
    maintenance_settings = get_maintenance_settings()
    maintenance_work_mem_gb = maintenance_settings["maintenance_work_mem_gb"]
    max_parallel_maintenance_workers = maintenance_settings["max_parallel_maintenance_workers"]
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Set PostgreSQL parameters for optimal index creation
        cur.execute(f"SET maintenance_work_mem = '{maintenance_work_mem_gb}GB';")
        cur.execute(f"SET max_parallel_maintenance_workers = {max_parallel_maintenance_workers};")
        if disable_sync_commit:
            cur.execute("SET synchronous_commit = OFF;")
        if hnsw_threads:
            cur.execute(f"SET hnsw.threads = {int(hnsw_threads)};")
        print(f"PostgreSQL parameters set: maintenance_work_mem = {maintenance_work_mem_gb}GB, "
              f"max_parallel_maintenance_workers = {max_parallel_maintenance_workers}")

        # Create index on document_id in documentblocks
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_documentblocks_document_id 
            ON documentblocks(document_id);
        """)
        print("Index on document_id in documentblocks created successfully.")

        # Dynamically create the vector index based on the selected index type (HNSW or IVFFlat)
        if index_type.lower() == "hnsw":
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_documentblocks_vector
                ON documentblocks USING hnsw (vector vector_l2_ops)
                WITH (m = {int(hnsw_m)}, ef_construction = {int(hnsw_ef_construction)});
            """)
            print("HNSW vector index on documentblocks created successfully.")
        elif index_type.lower() == "ivfflat":
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_documentblocks_vector
                ON documentblocks USING ivfflat (vector);
            """)
            print("IVFFlat vector index on documentblocks created successfully.")
        else:
            print(f"Unknown index type: {index_type}. Please use 'hnsw' or 'ivfflat'.")

        # Create index on role_id in PermissionAssignment
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_permissionassignment_role_id_document_id 
            ON PermissionAssignment(role_id, document_id);
        """)
        print("Index on role_id and document_id in PermissionAssignment created successfully.")

        # Create index on role_id in PermissionAssignment
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_permissionassignment_role_id 
            ON PermissionAssignment(role_id);
        """)
        print("Index on role_id in PermissionAssignment created successfully.")

        # Create index on user_id and role_id in UserRoles
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_userroles_user_id_role_id 
            ON UserRoles(user_id, role_id);
        """)
        print("Index on user_id and role_id in UserRoles created successfully.")

        conn.commit()

    except psycopg2.Error as e:
        print(f"Error creating indexes: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def drop_indexes():
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Drop index on document_id in documentblocks
        cur.execute("""
            DROP INDEX IF EXISTS idx_documentblocks_document_id;
        """)
        print("Index on document_id in documentblocks dropped successfully.")

        # Create index on vector field for fast retrieval
        cur.execute("""
            DROP INDEX IF EXISTS documents_embedding_idx;
        """)

        # Drop index on vector column in documentblocks
        cur.execute("""
            DROP INDEX IF EXISTS idx_documentblocks_vector;
        """)
        print("Vector index on documentblocks dropped successfully.")

        # Drop combination index on document_id and vector in documentblocks
        cur.execute("""
            DROP INDEX IF EXISTS idx_documentblocks_document_id_vector;
        """)
        print("Combined index on document_id and vector in documentblocks dropped successfully.")

        # Drop index on role_id and document_id in PermissionAssignment
        cur.execute("""
            DROP INDEX IF EXISTS idx_permissionassignment_role_id_document_id;
        """)
        print("Index on role_id and document_id in PermissionAssignment dropped successfully.")

        # Drop index on role_id in PermissionAssignment
        cur.execute("""
            DROP INDEX IF EXISTS idx_permissionassignment_role_id;
        """)
        print("Index on role_id in PermissionAssignment dropped successfully.")

        # Drop index on user_id and role_id in UserRoles
        cur.execute("""
            DROP INDEX IF EXISTS idx_userroles_user_id_role_id;
        """)
        print("Index on user_id and role_id in UserRoles dropped successfully.")

        conn.commit()

    except psycopg2.Error as e:
        print(f"Error dropping indexes: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


if __name__ == '__main__':
    #drop_indexes()
    create_indexes()
