import json
import os
import psycopg2
from psycopg2 import pool

def load_config():
    """
    Load configuration from a JSON file.
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
    with open(config_path, 'r') as config_file:
        return json.load(config_file)


config = load_config()

def get_dataset_path():
    """
    Get the dataset path from config, with fallback to default.
    """
    return config.get("dataset_path", "../dataset")

def get_maintenance_settings():
    """
    Retrieve maintenance-related PostgreSQL settings from config with safe defaults.
    """
    maintenance_work_mem_gb = int(config.get("maintenance_work_mem_gb", 1))
    max_parallel_workers = int(config.get("max_parallel_maintenance_workers", 1))
    return {
        "maintenance_work_mem_gb": max(1, maintenance_work_mem_gb),
        "max_parallel_maintenance_workers": max(1, max_parallel_workers),
    }

def get_db_connection():
    return psycopg2.connect(
        dbname=config["dbname"],
        user=config["user"],
        password=config["password"],
        host=config["host"],
        port=config["port"]
    )

# Create a global connection pool dictionary for user-specific pools
connection_pool = {}


def initialize_user_connections(minconn=1, maxconn=100):
    """
    Initialize a connection pool for each distinct user_id in the database.
    If no user_ids are found, the default connection pool will still be used.

    :param minconn: Minimum number of connections in the pool.
    :param maxconn: Maximum number of connections in the pool.
    """
    global connection_pool

    # Connect to the database to fetch all distinct user_ids
    with psycopg2.connect(
        dbname=config["dbname"],
        user=config["user"],
        password=config["password"],
        host=config["host"],
        port=config["port"],
    ) as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT user_id FROM userroles ORDER BY user_id;")
        user_ids = [row[0] for row in cur.fetchall()]  # Fetch all distinct user_ids

    # Initialize a connection pool for each user_id
    for user_id in user_ids:
        if user_id not in connection_pool:
            connection_pool[user_id] = pool.SimpleConnectionPool(
                minconn=minconn,
                maxconn=maxconn,
                dbname=config["dbname"],
                user=str(user_id),  # Use user_id as the username
                password=config["password"],
                host=config["host"],
                port=config["port"],
            )
            print(f"Connection pool created for user_id: {user_id}")

    # Create a default connection pool if needed
    if "default" not in connection_pool:
        connection_pool["default"] = pool.SimpleConnectionPool(
            minconn=minconn,
            maxconn=maxconn,
            dbname=config["dbname"],
            user=config["user"],  # Use the default user from the config
            password=config["password"],
            host=config["host"],
            port=config["port"],
        )
        print("Default connection pool created.")


def get_db_connection_from_pool(user_id=None, minconn=5, maxconn=100):
    """
    Get a database connection for the given user_id. If the connection pool is empty or insufficient,
    dynamically create more connections up to maxconn.

    :param user_id: Optional user identifier. If not provided, the default database pool is used.
    :param minconn: Minimum number of connections to create if the pool is empty.
    :param maxconn: Maximum number of connections that can be created for the pool.
    :return: A psycopg2 database connection.
    """
    global connection_pool
    pool_key = user_id if user_id else "default"

    # Check if the connection pool exists for the given user_id
    if pool_key not in connection_pool:
        # Create a new connection pool if it doesn't exist
        connection_pool[pool_key] = pool.SimpleConnectionPool(
            minconn=minconn,
            maxconn=maxconn,
            dbname=config["dbname"],
            user=str(user_id) if user_id else config["user"],
            password=config["password"],
            host=config["host"],
            port=config["port"],
        )
        print(f"Connection pool created for user_id: {pool_key} with minconn={minconn}, maxconn={maxconn}.")

    # Get a connection from the pool
    try:
        return connection_pool[pool_key].getconn()
    except pool.PoolError:
        # If the pool is exhausted, dynamically expand it
        print(f"Connection pool for user_id {pool_key} is exhausted. Expanding connections.")
        expand_connection_pool(pool_key, maxconn)
        return connection_pool[pool_key].getconn()


def expand_connection_pool(pool_key, maxconn):
    """
    Dynamically expand the connection pool for a specific user_id up to maxconn.

    :param pool_key: The identifier of the pool (user_id or "default").
    :param maxconn: Maximum number of connections for the pool.
    """
    if pool_key in connection_pool:
        old_pool = connection_pool[pool_key]
        current_minconn = len(old_pool._used) + len(old_pool._idle)  # Total current connections
        connection_pool[pool_key] = pool.SimpleConnectionPool(
            minconn=current_minconn,
            maxconn=maxconn,
            dbname=config["dbname"],
            user=str(pool_key) if pool_key != "default" else config["user"],
            password=config["password"],
            host=config["host"],
            port=config["port"],
        )
        print(f"Connection pool for user_id {pool_key} expanded to maxconn={maxconn}.")
    else:
        raise ValueError(f"No connection pool found for pool_key: {pool_key}.")

def release_db_connection(user_id, conn):
    """
    Release a connection back to the pool for a specific user_id.

    :param user_id: The user identifier whose pool the connection belongs to.
    :param conn: The connection object to release.
    """
    pool_key = user_id if user_id else "default"
    if pool_key in connection_pool:
        connection_pool[pool_key].putconn(conn)
    else:
        raise ValueError(f"No connection pool found for user_id: {pool_key}")


def get_document_vector_dimension(default=300):
    """
    Inspect the documentblocks.vector column and return its configured dimension.
    Falls back to `default` when the column is missing or dimension cannot be read.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT format_type(atttypid, atttypmod)
            FROM pg_attribute
            WHERE attrelid = 'documentblocks'::regclass
              AND attname = 'vector'
              AND NOT attisdropped;
        """)
        row = cur.fetchone()
        if row and row[0]:
            type_str = row[0]
            if "vector" in type_str:
                import re
                match = re.search(r'vector\((\d+)\)', type_str)
                if match:
                    return int(match.group(1))
        # Fallback: attempt to read a sample row if available
        try:
            cur.execute("SELECT vector_dims(vector) FROM documentblocks LIMIT 1;")
            sample = cur.fetchone()
            if sample and sample[0]:
                return int(sample[0])
        except psycopg2.Error:
            pass
    except psycopg2.Error as exc:
        print(f"Unable to determine vector dimension, defaulting to {default}: {exc}")
    finally:
        cur.close()
        conn.close()
    return default


def close_all_user_connections():
    """
    Close all connections in all user-specific connection pools.
    """
    for user_id, pool_instance in connection_pool.items():
        pool_instance.closeall()
        print(f"All connections closed for pool: {user_id}")
