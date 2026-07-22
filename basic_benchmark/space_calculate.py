import os
import re
import sys
from decimal import Decimal

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import quote_ident
from wasabi import table

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from services.config import get_db_connection
from controller.baseline.HQI.qd_tree import DEFAULT_QD_TREE_PARTITION_PREFIX

MB_DIVISOR = 1024 * 1024

BYTES_PER_COMPONENT = {
    "vector": 4,   # float32
    "halfvec": 2,  # float16
}


def _get_index_size_mb(cur, table_name):
    """
    Return the total size of indexes on the given table in MB.
    """
    try:
        cur.execute(
            """
            SELECT COALESCE(SUM(pg_total_relation_size(indexrelid)), 0)
            FROM pg_index
            WHERE indrelid = %s::regclass
            """,
            [table_name],
        )
    except psycopg2.errors.UndefinedTable:
        cur.connection.rollback()
        return 0.0

    index_bytes = cur.fetchone()[0] or 0
    if not index_bytes:
        return 0.0
    index_bytes = float(index_bytes)
    return index_bytes / MB_DIVISOR


def _get_vector_metadata(cur, table_name):
    """
    Fetch the vector column's base type name and declared dimension.

    Returns:
        tuple[str, int] | None: (base_type, dimension) if available.
    """
    try:
        cur.execute(
            """
            SELECT
                format_type(a.atttypid, a.atttypmod) AS column_type,
                a.atttypid::regtype::text AS type_name
            FROM pg_attribute a
            WHERE a.attrelid = %s::regclass
              AND a.attname = 'vector'
              AND NOT a.attisdropped
            """,
            [table_name],
        )
    except psycopg2.errors.UndefinedTable:
        cur.connection.rollback()
        return None

    result = cur.fetchone()
    if not result:
        return None

    column_type, type_name = result
    base_type = type_name.split(".")[-1] if type_name else None

    dimension = None
    if column_type:
        match = re.search(r"\((\d+)\)", column_type)
        if match:
            dimension = int(match.group(1))

    return base_type, dimension


def _safe_avg_dimension(cur, table_identifier):
    """
    Fallback to querying the average vector dimension if metadata is unavailable.
    """
    try:
        cur.execute(
            sql.SQL(
                "SELECT AVG(vector_dims(vector)) FROM {} WHERE vector IS NOT NULL"
            ).format(sql.SQL(table_identifier))
        )
    except psycopg2.errors.UndefinedFunction:
        cur.connection.rollback()
        return None

    avg_dim = cur.fetchone()[0]
    if isinstance(avg_dim, Decimal):
        avg_dim = float(avg_dim)
    return avg_dim


def _avg_column_bytes(cur, table_identifier):
    """
    Estimate bytes per vector using pg_column_size when type-specific math is unavailable.
    """
    cur.execute(
        sql.SQL(
            "SELECT AVG(pg_column_size(vector)) FROM {} WHERE vector IS NOT NULL"
        ).format(sql.SQL(table_identifier))
    )
    avg_bytes = cur.fetchone()[0]
    if isinstance(avg_bytes, Decimal):
        avg_bytes = float(avg_bytes)
    return avg_bytes


def calculate_size_in_mb(table_names):
    """
    Estimate the total storage space taken by vector columns in the specified tables.

    Args:
        table_names (list): List of table names to calculate size for.

    Returns:
        float: Total size in MB attributed to vectors across the tables.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    total_vector_bytes = 0
    for table in table_names:
        candidate_names = [table]
        lowered = table.lower()
        if lowered != table:
            candidate_names.append(lowered)

        for candidate in candidate_names:
            metadata = _get_vector_metadata(cur, candidate)
            if metadata is None:
                continue

            table_identifier = quote_ident(candidate.lower(), conn)

            cur.execute(
                sql.SQL(
                    "SELECT COUNT(*) FROM {} WHERE vector IS NOT NULL"
                ).format(sql.SQL(table_identifier))
            )
            vector_count = cur.fetchone()[0]
            if not vector_count:
                break
            vector_count = int(vector_count)

            base_type, dimension = metadata
            if dimension is None:
                dimension = _safe_avg_dimension(cur, table_identifier)
                if not dimension:
                    base_type = None  # force fallback to pg_column_size
                else:
                    dimension = int(round(dimension))
            else:
                dimension = int(dimension)

            if base_type:
                base_type = base_type.split(".")[-1]

            bytes_per_component = BYTES_PER_COMPONENT.get(base_type)
            if bytes_per_component and dimension:
                total_vector_bytes += vector_count * dimension * bytes_per_component
            else:
                avg_bytes = _avg_column_bytes(cur, table_identifier)
                if avg_bytes:
                    total_vector_bytes += vector_count * avg_bytes
            break

    cur.close()
    conn.close()

    return total_vector_bytes / (1024 * 1024)


def calculate_table_storage_in_mb(table_names):
    """
    Calculate the total physical storage used by the specified tables (all columns + indexes).

    Args:
        table_names (list): List of table names to calculate size for.

    Returns:
        float: Total size of the tables in MB.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    total_size = 0
    for table in table_names:
        cur.execute(sql.SQL("SELECT pg_total_relation_size(%s);"), [table])
        table_size = cur.fetchone()[0]
        total_size += table_size

    cur.close()
    conn.close()

    return total_size / (1024 * 1024)


def calculate_prefilter(condition, *, enable_index):
    """
    Calculate the total physical storage used by the prefilter partitions.

    Returns:
        float: The total size of the relevant tables in MB.
    """
    tables = [
        'Documents',
        'PermissionAssignment',
        'Roles',
        'UserRoles',
        'Users',
    ]

    conn = get_db_connection()
    cur = conn.cursor()
    if condition == "prefilter_partition_role":
        cur.execute("SELECT tablename FROM pg_tables WHERE tablename LIKE 'documentblocks_role_%';")
    elif condition == "prefilter_partition_combination":
        cur.execute("SELECT tablename FROM pg_tables WHERE tablename LIKE 'documentblocks_comb_%';")
    else:
        raise ValueError("Invalid condition specified")

    tables.extend(row[0] for row in cur.fetchall())
    cur.close()
    conn.close()

    return calculate_table_storage_in_mb(tables)


def calculate_postfilter(condition=None, *, enable_index=None):
    """
    Calculate the total storage space used by the postfilter scenario.

    Returns:
        float: The total size of the relevant tables in MB.
    """
    tables = [
        'Documents',
        'PermissionAssignment',
        'Roles',
        'UserRoles',
        'Users',
        'documentblocks'
    ]

    # Calculate and return the total size in MB
    return calculate_table_storage_in_mb(tables)


def calculate_rls(condition=None, *, enable_index=None):
    """
    Calculate the total physical storage used by the RLS scenario.

    Returns:
        float: Total size in MB, using full relation size for documentblocks.
    """
    base_tables = [
        'Documents',
        'PermissionAssignment',
        'Roles',
        'UserRoles',
        'Users',
        'documentblocks',
    ]

    total_size_mb = calculate_table_storage_in_mb(base_tables)
    rls_policy_size_mb = float(calculate_rls_policy_size())
    return total_size_mb + rls_policy_size_mb


def calculate_rls_policy_size(dynamic_partition=False):
    """
    Calculate the total storage space used by RLS policies and optionally include materialized views.

    Args:
        dynamic_partition (bool): If True, include the size of the materialized view.

    Returns:
        float: The total size of RLS policies and materialized views in MB.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # Calculate the size of RLS policies
    cur.execute("""
        SELECT pg_total_relation_size('pg_policy') / 1024.0 / 1024.0;
    """)
    rls_policy_size_mb = float(cur.fetchone()[0] or 0.0)

    # If dynamic_partition is True, include the size of the materialized view
    if dynamic_partition:
        cur.execute("SELECT to_regclass('user_accessible_documents') IS NOT NULL;")
        if cur.fetchone()[0]:
            cur.execute("""
                SELECT pg_total_relation_size('user_accessible_documents') / 1024.0 / 1024.0;
            """)
            materialized_view_size_mb = float(cur.fetchone()[0] or 0.0)
        else:
            materialized_view_size_mb = 0.0
        total_size_mb = rls_policy_size_mb + materialized_view_size_mb
    else:
        total_size_mb = rls_policy_size_mb

    cur.close()
    conn.close()

    return total_size_mb


def calculate_partition_proposal(condition, *, enable_index=None):
    """
    Calculate the total storage space used by the partition proposal scenario.

    Args:
        condition (str): The partitioning strategy ("lsh", "role_partition", "uniform").

    Returns:
        float: The total size of the relevant tables in MB.
    """
    tables = [
        'Documents',
        'PermissionAssignment',
        'Roles',
        'UserRoles',
        'Users'
    ]

    conn = get_db_connection()
    cur = conn.cursor()

    # Determine the specific partition tables to include based on the condition
    if condition == "lsh":
        cur.execute("SELECT tablename FROM pg_tables WHERE tablename LIKE 'lsh_partition_%';")
    elif condition == "role":
        cur.execute("SELECT tablename FROM pg_tables WHERE tablename LIKE 'role_document_partition_%';")
    elif condition == "uniform_disjoint_partition":
        cur.execute("SELECT tablename FROM pg_tables WHERE tablename LIKE 'documentblocks_partition_%';")
    else:
        raise ValueError("Invalid condition specified")

    partition_tables = cur.fetchall()

    # Add partition table names to the list of tables to calculate size for
    for table_name in partition_tables:
        tables.append(table_name[0])

    cur.close()
    conn.close()

    # Calculate and return the total size in MB
    return calculate_table_storage_in_mb(tables)


def calculate_dynamic_partition(condition=None, *, enable_index=None):
    """
    Calculate the total physical storage used by dynamic partition tables, including RolePartitions.

    Returns:
        float: The total size of the dynamic partition tables in MB.
    """
    mapping_table = os.environ.get("HONEYBEE_MAPPING_TABLE") or os.environ.get("DYNAMIC_PARTITION_MAPPING_TABLE") or "CombRolePartitions"
    table_prefix = os.environ.get("HONEYBEE_TABLE_PREFIX") or os.environ.get("DYNAMIC_PARTITION_TABLE_PREFIX")
    tables = [
        'Documents', 'PermissionAssignment', 'Roles',
        'UserRoles', 'Users', mapping_table
    ]

    conn = get_db_connection()
    cur = conn.cursor()
    like_pattern = f"{table_prefix}_partition_%" if table_prefix else "documentblocks_partition_%"
    cur.execute("SELECT tablename FROM pg_tables WHERE tablename LIKE %s;", [like_pattern])
    tables.extend(row[0] for row in cur.fetchall())
    cur.close()
    conn.close()

    total_storage_mb = calculate_table_storage_in_mb(tables)
    rls_policy_size_mb = float(calculate_rls_policy_size(dynamic_partition=True))
    return total_storage_mb + rls_policy_size_mb


def calculate_method_partition(condition=None, *, enable_index=None):
    """Calculate total physical storage for the standalone method_partition implementation."""
    base_tables = [
        'Documents', 'PermissionAssignment', 'Roles', 'UserRoles', 'Users',
        'dynamic_partition_current_plan', 'dynamic_partition_current_partitions',
        'dynamic_partition_current_partition_documents', 'dynamic_partition_current_logical_patterns',
        'dynamic_partition_current_dag_nodes',
    ]

    total_storage_mb = 0.0
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        existing_base_tables = []
        for table_name in base_tables:
            cur.execute('SELECT to_regclass(%s);', [table_name])
            if cur.fetchone()[0] is not None:
                existing_base_tables.append(table_name)
        if existing_base_tables:
            total_storage_mb += calculate_table_storage_in_mb(existing_base_tables)

        cur.execute(
            """
            SELECT tablename
            FROM pg_tables
            WHERE tablename LIKE 'workload_documentblocks_partition_%'
               OR tablename LIKE 'workload_documentblocks_overlay_%'
               OR tablename LIKE 'workload_documentblocks_access_overlay_%';
            """
        )
        partition_tables = [row[0] for row in cur.fetchall()]
        if partition_tables:
            total_storage_mb += calculate_table_storage_in_mb(partition_tables)
    finally:
        cur.close()
        conn.close()

    return total_storage_mb


def calculate_kmeans_partition(condition=None, *, enable_index=None):
    """Calculate total physical storage for the kmeans tenant-cluster partition implementation."""
    plan_table = os.environ.get('KMEANS_PLAN_TABLE', 'kmeans_current_plan')
    partition_table = os.environ.get('KMEANS_PARTITION_TABLE', 'kmeans_current_partitions')
    pattern_table = os.environ.get('KMEANS_PATTERN_TABLE', 'kmeans_current_patterns')
    route_table = os.environ.get('KMEANS_ROUTE_TABLE', 'kmeans_current_routes')
    base_tables = [
        'Documents', 'PermissionAssignment', 'Roles', 'UserRoles', 'Users',
        plan_table, partition_table, pattern_table, route_table,
    ]

    total_storage_mb = 0.0
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        existing_base_tables = []
        for table_name in base_tables:
            cur.execute('SELECT to_regclass(%s);', [table_name])
            if cur.fetchone()[0] is not None:
                existing_base_tables.append(table_name)
        if existing_base_tables:
            total_storage_mb += calculate_table_storage_in_mb(existing_base_tables)

        cur.execute(
            sql.SQL(
                """
                SELECT table_name
                FROM {}
                WHERE plan_id = (
                    SELECT plan_id FROM {} ORDER BY plan_id DESC LIMIT 1
                );
                """
            ).format(sql.Identifier(partition_table), sql.Identifier(plan_table))
        )
        partition_tables = [row[0] for row in cur.fetchall()]
        if partition_tables:
            total_storage_mb += calculate_table_storage_in_mb(partition_tables)
    finally:
        cur.close()
        conn.close()

    return total_storage_mb


def calculate_adaptive_tenant(condition=None, *, enable_index=None):
    """Calculate total physical storage for adaptive_tenant control and physical partitions."""
    base_tables = [
        'Documents', 'PermissionAssignment', 'Roles', 'UserRoles', 'Users',
        'adaptive_tenant_profiles', 'adaptive_tenant_windows',
        'adaptive_tenant_current_plan', 'adaptive_tenant_current_partitions',
        'adaptive_tenant_current_partition_members', 'adaptive_tenant_current_operations',
    ]

    total_storage_mb = 0.0
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        existing_base_tables = []
        for table_name in base_tables:
            cur.execute('SELECT to_regclass(%s);', [table_name])
            if cur.fetchone()[0] is not None:
                existing_base_tables.append(table_name)
        if existing_base_tables:
            total_storage_mb += calculate_table_storage_in_mb(existing_base_tables)

        cur.execute(
            """
            SELECT tablename
            FROM pg_tables
            WHERE tablename LIKE 'adaptive_documentblocks_partition_%';
            """
        )
        partition_tables = [row[0] for row in cur.fetchall()]
        if partition_tables:
            total_storage_mb += calculate_table_storage_in_mb(partition_tables)
    finally:
        cur.close()
        conn.close()

    return total_storage_mb


def calculate_latent_access(condition=None, *, enable_index=None):
    """Calculate total physical storage for latent_access control data and materialized partitions."""
    base_tables = [
        'Documents', 'PermissionAssignment', 'Roles', 'UserRoles', 'Users',
        'latent_access_current_plan', 'latent_access_current_partitions',
        'latent_access_current_partition_documents', 'latent_access_current_atom_tenants',
    ]

    total_storage_mb = 0.0
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        existing_base_tables = []
        for table_name in base_tables:
            cur.execute('SELECT to_regclass(%s);', [table_name])
            if cur.fetchone()[0] is not None:
                existing_base_tables.append(table_name)
        if existing_base_tables:
            total_storage_mb += calculate_table_storage_in_mb(existing_base_tables)

        cur.execute(
            """
            SELECT tablename
            FROM pg_tables
            WHERE tablename LIKE 'latent_documentblocks_partition_%';
            """
        )
        partition_tables = [row[0] for row in cur.fetchall()]
        if partition_tables:
            total_storage_mb += calculate_table_storage_in_mb(partition_tables)
    finally:
        cur.close()
        conn.close()

    return total_storage_mb


def calculate_qd_tree_storage(condition: str = None, partition_prefix: str = None, *, enable_index=None) -> float:
    """
    Calculate the total physical storage used by the QD-tree partitioning scheme.

    Args:
        condition (str): Unused placeholder to maintain compatibility with the benchmark harness.
        partition_prefix (str): Prefix used when materializing QD-tree partitions.

    Returns:
        float: Total size in MB of core tables, QD-tree partitions, and associated RLS metadata.
    """
    prefix = partition_prefix or DEFAULT_QD_TREE_PARTITION_PREFIX
    tables = ['Documents', 'PermissionAssignment', 'Roles', 'UserRoles', 'Users']

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT tablename
        FROM pg_tables
        WHERE tablename LIKE %s;
        """,
        (f"{prefix}_%",),
    )
    tables.extend(row[0] for row in cur.fetchall())
    cur.close()
    conn.close()

    total_storage_mb = calculate_table_storage_in_mb(tables)
    rls_policy_size_mb = float(calculate_rls_policy_size(dynamic_partition=False))
    return total_storage_mb + rls_policy_size_mb


def calculate_sieve(condition=None, *, enable_index=None):
    """Calculate the total physical storage for the PostgreSQL SIEVE baseline."""
    tables = [
        'Documents', 'PermissionAssignment', 'Roles', 'UserRoles', 'Users',
        'sieve_current_plan', 'sieve_current_candidates', 'sieve_current_partitions',
        'sieve_current_edges', 'sieve_current_hasse_edges', 'sieve_documentblocks_root',
    ]

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT tablename
            FROM pg_tables
            WHERE tablename LIKE 'sieve_documentblocks_partition_%';
        """)
        tables.extend(row[0] for row in cur.fetchall())
    finally:
        cur.close()
        conn.close()

    return calculate_table_storage_in_mb(tables)



def calculate_veda(condition=None, *, enable_index=None):
    """Calculate total physical storage for the Veda/EffVeda access-aware lattice baseline."""
    plan_table = os.environ.get('VEDA_PLAN_TABLE', 'veda_current_plan')
    pattern_table = os.environ.get('VEDA_PATTERN_TABLE', 'veda_current_patterns')
    node_table = os.environ.get('VEDA_NODE_TABLE', 'veda_current_nodes')
    route_table = os.environ.get('VEDA_ROUTE_TABLE', 'veda_current_user_routes')
    tables = [
        'Documents', 'PermissionAssignment', 'Roles', 'UserRoles', 'Users',
        plan_table, pattern_table, node_table, route_table,
    ]

    normalized = str(condition or "").strip().lower()
    if normalized == "veda":
        node_table_like = "veda_documentblocks_node_veda_%"
    elif normalized == "effveda":
        node_table_like = "veda_documentblocks_node_effveda_%"
    else:
        node_table_like = "veda_documentblocks_node_%"

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        existing_tables = []
        for table_name in tables:
            cur.execute('SELECT to_regclass(%s);', [table_name])
            if cur.fetchone()[0] is not None:
                existing_tables.append(table_name)

        if os.environ.get('VEDA_NODE_TABLE'):
            cur.execute(
                sql.SQL(
                    """
                    SELECT table_name
                    FROM {}
                    WHERE plan_id = (
                        SELECT plan_id FROM {} ORDER BY plan_id DESC LIMIT 1
                    );
                    """
                ).format(sql.Identifier(node_table), sql.Identifier(plan_table))
            )
            existing_tables.extend(row[0] for row in cur.fetchall())
        else:
            cur.execute(
                """
                SELECT tablename
                FROM pg_tables
                WHERE tablename LIKE %s;
                """,
                [node_table_like],
            )
            existing_tables.extend(row[0] for row in cur.fetchall())
    finally:
        cur.close()
        conn.close()

    if not existing_tables:
        return 0.0
    return calculate_table_storage_in_mb(existing_tables)


def calculate_acl_partition(condition=None, *, enable_index=None):
    """Calculate total physical storage for the ACL-per-partition baseline."""
    tables = [
        'Documents', 'PermissionAssignment', 'Roles', 'UserRoles', 'Users',
        'acl_partition_current_plan', 'acl_partition_current_patterns',
        'acl_partition_current_routes',
    ]

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        existing_tables = []
        for table_name in tables:
            cur.execute('SELECT to_regclass(%s);', [table_name])
            if cur.fetchone()[0] is not None:
                existing_tables.append(table_name)

        cur.execute("""
            SELECT tablename
            FROM pg_tables
            WHERE tablename LIKE 'acl_documentblocks_partition_%';
        """)
        existing_tables.extend(row[0] for row in cur.fetchall())
    finally:
        cur.close()
        conn.close()

    if not existing_tables:
        return 0.0
    return calculate_table_storage_in_mb(existing_tables)
