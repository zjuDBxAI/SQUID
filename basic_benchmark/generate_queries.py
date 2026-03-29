import os
import sys
import json
import time
from psycopg2 import sql


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(sys.path)
from services.config import get_db_connection


def calculate_block_selectivity(user_id):
    """
    Executes the provided SQL query and calculates the block selectivity.
    :param user_id: The user ID for which the block selectivity is calculated.
    :return: The block selectivity (ratio of blocks accessed to total blocks in the table).
    """
    # Step 1: Connect to the database
    conn = get_db_connection()
    cur = conn.cursor()

    # Step 2: Define the SQL query to count distinct block_id based on the user_id
    sql_query = """
        SELECT COUNT(db.block_id)
        FROM PermissionAssignment pa
        JOIN UserRoles ur ON pa.role_id = ur.role_id
        JOIN documentblocks db ON db.document_id = pa.document_id
        WHERE ur.user_id = %s;
    """

    # Step 3: Execute the SQL query to get the number of accessed distinct block_id
    cur.execute(sql_query, (user_id,))
    accessed_blocks_result = cur.fetchone()

    if accessed_blocks_result is None or accessed_blocks_result[0] is None:
        accessed_blocks = 0
    else:
        accessed_blocks = accessed_blocks_result[0]  # Get the count of distinct block_id

    # Step 4: Get the total distinct block_id in the documentblocks table
    cur.execute("SELECT COUNT(block_id) FROM documentblocks;")
    total_blocks = cur.fetchone()[0]

    # Step 5: Calculate selectivity
    if total_blocks == 0:
        block_selectivity = 0
    else:
        block_selectivity = accessed_blocks / total_blocks

    # Step 6: Close the database connection
    cur.close()
    conn.close()

    return block_selectivity


def add_query_block_selectivity_to_json(file_path):
    with open(file_path, 'r') as file:
        query_data = json.load(file)

    for query in query_data:
        user_id = query["user_id"]

        query_block_selectivity = calculate_block_selectivity(user_id)

        query["query_block_selectivity"] = query_block_selectivity

    with open(file_path, 'w') as file:
        json.dump(query_data, file, indent=4)


if __name__ == '__main__':
    if __name__ == '__main__':
        import argparse
        from services.read_dataset_function import generate_query_dataset, generate_query_dataset_for_roles, \
            generate_query_dataset_for_cache, \
            generate_query_dataset_with_roles_and_repetitions
        from basic_benchmark.common_function import clear_ground_truth_cache

        # Set up command-line argument parser with only the essential parameters
        parser = argparse.ArgumentParser(description='Generate query dataset')
        parser.add_argument('--num_queries', type=int, default=1000, help='Number of queries to generate')
        parser.add_argument('--topk', type=int, default=10, help='Top K parameter')
        parser.add_argument('--num_threads', type=int, default=1, help='Number of threads to use')

        # Parse arguments
        args = parser.parse_args()

        # Clear ground truth cache when regenerating queries (since queries changed)
        print("Clearing ground truth cache (queries will be regenerated)...")
        clear_ground_truth_cache()

        # Keep the other parameters as default
        generate_query_dataset(
            num_queries=args.num_queries,
            topk=args.topk,
            output_file="query_dataset.json",
            zipf_param=0,
            num_threads=args.num_threads
        )

        print("✓ Query generation complete!")