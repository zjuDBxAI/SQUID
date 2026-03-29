import argparse
import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(sys.path)

from controller.prepare_database import clear_db
from controller.initialize_main_tables import initialize_database_deduplication
from services.read_dataset_function import read_and_store_dataset_parallel

DEFAULT_WORKERS = os.cpu_count() or 1

DATASET_VECTOR_DIMENSIONS = {
    "sift-128-euclidean": 128,
    "sift10m": 128,
    "wikipedia-22-12": 300,
    "arxiv": 300,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare database and load document blocks for benchmarks.")
    parser.add_argument("--dataset", choices=DATASET_VECTOR_DIMENSIONS.keys(), default="wikipedia-22-12",
                        help="Dataset to ingest.")
    parser.add_argument("--load-number", type=int, default=1_000_000,
                        help="Number of records to load (use a large number to load entire dataset).")
    parser.add_argument("--start-row", type=int, default=0, help="Starting row within the dataset.")
    parser.add_argument("--num-threads", type=int, default=DEFAULT_WORKERS,
                        help="Number of worker processes for dataset ingestion.")

    args = parser.parse_args()
    vector_dimension = DATASET_VECTOR_DIMENSIONS.get(args.dataset, 300)

    # init database
    clear_db()
    initialize_database_deduplication(enable_index=True, vector_dimension=vector_dimension)
    start_time = time.time()
    print(f"Using {args.num_threads} workers to load dataset '{args.dataset}' "
          f"(vector dimension {vector_dimension}).")
    read_and_store_dataset_parallel(
        load_number=args.load_number,
        start_row=args.start_row,
        num_threads=args.num_threads,
        dataset=args.dataset
    )
    elapsed_time = time.time() - start_time
    print(f"Time taken to execute read_dataset_deduplication_rbac: {elapsed_time:.2f} seconds")
