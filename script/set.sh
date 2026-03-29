cd /data/Multitenanthakes/services/rbac_generator
python3 store_tree_based_rbac_generate_data.py

# init
cd /data/Multitenanthakes/basic_benchmark
python3 initialize_role_partition_tables.py --index_type hnsw
python3 initialize_combination_role_partition_tables.py --index_type hnsw
python3 generate_queries.py --num_queries 1000 --topk 100 --num_threads 4
rm -f ground_truth_cache.json
python3 compute_ground_truth.py --force


# set ef value.
cd /data/Multitenanthakes/controller/dynamic_partition/hnsw
rm -f parameter_hnsw.json
python3 AnonySys_dynamic_partition.py --storage 2.0 --recall 0.95


# init partition
cd /data/Multitenanthakes/basic_benchmark
python3 initialize_dynamic_partition_tables.py