# Experiment Code
## Prerequisites

- PostgreSQL
- psycopg2

### Clone the Repository

Clone this repository to your local machine.

### Install PostgreSQL 16 or higher and Development Tools
Ensure PostgreSQL and necessary development tools are installed

```sh
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib postgresql-server-dev-all build-essential
```
```shell
sudo apt install libpq-dev
```

### Install and Setup pgvector

The pgvector repository is already included in this project.

**Initial compilation and installation:**
```sh
chmod +x compile_pgvector.sh
./compile_pgvector.sh
```

This will compile pgvector in **debug mode** (`-g -O0`) which is useful for development and debugging.

**After modifying pgvector source code:**
Simply run the compile script again:
```sh
./compile_pgvector.sh
```

Then restart PostgreSQL to load the updated extension:
```sh
sudo service postgresql restart
```

### Setup Database

Start PostgreSQL service:
```sh
sudo service postgresql start
```

Create database user and database with pgvector extension:
```sh
chmod +x setup_db.sh
./setup_db.sh
```

This script will:
- Create user `xx` (with no password, or configure your own in the script)
- Create database `rbacdatabase_treebase`
- Install pgvector extension

**Note**: You may need to modify `setup_db.sh` to match your desired database name, username, and password. Make sure to update `config.json` accordingly.

### Setup Python Environment
Create a virtual environment and install dependencies:

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Setup embedding model

```sh
python -m spacy download en_core_web_md
```

### Download Dataset

Download the dataset to {project directory}/dataset/:

- [Cohere Wikipedia 22-12 Dataset](https://huggingface.co/datasets/Cohere/wikipedia-22-12)
```shell
mkdir dataset
cd dataset
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/datasets/Cohere/wikipedia-22-12
```

- [SIFT10M features](https://people.otago.ac.nz/xipingfu/SIFT10M.html) (Fu et al.)
  - Download `SIFT10M.tar.gz` and place it in the directory pointed to by `dataset_path` (defaults to `/data`).
  - The loader extracts `SIFT10M/SIFT10Mfeatures.mat` automatically on first run, or you can run  
    `tar -xf SIFT10M.tar.gz SIFT10M/SIFT10Mfeatures.mat`.


### Configure database config file

Edit `config.json` in project root directory to match your database setup:

```json
{
    "dbname": "rbacdatabase_treebase",
    "user": "x",
    "password": "123",
    "host": "localhost",
    "port": "5432",
    "dataset_path": "/data",
    "use_gpu_groundtruth": false
}
```

**Configuration Options:**
- `use_gpu_groundtruth`:
  - `false` (recommended): Use PostgreSQL for ground truth computation. Slower but no setup required.
  - `true`: Use FAISS GPU for ground truth computation. First run is slow (builds indexes), subsequent runs are much faster.

**Note**: If you used `setup_db.sh`, the default configuration is username `xx` with no password and database `rbacdatabase_treebase`. Adjust these values as needed.

### Prepare Data
```sh
cd basic_benchmark
# Load the default Wikipedia dataset (1M rows by default)
python3 common_prepare_pipeline.py --dataset wikipedia-22-12

# Example: load the SIFT 1M benchmark vectors (load-number 0 loads the entire file)
python3 common_prepare_pipeline.py --dataset sift-128-euclidean --load-number 0

# Example: load the 10M SIFT feature matrix (auto-extracts SIFT10Mfeatures.mat if needed)
python3 common_prepare_pipeline.py --dataset sift10m --load-number 0

# Flags:
#   --dataset       One of {wikipedia-22-12, arxiv, sift-128-euclidean, sift10m}
#   --load-number   Number of rows to ingest (0 or negative means “all remaining rows”)
#   --start-row     Offset within the dataset before loading
#   --num-threads   Worker processes used for ingestion (defaults to CPU count)

go to controller directory
# prepare main tables(user/role/permission) index
python3 initialize_main_tables.py
```

### Generate Permission
```sh
go to services/rbac_generator diretory

# Taking treebased as an example
python3 store_tree_based_rbac_generate_data.py
```
### Initilize partition and prepare for queries
```shell
go to basic_benchmark
# initialize role partition
python3 initialize_role_partition_tables.py --index_type hnsw

# (optional) initialize user partition
python3 initialize_combination_role_partition_tables.py --index_type hnsw

# generate queries
python3 generate_queries.py --num_queries 1000 --topk 100 --num_threads 4

# compute ground truth (pointer benchmark & tests share cache)
python3 compute_ground_truth.py
```

**Ground Truth Caching:**
- Ground truth results are automatically cached in `basic_benchmark/ground_truth_cache.json`
- Subsequent test runs with the same queries will load from cache (instant)
- Cache is automatically cleared when regenerating queries with `generate_queries.py`
- To manually clear cache: `rm basic_benchmark/ground_truth_cache.json`
```

### Initilize dynamic partition
```sh
# initilize dynamic
go to controller/dynamic_partition/hnsw
# if needed, delete parameter_hnsw.json from hnsw directory to regenerate parameters
python3 AnonySys_dynamic_partition.py --storage 2.0 --recall 0.95

Run(HNSW index)
```sh
go to basic_benchmark directory

# example:
python test_all.py --algorithm RLS --efs 500
python test_all.py --algorithm ROLE --efs 20
python test_all.py --algorithm USER --efs 20
python test_all.py --algorithm AnonySys --efs 40
```


Run(ACORN index)
```sh
go to acorn_benchmark directory

#modify efs value from main.cpp
build C++ project and run
```

Before building, create `acorn_benchmark/config.json` to point benchmarks at the shared index location:

```json
{
    "index_storage_path": "/pgsql_data/acorn/"
}
```

Make sure `/pgsql_data/acorn/` exists ahead of time; both ACORN and dynamic-partition indexes are persisted there.
