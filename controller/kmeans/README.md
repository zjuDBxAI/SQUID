# KMeans ACL Partition

`controller/kmeans` 是当前项目中的多租户向量检索分区方案实现。它不是传统语义 k-means，而是基于 ACL pattern、租户共现关系、存储预算和查询代价模型构建 PostgreSQL/pgvector 物理分区。

核心目标：

- 降低单次查询需要访问的分区数。
- 提高分区内权限选择性，减少 PostgreSQL HNSW 后过滤开销。
- 在允许的数据复制预算下，用更多空间换取更低延迟和更高召回。

## 文件结构

```text
controller/kmeans/
├── __init__.py              # 对外导出的 planner/materialize/search API
├── common.py                # 常量、表名、dataclass、partition table 命名
├── planner.py               # 兼容入口 TenantKMeansPlanner
├── repository.py            # 从 PostgreSQL 读取 ACL pattern 和租户-权限信息
├── hybrid_planner.py        # 核心分区规划算法和 cost model
├── storage.py               # 分区计划落库、物理分区表创建、索引构建
├── search.py                # 查询路由、分区内 pgvector 检索、结果合并
├── debug/                   # 分区质量和租户分布分析脚本
└── version/                 # 历史方案说明文档，不属于运行主链路
```

## 核心实现位置

### 1. 数据读取

实现位置：`repository.py`

`KMeansRepository.fetch_acl_rows()` 从数据库中读取文档 ACL，并按相同访问权限聚合成 ACL pattern：

- `PermissionAssignment` 提供 document 到 role 的权限关系。
- `UserRoles` 将 role 展开为 user/tenant。
- `documentblocks` 提供每个 document 的 block/vector 数量。

返回格式为：

```python
(pattern_id, tenant_ids, document_ids, vector_count)
```

其中 `tenant_ids` 表示一个 ACL pattern，`document_ids` 是具有相同 ACL 的文档集合，`vector_count` 是这些文档对应的向量数量。

### 2. 分区规划

实现位置：`hybrid_planner.py`

核心入口是：

```python
HybridACLKMeansPlanner.build_plan(...)
```

规划流程：

1. `_build_patterns()`：把数据库读取出的 ACL rows 转换为 `ACLPattern`，并计算 pattern 的 score/weight。
2. `_split_patterns()`：根据共享程度、查询代价和复制预算，把 ACL pattern 划分为 shared zone 和 private zone。
3. `_assign_shared_groups_by_cost_split()`：对 shared ACL pattern 做 bottom-up merge。候选 merge 以查询代价下降为收益，只有 `gain > 0` 时才继续合并。
4. `_cluster_private_tenants_by_cost_split()`：对 private zone 中的租户做 bottom-up merge。它先根据 ACL 共现图生成候选边，再按 `query_cost_increase / storage_saved` 选择合并，直到满足分区数或复制预算。
5. `_build_partitions()`：根据 shared/private 的规划结果生成物理分区定义。
6. `_build_routes()`：为每个 tenant 生成查询路由，即该 tenant 需要访问哪些分区，以及每个分区内允许的 `pattern_id`。

主要 cost model 在：

```python
HybridACLKMeansPlanner._partition_query_cost(...)
```

它同时考虑：

- 分区规模 `partition_vectors`
- 租户在分区内可见向量数 `accessible_vectors`
- 租户查询权重 `tenant_weight`
- 全局规模 `total_vectors`
- HNSW 搜索参数 `ef_search`

### 3. 计划落库和物理分区

实现位置：`storage.py`

主要入口：

```python
build_and_materialize_kmeans_plan(...)
```

它会执行完整流程：

1. 读取 ACL rows。
2. 调用 `HybridACLKMeansPlanner.build_plan()` 生成逻辑计划。
3. 调用 `materialize_plan()` 保存计划元数据。
4. 为每个分区创建物理表 `kmeans_documentblocks_partition_*`。
5. 可选创建 pgvector 索引。

元数据表：

```text
kmeans_current_plan
kmeans_current_partitions
kmeans_current_patterns
kmeans_current_routes
```

物理分区表前缀：

```text
kmeans_documentblocks_partition_
```

索引创建入口：

```python
create_indexes_for_materialized_partitions(index_type="hnsw")
drop_indexes_for_materialized_partitions()
```

### 4. 查询执行

实现位置：`search.py`

benchmark 调用入口：

```python
kmeans_partition_search(user_id, query_vector, topk=5, statistics_type="sql")
```

查询流程：

1. `load_tenant_routes(user_id)` 读取该 tenant 的路由。
2. 每个 route 设置自适应 `hnsw.ef_search`。
3. 在对应分区表中执行 pgvector 查询。
4. SQL 中使用 `pattern_id = ANY(...)` 做权限过滤。
5. 合并所有分区结果，按距离排序返回 top-k。

分区查询 SQL 在：

```python
_build_partition_query(...)
```

## 测试入口

主要测试脚本：

```text
basic_benchmark/test_kmeans_partition.py
```

benchmark 配置映射：

```text
basic_benchmark/condition_config.py
```

其中 `kmeans_partition` 对应：

```python
search_func_path = "controller.kmeans.kmeans_partition_search"
space_calc_func_path = "basic_benchmark.space_calculate.calculate_kmeans_partition"
```

空间统计实现：

```text
basic_benchmark/space_calculate.py
```

函数：

```python
calculate_kmeans_partition(...)
```

## 常用测试命令

第一次运行或需要重建分区时：

```bash
cd /data/Multitenanthakes
python basic_benchmark/test_kmeans_partition.py \
  --prepare true \
  --enable-index true \
  --index-type hnsw \
  --statistics-type sql \
  --query-num 1000 \
  --private-replication-budget-ratio 0.0 \
  --ef-search 120 \
```
核心参数private-replication-budget-ratio： 内存限制，指的是能够有多少跨分区复制。

如果分区已经 materialize，只想复用当前分区和索引跑测试：

```bash
cd /data/Multitenanthakes
python basic_benchmark/test_kmeans_partition.py \
  --prepare false \
  --enable-index true \
  --index-type hnsw \
  --statistics-type sql \
  --iterations 1 \
  --query-num 1000 \
  --ef-search 120
```

扫不同 `ef_search` 时可以使用：

```bash
cd /data/Multitenanthakes/basic_benchmark/script
bash run_ours.sh
```

## 重要参数

| 参数 | 作用 |
| `--private-replication-budget-ratio` | private zone 允许的数据复制预算，越大通常空间越高、路由和过滤代价越低。 |
| `--ef-search` | HNSW 查询搜索参数，同时参与 planner cost model 和实际查询。 |


## 调试脚本

`debug/` 下的脚本用于分析分区质量，不在 benchmark 主链路中直接调用：

```text
debug/analyze_cluster_similarity.py
debug/analyze_tenant_partitions.py
```

这些脚本通常用于检查：

- tenant 被分到哪些 partition。
- shared/private 分区大小是否异常。
- ACL pattern 是否被过度复制。
- route count 是否过高。

## 对外 API

`__init__.py` 导出的主要 API：

```python
build_and_materialize_kmeans_plan
clear_current_plan
create_indexes_for_materialized_partitions
drop_indexes_for_materialized_partitions
get_current_plan_summary
kmeans_partition_search
list_materialized_partition_tables
load_current_partitions
load_tenant_routes
```

通常只需要直接调用：

```python
from controller.kmeans import build_and_materialize_kmeans_plan, kmeans_partition_search
```

