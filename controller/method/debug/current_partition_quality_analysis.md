# 当前已建分区质量分析

分析对象：当前数据库中已经物化的 method plan。

当前结果：

```text
Average Recall: 0.9554
Average Query Time: 0.0165 seconds
Space used: 2078.20 MB
```

当前 plan 元信息：

```text
plan_id: 46
target_partition_count: 100
actual partition_count: 100
logical_pattern_count: 898
document_count: 10000
protection_overlay_space_ratio: 0.5
overlay_budget_vectors: 500000
protection_overlay_selected_vectors: 452900
shared_protection_group_count: 3
shared_protection_protected_tenant_count: 9
shared_protection_skipped_low_fanout_count: 503
```

## 1. 总体判断

这一版的 recall 已经比较高：

```text
Recall = 0.9554
```

但是 query latency 偏高：

```text
Average Query Time = 0.0165s
```

主要原因不是单一的 HNSW 参数，而是当前分区结构和保护分区结构共同导致：

1. 基础分区大小极不均衡。
2. 大量租户本身 fanout 很高。
3. 保护分区只覆盖了少量租户，没有覆盖最高 fanout 的租户。
4. 保护分区表本身偏大，查询保护表也会产生较高 HNSW / SQL 成本。
5. 很多大分区只包含 1-2 个 ACL pattern，说明当前分区数增加后并没有真正把大 pattern 拆小。

一句话结论：

> 当前版本的保护分区确实避免了之前“131 个不相似租户塞进一张巨表”的问题，但它现在保护得太保守，而且基础分区内部仍然存在严重的大 pattern / 大分区问题，所以平均延迟仍然偏高。

## 2. 基础分区大小质量

当前 100 个基础分区的向量数统计：

```text
partition_count: 100
min_vectors: 100
p25_vectors: 175
p50_vectors: 3400
p75_vectors: 20425
p90_vectors: 23400
max_vectors: 25600
total_vectors: 1000000
```

这个分布非常不均衡。

可以看成两类分区：

- 很多极小分区：100-1500 vectors。
- 一批很大的分区：16000-25600 vectors。

按 bucket 统计：

```text
0-2000 vectors:      48 partitions, 15300 vectors
2300-3900 vectors:    3 partitions,  9100 vectors
6200 vectors:         1 partition,   6200 vectors
15700-15900 vectors:  2 partitions, 31600 vectors
16000-16800 vectors: 13 partitions, 213800 vectors
18000-19800 vectors:  5 partitions, 94500 vectors
20200-21900 vectors: 13 partitions, 273600 vectors
22300-23900 vectors: 11 partitions, 255800 vectors
24800-25600 vectors:  4 partitions, 100100 vectors
```

这说明目标分区数变成 100 后，并没有得到“100 个大小接近的小分区”。

实际效果是：

```text
很多小分区 + 一批仍然很大的分区
```

这会带来两个延迟问题：

1. 如果 query 命中很多小分区，会产生大量 SQL union / HNSW 启动成本。
2. 如果 query 命中大分区，单个 HNSW 表成本仍然不低。

因此分区数增加并不自动降低查询延迟。

## 3. 最大分区情况

当前最大的 20 个基础分区：

```text
p0   25600 vectors, 2 patterns, 135 tenants
p1   24900 vectors, 2 patterns, 233 tenants
p2   24800 vectors, 2 patterns, 267 tenants
p3   24800 vectors, 2 patterns, 197 tenants
p4   23900 vectors, 2 patterns, 207 tenants
p5   23900 vectors, 2 patterns, 368 tenants
p6   23800 vectors, 2 patterns, 82 tenants
p7   23700 vectors, 2 patterns, 195 tenants
p8   23500 vectors, 2 patterns, 267 tenants
p9   23400 vectors, 2 patterns, 273 tenants
p10  23400 vectors, 2 patterns, 194 tenants
p11  22800 vectors, 2 patterns, 325 tenants
p12  22700 vectors, 2 patterns, 202 tenants
p13  22400 vectors, 1 pattern, 35 tenants
p14  22300 vectors, 2 patterns, 186 tenants
p15  21900 vectors, 1 pattern, 77 tenants
p16  21700 vectors, 2 patterns, 193 tenants
p17  21600 vectors, 2 patterns, 282 tenants
p18  21400 vectors, 2 patterns, 160 tenants
p19  21300 vectors, 2 patterns, 250 tenants
```

这里最关键的现象是：

```text
大分区通常只包含 1-2 个 ACL pattern。
```

这说明这些大分区并不是因为多个小 pattern 被错误合并造成的，而是因为某些 exact ACL pattern 本身就很大。

当前 method 把 exact ACL pattern 当作不可拆最小权限单元，所以这些大 pattern 无法再被 K-cut 分小。

这会导致：

- 增加 `target_partition_count` 也不能有效缩小这些大 pattern。
- 保护分区如果复制这些 pattern，也会很大。
- HNSW build 和查询都会被这些大 pattern 主导。

## 4. pattern 数和向量量关系

按每个分区包含的 pattern 数统计：

```text
1 pattern:  29 partitions,  88100 vectors
2 patterns: 36 partitions, 639900 vectors
<=2 patterns: 65 partitions, 728000 vectors
```

也就是说：

```text
65% 的分区只包含 1-2 个 pattern
这些分区占 72.8% 的总向量
```

这说明当前分区质量的核心瓶颈不是“pattern 合并太多”，而是：

```text
pattern 本身太大，而且不能被拆分。
```

如果继续坚持 exact ACL pattern 绝对不可拆，那么基础分区的大小均衡能力会受到很强限制。

## 5. 租户 fanout 分布

当前 1000 个租户的基础分区 fanout 统计：

```text
all tenants:
branch min/p50/p75/p90/max = 8 / 40 / 49 / 56 / 68
vectors p50/p90/max = 122900 / 209300 / 284700

protected tenants:
branch min/p50/p75/p90/max = 41 / 43 / 44 / 49 / 50
vectors p50/p90/max = 113000 / 136700 / 149800

unprotected tenants:
branch min/p50/p75/p90/max = 8 / 40 / 49 / 56 / 68
vectors p50/p90/max = 122900 / 209300 / 284700
```

这个结果非常重要。

当前保护层只保护了 9 个租户，但全体租户的 median fanout 已经是 40，p90 fanout 是 56。

也就是说：

```text
高 fanout 不是少数极端租户现象，而是当前基础分区下的普遍现象。
```

同时，保护层没有覆盖最高 fanout 的租户。

最高 fanout 的租户示例：

```text
tenant 375: 68 branches, 284700 vectors, protected=False
tenant 446: 68 branches, 283600 vectors, protected=False
tenant 975: 67 branches, 273200 vectors, protected=False
tenant 253: 66 branches, 225500 vectors, protected=False
tenant 884: 65 branches, 265300 vectors, protected=False
tenant 152: 64 branches, 269500 vectors, protected=False
tenant 661: 64 branches, 268200 vectors, protected=False
tenant 967: 64 branches, 251500 vectors, protected=False
```

这说明当前保护分区选择策略虽然避免了低相似租户乱合并，但也过度依赖“租户之间权限相似度”和“保护 group 收益密度”，导致最高 fanout 租户没有被优先保护。

## 6. 保护分区质量

当前保护分区有 3 张表：

```text
protect_2:
  tenants: 4
  patterns: 344
  base_partitions: 60
  vector_count: 216300
  document_count: 2163
  avg_tenant_selectivity: 0.5850
  min_tenant_selectivity: 0.4535
  table_size: 299 MB

protect_1:
  tenants: 1
  patterns: 222
  base_partitions: 44
  vector_count: 123600
  document_count: 1236
  avg_tenant_selectivity: 1.0000
  min_tenant_selectivity: 1.0000
  table_size: 170 MB

protect_0:
  tenants: 4
  patterns: 206
  base_partitions: 44
  vector_count: 113000
  document_count: 1130
  avg_tenant_selectivity: 0.8633
  min_tenant_selectivity: 0.8177
  table_size: 156 MB
```

保护表总向量：

```text
452900 vectors
```

保护表总空间约：

```text
299 MB + 170 MB + 156 MB = 625 MB
```

保护层现在的优点是：

- 不再是之前那种 131 个不相似租户合成一张巨表。
- group 内 tenant selectivity 明显更高。
- 保护的租户确实是 high fanout 租户，branch count 大约 41-50。

保护层现在的问题是：

- 只保护了 9 个租户，覆盖面太小。
- 保护表仍然很大，查询保护表本身并不便宜。
- 保护表消耗了约 625MB 额外空间，但对平均延迟的帮助有限。
- 最高 fanout 租户没有被保护。

## 7. 空间结构问题

当前最大表空间主要来自保护表：

```text
workload_documentblocks_access_overlay_cover_protect_210ad90a3d: 299 MB
workload_documentblocks_access_overlay_cover_protect_fa82dffd0e: 170 MB
workload_documentblocks_access_overlay_cover_protect_ed6f445c64: 156 MB
```

基础大分区表大约在 28-36MB：

```text
p0: 36 MB
p1: 35 MB
p2: 35 MB
p3: 35 MB
p4-p12: 32-33 MB
```

这说明当前额外空间主要被保护层吃掉。

如果保护表不能显著降低 query fanout 或 query time，那么这 625MB 空间的性价比不高。

## 8. 当前延迟高的主要原因

结合当前统计，平均延迟高的原因可以拆成四个层面。

### 8.1 基础分区 fanout 普遍偏高

全体租户 median fanout 是 40。

这意味着即使没有保护分区，普通租户查询也经常面对很多候选分区。

当前 `route_limit=16` 会限制实际查询分支，但为了保持 recall，coverage guard 和候选排序仍然会处理大量可访问分区信息。

### 8.2 基础分区大小不均衡

48 个小分区只包含 15300 vectors。

但 46 个左右的大分区包含绝大多数向量。

这说明很多 query 即使命中较少分区，也可能命中大表。

### 8.3 大 exact ACL pattern 不可拆

65 个分区只包含 1-2 个 pattern，却占 72.8% 向量。

这意味着当前 K-cut 无法解决这些大 pattern。

如果不允许 pattern 内部拆分，那么基础分区很难继续变均衡。

### 8.4 保护分区表过大

保护表最大 216300 vectors，空间 299MB。

这种表虽然能把 40-60 个基础分区查询变成一个保护表查询，但单表 HNSW 查询成本和 pattern filter 成本也会上升。

所以保护分区不是越大越好。

## 9. 对当前方案的启发

### 9.1 保护分区选择要从“相似优先”改成“fanout 优先 + 相似约束”

当前保护选择已经加入相似约束后，确实避免了乱合并。

但它漏掉了最高 fanout 租户。

更合理的目标应该是：

```text
先按 fanout pain 找到最需要保护的租户
再在这些租户里寻找可共享保护表的机会
```

而不是：

```text
先找最相似、收益密度最高的 group
```

推荐新的优先级：

```text
tenant pain(u) =
  query_weight(u)
  * branch_count(u)
  * small_branch_ratio(u)
  * estimated_base_cost(u)
```

预算选择时先覆盖 pain 最大的租户，再考虑 group 共享。

### 9.2 对高 fanout 但不相似的租户，应允许 singleton 保护

当前 `protect_1` 是 singleton，说明这个方向是合理的。

对最高 fanout 租户，如果找不到相似 tenant，也应该允许单独保护。

否则这些租户继续走 60 多个基础分区的 route，平均延迟可能仍然被拉高。

### 9.3 保护表应限制最大规模

当前最大保护表 216300 vectors，已经很大。

建议保护表规模不要只受总预算约束，还要受单表规模约束。

可选设计：

```text
max_protection_group_vectors = min(
    budget_vectors * 0.25,
    p90_tenant_vector_count
)
```

这样避免一个保护表过大。

### 9.4 基础分区需要重新考虑大 pattern

当前最大问题之一是：

```text
大 exact ACL pattern 不可拆
```

如果继续坚持 pattern 不拆，那么大分区无法消除。

如果允许有限拆分，应只对大 pattern 做 controlled split：

- 只拆 vector_count 超过阈值的大 pattern。
- 拆分后每个子块仍保留同一个 pattern_id。
- 查询权限仍按 pattern_id 判断。
- route metadata 要知道同一个 pattern 可能出现在多个 physical shard。

这个改动会增加 route fanout，但能显著改善大表 HNSW 成本。

是否值得做，需要单独实验。

## 10. 建议下一步

短期优先改保护层：

1. 保护候选按 fanout pain 排序。
2. 最高 fanout 租户即使没有相似伙伴，也允许 singleton 保护。
3. 对保护 group 加单表最大向量数约束。
4. 预算选择不再只看 benefit density，而是看：

```text
saved_fanout_per_vector
```

或：

```text
query_weight * reduced_branch_count / extra_vectors
```

中期再改基础分区：

1. 分析大 exact ACL pattern 的数量和来源。
2. 尝试只对大 pattern 做 controlled semantic split。
3. 对比：

```text
不拆 pattern + 保护分区
只拆大 pattern
大 pattern split + 保护分区
```

当前这版的关键问题不是 recall，而是：

```text
用 625MB 保护层空间，只保护了 9 个租户，而且没有覆盖最高 fanout 租户。
```

因此下一轮优化应优先让保护分区真正服务最痛的 high-fanout tenants。
