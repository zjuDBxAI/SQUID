# Latent Access Engineering Skill

每次修改 `latent_access` 前，先对照这个文档，避免把系统再次改回低召回但看起来更“干净”的状态。

## 硬约束

- 只改 `latent_access` 自己的训练、规划、物化、搜索和文档链路。
- 不要伤害其它 baseline，尤其不要修改 `dynamic_partition`、`adaptive_tenant`、`predicate_*` 的默认行为。
- benchmark 结果必须区分“训练问题”和“执行链路问题”，不要把低召回全部归因于训练器。
- 任何改动都要优先看 block-level benchmark 的真实工作链条，而不是只看 document-level 直觉。

## 当前已经验证有效的改进

### 1. 分区语义要看真实 block，而不是只看文档代表向量

当前实现已经做了这件事：

- 在 `load_result_to_database.py` 里，为每个 partition 读取真实 block vectors
- 计算 `semantic_centroid`
- 计算多个 `semantic_anchor_vectors`
- 在 plan metadata 中写入 `semantic_metadata_mode = block_anchors`

结论：

- `latent_access` 的 benchmark 是 block-level top-k
- 如果 partition 语义只由 document representative vector 决定，route score 很容易偏离真实 top-k block 分布
- 所以后续任何 route scoring 改进，都应该优先基于 block anchors，而不是退回单 centroid

### 2. residual 分区不是噪声桶，而是召回兜底桶

已经做过离线验证：

- 把 cell 内 residual 强行后置，会让 ground-truth 文档覆盖率下降
- 当前结构下，residual 在许多 query 上承担跨 atom 兜底作用

结论：

- 不要简单把 residual 从 route queue 顶部移走
- 如果要弱化 residual，必须同时提供新的高召回 fallback 机制

### 3. route depth 必须自适应，而不是死卡一个固定值

当前实现已经做了：

- 默认 `base_route_limit = 16`
- 如果 query 在多个 semantic cells 上分数很接近，就自动扩到更大的 route limit
- 当前默认扩容阈值是 `latent_route_expansion_margin = 0.15`
- 当前默认最大扩容目标是 `latent_route_limit_max = 24`

结论：

- 低召回并不一定说明 trainer 错了，也可能只是路由截断过早
- 后续调参时，先看 route coverage，再看 benchmark recall

### 4. 多分区查询不能逐分区串行执行

当前实现已经改成：

- 将 routed partitions 组装成单次 `UNION ALL` 查询
- 在一个 SQL 中完成 routed search，再做全局 top-k merge

结论：

- 如果 route fanout 提高，逐分区串行 `EXPLAIN ANALYZE + SELECT` 会把时间线性放大
- 所以后续只要路由分区数增加，就必须优先保住单查询执行链路

### 5. 默认 benchmark 参数不能太保守

当前默认值：

- `route_limit = 16`
- `partition_fetch_multiplier = 6`

结论：

- 对 `latent_access` 来说，默认 `8 / 3` 太保守，会把 prototype 的真实上限压低
- 如果要重新缩小默认值，必须先证明 recall 不会明显掉

## 当前代码状态的核心判断

截至 2026-04-11：

- trainer 已经不是单纯聚类代理，而是可运行的 `A ~= ZB` 稀疏交替训练器
- planner 已经有 support-aware 的 cell-local atom budget
- materialization 已经会把 block-level 语义锚点写进控制面
- search 已经有 anchor-aware routing、自适应扩路由和 `UNION ALL` 执行链路

这意味着：

- 当前瓶颈不再只是“代码没实现”，而是“latent partition 质量还不够接近 dynamic_partition”
- 继续优化时，应该优先做 route coverage 分析、partition purity 分析、residual 占比分析，而不是盲目重写 trainer

## 修改优先级

如果下一轮继续优化，优先按这个顺序推进：

1. 先看 route coverage
2. 再看 residual 占比和 stable atom 支持度
3. 再看 partition 内 block 分布是否过杂
4. 最后才考虑重写 trainer objective

## 每次改动后至少检查这几件事

1. `cli.py summary` 里是否还能看到：
   - `semantic_metadata_mode = block_anchors`
   - `semantic_anchor_count`
2. `benchmark` 默认参数是否仍然是较强的默认值：
   - `route_limit = 16`
   - `partition_fetch_multiplier = 6`
3. search 是否仍然走单次 `UNION ALL` 查询，而不是回退成逐表串行查询
4. 没有改坏其它 baseline

## 当前阶段最值得继续做的事

- 分析 query 的 ground-truth 文档有多少落入 routed partitions
- 分析哪些 tenants / queries 主要失败在 residual 之外
- 分析哪些 semantic cells 内部仍然混入太多不相关 block
- 在不伤害其它 baseline 的前提下，继续提高 `latent_access` 的 recall/time 比
