import matplotlib.pyplot as plt
# 实验数据（已剔除 top_d=2）
top_d = [4, 8, 16, 32, 64]
query_ms = [0.3663, 0.4945, 0.3149, 0.4163, 0.3468]

# 绘制柱状图（橙色系，与规划时间图配色区分）
plt.figure(figsize=(6, 4))
bars = plt.bar(
    [str(x) for x in top_d],
    query_ms,
    color="#ff7f0e",
    edgecolor="black",
    width=0.6
)

plt.xlabel("top-d")
plt.ylabel("Query Time (ms)")
plt.ylim(0, max(query_ms) * 1.18)

# 柱顶标注数值
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.008,
        f"{height:.4f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()

# 同时保存 PNG + PDF
plt.savefig("query_time.png")
plt.savefig("query_time.pdf")
plt.close()
