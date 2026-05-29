import matplotlib.pyplot as plt
import numpy as np

# =========================
# 原始数据（query time 单位：秒）
# 会在后面自动转成 ms
# =========================
raw_data = {
    "SQUID(ours)": [
        {"avg_recall": 0.8545000000000003, "avg_query_time": 0.0064873921871185306},
        {"avg_recall": 0.9635000000000008, "avg_query_time": 0.006687794923782349},
        {"avg_recall": 0.9635000000000008, "avg_query_time": 0.006687794923782349},
        {"avg_recall": 0.9840000000000004, "avg_query_time": 0.006704506874084473},
        {"avg_recall": 0.9895000000000005, "avg_query_time": 0.00715760588645935},
        {"avg_recall": 0.9900000000000004, "avg_query_time": 0.007086875438690185},
        {"avg_recall": 0.9930000000000001, "avg_query_time": 0.007843213081359863},
    ],
    "HONEYBEE": [
        {"avg_recall": 0.8232999999999965, "avg_query_time": 0.0051584444046020505},
        {"avg_recall": 0.9022999999999952, "avg_query_time": 0.0058180949687957765},
        {"avg_recall": 0.953399999999997, "avg_query_time": 0.006855705690383911},
        {"avg_recall": 0.970999999999998, "avg_query_time": 0.00718748927116394},
        {"avg_recall": 0.9811999999999981, "avg_query_time": 0.007559278249740601},
        {"avg_recall": 0.9918999999999993, "avg_query_time": 0.00834829568862915},
    ],
    "RLS": [
        {"avg_recall": 0.6884999999999999, "avg_query_time": 0.03298648238182068},
        {"avg_recall": 0.7895, "avg_query_time": 0.03279973983764648},
        {"avg_recall": 0.8610000000000002, "avg_query_time": 0.032163386344909665},
        {"avg_recall": 0.9230000000000002, "avg_query_time": 0.03492489457130432},
        {"avg_recall": 0.9425000000000004, "avg_query_time": 0.036152442693710325},
        {"avg_recall": 0.9760000000000001, "avg_query_time": 0.0388651180267334},
        {"avg_recall": 0.983, "avg_query_time": 0.043089878559112546},
    ],
    "USER": [
        {"avg_recall": 0.8954999999999955, "avg_query_time": 0.004164904832839966},
        {"avg_recall": 0.9344999999999959, "avg_query_time": 0.004489888429641723},
        {"avg_recall": 0.9585999999999963, "avg_query_time": 0.004895850658416748},
        {"avg_recall": 0.9771999999999977, "avg_query_time": 0.005066277265548706},
        {"avg_recall": 0.9879999999999985, "avg_query_time": 0.00521192479133606},
        {"avg_recall": 0.9904999999999988, "avg_query_time": 0.00518737530708313},
    ]
}

# =========================
# 风格设置：尽量贴近你的样板图
# =========================
style_map = {
    "RLS": {
        "color": "#0b3a53",   # 深蓝
        "marker": "^",
        "size": 70,
        "label": "RLS"
    },
    "HONEYBEE": {
        "color": "#e0a03a",   # 橙色
        "marker": "s",
        "size": 62,
        "label": "HONEYBEE"
    },
    "SQUID(ours)": {
        "color": "#b0302f",   # 红色
        "marker": "X",
        "size": 70,
        "label": "SQUID(ours)"
    },
    "USER": {
        "color": "#0a8f08",   # 绿色
        "marker": "o",
        "size": 68,
        "label": "USER"
    },
}

plot_order = ["RLS", "HONEYBEE", "SQUID(ours)", "USER"]

# =========================
# 数据预处理
# 1. 秒 -> ms
# 2. 按 recall 排序
# 3. 去掉完全重复点（SQUID 里有一个重复点）
# =========================
plot_data = {}

for method, points in raw_data.items():
    new_points = []
    seen = set()
    for p in points:
        recall = float(p["avg_recall"])
        time_ms = float(p["avg_query_time"]) * 1000.0
        key = (round(recall, 12), round(time_ms, 12))
        if key not in seen:
            seen.add(key)
            new_points.append((recall, time_ms))

    new_points = sorted(new_points, key=lambda x: x[0])
    plot_data[method] = new_points

# =========================
# 画图
# =========================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 13,
    "axes.labelsize": 17,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 14,
})

fig, ax = plt.subplots(figsize=(8.8, 3.6), dpi=220)

# 浅灰背景，接近样板图
bg_color = "#e9e9e9"
fig.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

for method in plot_order:
    pts = plot_data[method]
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])

    st = style_map[method]

    # 虚线连接
    ax.plot(
        x, y,
        linestyle=":",
        linewidth=1.0,
        color=st["color"],
        zorder=2
    )

    # 散点
    ax.scatter(
        x, y,
        s=st["size"],
        marker=st["marker"],
        color=st["color"],
        edgecolors="none",
        label=st["label"],
        zorder=3
    )

# =========================
# 坐标轴与范围
# =========================
all_x = np.concatenate([np.array([p[0] for p in plot_data[m]]) for m in plot_order])
all_y = np.concatenate([np.array([p[1] for p in plot_data[m]]) for m in plot_order])

ax.set_xlim(0.68, 1.00)
ax.set_ylim(0, all_y.max() * 1.10)

ax.set_xlabel("Recall@10")
ax.set_ylabel("Query Time (ms)")

# 网格
ax.grid(True, which="major", color="#cfcfcf", linestyle="-", linewidth=0.55, alpha=0.8)

# 边框
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color("#777777")

ax.tick_params(axis="both", length=3.2, width=0.8, color="#666666")

# 顶部 legend
leg = ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.28),
    ncol=4,
    frameon=True,
    fancybox=True,
    framealpha=1.0,
    columnspacing=1.5,
    handletextpad=0.6
)
leg.get_frame().set_facecolor("#f8f8f8")
leg.get_frame().set_edgecolor("#333333")
leg.get_frame().set_linewidth(0.8)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("single_panel_sample_style.png", dpi=300, bbox_inches="tight")
plt.savefig("single_panel_sample_style.pdf", bbox_inches="tight")
plt.show()