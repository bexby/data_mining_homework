import json
import matplotlib.pyplot as plt
import itertools
import numpy as np
from typing import List, Dict, Tuple, Set

def _normalize_step(step: Dict) -> Dict[str, List[Tuple[float,float]]]:
    """把一步的 cluster dict 归一化：key->str, 点->tuple"""
    norm = {}
    for k,v in step.items():
        key = str(k)
        pts = [tuple(p) for p in v]  # v 预期为 [[x,y], ...]
        norm[key] = pts
    return norm

def plot_hc_and_k_clusters(
    history: List[Dict[str, List[List[float]]]],
    k: int,
    max_steps: int = 20
):
    """
    history: list of cluster dicts, ordered from older -> newer
             each element: {cluster_name(str): [[x,y], ...], ...}
    k:        number of clusters to visualize (must exist in history)
    """

    if not history:
        raise ValueError("history 为空")

    # ========= 取最后 max_steps 步 =========
    steps = history[-max_steps:] if len(history) > max_steps else history[:]

    # 归一化：点 -> tuple
    norm_steps = []
    for step in steps:
        norm = {}
        for kname, pts in step.items():
            norm[str(kname)] = [tuple(p) for p in pts]
        norm_steps.append(norm)

    # ========= 找到“簇数 = k”的那一步 =========
    k_step = None
    for step in norm_steps:
        if len(step) == k:
            k_step = step
            break
    if k_step is None:
        raise ValueError(f"history 中不存在簇数为 k={k} 的步骤")

    # ===============================
    # 一、绘制层次聚类树
    # ===============================

    first = norm_steps[0]
    leaf_names = list(first.keys())
    n_leaf = len(leaf_names)

    cluster_to_node = {name: i for i, name in enumerate(leaf_names)}
    next_node = n_leaf

    node_info = {}
    for i, name in enumerate(leaf_names):
        node_info[i] = {
            "x": float(i),      # 等距排开
            "y": 0.0,
            "label": name,
            "size": len(first[name])
        }

    merges = []  # (a, b, parent, height)

    prev = norm_steps[0]

    for step_i in range(1, len(norm_steps)):
        curr = norm_steps[step_i]
        prev_sets = {k: set(v) for k, v in prev.items()}
        curr_sets = {k: set(v) for k, v in curr.items()}

        for cname, cset in curr_sets.items():
            contained = [
                pname for pname, pset in prev_sets.items()
                if pset and pset.issubset(cset)
            ]

            if len(contained) >= 2:
                nodes = [cluster_to_node[p] for p in contained]

                a = nodes[0]
                for b in nodes[1:]:
                    xa, xb = node_info[a]["x"], node_info[b]["x"]
                    height = step_i

                    parent = next_node
                    next_node += 1

                    node_info[parent] = {
                        "x": (xa + xb) / 2,
                        "y": float(height),
                        "label": cname,
                        "size": node_info[a]["size"] + node_info[b]["size"]
                    }

                    merges.append((a, b, parent, height))
                    a = parent

                cluster_to_node[cname] = a

        # 继承未变化簇
        for cname, cset in curr_sets.items():
            if cname in cluster_to_node:
                continue
            for pname, pset in prev_sets.items():
                if pset == cset and pname in cluster_to_node:
                    cluster_to_node[cname] = cluster_to_node[pname]
                    break

        prev = curr

    if not merges:
        raise RuntimeError("未检测到任何合并")

    # ===== 画图：dendrogram =====
    # fig, (ax_tree, ax_scatter) = plt.subplots(
    #     1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2, 1]}
    # )
    fig_tree, ax_tree = plt.subplots(figsize=(10, 6))
    fig_scatter, ax_scatter = plt.subplots(figsize=(6, 6))

    # 叶节点
    for i in range(n_leaf):
        x = node_info[i]["x"]
        ax_tree.scatter(x, 0, color="k")
        ax_tree.text(
            x, -0.1, node_info[i]["label"],
            ha="center", va="top", fontsize=9, rotation=45
        )

    # 合并线 + 合并标签
    for a, b, parent, h in merges:
        xa, ya = node_info[a]["x"], node_info[a]["y"]
        xb, yb = node_info[b]["x"], node_info[b]["y"]
        xp, yp = node_info[parent]["x"], node_info[parent]["y"]

        ax_tree.plot([xa, xa], [ya, yp], color="k")
        ax_tree.plot([xb, xb], [yb, yp], color="k")
        ax_tree.plot([xa, xb], [yp, yp], color="k")

        ax_tree.text(
            xp, yp + 0.05, node_info[parent]["label"],
            ha="center", va="bottom", fontsize=9, color="magenta"
        )

    ax_tree.set_ylabel("Merge step")
    ax_tree.set_ylim(-2, max(h for *_, h in merges) + 1)
    ax_tree.set_yticks(range(0, int(max(h for *_, h in merges)) + 1))
    ax_tree.set_xticks([])
    ax_tree.set_title("Hierarchical Clustering Tree")
    ax_tree.grid(axis="y", linestyle=":", alpha=0.3)

    # ===============================
    # 二、绘制 k 个簇的散点图
    # ===============================

    cluster_names = list(k_step.keys())
    num_clusters = len(cluster_names)

    cmap = plt.cm.get_cmap("tab20", num_clusters)  # 或 "hsv", "rainbow"

    for i, cname in enumerate(cluster_names):
        pts = k_step[cname]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax_scatter.scatter(
            xs, ys,
            label=cname,
            s=30,
            color=cmap(i)
        )

    ax_scatter.set_title(f"{k} clusters")
    ax_scatter.set_xlabel("x")
    ax_scatter.set_ylabel("y")
    ax_scatter.legend(fontsize=8)
    ax_scatter.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()



"================================================"


import numpy as np
import matplotlib.pyplot as plt

# sklearn 指标
from sklearn.metrics import silhouette_score, calinski_harabasz_score


def _assemble_X_labels(cluster_dict):
    """把 {cluster_id: [[x,y], ...]} 转成 X, labels"""
    X_list, y_list = [], []
    label_map = {}
    cur = 0
    for key, pts in cluster_dict.items():
        if key not in label_map:
            label_map[key] = cur
            cur += 1
        lbl = label_map[key]
        pts = np.asarray(pts)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        X_list.append(pts)
        y_list.append(np.full(len(pts), lbl, dtype=int))
    if not X_list:
        return np.empty((0, 2)), np.array([], dtype=int)
    return np.vstack(X_list), np.concatenate(y_list)


def plot_silhouette_and_ch(cluster_history, figsize=(14, 5), savepath=None, show=True):
    """
    cluster_history: list[dict], 每个 dict 形如 {"4446": [[x,y], ...], ...}
    """
    Ks = []
    ch_list = []
    sil_list = []

    for step in cluster_history:
        K = len(step)
        Ks.append(K)

        X, labels = _assemble_X_labels(step)

        # --- CH 指数 ---
        try:
            if len(np.unique(labels)) >= 2 and len(labels) > len(np.unique(labels)):
                ch = calinski_harabasz_score(X, labels)
            else:
                ch = np.nan
        except Exception:
            ch = np.nan
        ch_list.append(ch)

        # --- Silhouette ---
        try:
            if len(np.unique(labels)) >= 2 and len(np.unique(labels)) < len(labels):
                sil = silhouette_score(X, labels)
            else:
                sil = np.nan
        except Exception:
            sil = np.nan
        sil_list.append(sil)

    # 按 K 递增排序，便于观察
    order = np.argsort(Ks)
    Ks = np.array(Ks)[order]
    ch_list = np.array(ch_list)[order]
    sil_list = np.array(sil_list)[order]

    # 最优 K
    best_ch_k = None if np.all(np.isnan(ch_list)) else int(Ks[np.nanargmax(ch_list)])
    best_sil_k = None if np.all(np.isnan(sil_list)) else int(Ks[np.nanargmax(sil_list)])

    # ================== 画图 ==================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 左：CH 指数
    ax1.plot(Ks, ch_list, marker="o")
    ax1.set_xlabel("Number of clusters (K)")
    ax1.set_ylabel("Calinski–Harabasz Index")
    ax1.set_title("CH index vs K (higher is better)")

    if best_ch_k is not None:
        y = ch_list[np.where(Ks == best_ch_k)][0]
        ax1.scatter([best_ch_k], [y], s=100, marker="x")
        ax1.annotate(f"best K={best_ch_k}",
                     xy=(best_ch_k, y),
                     xytext=(0, -30),
                     textcoords="offset points",
                     ha="center")

    # 右：Silhouette
    ax2.plot(Ks, sil_list, marker="o")
    ax2.set_xlabel("Number of clusters (K)")
    ax2.set_ylabel("Average Silhouette Score")
    ax2.set_title("Silhouette vs K")

    if best_sil_k is not None:
        y = sil_list[np.where(Ks == best_sil_k)][0]
        ax2.scatter([best_sil_k], [y], s=100, marker="x")
        ax2.annotate(f"best K={best_sil_k}",
                     xy=(best_sil_k, y),
                     xytext=(0, -30),
                     textcoords="offset points",
                     ha="center")

    fig.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return {
        "Ks": Ks.tolist(),
        "CH": np.where(np.isnan(ch_list), None, ch_list).tolist(),
        "silhouette": np.where(np.isnan(sil_list), None, sil_list).tolist(),
        "best_ch_k": best_ch_k,
        "best_sil_k": best_sil_k,
    }



def plot_hc_from_jsonl(path: str, last_k: int = None):
    """从 jsonl 文件读取历史并绘图。若 last_k 指定，则取最后 last_k 行（保持 older->newer 顺序）"""
    steps = []
    with open(path, "r") as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            steps.append(json.loads(line))
    if not steps:
        raise ValueError("文件为空或未读取到任何行")
    # steps currently in file order = earliest appended -> latest appended
    if last_k is not None and last_k < len(steps):
        steps = steps[-last_k:]
    # 调用绘图主函数
    # for c in [10, 15, 20]:
    #     plot_hc_and_k_clusters(steps, c)
    # plot_hc_and_k_clusters(steps, 15)
    plot_silhouette_and_ch(steps)


# 示例调用
plot_hc_from_jsonl("./result/HC/HC_log_min.jsonl")
