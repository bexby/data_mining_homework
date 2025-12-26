import numpy as np
import json
import matplotlib.pyplot as plt

def multivariate_normal_pdf(x, mean, cov, eps=1e-8):
    """计算单点或多点的多元正态密度。
    x: (N,2) 或 (2,) ; mean: (2,) ; cov: (2,2)
    返回 shape (N,) 或 标量
    """
    x_arr = np.atleast_2d(x)  # (N,2)
    d = mean.shape[0]
    cov_reg = cov + np.eye(d) * eps
    det = np.linalg.det(cov_reg)
    inv = np.linalg.inv(cov_reg)
    diff = x_arr - mean.reshape(1, -1)
    exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)  # (N,)
    coef = 1.0 / (np.sqrt((2 * np.pi) ** d * det))
    return coef * np.exp(exponent)

def plot_gmm(data, means, covs, weights=None, ax=None, cmap='tab10', show_legend=True):
    """
    data: list or array shape (N,2)
    means: list/array shape (K,2)
    covs: list/array shape (K,2,2)
    weights: optional list shape (K,) - 如果 None 则均匀权重
    ax: matplotlib.axes（可选）
    返回: responsibilities (N,K)
    """
    data = np.asarray(data, dtype=float)
    means = np.asarray(means, dtype=float)
    covs = np.asarray(covs, dtype=float)
    N = data.shape[0]
    K = means.shape[0]

    if weights is None:
        weights = np.ones(K) / K
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

    # 计算每个点属于每个分量的未归一化后验： w_k * N(x|mu_k, cov_k)
    pdfs = np.zeros((N, K))
    for k in range(K):
        pdfs[:, k] = multivariate_normal_pdf(data, means[k], covs[k])
    numer = pdfs * weights.reshape(1, -1)
    denom = numer.sum(axis=1, keepdims=True)
    # 防止除零
    denom[denom == 0] = 1e-12
    responsibilities = numer / denom  # shape (N,K)

    # 每个点的标签 = 后验最大分量
    labels = responsibilities.argmax(axis=1)

    # plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    cmap_obj = plt.get_cmap(cmap)
    colors = [cmap_obj(i % cmap_obj.N) for i in range(K)]

    # 散点：按标签着色；为美观设置alpha
    for k in range(K):
        mask = labels == k
        if mask.any():
            ax.scatter(data[mask, 0], data[mask, 1], s=20, alpha=0.7, color=colors[k], label=f'comp {k}')

    
    # 画每个高斯的 1-sigma 椭圆（黑色虚线）
    theta = np.linspace(0, 2 * np.pi, 200)
    unit_circle = np.vstack((np.cos(theta), np.sin(theta)))  # (2, T)

    for k in range(K):
        cov_k = covs[k]
        mu_k = means[k]

        # 特征分解
        eigvals, eigvecs = np.linalg.eigh(cov_k)
        eigvals[eigvals < 0] = 0.0
        axes = np.sqrt(eigvals)  # 1-sigma 半轴

        transform = eigvecs @ np.diag(axes)
        ellipse_pts = (transform @ unit_circle).T + mu_k.reshape(1, -1)

        # 黑色虚线边界
        ax.plot(
            ellipse_pts[:, 0],
            ellipse_pts[:, 1],
            linestyle='--',
            linewidth=1.8,
            color='black',
            alpha=0.9
        )

        # 均值点：黑色 X，更醒目
        ax.scatter(
            mu_k[0],
            mu_k[1],
            marker='x',
            s=90,
            linewidths=2.2,
            color='black',
            zorder=5
        )


    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if show_legend:
        ax.legend()
    ax.grid(True)
    return responsibilities

# -----------------------
# 示例用法（可删去）
if __name__ == "__main__":
    # 生成一些示例数据用于演示

    with open("GMM_log.jsonl", "r") as fr:
        gauss = [json.loads(line) for line in fr.readlines()]
    
    gauss = gauss[-1]
    rng = np.random.RandomState(0)
    means = np.array(gauss["mu"])
    covs = np.array(gauss["sigma"])
    with open("data.txt", "r") as fr:
        data = [list(map(float, line.split())) for line in fr.readlines()]

    resp = plot_gmm(data, means, covs)
    plt.title("GMM components (points colored by argmax posterior) and 1-sigma ellipses")
    plt.show()
