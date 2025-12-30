# plot_bic_aic.py
# 从 bic_aic_results.json 读取数据并绘制 BIC / AIC 曲线，保存图片 bic_aic_curve.png
#
# 使用示例： python plot_bic_aic.py

import json
from pathlib import Path
import matplotlib.pyplot as plt

RESULTS_PATH = "bic_aic_results.json"
OUT_PNG = "bic_aic_curve.png"

def plot_from_file(results_path=RESULTS_PATH, out_png=OUT_PNG):
    p = Path(results_path)
    if not p.exists():
        raise FileNotFoundError(f"{results_path} not found. Run gmm_bic_aic_scan.py first.")

    with open(results_path, "r") as fr:
        results = json.load(fr)

    Ks = [r["K"] for r in results]
    BICs = [r["BIC"] for r in results]
    AICs = [r["AIC"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(Ks, BICs, marker='o', label='BIC')
    plt.plot(Ks, AICs, marker='s', label='AIC')

    plt.xlabel('Number of Components (K)')
    plt.ylabel('Score')
    plt.xticks(Ks)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_png, dpi=150)
    print(f"Saved figure to {out_png}")
    plt.show()

    # 额外打印最小值对应的 K
    bic_min = min(results, key=lambda r: r["BIC"])
    aic_min = min(results, key=lambda r: r["AIC"])
    print(f"Best K by BIC = {bic_min['K']} (BIC={bic_min['BIC']:.3f})")
    print(f"Best K by AIC = {aic_min['K']} (AIC={aic_min['AIC']:.3f})")

if __name__ == "__main__":
    plot_from_file()
