#!/usr/bin/env python3
"""
hyperslic 演示 — PaviaU 超像素分割

MATLAB 原版:
  hcube = imhypercube("paviaU.dat");
  [L, numLabels] = hyperslic(hcube, 185);
  BW = boundarymask(L);
  rgbImg = colorize(hcube, Method="rgb", ContrastStretching=true);
  I = imoverlay(rgbImg, BW, "cyan");
  figure; imshow(I)
"""

import os, sys, numpy as np, scipy.io
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.slic import hyperslic
from src.colorize import colorize


def boundarymask(L):
    """找出超像素之间的边界像素。"""
    rows, cols = L.shape
    mask = np.zeros((rows, cols), dtype=bool)
    # 检查 4 邻域是否有不同标签
    mask[1:, :] |= (L[1:, :] != L[:-1, :])
    mask[:-1, :] |= (L[:-1, :] != L[1:, :])
    mask[:, 1:] |= (L[:, 1:] != L[:, :-1])
    mask[:, :-1] |= (L[:, :-1] != L[:, 1:])
    return mask


def imoverlay(rgb, mask, color='cyan'):
    """在 RGB 图上叠加边界掩膜。"""
    overlay = rgb.copy()
    colors = {'cyan': [0, 1, 1], 'red': [1, 0, 0],
              'yellow': [1, 1, 0], 'green': [0, 1, 0]}
    c = np.array(colors.get(color, [0, 1, 1]))
    for i in range(3):
        overlay[mask, i] = c[i]
    return overlay


def main():
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mat_path = os.path.join(BASE, "data", "paviaU", "PaviaU.mat")

    # ---- MATLAB: hcube = imhypercube("paviaU.dat") ----
    mat = scipy.io.loadmat(mat_path)
    data = mat['paviaU']
    wl = np.linspace(430, 860, 103, dtype=np.float64)
    hcube = SimpleNamespace(data=data, wavelengths=wl, shape=data.shape)
    print(f"PaviaU: {data.shape[0]}×{data.shape[1]}×{data.shape[2]}\n")

    # ---- MATLAB: [L, numLabels] = hyperslic(hcube, 185) ----
    print("MATLAB: [L, numLabels] = hyperslic(hcube, 185)")
    L, numLabels = hyperslic(hcube, K=185)

    # ---- MATLAB: BW = boundarymask(L) ----
    BW = boundarymask(L)

    # ---- MATLAB: rgbImg = colorize(hcube, Method="rgb", ContrastStretching=true) ----
    rgbImg, idx = colorize(hcube, Method="rgb", ContrastStretching=True)

    # ---- MATLAB: I = imoverlay(rgbImg, BW, "cyan") ----
    I = imoverlay(rgbImg, BW, 'cyan')

    # ---- 显示 ----
    for f in fm.fontManager.ttflist:
        if 'pingfang' in f.name.lower():
            plt.rcParams['font.family'] = f.name
            break

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 左: RGB 参考
    axes[0].imshow(rgbImg)
    axes[0].set_title("RGB Image of Data Cube\ncolorize(Method='rgb')",
                      fontsize=11, fontweight='bold')
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # 中: 超像素边界叠加
    axes[1].imshow(I)
    axes[1].set_title(f"Superpixel Boundaries (cyan)\n"
                      f"hyperslic(hcube, K=185) → {numLabels} superpixels",
                      fontsize=11, fontweight='bold')
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # 右: 标签图 (随机着色)
    rng = np.random.RandomState(42)
    colors = rng.randint(50, 256, (numLabels, 3)) / 255.0
    label_rgb = colors[L % numLabels]
    axes[2].imshow(label_rgb)
    axes[2].set_title(f"Superpixel Labels\n{numLabels} regions (random colors)",
                      fontsize=11, fontweight='bold')
    axes[2].set_xticks([]); axes[2].set_yticks([])

    plt.suptitle("PaviaU — hyperslic Superpixel Oversegmentation",
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()

    out = os.path.join(BASE, "output", "hyperslic_paviaU.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ {out}")

    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
