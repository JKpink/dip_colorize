#!/usr/bin/env python3
"""
Indian Pines NDVI 演示 — 对齐 MATLAB ndvi 示例

MATLAB 原版:
  hcube = hypercube("indian_pines.dat");
  ndviImg = ndvi(hcube);
  rgbImg = colorize(hcube, Method="rgb");
  fig = figure(Position=[0 0 1200 600]);
  axes1 = axes(Parent=fig, Position=[0 0.1 0.4 0.8]);
  imshow(rgbImg, Parent=axes1)
  title("RGB Image of Data Cube")
  axes2 = axes(Parent=fig, Position=[0.45 0.1 0.4 0.8]);
  imagesc(ndviImg, Parent=axes2)    % imagesc = 自动拉伸 + jet colormap
  colorbar
  title("NDVI Image")
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indices import ndvi
from src.colorize import colorize


def main():
    import scipy.io
    from types import SimpleNamespace
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE, "data", "indian_pines")
    OUT_DIR = os.path.join(BASE, "output")
    MAT_PATH = os.path.join(DATA_DIR, "Indian_pines_corrected.mat")

    # ---- 加载数据 ----
    print("=" * 60)
    print("MATLAB: hcube = hypercube(\"indian_pines.dat\")")
    print("=" * 60)

    if not os.path.exists(MAT_PATH):
        print(f"⚠ 数据文件不存在: {MAT_PATH}")
        sys.exit(1)

    mat = scipy.io.loadmat(MAT_PATH)
    data = mat['indian_pines_corrected']  # (145, 145, 200)
    print(f"Indian Pines: {data.shape[0]}×{data.shape[1]}×{data.shape[2]} 波段")
    print(f"数据类型: {data.dtype}, 值域 [{data.min():.0f}, {data.max():.0f}]")

    # AVIRIS 波长: 400-2500nm, 200 波段 (去水汽吸收)
    wavelengths = np.linspace(400, 2500, data.shape[2], dtype=np.float64)
    print(f"波长: {wavelengths[0]:.0f}-{wavelengths[-1]:.0f}nm, "
          f"~{(wavelengths[-1]-wavelengths[0])/(data.shape[2]-1):.1f}nm/波段")

    hcube = SimpleNamespace(data=data, wavelengths=wavelengths, shape=data.shape)

    # ---- ndviImg = ndvi(hcube) ----
    print(f"\n{'='*60}")
    print("MATLAB: ndviImg = ndvi(hcube)")
    print("=" * 60)
    ndvi_img, nir_idx, red_idx = ndvi(hcube)

    # ---- rgbImg = colorize(hcube, Method="rgb") ----
    print(f"\n{'='*60}")
    print("MATLAB: rgbImg = colorize(hcube, Method=\"rgb\")")
    print("=" * 60)
    rgb_img, _ = colorize(hcube, Method="rgb")

    # ---- 双图布局 (对齐 MATLAB 示例) ----
    for f in fm.fontManager.ttflist:
        if 'pingfang' in f.name.lower():
            plt.rcParams['font.family'] = f.name
            break

    # MATLAB: figure(Position=[0 0 1200 600])
    fig = plt.figure(figsize=(12, 5.5))

    # MATLAB: axes1(Position=[0 0.1 0.4 0.8])
    ax1 = fig.add_axes([0.02, 0.10, 0.45, 0.82])
    ax1.imshow(rgb_img)
    # MATLAB: title("RGB Image of Data Cube")
    ax1.set_title("RGB Image of Data Cube", fontsize=11, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # MATLAB: axes2(Position=[0.45 0.1 0.4 0.8])
    ax2 = fig.add_axes([0.52, 0.10, 0.45, 0.82])
    # MATLAB: imagesc(ndviImg) — 自动拉伸 + jet colormap
    im = ax2.imshow(ndvi_img, cmap='jet')
    # MATLAB: colorbar — 对齐 MATLAB 样式 (薄、紧贴、刻度在外)
    cbar = plt.colorbar(im, ax=ax2, fraction=0.04, pad=0.02,
                        ticks=np.linspace(np.nanmin(ndvi_img), np.nanmax(ndvi_img), 5))
    cbar.ax.tick_params(labelsize=8, direction='out', length=4, width=0.8)
    cbar.outline.set_linewidth(0.6)
    cbar.set_label('NDVI', fontsize=9, labelpad=2)
    # MATLAB: title("NDVI Image")
    ax2.set_title("NDVI Image", fontsize=11, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.suptitle(f"Indian Pines — ndvi(hcube)  |  "
                 f"NIR=B{nir_idx+1}({wavelengths[nir_idx]:.0f}nm)  "
                 f"Red=B{red_idx+1}({wavelengths[red_idx]:.0f}nm)",
                 fontsize=12, fontweight='bold', y=0.98)

    out_path = os.path.join(OUT_DIR, "ndvi_indian_pines.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ 已保存: {out_path}")

    try:
        plt.show()
    except Exception:
        pass

    print(f"\nNDVI 范围: [{np.nanmin(ndvi_img):.4f}, {np.nanmax(ndvi_img):.4f}]")
    print(f"NIR={wavelengths[nir_idx]:.0f}nm  Red={wavelengths[red_idx]:.0f}nm")


if __name__ == "__main__":
    main()
