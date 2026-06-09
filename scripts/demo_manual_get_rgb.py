#!/usr/bin/env python3
"""
手写实现 SPy 的 get_rgb / 百分位拉伸 / 波段提取流水线
对照 graphics.py 的源码，逐函数还原，最后用 matplotlib imshow 展示

核心调用链回顾：
  imshow(img, bands=(3,2,1))
    └─ get_rgb(source, bands, stretch=0.02)
         └─ get_rgb_meta(source, bands, stretch=0.02)
              ├─ Step 1: 波段提取  (np.take)
              ├─ Step 2: 拉伸参数计算 (histogram CDF / np.percentile)
              └─ Step 3: 线性拉伸 + clip 到 [0,1]
"""

import os
import sys
import numpy as np


# ============================================================
# 手写实现 — 对应 SPy graphics.py 的核心函数
# ============================================================

def get_histogram_cdf_points(band, stretch):
    """
    对应 graphics.py 中的 get_histogram_cdf_points。
    给定一个波段数组和尾部比例 stretch=(low_tail, high_tail)，
    返回 (lower_bound, upper_bound) 两个 DN 值。

    算法：用 np.percentile 近似累积直方图的分位点。

    Args:
        band: 1D 或 2D 数组，单波段数据
        stretch: (low, high) 尾部比例，如 (0.02, 0.98)

    Returns:
        (lower, upper): 低/高分位点对应的 DN 值
    """
    low, high = stretch
    lower = np.percentile(band, low * 100)
    upper = np.percentile(band, high * 100)
    return lower, upper


def my_get_rgb(source, bands, stretch=0.02, stretch_all=False):
    """
    手写实现 SPy get_rgb / get_rgb_meta 的核心逻辑。
    对照 graphics.py:531-705 行。

    流程:
      1. 从多波段立方体中按 bands 索引提取 3 个波段 → R/G/B
      2. 对每个通道分别计算百分位拉伸的上下界
      3. 线性缩放 + clip 到 [0, 1]

    Args:
        source: numpy 数组 (R, C, B) 或 SPy SpyFile 对象
        bands: 3 元组，指定 R/G/B 对应的波段索引 (0-based)
        stretch: 尾部比例，默认 0.02 (即 2%-98%)
        stretch_all: 如果 True，各通道独立拉伸；否则统一上下界

    Returns:
        rgb: (R, C, 3) 的 float 数组，值域 [0, 1]
        meta: 字典，包含 bands, rgb_lims, mode
    """
    # ---- Step 1: 波段提取 (对应 graphics.py:572) ----
    # 如果 source 是 SpyFile，先读入内存
    if hasattr(source, 'read_bands'):
        rgb = source.read_bands(bands).astype(np.float64)
    elif hasattr(source, 'load'):
        rgb = source.load().read_bands(bands).astype(np.float64)
    else:
        # numpy 数组: 沿第 2 轴取索引
        rgb = np.take(source, bands, axis=2).astype(np.float64)

    meta = {'bands': list(bands), 'mode': 'rgb'}

    print(f"  [Step 1] 波段提取: indices={bands} → shape={rgb.shape}")

    # ---- Step 2: 计算拉伸范围 (对应 graphics.py:644-678) ----
    if isinstance(stretch, (int, float)):
        stretch = (stretch, 1.0 - stretch)

    # 对 3 个通道分别计算 p_low / p_high
    lims = np.array([
        get_histogram_cdf_points(rgb[:, :, i], stretch)
        for i in range(3)
    ])
    # lims 现在是 (3, 2): [[lower_R, upper_R], [lower_G, upper_G], [lower_B, upper_B]]

    if stretch_all:
        # 各通道独立拉伸 (对应 graphics.py:667-670)
        rgb_lims = lims
        method = "各通道独立"
    else:
        # 统一拉伸: 取最小下限、最大上限 (对应 graphics.py:672-678)
        # 这样可以避免色偏
        min_lower = lims[:, 0].min()
        max_upper = lims[:, 1].max()
        rgb_lims = np.tile([min_lower, max_upper], (3, 1))
        method = "统一(min_lower, max_upper)"

    meta['rgb_lims'] = rgb_lims.tolist()

    print(f"  [Step 2] 拉伸参数 (stretch={stretch}):")
    print(f"           各通道 p2/p98: R={lims[0]}, G={lims[1]}, B={lims[2]}")
    print(f"           策略: {method} → {rgb_lims[0]}")

    # ---- Step 3: 线性拉伸 + clip (对应 graphics.py:698-704) ----
    for i in range(3):
        lower, upper = rgb_lims[i]
        span = upper - lower
        if span == 0:
            rgb[:, :, i] = 0
        else:
            rgb[:, :, i] = np.clip((rgb[:, :, i] - lower) / span, 0.0, 1.0)

    print(f"  [Step 3] 线性拉伸完成, 值域: [{rgb.min():.4f}, {rgb.max():.4f}]")
    return rgb, meta


# ============================================================
# 可视化 — 用 matplotlib imshow 展示
# ============================================================
def show_all_composites(img, composites, title_prefix="", out_png=None):
    """
    用 matplotlib imshow 展示多种伪彩色合成方案。
    每一张图都通过我们手写的 my_get_rgb 生成。

    Args:
        img: SPy SpyFile 对象 (如 envi.open 返回的)
        composites: {名称: (R_idx, G_idx, B_idx)} 字典
        title_prefix: 标题前缀
        out_png: 若提供，则保存到这个路径
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # 中文字体
    for f in fm.fontManager.ttflist:
        if 'pingfang' in f.name.lower() or 'heiti' in f.name.lower():
            plt.rcParams['font.family'] = f.name
            break

    n = len(composites)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    if n == 1:
        axes = [axes]

    for ax, (name, (r, g, b)) in zip(axes, composites.items()):
        # ★ 调用手写的 my_get_rgb ★
        rgb, meta = my_get_rgb(img, bands=[r, g, b])
        ax.imshow(rgb)
        ax.set_title(f"{title_prefix}{name}", fontsize=12, fontweight='bold')
        ax.set_xlabel(f"R:B{r+1} G:B{g+1} B:B{b+1}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n✓ 保存到: {out_png}")
    plt.show()


# ============================================================
# 主流程
# ============================================================
def main():
    import spectral.io.envi as envi

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    HDR_PATH = os.path.join(BASE_DIR, "output", "landsat8_cube.hdr")

    print("=" * 60)
    print("手写 SPy get_rgb 流水线 — 逐步讲解")
    print("=" * 60)

    # 用 SPy 读取 ENVI 文件（读取的部分还是用 SPy，不重复造轮子）
    img = envi.open(HDR_PATH)
    print(f"\n原始数据: {img.shape}  (rows × cols × bands)")

    # -------------------------------------------------------
    # 演示 1: 自然彩色 — 逐步打印每一步的中间结果
    # -------------------------------------------------------
    print("\n" + "-" * 40)
    print("演示: 自然彩色 (B4→R, B3→G, B2→B)")
    print("-" * 40)
    rgb_natural, meta = my_get_rgb(img, bands=[3, 2, 1])

    print(f"\n最终 RGB 数组: shape={rgb_natural.shape}, "
          f"dtype={rgb_natural.dtype}, "
          f"min={rgb_natural.min():.4f}, max={rgb_natural.max():.4f}")
    print(f"元数据: {meta}")

    # -------------------------------------------------------
    # 演示 2: 对比 stretch_all=True vs False
    # -------------------------------------------------------
    print("\n" + "-" * 40)
    print("演示: 对比 stretch_all (统一拉伸 vs 独立拉伸)")
    print("-" * 40)
    rgb_unified, _ = my_get_rgb(img, bands=[3, 2, 1], stretch_all=False)
    rgb_perband, _ = my_get_rgb(img, bands=[3, 2, 1], stretch_all=True)

    # -------------------------------------------------------
    # 演示 3: 不同 stretch 参数的效果
    # -------------------------------------------------------
    print("\n" + "-" * 40)
    print("演示: 不同 stretch 参数 (0.02 vs 0.05 vs 0.10)")
    print("-" * 40)
    rgb_s02, _ = my_get_rgb(img, bands=[4, 3, 2], stretch=0.02)   # 2%-98%
    rgb_s05, _ = my_get_rgb(img, bands=[4, 3, 2], stretch=0.05)   # 5%-95%
    rgb_s10, _ = my_get_rgb(img, bands=[4, 3, 2], stretch=0.10)   # 10%-90%

    # -------------------------------------------------------
    # 多图对比展示
    # -------------------------------------------------------
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    for f in fm.fontManager.ttflist:
        if 'pingfang' in f.name.lower() or 'heiti' in f.name.lower():
            plt.rcParams['font.family'] = f.name
            break

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

    # 第一行: 不同伪彩色合成
    composites_row1 = [
        ("自然彩色 B4-B3-B2", [3, 2, 1]),
        ("标准假彩色 B5-B4-B3", [4, 3, 2]),
        ("SWIR假彩色 B7-B6-B4", [6, 5, 3]),
    ]
    for ax, (name, b) in zip(axes[0], composites_row1):
        rgb, _ = my_get_rgb(img, bands=b)
        ax.imshow(rgb)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel(f"R:B{b[0]+1} G:B{b[1]+1} B:B{b[2]+1}")
        ax.set_xticks([]); ax.set_yticks([])

    # 第二行: stretch_all 对比 (NIR假彩色)
    titles_row2 = [
        ("统一拉伸 (stretch_all=False)", False),
        ("独立拉伸 (stretch_all=True)", True),
        ("独立拉伸, 不同stretch=(0.05,0.95)", True),
    ]
    for ax, (title, sa) in zip(axes[1], titles_row2):
        if "不同stretch" in title:
            rgb, _ = my_get_rgb(img, bands=[4, 3, 2], stretch=0.05, stretch_all=sa)
        else:
            rgb, _ = my_get_rgb(img, bands=[4, 3, 2], stretch_all=sa)
        ax.imshow(rgb)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    # 第三行: 不同 stretch 参数 (NIR假彩色)
    stretch_params = [0.02, 0.05, 0.10]
    for ax, st in zip(axes[2], stretch_params):
        rgb, _ = my_get_rgb(img, bands=[4, 3, 2], stretch=st)
        ax.imshow(rgb)
        ax.set_title(f"stretch={st} ({st*100:.0f}%-{100-st*100:.0f}%)", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("手写 SPy get_rgb 流水线 — 参数对比",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()

    out_path = os.path.join(BASE_DIR, "output", "manual_get_rgb_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ 9 宫格对比图已保存: {out_path}")

    # 尝试弹出交互窗口
    try:
        plt.show()
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("完成！手写流水线成功复现了 SPy 的核心 get_rgb 逻辑。")
    print("=" * 60)


if __name__ == "__main__":
    main()
