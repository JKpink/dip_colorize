#!/usr/bin/env python3
"""
Landsat 8 多光谱数据可视化
使用 SPy (Spectral Python) 进行伪彩色合成和光谱展示

数据: LC08_L1TP_113082_20211206 — 7721×7651, 30m, B1-B7
"""

import os
import sys
import numpy as np

# ============================================================
# 配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "landsat8")
ENVI_BASENAME = os.path.join(BASE_DIR, "output", "landsat8_cube")

# 要读取的波段及对应文件名后缀
BANDS = {
    "B1": "Coastal/Aerosol (443 nm)",
    "B2": "Blue (482 nm)",
    "B3": "Green (562 nm)",
    "B4": "Red (655 nm)",
    "B5": "Near Infrared (865 nm)",
    "B6": "SWIR 1 (1609 nm)",
    "B7": "SWIR 2 (2201 nm)",
}

# 常见伪彩色合成方案 (R, G, B 波段索引)
COMPOSITES = {
    "自然彩色 (True Color)":       (3, 2, 1),   # B4,B3,B2
    "标准假彩色 (NIR-R-G)":        (4, 3, 2),   # B5,B4,B3  — 植被→红色
    "SWIR 假彩色 (SWIR2-SWIR1-R)": (6, 5, 3),   # B7,B6,B4  — 地质/矿物
    "农业监测 (NIR-SWIR1-Blue)":   (4, 5, 1),   # B5,B6,B2
}

# 降采样因子（全分辨率约 59M 像素/波段, 降 4 倍后约 3.7M 像素）
DOWNSAMPLE = 4


# ============================================================
# Step 1: 读取波段并堆叠
# ============================================================
def read_bands(data_dir, bands_dict, downsample=1):
    """用 rasterio 读取各波段 GeoTIFF，堆叠为 3D numpy 数组。

    Args:
        data_dir: 数据目录路径
        bands_dict: {波段名: 描述} 字典
        downsample: 降采样因子 (1=全分辨率, 4=1/4 宽高)

    Returns:
        cube: (rows, cols, num_bands) float32 数组
        wavelengths: 各波段中心波长 (nm)
    """
    import rasterio

    band_keys = list(bands_dict.keys())
    cube = None
    profile = None

    for i, band_name in enumerate(band_keys):
        # 定位波段文件
        tif_files = [
            f for f in os.listdir(data_dir)
            if f.endswith(f"_{band_name}.TIF")
        ]
        if not tif_files:
            print(f"⚠ 未找到 {band_name} 的文件，跳过")
            continue

        tif_path = os.path.join(data_dir, tif_files[0])

        with rasterio.open(tif_path) as src:
            if profile is None:
                profile = src.profile
                out_shape = (
                    src.height // downsample,
                    src.width // downsample,
                )
                print(f"原始尺寸: {src.width}×{src.height}")
                print(f"降采样因子: {downsample}")
                print(f"输出尺寸: {out_shape[1]}×{out_shape[0]}")

            # 读取并降采样
            band_data = src.read(
                1,
                out_shape=out_shape,
                resampling=rasterio.enums.Resampling.average,
            ).astype(np.float32)

            print(f"✓ 读取 {band_name}: {tif_files[0]} "
                  f"({band_data.shape[0]}×{band_data.shape[1]})")

            if cube is None:
                cube = np.zeros(
                    (band_data.shape[0], band_data.shape[1], len(band_keys)),
                    dtype=np.float32,
                )
            cube[:, :, i] = band_data

    # 波长 (Landsat 8 各波段中心波长, nm)
    wavelengths = np.array([443, 482, 562, 655, 865, 1609, 2201], dtype=np.float32)

    return cube, wavelengths, profile


# ============================================================
# Step 2: 保存为 ENVI 格式
# ============================================================
def save_envi(cube, basename):
    """将 numpy 数组保存为 ENVI 格式 (.hdr + .dat)"""
    import spectral.io.envi as envi

    # 对每个波段做 2%-98% 百分位拉伸到 uint8
    print("\n应用百分位拉伸 (2%-98%) ...")
    cube_scaled = np.zeros(cube.shape, dtype=np.uint8)
    for i in range(cube.shape[2]):
        band = cube[:, :, i]
        p2 = np.percentile(band, 2)
        p98 = np.percentile(band, 98)
        # 线性拉伸
        stretched = (band - p2) / (p98 - p2) * 255.0
        stretched = np.clip(stretched, 0, 255)
        cube_scaled[:, :, i] = stretched.astype(np.uint8)
        print(f"  波段 {i+1}: p2={p2:.0f}, p98={p98:.0f}")

    envi.save_image(f"{basename}.hdr", cube_scaled, dtype=np.uint8, force=True)
    # SPy 默认使用 .img 扩展名
    print(f"✓ ENVI 文件已保存: {basename}.hdr + .img")
    return cube_scaled


# ============================================================
# Step 3a: SPy 交互式可视化 (wxPython)
# ============================================================
def view_with_spy(hdr_path):
    """使用 SPy 的 ImageView 进行交互式可视化"""
    import spectral
    from spectral import imshow, view_cube

    img = spectral.open_image(hdr_path)
    print(f"\nSPy 图像加载成功: {img.shape}")

    try:
        # 尝试交互式视图 — 点击像素可看光谱
        view = imshow(img, (3, 2, 1))  # 默认 True Color
        print("SPy ImageView 已启动。")
        print("  用法: 点击像素查看光谱曲线, 拖动浏览")
        print("  关闭窗口后继续...")
        view.show()
        return True
    except Exception as e:
        print(f"SPy 交互视图失败: {e}")
        return False


# ============================================================
# Step 3b: Matplotlib 静态可视化 (备选)
# ============================================================
def view_with_matplotlib(cube_scaled, wavelengths, cube_raw, out_png):
    """使用 matplotlib 生成多子图静态可视化"""
    import matplotlib
    # 关闭之前可能已打开的 figure，避免 backend 切换警告
    import matplotlib.pyplot as plt
    plt.close("all")
    matplotlib.use("Agg")  # 非交互后端
    import matplotlib.font_manager as fm
    from matplotlib.gridspec import GridSpec

    # ---- 配置中文字体 ----
    cjk_fonts = [f for f in fm.fontManager.ttflist
                 if any(k in f.name.lower() for k in
                        ['heiti', 'pingfang', 'stheit', 'hiragino sans gb',
                         'lantinghei', 'songti', 'stfangsong'])]
    if cjk_fonts:
        cjk_prop = fm.FontProperties(fname=cjk_fonts[0].fname)
        plt.rcParams['font.family'] = cjk_fonts[0].name
        print(f"  使用中文字体: {cjk_fonts[0].name}")
    else:
        cjk_prop = None
        print("  ⚠ 未找到中文字体，中文标题可能显示异常")

    rows, cols, num_bands = cube_scaled.shape
    print(f"\n生成 Matplotlib 静态图 ({num_bands} 波段, {rows}×{cols})...")

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ---- 伪彩色合成 (左侧 2×3) ----
    composite_list = list(COMPOSITES.items())
    for idx, (name, (r_idx, g_idx, b_idx)) in enumerate(composite_list):
        ax = fig.add_subplot(gs[idx // 3 * 2, idx % 3])  # 2 rows, 3 cols
        rgb = np.stack([
            cube_scaled[:, :, r_idx],
            cube_scaled[:, :, g_idx],
            cube_scaled[:, :, b_idx],
        ], axis=-1)
        ax.imshow(rgb)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        # 标注波段组合
        band_keys = list(BANDS.keys())
        ax.set_xlabel(f"R:{band_keys[r_idx]} G:{band_keys[g_idx]} B:{band_keys[b_idx]}",
                      fontsize=8)

    # ---- 光谱曲线: 随机采样 50 个像素 ----
    ax_spec = fig.add_subplot(gs[:, 2])  # 右列 3 行合并
    rng = np.random.RandomState(42)
    n_samples = 50
    y_idx = rng.randint(0, rows, n_samples)
    x_idx = rng.randint(0, cols, n_samples)

    for i in range(n_samples):
        spectrum = cube_raw[y_idx[i], x_idx[i], :]
        ax_spec.plot(wavelengths, spectrum, alpha=0.3, linewidth=0.8, color="gray")

    # 均值光谱
    mean_spectrum = cube_raw.reshape(-1, num_bands).mean(axis=0)
    ax_spec.plot(wavelengths, mean_spectrum, linewidth=2.5, color="red",
                 label="Mean Spectrum")

    ax_spec.set_xlabel("Wavelength (nm)", fontsize=11)
    ax_spec.set_ylabel("DN Value", fontsize=11)
    ax_spec.set_title(f"Spectral Profiles ({n_samples} random pixels)", fontsize=11)
    ax_spec.legend(fontsize=9)
    ax_spec.grid(True, alpha=0.3)

    # ---- 各波段直方图预览 ----
    # 在底部额外说明
    fig.text(0.5, 0.02,
             f"Landsat 8 — Path 113 Row 82 — 2021-12-06 | "
             f"Dimension: {cols}×{rows}×{num_bands} | "
             f"Wavelength range: {wavelengths[0]:.0f}–{wavelengths[-1]:.0f} nm",
             ha="center", fontsize=9, style="italic")

    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✓ 静态图已保存: {out_png}")
    plt.close()


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("Landsat 8 多光谱数据可视化 (SPy)")
    print("=" * 60)

    # ---- 读取数据 ----
    print("\n[1/4] 读取波段数据...")
    cube_raw, wavelengths, profile = read_bands(DATA_DIR, BANDS, DOWNSAMPLE)
    print(f"数据立方体: {cube_raw.shape}, dtype={cube_raw.dtype}, "
          f"内存 {cube_raw.nbytes / 1024**2:.0f} MB")

    # ---- 保存 ENVI ----
    print("\n[2/4] 保存为 ENVI 格式...")
    cube_scaled = save_envi(cube_raw, ENVI_BASENAME)

    # ---- 尝试 SPy 交互式 ----
    print("\n[3/4] 启动 SPy 可视化...")
    hdr_path = f"{ENVI_BASENAME}.hdr"
    spy_ok = view_with_spy(hdr_path)

    # ---- 备选: Matplotlib ----
    out_png = os.path.join(BASE_DIR, "output", "landsat8_preview.png")
    print(f"\n[4/4] 生成静态预览图 {out_png} ...")
    view_with_matplotlib(cube_scaled, wavelengths, cube_raw, out_png)

    print("\n" + "=" * 60)
    print("完成！")
    print(f"  ENVI 文件: {ENVI_BASENAME}.hdr / .img")
    print(f"  预览图:    {out_png}")
    if spy_ok:
        print("  交互查看: 请关闭 ImageView 窗口退出")
    print("=" * 60)


if __name__ == "__main__":
    main()
