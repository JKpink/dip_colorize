#!/usr/bin/env python3
"""
MATLAB colorize 兼容的 Python 实现

对照 MATLAB R2026a hyper.io.hypercube.colorize API:
  colorize(spcube, band, Method="falsecolored", ContrastStretching=false)

Method 参数:
  "falsecolored" — 自动选信息量最大的 3 个波段
  "rgb"         — 波长匹配 R(650nm)/G(550nm)/B(480nm)
  "cir"         — 彩红外: NIR→R, Red→G, Green→B

依赖: numpy + matplotlib (核心算法无外部依赖，便于 C++ 移植)
"""

import os
import sys
import numpy as np

# ============================================================
# Landsat 8 波段波长定义 (nm) — 用于演示，实际产品中来自元数据
# ============================================================
LANDSAT8_WAVELENGTHS = np.array([443, 482, 562, 655, 865, 1609, 2201], dtype=np.float64)
#                                B1   B2   B3   B4   B5   B6    B7

# RGB 目标波长 (nm)
RGB_TARGETS = {'R': 650.0, 'G': 550.0, 'B': 480.0}
# CIR 目标波长: NIR→R, Red→G, Green→B
CIR_TARGETS = {'R': 850.0, 'G': 650.0, 'B': 550.0}

# ============================================================
# 工具函数
# ============================================================

def find_nearest_band(wavelengths, target_nm):
    """找到 wavelengths 中最接近 target_nm 的波段索引 (0-based)。

    Args:
        wavelengths: 波段中心波长数组 (nm)
        target_nm: 目标波长 (nm)

    Returns:
        int: 最接近的波段索引
    """
    return int(np.argmin(np.abs(np.array(wavelengths) - target_nm)))


def _get_wavelengths(spcube):
    """从 hypercube 对象中提取波长信息。

    支持 SPy SpyFile (bands.centers) 和自定义对象 (wavelengths 属性)。
    返回 None 表示无波长元数据。
    """
    # SPy SpyFile 对象
    if hasattr(spcube, 'bands') and hasattr(spcube.bands, 'centers'):
        centers = spcube.bands.centers
        if centers is not None and len(centers) > 0:
            return np.array(centers, dtype=np.float64)

    # 自定义对象
    if hasattr(spcube, 'wavelengths'):
        w = spcube.wavelengths
        if w is not None and len(w) > 0:
            return np.array(w, dtype=np.float64)

    return None


def _get_band_stddevs(cube):
    """计算每个波段的标准差。

    Args:
        cube: (rows, cols, bands) 的 numpy 或 memmap 数组

    Returns:
        list of (band_index, stddev) 按 stddev 降序排列
    """
    n_bands = cube.shape[2]
    stds = []
    # 采样计算以加速 (大图全图计算太慢)
    sample_size = min(10000, cube.shape[0] * cube.shape[1])
    rng = np.random.RandomState(42)
    flat_idx = rng.choice(cube.shape[0] * cube.shape[1], sample_size, replace=False)

    for b in range(n_bands):
        band_data = cube[:, :, b].ravel()[flat_idx]
        stds.append((b, np.std(band_data)))

    stds.sort(key=lambda x: x[1], reverse=True)
    return stds


# ============================================================
# 波段选择策略 (对应 MATLAB Method 参数)
# ============================================================

def select_bands_falsecolored(spcube):
    """Method="falsecolored": PCA 选 3 个信息量最大的波段。

    MATLAB 文档描述: "three most informative bands"

    算法: PCA → 前 3 个主成分 → 每成分取绝对载荷最大的原始波段
    → 按波长排序 → R/G/B

    这比简单取方差 Top3 更好，因为 PCA 保证选出的波段在光谱上不冗余
    (方差最大的 3 个波段可能都在 NIR 区，导致接近灰度图)。
    """
    # 获取数据
    if hasattr(spcube, 'read_bands'):
        data = spcube.read_subregion([0, min(200, spcube.shape[0])],
                                     [0, min(200, spcube.shape[1])])
    elif hasattr(spcube, 'data'):
        data = spcube.data
    else:
        data = spcube

    rows, cols, n_bands = data.shape
    # 采样以加速 (大图全像素 PCA 太慢)
    n_sample = min(10000, rows * cols)
    rng = np.random.RandomState(42)
    flat_idx = rng.choice(rows * cols, n_sample, replace=False)
    pixels = data.reshape(-1, n_bands)[flat_idx].astype(np.float64)

    # 去均值 (PCA 前提)
    mean = pixels.mean(axis=0)
    pixels_centered = pixels - mean

    # SVD 分解 (等价于 PCA，numpy 原生，无外部依赖)
    U, S, Vt = np.linalg.svd(pixels_centered, full_matrices=False)
    # Vt: (n_bands, n_bands)，行 = 主成分方向 (载荷)

    # 前 3 个主成分，每个取绝对载荷最大的波段
    top_bands = []
    used = set()
    for pc_idx in range(3):
        loadings = np.abs(Vt[pc_idx])  # 该主成分对各原始波段的权重
        # 跳过已选波段，找下一个最大
        for b in np.argsort(-loadings):
            if b not in used:
                top_bands.append(b)
                used.add(b)
                break

    # 按波长排序
    top_bands.sort()

    wavelengths = _get_wavelengths(spcube)

    # 同时打印方差 Top3 做对比
    stds = _get_band_stddevs(data)
    print(f"  [falsecolored] 方差 Top3:    {[f'B{b+1}' for b, _ in stds[:3]]}")
    print(f"  [falsecolored] PCA 选出:     {[f'B{b+1}' for b in top_bands]}")
    print(f"  [falsecolored] PCA 解释方差: {[f'{S[i]**2/n_sample*100:.1f}%' for i in range(3)]}")
    if wavelengths is not None:
        print(f"  [falsecolored] 对应波长:     {wavelengths[top_bands]} nm")

    return top_bands  # [R_idx, G_idx, B_idx] sorted by wavelength


def select_bands_rgb(spcube):
    """Method="rgb": 波长匹配自然彩色 R(650nm)/G(550nm)/B(480nm)。"""
    wavelengths = _get_wavelengths(spcube)

    if wavelengths is None:
        print("  [rgb] ⚠ 无 wavelength 元数据，降级为 [0, n//2, n-1]")
        n = spcube.shape[2] if hasattr(spcube, 'shape') else spcube.nbands
        return [0, n // 2, n - 1]

    r_idx = find_nearest_band(wavelengths, RGB_TARGETS['R'])
    g_idx = find_nearest_band(wavelengths, RGB_TARGETS['G'])
    b_idx = find_nearest_band(wavelengths, RGB_TARGETS['B'])

    bands = [r_idx, g_idx, b_idx]
    print(f"  [rgb] 目标 R={RGB_TARGETS['R']}nm G={RGB_TARGETS['G']}nm B={RGB_TARGETS['B']}nm")
    print(f"  [rgb] 匹配: R→B{r_idx+1}({wavelengths[r_idx]}nm), "
          f"G→B{g_idx+1}({wavelengths[g_idx]}nm), "
          f"B→B{b_idx+1}({wavelengths[b_idx]}nm)")
    return bands


def select_bands_cir(spcube):
    """Method="cir": 彩红外 NIR≈850nm→R, Red≈650nm→G, Green≈550nm→B。"""
    wavelengths = _get_wavelengths(spcube)

    if wavelengths is None:
        print("  [cir] ⚠ 无 wavelength 元数据，降级为 [n-3, n-2, n-1]")
        n = spcube.shape[2] if hasattr(spcube, 'shape') else spcube.nbands
        return [n - 3, n - 2, n - 1]

    r_idx = find_nearest_band(wavelengths, CIR_TARGETS['R'])  # NIR→R
    g_idx = find_nearest_band(wavelengths, CIR_TARGETS['G'])  # Red→G
    b_idx = find_nearest_band(wavelengths, CIR_TARGETS['B'])  # Green→B

    bands = [r_idx, g_idx, b_idx]
    print(f"  [cir] 目标 NIR={CIR_TARGETS['R']}nm→R, "
          f"Red={CIR_TARGETS['G']}nm→G, Green={CIR_TARGETS['B']}nm→B")
    print(f"  [cir] 匹配: R→B{r_idx+1}({wavelengths[r_idx]}nm), "
          f"G→B{g_idx+1}({wavelengths[g_idx]}nm), "
          f"B→B{b_idx+1}({wavelengths[b_idx]}nm)")
    return bands


# ============================================================
# 百分位拉伸 (对应 ContrastStretching=true)
# ============================================================

def apply_contrast_stretch(rgb, stretch=(0.02, 0.98)):
    """对 RGB 图像的 3 个通道分别做百分位拉伸。

    这就是 MATLAB ContrastStretching=true 做的事。

    Args:
        rgb: (rows, cols, 3) float 或 int 数组
        stretch: (low_tail, high_tail) 尾部比例

    Returns:
        (rows, cols, 3) float [0,1]
    """
    rgb = rgb.astype(np.float64)
    lims = []
    for i in range(3):
        band = rgb[:, :, i].ravel()
        low_val = np.percentile(band, stretch[0] * 100)
        high_val = np.percentile(band, stretch[1] * 100)
        lims.append((low_val, high_val))

    # 统一拉伸 (与 SPy 默认一致，避免色偏)
    lower = min(l[0] for l in lims)
    upper = max(l[1] for l in lims)
    span = upper - lower

    print(f"  [stretch] 各通道 p{stretch[0]*100:.0f}/p{stretch[1]*100:.0f}: "
          f"R={lims[0]}, G={lims[1]}, B={lims[2]}")
    print(f"  [stretch] 统一拉伸: [{lower:.1f}, {upper:.1f}]")

    if span == 0:
        return np.zeros_like(rgb, dtype=np.float64)
    return np.clip((rgb - lower) / span, 0.0, 1.0)


def linear_stretch(rgb):
    """不做百分位拉伸，仅线性映射 min→max 到 [0,1]。

    对应 MATLAB ContrastStretching=false。
    """
    rgb = rgb.astype(np.float64)
    lower = rgb.min()
    upper = rgb.max()
    span = upper - lower
    print(f"  [linear] min={lower:.1f}, max={upper:.1f}")
    if span == 0:
        return np.zeros_like(rgb, dtype=np.float64)
    return (rgb - lower) / span


# ============================================================
# 主函数: colorize
# ============================================================

def colorize(spcube, band=None, *, Method="falsecolored", ContrastStretching=False):
    """MATLAB colorize 兼容的 Python 实现。

    Args:
        spcube: SPy SpyFile 或 (R,C,B) numpy 数组
        band:   可选，3 元素波段列表 (1-based，与 MATLAB 一致)。
                若提供，直接使用，忽略 Method。
        Method: "falsecolored" | "rgb" | "cir"
        ContrastStretching: True (2%-98% 百分位拉伸) | False (线性 min-max)

    Returns:
        (rgb, indices) 元组:
            rgb:     (rows, cols, 3) float [0,1] 的彩色图像
            indices: 实际使用的波段索引列表 (0-based)
    """
    print(f"\n{'='*50}")
    print(f"colorize(Method=\"{Method}\", ContrastStretching={ContrastStretching}"
          f"{', band=' + str(band) if band else ''})")
    print(f"{'='*50}")

    # ---- 确定波段索引 (1-based → 0-based) ----
    if band is not None:
        # 手动指定: MATLAB 是 1-based，转 0-based
        if len(band) != 3:
            raise ValueError(f"band 必须是 3 元素列表，收到 {len(band)}")
        indices = [b - 1 for b in band]
        print(f"  [custom] 手动指定: {band} → 0-based: {indices}")
    elif Method == "rgb":
        indices = select_bands_rgb(spcube)
    elif Method == "cir":
        indices = select_bands_cir(spcube)
    elif Method == "falsecolored":
        indices = select_bands_falsecolored(spcube)
    else:
        raise ValueError(f"未知 Method: {Method}，有效值: falsecolored, rgb, cir")

    # ---- 读取波段数据 ----
    if hasattr(spcube, 'read_bands'):
        rgb_data = spcube.read_bands(indices).astype(np.float64)
    elif hasattr(spcube, 'load'):
        rgb_data = spcube.load().read_bands(indices).astype(np.float64)
    else:
        # 可能是包装类或裸 numpy 数组
        src = spcube.data if hasattr(spcube, 'data') else spcube
        rgb_data = np.take(src, indices, axis=2).astype(np.float64)

    print(f"  [data] RGB cube shape: {rgb_data.shape}, "
          f"值域: [{rgb_data.min():.1f}, {rgb_data.max():.1f}]")

    # ---- 拉伸 ----
    if ContrastStretching:
        rgb = apply_contrast_stretch(rgb_data)
    else:
        rgb = linear_stretch(rgb_data)

    print(f"  [output] 最终 RGB 值域: [{rgb.min():.4f}, {rgb.max():.4f}]")
    return rgb, indices


# ============================================================
# 演示: 3 种 Method × 2 种拉伸 = 6 图对比
# ============================================================
def main():
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # 中文字体
    for f in fm.fontManager.ttflist:
        if 'pingfang' in f.name.lower() or 'heiti' in f.name.lower():
            plt.rcParams['font.family'] = f.name
            break

    # ---- 数据准备: 用 rasterio 读原始 16-bit TIF (展示真实的百分位拉伸效果) ----
    import rasterio

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data", "landsat8")
    BAND_KEYS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]

    print("=" * 60)
    print("读取原始 16-bit Landsat 8 数据 (百分位拉伸效果更明显)")
    print("=" * 60)

    # 降采样读取
    downsample = 4
    cube_raw = None
    for i, bk in enumerate(BAND_KEYS):
        tif_files = [f for f in os.listdir(DATA_DIR) if f.endswith(f"_{bk}.TIF")]
        if not tif_files:
            continue
        tif_path = os.path.join(DATA_DIR, tif_files[0])
        with rasterio.open(tif_path) as src:
            band_data = src.read(
                1,
                out_shape=(src.height // downsample, src.width // downsample),
                resampling=rasterio.enums.Resampling.average,
            ).astype(np.float64)
            if cube_raw is None:
                cube_raw = np.zeros((band_data.shape[0], band_data.shape[1], len(BAND_KEYS)),
                                     dtype=np.float64)
            cube_raw[:, :, i] = band_data
        print(f"✓ B{i+1} ({LANDSAT8_WAVELENGTHS[i]}nm): {band_data.shape}, "
              f"DN [{band_data.min():.0f}, {band_data.max():.0f}]")

    print(f"\n数据立方体: {cube_raw.shape}, 值域 [{cube_raw.min():.0f}, {cube_raw.max():.0f}]")

    # 用简单包装类注入 wavelength 信息 (模拟带元数据的 hypercube)
    from types import SimpleNamespace
    wrapped = SimpleNamespace()
    wrapped.data = cube_raw
    wrapped.wavelengths = LANDSAT8_WAVELENGTHS
    wrapped.shape = cube_raw.shape

    cube_raw = wrapped

    # ---- 生成 6 张图 ----
    methods = ["falsecolored", "rgb", "cir"]
    stretches = [False, True]

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    for row, method in enumerate(methods):
        for col, use_stretch in enumerate(stretches):
            ax = axes[row, col]
            rgb, indices = colorize(
                cube_raw,
                Method=method,
                ContrastStretching=use_stretch,
            )

            ax.imshow(rgb)
            stretch_label = "有拉伸 (2%-98%)" if use_stretch else "无拉伸 (min-max)"
            bands_used = [f"B{i+1}" for i in indices]
            ax.set_title(f"Method=\"{method}\" | {stretch_label}",
                         fontsize=11, fontweight='bold')
            ax.set_xlabel(f"波段: {', '.join(bands_used)}  "
                          f"({LANDSAT8_WAVELENGTHS[indices]} nm)")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("MATLAB colorize 兼容实现 — 3 Method × 2 Stretch 对比",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()

    out_path = os.path.join(BASE_DIR, "output", "colorize_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ 对比图已保存: {out_path}")

    try:
        plt.show()
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("完成！colorize 函数实现成功。")
    print("=" * 60)


if __name__ == "__main__":
    main()
