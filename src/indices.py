#!/usr/bin/env python3
"""
植被指数 — MATLAB 兼容实现

对照 MATLAB Image Processing Toolbox:
  ndvi — Normalized Difference Vegetation Index
"""

import numpy as np


def _get_wavelengths(spcube):
    """从 hypercube 对象中提取波长信息。"""
    if hasattr(spcube, 'bands') and hasattr(spcube.bands, 'centers'):
        centers = spcube.bands.centers
        if centers is not None and len(centers) > 0:
            return np.array(centers, dtype=np.float64)
    if hasattr(spcube, 'wavelengths'):
        w = spcube.wavelengths
        if w is not None and len(w) > 0:
            return np.array(w, dtype=np.float64)
    return None


def _find_nearest_band(wavelengths, target_nm):
    """找到最接近 target_nm 的波段索引 (0-based)。"""
    return int(np.argmin(np.abs(wavelengths - target_nm)))


def _get_band_data(spcube, band_idx):
    """从 spcube 中提取单个波段的数据。

    支持: SPy SpyFile, SimpleNamespace (带 .data), numpy ndarray
    """
    if hasattr(spcube, 'read_band'):
        return spcube.read_band(band_idx).astype(np.float64)
    elif hasattr(spcube, 'read_bands'):
        data = spcube.read_bands([band_idx])
        return data[:, :, 0].astype(np.float64)
    elif hasattr(spcube, 'data'):
        return spcube.data[:, :, band_idx].astype(np.float64)
    else:
        return spcube[:, :, band_idx].astype(np.float64)


def ndvi(spcube, nir_band=None, red_band=None):
    """计算 NDVI (Normalized Difference Vegetation Index)。

    MATLAB 兼容: ndviImg = ndvi(hcube)
                   ndviImg = ndvi(hcube, nirBand, redBand)

    NDVI = (NIR - Red) / (NIR + Red)

    Args:
        spcube:   hypercube 对象 (带 wavelengths 元数据)
        nir_band: NIR 波段索引 (0-based)。None 时自动匹配 ~850nm。
        red_band: Red 波段索引 (0-based)。None 时自动匹配 ~650nm。

    Returns:
        ndvi_img: (rows, cols) float [-1, 1]
        nir_idx:  使用的 NIR 波段索引
        red_idx:  使用的 Red 波段索引
    """
    wavelengths = _get_wavelengths(spcube)

    # 自动匹配波段
    if nir_band is None:
        if wavelengths is not None:
            nir_band = _find_nearest_band(wavelengths, 850.0)
            print(f"  [ndvi] NIR: 自动匹配 → B{nir_band+1} ({wavelengths[nir_band]:.0f}nm)")
        else:
            n_bands = spcube.shape[2] if hasattr(spcube, 'shape') else spcube.nbands
            nir_band = n_bands - 10  # 降级：假设 NIR 在靠后波段
            print(f"  [ndvi] NIR: 无波长元数据，降级 → B{nir_band+1}")

    if red_band is None:
        if wavelengths is not None:
            red_band = _find_nearest_band(wavelengths, 650.0)
            print(f"  [ndvi] Red: 自动匹配 → B{red_band+1} ({wavelengths[red_band]:.0f}nm)")
        else:
            n_bands = spcube.shape[2] if hasattr(spcube, 'shape') else spcube.nbands
            red_band = n_bands // 3  # 降级：红色在可见光中段
            print(f"  [ndvi] Red: 无波长元数据，降级 → B{red_band+1}")

    # 读取波段数据
    nir = _get_band_data(spcube, nir_band)
    red = _get_band_data(spcube, red_band)

    print(f"  [ndvi] NIR DN: [{nir.min():.0f}, {nir.max():.0f}]")
    print(f"  [ndvi] Red DN: [{red.min():.0f}, {red.max():.0f}]")

    # NDVI = (NIR - Red) / (NIR + Red)
    denominator = nir + red
    ndvi_img = np.full_like(nir, np.nan, dtype=np.float64)

    valid = denominator != 0
    ndvi_img[valid] = (nir[valid] - red[valid]) / denominator[valid]

    print(f"  [ndvi] NDVI 值域: [{np.nanmin(ndvi_img):.4f}, {np.nanmax(ndvi_img):.4f}]")
    print(f"  [ndvi] 有效像素: {valid.sum()}/{valid.size}")

    return ndvi_img, nir_band, red_band
