#!/usr/bin/env python3
"""
ENVI 格式读写 + Landsat GeoTIFF 波段读取

从 demo_spy_view.py 抽取的通用函数，供 scripts/ 和 src/ 使用。
"""

import os
import numpy as np


def read_landsat_bands(data_dir, band_keys, downsample=1):
    """用 rasterio 读取 Landsat 8 各波段 GeoTIFF，堆叠为 3D numpy 数组。

    Args:
        data_dir:   包含 .TIF 波段文件的目录路径
        band_keys:  波段名列表，如 ["B1", "B2", ..., "B7"]
        downsample: 降采样因子 (1=全分辨率, 4=1/4 宽高)

    Returns:
        cube:        (rows, cols, num_bands) float32 数组
        wavelengths: 各波段中心波长 (nm)，Landsat 8 固定值
        profile:     rasterio profile (首波段)
    """
    import rasterio

    # Landsat 8 各波段中心波长 (nm)
    _WAVELENGTHS = {
        "B1": 443, "B2": 482, "B3": 562, "B4": 655,
        "B5": 865, "B6": 1609, "B7": 2201,
    }

    cube = None
    profile = None
    wavelengths = []

    for band_name in band_keys:
        tif_files = [f for f in os.listdir(data_dir)
                     if f.endswith(f"_{band_name}.TIF")]
        if not tif_files:
            print(f"⚠ 未找到 {band_name} 的文件，跳过")
            continue

        tif_path = os.path.join(data_dir, tif_files[0])

        with rasterio.open(tif_path) as src:
            if profile is None:
                profile = src.profile
                out_shape = (src.height // downsample,
                             src.width // downsample)
                print(f"原始尺寸: {src.width}×{src.height}")
                print(f"降采样因子: {downsample}")
                print(f"输出尺寸: {out_shape[1]}×{out_shape[0]}")

            band_data = src.read(
                1, out_shape=out_shape,
                resampling=rasterio.enums.Resampling.average,
            ).astype(np.float32)

            print(f"✓ {band_name}: {band_data.shape[0]}×{band_data.shape[1]}")

            if cube is None:
                cube = np.zeros(
                    (band_data.shape[0], band_data.shape[1], len(band_keys)),
                    dtype=np.float32,
                )
            idx = list(band_keys).index(band_name)
            cube[:, :, idx] = band_data
            wavelengths.append(_WAVELENGTHS.get(band_name, 0))

    return cube, np.array(wavelengths, dtype=np.float32), profile


def save_envi(cube, basename):
    """将 numpy 数组 (rows, cols, bands) 保存为 ENVI 格式。

    自动对每个波段做 2%-98% 百分位拉伸到 uint8。

    Args:
        cube:     (R, C, B) 的 float/int 数组
        basename: 输出文件基础名 (不含扩展名)

    Returns:
        cube_scaled: 拉伸后的 uint8 数组
    """
    import spectral.io.envi as envi

    print("\n应用百分位拉伸 (2%-98%) ...")
    cube_scaled = np.zeros(cube.shape, dtype=np.uint8)
    for i in range(cube.shape[2]):
        band = cube[:, :, i]
        p2 = np.percentile(band, 2)
        p98 = np.percentile(band, 98)
        stretched = (band - p2) / (p98 - p2) * 255.0
        stretched = np.clip(stretched, 0, 255)
        cube_scaled[:, :, i] = stretched.astype(np.uint8)
        print(f"  波段 {i+1}: p2={p2:.0f}, p98={p98:.0f}")

    envi.save_image(f"{basename}.hdr", cube_scaled, dtype=np.uint8, force=True)
    print(f"✓ ENVI 文件已保存: {basename}.hdr + .img")
    return cube_scaled
