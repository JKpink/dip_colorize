#!/usr/bin/env python3
"""
hyperslic — 高光谱 SLIC 超像素分割 (MATLAB 兼容)

对照 MATLAB hyperslic 算法:
  Achanta et al. 2012 SLIC + PCA 降维预处理

用法:
  L, numLabels = hyperslic(hcube, K=100)
  L, numLabels = hyperslic(hcube, K=100, IsInputDimReduced=False, NumIterations=10)
"""

import numpy as np


# ============================================================
# Phase 1: PCA 光谱降维
# ============================================================

def _pca_reduce(pixels, var_threshold=0.99, min_bands=3):
    """PCA 降维，保留足够解释方差的主成分。

    先做波段级 min-max 归一化 → PCA → Z-score 标准化各主成分，
    保证 SLIC 距离中光谱分量和空间分量权重平衡。

    Args:
        pixels:       (N, B) 像素矩阵, N=行×列, B=原始波段数
        var_threshold: 累积解释方差阈值 (默认 99%)
        min_bands:     最少保留的波段数

    Returns:
        reduced:  (N, B') 降维后像素矩阵 (每列 ~N(0,1))
        n_kept:   保留的主成分数
    """
    N, B = pixels.shape

    # 清理输入中的 NaN/Inf
    pixels = np.nan_to_num(pixels, nan=0.0, posinf=0.0, neginf=0.0)

    # 1. 逐波段 min-max 归一化
    p_min = pixels.min(axis=0)
    p_max = pixels.max(axis=0)
    p_max[p_max == p_min] = 1.0
    p_norm = (pixels - p_min) / (p_max - p_min)

    # 2. 去均值
    mean = p_norm.mean(axis=0)
    centered = p_norm - mean
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. 协方差矩阵特征分解 (eigh 对 103×103 矩阵非常稳定)
    with np.errstate(all='ignore'):
        cov = (centered.T @ centered) / (N - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigh 返回升序，取降序
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]  # (B, B), 列=主成分方向

    # 4. 确定保留主成分数
    total_var = np.sum(np.maximum(eigenvalues, 0))
    cumsum = np.cumsum(np.maximum(eigenvalues, 0)) / max(total_var, 1e-10)
    n_kept = max(min_bands, int(np.searchsorted(cumsum, var_threshold) + 1))
    n_kept = min(n_kept, B)

    # 5. 投影 (用前 n_kept 个特征向量)
    with np.errstate(all='ignore'):
        scores = centered @ eigenvectors[:, :n_kept]
    pc_std = scores.std(axis=0)
    pc_std[pc_std < 1e-10] = 1.0
    reduced = scores / pc_std
    reduced = np.nan_to_num(reduced, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  [PCA] {B} 波段 -> {n_kept} 主成分 "
          f"(累积方差 {cumsum[n_kept-1]*100:.1f}%)")

    return reduced, n_kept


# ============================================================
# Phase 2 & 3: SLIC 聚类 + 连通性强制
# ============================================================

def _init_centers(rows, cols, K, pixels_reduced):
    """在 3×3 邻域内初始化聚类中心到梯度最小位置。

    Returns:
        centers:      (K, B') 聚类中心光谱
        centers_xy:   (K, 2) 聚类中心坐标
        grid_step:    网格间隔 S
        labels:       (rows*cols,) 像素 → 聚类标签
        distances:    (rows*cols,) 最小距离
    """
    N = rows * cols
    S = int(np.sqrt(N / K))
    grid_step = max(1, S)

    # 在网格上均匀撒种子
    centers = []
    centers_xy = []
    for r in range(grid_step // 2, rows, grid_step):
        for c in range(grid_step // 2, cols, grid_step):
            # 3×3 邻域内找梯度最小位置
            r0, c0 = max(0, r-1), max(0, c-1)
            r1, c1 = min(rows, r+2), min(cols, c+2)
            patch = pixels_reduced.reshape(rows, cols, -1)[r0:r1, c0:c1]
            # 梯度 = 邻域内光谱差异
            if patch.size > 0:
                grad = np.zeros(patch.shape[:2])
                for i in range(1, patch.shape[0]):
                    grad[i, :] += np.sum((patch[i] - patch[i-1])**2, axis=1)
                for j in range(1, patch.shape[1]):
                    grad[:, j] += np.sum((patch[:, j] - patch[:, j-1])**2, axis=1)
                min_pos = np.unravel_index(np.argmin(grad), grad.shape[:2])
                best_r, best_c = r0 + min_pos[0], c0 + min_pos[1]
            else:
                best_r, best_c = r, c

            centers.append(pixels_reduced[best_r * cols + best_c])
            centers_xy.append([best_r, best_c])

    centers = np.array(centers)
    centers_xy = np.array(centers_xy)
    actual_K = len(centers)

    labels = -np.ones(N, dtype=np.int32)
    distances = np.full(N, np.inf, dtype=np.float64)

    return centers, centers_xy, grid_step, labels, distances, actual_K


def _slic_iterate(pixels_reduced, rows, cols, centers, centers_xy,
                  grid_step, m, labels, distances, n_iters):
    """SLIC 迭代聚类。

    距离公式: D = sqrt( (d_spectral/m)^2 + (d_xy/S)^2 )
    """
    N = rows * cols
    S = float(grid_step)

    for it in range(n_iters):
        # ---- 赋值: 搜索 2S×2S 邻域 ----
        for k, (center_spec, (cr, cc)) in enumerate(zip(centers, centers_xy)):
            # 搜索窗口: [r_low, r_high) × [c_low, c_high)
            r_low = max(0, int(cr - S))
            r_high = min(rows, int(cr + S + 1))
            c_low = max(0, int(cc - S))
            c_high = min(cols, int(cc + S + 1))

            # 提取窗口内像素的光谱
            patch_shape = (r_high - r_low, c_high - c_low)
            patch_pixels = pixels_reduced.reshape(rows, cols, -1)[r_low:r_high, c_low:c_high]
            patch_pixels = patch_pixels.reshape(-1, patch_pixels.shape[-1])

            # 光谱距离
            d_spec = np.sqrt(np.sum((patch_pixels - center_spec)**2, axis=1))

            # 空间距离
            ys, xs = np.mgrid[r_low:r_high, c_low:c_high]
            ys = ys.ravel() - cr
            xs = xs.ravel() - cc
            d_xy = np.sqrt(ys**2 + xs**2)

            # 组合距离
            D = np.sqrt((d_spec / m)**2 + (d_xy / S)**2)

            # 更新标签和距离
            patch_idx = (ys + cr).astype(int) * cols + (xs + cc).astype(int)
            better = D < distances[patch_idx]
            labels[patch_idx[better]] = k
            distances[patch_idx[better]] = D[better]

        # ---- 更新中心 ----
        for k in range(len(centers)):
            mask = labels == k
            if mask.sum() > 0:
                centers[k] = pixels_reduced[mask].mean(axis=0)
                ys = mask.reshape(rows, cols).nonzero()[0]
                xs = mask.reshape(rows, cols).nonzero()[1]
                centers_xy[k] = [ys.mean(), xs.mean()]

    return labels


def _enforce_connectivity(labels, rows, cols, min_size=None,
                           pixels_reduced=None):
    """连通性强制 + 光谱相似合并。

    1. BFS 找连通分量，面积 < min_size 的合并到邻接区域
    2. 遍历所有相邻超像素对，光谱距离 < merge_threshold 的自动合并
    """
    if min_size is None:
        S = int(np.sqrt(rows * cols / max(1, labels.max() + 1)))
        min_size = max(1, (S * S) // 4)

    new_labels = np.full_like(labels, -1)
    label_map = {}
    next_label = 0
    adj = ((-1, 0), (1, 0), (0, -1), (0, 1))

    # ---- Pass 1: 连通分量扫描 + 小碎片合并 ----
    visited = np.zeros(rows * cols, dtype=bool)
    for start in range(rows * cols):
        if visited[start]:
            continue
        r, c = start // cols, start % cols
        cluster_label_val = labels[start]

        queue = [(r, c)]
        component = []
        while queue:
            cr, cc = queue.pop(0)
            idx = cr * cols + cc
            if visited[idx] or labels[idx] != cluster_label_val:
                continue
            visited[idx] = True
            component.append(idx)
            for dr, dc in adj:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    queue.append((nr, nc))

        if len(component) >= min_size:
            new_id = next_label
            next_label += 1
        else:
            # 小碎片：找最大接触面积的邻接区域
            neighbor_labels = {}
            for idx in component:
                cr, cc = idx // cols, idx % cols
                for dr, dc in adj:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        nl = labels[nr * cols + nc]
                        if nl != cluster_label_val:
                            neighbor_labels[nl] = neighbor_labels.get(nl, 0) + 1
            if neighbor_labels:
                merge_to = max(neighbor_labels, key=neighbor_labels.get)
                new_id = label_map.get(merge_to, -1)
                if new_id < 0:
                    new_id = next_label
                    next_label += 1
                    label_map[merge_to] = new_id
            else:
                new_id = next_label
                next_label += 1

        for idx in component:
            new_labels[idx] = new_id
        label_map[cluster_label_val] = new_labels[component[0]]

    # 重整标签
    unique = np.unique(new_labels)
    remap = {old: new for new, old in enumerate(unique)}
    L = np.array([remap[l] for l in new_labels], dtype=np.int32)
    num_labels = len(unique)

    # ---- Pass 2: 光谱相似合并 ----
    if pixels_reduced is not None:
        L, num_labels = _merge_spectrally_similar(L, rows, cols,
                                                    pixels_reduced)

    return L, num_labels


def _merge_spectrally_similar(L, rows, cols, pixels_reduced):
    """自动合并光谱相似的小区域。

    对面积 < S²/4 的区域，如果其平均光谱与相邻区域的光谱距离
    小于阈值（所有区域间光谱距离的中位数），则合并。
    """
    N = rows * cols
    unique_labels = np.unique(L)
    num_labels = len(unique_labels)
    K = num_labels
    S = int(np.sqrt(N / max(1, K)))
    small_threshold = max(1, S * S // 4)

    # 计算每个超像素的平均光谱
    region_spectra = {}
    region_sizes = {}
    for lb in unique_labels:
        mask = L.reshape(-1) == lb
        region_sizes[lb] = int(mask.sum())
        region_spectra[lb] = pixels_reduced[mask].mean(axis=0)

    # 找邻接关系
    adjacency = {lb: set() for lb in unique_labels}
    L2d = L.reshape(rows, cols)
    for r in range(rows - 1):
        for c in range(cols - 1):
            a, b = L2d[r, c], L2d[r, c+1]
            if a != b:
                adjacency[a].add(b)
                adjacency[b].add(a)
            a, b = L2d[r, c], L2d[r+1, c]
            if a != b:
                adjacency[a].add(b)
                adjacency[b].add(a)

    # 计算所有邻接对的光谱距离，取中位数作为合并阈值
    all_dists = []
    for a, neighbors in adjacency.items():
        for b in neighbors:
            if a < b:
                d = np.sqrt(np.sum((region_spectra[a] - region_spectra[b])**2))
                all_dists.append(d)
    merge_threshold = np.median(all_dists) * 0.5 if all_dists else 0

    # 从小到大遍历区域，合并小且光谱相似的对
    # 使用 Union-Find
    parent = {lb: lb for lb in unique_labels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for lb in sorted(unique_labels, key=lambda x: region_sizes[x]):
        if region_sizes[lb] >= small_threshold * 2:
            continue  # 跳过足够大的区域
        best_neighbor = None
        best_dist = float('inf')
        for nb in adjacency[lb]:
            d = np.sqrt(np.sum((region_spectra[lb] - region_spectra[nb])**2))
            if d < best_dist:
                best_dist = d
                best_neighbor = nb
        if best_dist < merge_threshold and best_neighbor is not None:
            union(lb, best_neighbor)

    # 应用合并
    root_to_new = {}
    next_id = 0
    new_L = np.zeros_like(L)
    for lb in unique_labels:
        root = find(lb)
        if root not in root_to_new:
            root_to_new[root] = next_id
            next_id += 1
        new_L[L == lb] = root_to_new[root]

    n_merged = num_labels - next_id
    if n_merged > 0:
        print(f"  [merge] 光谱相似合并: {n_merged} 个区域")

    return new_L, next_id


# ============================================================
# 主函数
# ============================================================

def hyperslic(hcube, K, IsInputDimReduced=False, NumIterations=10, m=None):
    """高光谱 SLIC 超像素分割 (MATLAB hyperslic 兼容)。

    Args:
        hcube:             SPy SpyFile 或 (R,C,B) numpy 或 SimpleNamespace (带 .data)
        K:                 目标超像素数
        IsInputDimReduced: True -> 跳过 PCA 降维
        NumIterations:     迭代次数 (默认 10)
        m:                 紧凑度参数。None 时自动 = sqrt(B')，平衡光谱/空间权重。

    Returns:
        L:          (rows, cols) 标签矩阵, 值 0..numLabels-1
        numLabels:  实际超像素数
    """
    # 获取数据
    if hasattr(hcube, 'data'):
        cube = hcube.data
    elif hasattr(hcube, 'read_bands'):
        cube = hcube.load() if hasattr(hcube, 'load') else hcube[:, :, :]
    else:
        cube = hcube

    if isinstance(cube, np.ndarray):
        cube = cube.astype(np.float64)
    rows, cols, bands = cube.shape
    N = rows * cols

    print(f"hyperslic: {rows}×{cols}×{bands} bands, K={K}")

    # ---- Phase 1: PCA 降维 ----
    if IsInputDimReduced or bands <= 3:
        pixels = cube.reshape(N, bands)
        n_kept = bands
        if not IsInputDimReduced and bands <= 3:
            print(f"  [PCA] 波段数 ≤ 3, 跳过降维")
    else:
        pixels = cube.reshape(N, bands)
        pixels, n_kept = _pca_reduce(pixels, var_threshold=0.99, min_bands=3)

    # ---- Auto-m: 根据降维后主成分数自动选紧凑度 ----
    if m is None:
        # m ≈ sqrt(B') 保证 d_spec/m 和 d_xy/S 在同一量级
        m = max(1, int(np.sqrt(n_kept)))
        print(f"  [m] 自动: sqrt({n_kept}) = {m}")

    # ---- Phase 2: SLIC ----
    centers, centers_xy, grid_step, labels, distances, actual_K = \
        _init_centers(rows, cols, K, pixels)

    labels = _slic_iterate(pixels, rows, cols, centers, centers_xy,
                           grid_step, m, labels, distances, NumIterations)

    # ---- Phase 3: 连通性强制 ----
    labels_2d, num_labels = _enforce_connectivity(labels, rows, cols,
                                                    pixels_reduced=pixels)
    L = labels_2d.reshape(rows, cols)

    print(f"  ✓ 实际超像素: {num_labels} (目标 K={K})")
    return L, num_labels
