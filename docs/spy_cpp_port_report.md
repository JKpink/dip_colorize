# SPy 数据解析完整技术报告 — C++ 复现指南

## 1. 总览：从文件到像素的完整流水线

```
磁盘上的 .hdr + .img 文件
  │
  ├─ [Phase A] ENVI 头文件解析 ──→ 元数据结构体 (宽高、波段数、数据类型、字节序、interleave)
  │
  ├─ [Phase B] 数据文件定位 ──→ 根据头文件名推断 .img/.dat 文件路径
  │
  ├─ [Phase C] 波段读取 ──→ 根据 interleave 计算文件偏移，mmap 或 seek+read
  │
  ├─ [Phase D] get_rgb ──→ 波段提取 + 百分位拉伸 + clip
  │
  └─ 显示
```

---

## 2. ENVI 头文件格式 (.hdr)

### 2.1 文件示例 (我们的 landsat8_cube.hdr)

```
ENVI
samples = 1912
lines = 1930
bands = 7
header offset = 0
file type = ENVI Standard
data type = 1
interleave = bip
byte order = 0 
```

### 2.2 解析规则

| 规则 | 说明 |
|---|---|
| 第一行 | 必须以 `ENVI` 开头 |
| 注释 | `;` 开头的行跳过 |
| 键值对 | `key = value`，键大小写不敏感（统一转小写） |
| 多值 | `{val1, val2, val3}` 花括号包围，逗号分隔 |
| 空行 | 忽略 |

### 2.3 必需字段

| 字段 | 含义 |
|---|---|
| `samples` | 图像宽度 (列数) |
| `lines` | 图像高度 (行数) |
| `bands` | 波段数 |
| `data type` | 数据类型编号 (见下表) |
| `byte order` | 0=小端, 1=大端 |
| `interleave` | `bip`, `bil`, 或 `bsq` |
| `header offset` | 数据起始偏移字节数 (通常为 0) |

### 2.4 data type 映射表

| ENVI编号 | C++ 类型 | 字节数 | numpy |
|---|---|---|---|
| 1 | `uint8_t` | 1 | uint8 |
| 2 | `int16_t` | 2 | int16 |
| 3 | `int32_t` | 4 | int32 |
| 4 | `float` | 4 | float32 |
| 5 | `double` | 8 | float64 |
| 12 | `uint16_t` | 2 | uint16 |
| 13 | `uint32_t` | 4 | uint32 |
| 14 | `int64_t` | 8 | int64 |
| 15 | `uint64_t` | 8 | uint64 |

### 2.5 C++ 头文件解析伪代码

```cpp
struct EnviHeader {
    int samples, lines, bands;
    int data_type, byte_order, header_offset;
    std::string interleave;  // "bip", "bil", or "bsq"
    std::string data_file;
    // 可选字段
    std::vector<double> wavelengths;
    std::vector<int> default_bands;
};

EnviHeader parse_hdr(const std::string& hdr_path) {
    std::ifstream f(hdr_path);
    std::string first_line;
    std::getline(f, first_line);
    if (!first_line.starts_with("ENVI"))
        throw std::runtime_error("Not an ENVI header");

    std::map<std::string, std::string> kv;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == ';') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(line.substr(0, eq));
        std::transform(key.begin(), key.end(), key.begin(), ::tolower);
        std::string val = trim(line.substr(eq + 1));
        // 处理花括号多值: {v1, v2, v3}
        if (!val.empty() && val[0] == '{') {
            while (val.back() != '}') {
                std::string more;
                std::getline(f, more);
                val += "\n" + trim(more);
            }
            val = val.substr(1, val.size() - 2);  // 去掉 {}
        }
        kv[key] = val;
    }

    EnviHeader h;
    h.samples  = std::stoi(kv["samples"]);
    h.lines    = std::stoi(kv["lines"]);
    h.bands    = std::stoi(kv["bands"]);
    h.data_type = std::stoi(kv["data type"]);
    h.byte_order = std::stoi(kv["byte order"]);
    h.header_offset = kv.contains("header offset") ? std::stoi(kv["header offset"]) : 0;
    h.interleave = kv["interleave"];
    // tolower interleave
    std::transform(h.interleave.begin(), h.interleave.end(),
                   h.interleave.begin(), ::tolower);

    // 可选: wavelength
    if (kv.contains("wavelength")) {
        // 解析逗号分隔的浮点数列表
        h.wavelengths = parse_float_list(kv["wavelength"]);
    }

    return h;
}
```

---

## 3. 数据文件定位

### 3.1 文件名推断规则

去掉 `.hdr` 扩展名，依次尝试以下扩展名（大小写不敏感）：

```
无扩展名 → .img → .dat → .sli → .raw → .hyspex → interleave字符串
→ .IMG → .DAT → ...(大写版本)
```

### 3.2 C++ 实现

```cpp
std::string find_data_file(const std::string& hdr_path,
                           const std::string& interleave) {
    // 去掉 .hdr
    std::string base = hdr_path;
    if (base.ends_with(".hdr"))
        base = base.substr(0, base.size() - 4);

    std::vector<std::string> exts = {
        "", ".img", ".dat", ".sli", ".raw", ".hyspex",
        "." + interleave
    };

    for (auto& ext : exts) {
        if (std::filesystem::exists(base + ext))
            return base + ext;
        // 再试大写
        std::string upper = ext;
        std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
        if (std::filesystem::exists(base + upper))
            return base + upper;
    }
    throw std::runtime_error("Data file not found");
}
```

---

## 4. Interleave 与内存布局 — 这是 C++ 复现的核心

三种 interleave 决定了像素数据在二进制文件中的排列顺序。

### 4.1 BIP (Band Interleaved by Pixel) — 像素交错

```
文件字节序列:
[p0_b0][p0_b1]...[p0_bN][p1_b0][p1_b1]...[p1_bN]...
│←── 像素0的所有波段 ──→│←── 像素1的所有波段 ──→│

文件偏移公式:
offset(row, col, band) = header_offset
    + sample_size * (nbands * (row * ncols + col) + band)

逻辑形状: (rows, cols, bands)  ← memmap 直接映射，无需转置
```

```cpp
// BIP: 读取指定波段到 (rows, cols) 的二维数组
template<typename T>
void read_band_bip(const EnviHeader& h, const T* mmap_data,
                   int band, T* output) {
    int sample_size = sizeof(T);
    for (int r = 0; r < h.lines; r++) {
        for (int c = 0; c < h.samples; c++) {
            size_t offset = h.nbands * (r * h.samples + c) + band;
            output[r * h.samples + c] = mmap_data[offset];
        }
    }
}

// BIP: 读取多个波段到 (rows, cols, bands) 的三维数组
template<typename T>
void read_bands_bip(const EnviHeader& h, const T* mmap_data,
                    const std::vector<int>& bands, T* output) {
    for (int r = 0; r < h.lines; r++) {
        for (int c = 0; c < h.samples; c++) {
            size_t pixel_base = h.nbands * (r * h.samples + c);
            for (size_t bi = 0; bi < bands.size(); bi++) {
                output[(r * h.samples + c) * bands.size() + bi] =
                    mmap_data[pixel_base + bands[bi]];
            }
        }
    }
}
```

### 4.2 BIL (Band Interleaved by Line) — 行交错

```
文件字节序列:
[row0_b0_allcols][row0_b1_allcols]...[row0_bN_allcols]
[row1_b0_allcols][row1_b1_allcols]...[row1_bN_allcols]
...
│←────────── 第0行的所有波段 ──────────→│←── 第1行...──→│

文件偏移公式:
offset(row, col, band) = header_offset
    + sample_size * (nbands * ncols * row + ncols * band + col)

逻辑形状: (rows, bands, cols)  ← memmap 需要 transpose(0,2,1) 变成 (rows, cols, bands)
```

```cpp
// BIL: 读取多个波段
template<typename T>
void read_bands_bil(const EnviHeader& h, const T* mmap_data,
                    const std::vector<int>& bands, T* output) {
    for (int r = 0; r < h.lines; r++) {
        size_t row_base = h.nbands * h.samples * r;
        for (int c = 0; c < h.samples; c++) {
            for (size_t bi = 0; bi < bands.size(); bi++) {
                size_t offset = row_base + h.samples * bands[bi] + c;
                output[(r * h.samples + c) * bands.size() + bi] = mmap_data[offset];
            }
        }
    }
}
```

### 4.3 BSQ (Band Sequential) — 波段顺序

```
文件字节序列:
[band0_all_rows_all_cols][band1_all_rows_all_cols]...[bandN_all_rows_all_cols]
│←──────── 波段0的全部像素 ────────→│←── 波段1...──→│

文件偏移公式:
offset(row, col, band) = header_offset
    + sample_size * (band * nrows * ncols + row * ncols + col)

逻辑形状: (bands, rows, cols)  ← memmap 需要 transpose(1,2,0) 变成 (rows, cols, bands)
```

```cpp
// BSQ: 读取多个波段
template<typename T>
void read_bands_bsq(const EnviHeader& h, const T* mmap_data,
                    const std::vector<int>& bands, T* output) {
    size_t band_size = h.lines * h.samples;  // 每个波段的元素数
    for (size_t bi = 0; bi < bands.size(); bi++) {
        size_t band_offset = bands[bi] * band_size;
        for (int r = 0; r < h.lines; r++) {
            for (int c = 0; c < h.samples; c++) {
                size_t offset = band_offset + r * h.samples + c;
                output[(r * h.samples + c) * bands.size() + bi] = mmap_data[offset];
            }
        }
    }
}
```

### 4.4 Interleave 判断总结

```
        最快变化维度   次快       最慢
BIP     band         column     row      → offset = band + nbands*col + nbands*ncols*row
BIL     column       band       row      → offset = col + ncols*band + ncols*nbands*row
BSQ     column       row        band     → offset = col + ncols*row + ncols*nrows*band
```

---

## 5. get_rgb 流水线 — 波段提取 + 拉伸

### 5.1 算法流程

```
输入: (rows × cols × B) 的立方体, bands=[R_idx, G_idx, B_idx], stretch=(low, high)
默认: stretch=(0.02, 0.98)

Step 1: 波段提取
    R_plane = cube[:, :, bands[0]]
    G_plane = cube[:, :, bands[1]]
    B_plane = cube[:, :, bands[2]]

Step 2: 百分位拉伸参数计算
    # 对每个通道，计算累积直方图的 low*100% 和 high*100% 分位点
    R_low  = percentile(R_plane, low * 100)
    R_high = percentile(R_plane, high * 100)
    G_low  = percentile(G_plane, low * 100)
    G_high = percentile(G_plane, high * 100)
    B_low  = percentile(B_plane, low * 100)
    B_high = percentile(B_plane, high * 100)

    # 统一拉伸 (stretch_all=false, 默认):
    lower = min(R_low, G_low, B_low)
    upper = max(R_high, G_high, B_high)

    # 独立拉伸 (stretch_all=true):
    # 每个通道用自己的 lower/upper

Step 3: 线性拉伸 + clip 到 [0, 1]
    for each pixel p in each channel:
        p = clamp((p - lower) / (upper - lower), 0.0, 1.0)

返回: (rows × cols × 3) 的 float RGB 数组, 值域 [0, 1]
```

### 5.2 bands 默认选择策略

```
bands 为空时:
  ├─ metadata 中有 "default bands" → 用它
  ├─ B == 1 → bands = [0]
  ├─ B == 3 → bands = [0, 1, 2]    // 就当是 RGB 了
  └─ B > 3  → bands = [0, B/2, B-1] // 第一个、中间、最后一个
```

### 5.3 C++ 百分位计算 (无需排序全数组)

```cpp
// 用 nth_element 计算单个百分位点，O(n)
double percentile(float* data, size_t n, double pct) {
    size_t idx = static_cast<size_t>(n * pct / 100.0);
    if (idx >= n) idx = n - 1;
    std::nth_element(data, data + idx, data + n);
    return data[idx];
}

// 或者更精确的线性插值版本
double percentile_interp(const std::vector<float>& sorted, double pct) {
    double pos = pct / 100.0 * (sorted.size() - 1);
    size_t lo = static_cast<size_t>(pos);
    size_t hi = lo + 1;
    double frac = pos - lo;
    if (hi >= sorted.size()) return sorted[lo];
    return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}
```

### 5.4 C++ get_rgb 完整实现

```cpp
struct RgbResult {
    std::vector<float> data;  // rows * cols * 3, 值域 [0,1]
    int rows, cols;
    std::array<int, 3> bands_used;
};

RgbResult get_rgb(const float* cube, int rows, int cols, int num_bands,
                  const std::array<int, 3>& bands,
                  double stretch_low = 0.02, double stretch_high = 0.98,
                  bool stretch_all = false) {
    // Step 1: 提取 3 个波段
    int n_pixels = rows * cols;
    std::vector<float> R(n_pixels), G(n_pixels), B(n_pixels);
    for (int i = 0; i < n_pixels; i++) {
        R[i] = cube[i * num_bands + bands[0]];
        G[i] = cube[i * num_bands + bands[1]];
        B[i] = cube[i * num_bands + bands[2]];
    }

    // Step 2: 百分位计算 (需要排序)
    std::vector<float> R_sorted = R, G_sorted = G, B_sorted = B;
    std::sort(R_sorted.begin(), R_sorted.end());
    std::sort(G_sorted.begin(), G_sorted.end());
    std::sort(B_sorted.begin(), B_sorted.end());

    auto pct = [&](const std::vector<float>& sorted, double p) {
        return percentile_interp(sorted, p);
    };
    double R_low = pct(R_sorted, stretch_low * 100);
    double R_high = pct(R_sorted, stretch_high * 100);
    double G_low = pct(G_sorted, stretch_low * 100);
    double G_high = pct(G_sorted, stretch_high * 100);
    double B_low = pct(B_sorted, stretch_low * 100);
    double B_high = pct(B_sorted, stretch_high * 100);

    double R_lower, R_upper, G_lower, G_upper, B_lower, B_upper;
    if (stretch_all) {
        R_lower = R_low; R_upper = R_high;
        G_lower = G_low; G_upper = G_high;
        B_lower = B_low; B_upper = B_high;
    } else {
        double min_lower = std::min({R_low, G_low, B_low});
        double max_upper = std::max({R_high, G_high, B_high});
        R_lower = G_lower = B_lower = min_lower;
        R_upper = G_upper = B_upper = max_upper;
    }

    // Step 3: 线性拉伸 + clip
    std::vector<float> rgb(n_pixels * 3);
    auto stretch_channel = [&](const std::vector<float>& ch,
                                double lower, double upper, int ch_idx) {
        double span = upper - lower;
        for (int i = 0; i < n_pixels; i++) {
            float v = (span == 0) ? 0.0f :
                      static_cast<float>((ch[i] - lower) / span);
            rgb[i * 3 + ch_idx] = std::clamp(v, 0.0f, 1.0f);
        }
    };
    stretch_channel(R, R_lower, R_upper, 0);
    stretch_channel(G, G_lower, G_upper, 1);
    stretch_channel(B, B_lower, B_upper, 2);

    return {rgb, rows, cols, bands};
}
```

---

## 6. 推荐 C++ 架构

```
class HsiImage {
    EnviHeader header_;
    void* mmap_data_;         // mmap 映射的数据文件
    size_t data_size_;

public:
    // 工厂方法
    static HsiImage open(const std::string& hdr_path);

    // 属性
    int rows()    const { return header_.lines; }
    int cols()    const { return header_.samples; }
    int bands()   const { return header_.bands; }
    const std::string& interleave() const { return header_.interleave; }

    // 数据读取
    template<typename T>
    void read_bands(const std::vector<int>& band_indices, T* output) const;

    template<typename T>
    void read_band(int band, T* output) const;

    // 像素读取
    template<typename T>
    std::vector<T> read_pixel(int row, int col) const;

    // get_rgb
    RgbResult get_rgb(const std::array<int, 3>& bands,
                      double stretch_low = 0.02,
                      double stretch_high = 0.98,
                      bool stretch_all = false) const;
};
```

---

## 8. colorize 高层 API (Method 参数)

SPy 的 `get_rgb` 需要手动指定波段索引，MATLAB `colorize` 多了一层"意图驱动"抽象。

### 8.1 Method="rgb" — 真彩色

```
目标波长: R≈650nm, G≈550nm, B≈480nm
算法: 在 wavelengths[] 中找最接近目标波长的 3 个波段索引
```

```cpp
int find_nearest_band(const std::vector<double>& wavelengths, double target_nm) {
    int best = 0;
    double best_dist = std::abs(wavelengths[0] - target_nm);
    for (size_t i = 1; i < wavelengths.size(); i++) {
        double dist = std::abs(wavelengths[i] - target_nm);
        if (dist < best_dist) { best_dist = dist; best = i; }
    }
    return best;
}

// 调用
int r = find_nearest_band(wavelengths, 650.0);
int g = find_nearest_band(wavelengths, 550.0);
int b = find_nearest_band(wavelengths, 480.0);
auto rgb = get_rgb(cube, {r, g, b}, 0.02, false);
```

### 8.2 Method="cir" — 彩红外

```
NIR≈850nm→R, Red≈650nm→G, Green≈550nm→B
```

同 `find_nearest_band`，只是目标波长不同。

### 8.3 Method="falsecolored" — PCA 自动选择

SPy 的方案：取方差最大的 3 个波段 → 全挤在 NIR 区，视觉单调。
MATLAB 的方案：PCA 保证光谱正交性。

```
算法:
1. 采样 n 个像素 (n ≈ 2000-10000)
2. 去均值: X_centered = X - mean(X)
3. SVD: X_centered = U·S·V^T
4. 对 V^T 的前 3 行 (前3个主成分的载荷):
   - 取 |loading| 最大的波段索引 (跳过已选的)
5. 将 3 个索引按波长排序 → R/G/B
```

```cpp
// 简化版: 用 Eigen 做 SVD
#include <Eigen/SVD>

std::array<int, 3> select_bands_pca(
    const float* cube, int rows, int cols, int nbands,
    const std::vector<double>& wavelengths)
{
    int n_sample = std::min(10000, rows * cols);
    // 1. 随机采样
    Eigen::MatrixXf X(n_sample, nbands);
    // ... 填充 X ...

    // 2. 去均值
    Eigen::RowVectorXf mean = X.colwise().mean();
    X.rowwise() -= mean;

    // 3. SVD
    Eigen::BDCSVD<Eigen::MatrixXf> svd(X, Eigen::ComputeThinV);
    // V 的列 = 主成分方向 (载荷)

    // 4. 选前3个PC中载荷最大的波段
    std::set<int> used;
    std::vector<int> top3;
    for (int pc = 0; pc < 3 && top3.size() < 3; pc++) {
        auto loadings = svd.matrixV().col(pc).cwiseAbs();
        // 找 max loading 的波段
        int best_band = -1;
        float best_val = -1;
        for (int b = 0; b < nbands; b++) {
            if (used.count(b)) continue;
            if (loadings(b) > best_val) {
                best_val = loadings(b);
                best_band = b;
            }
        }
        if (best_band >= 0) { top3.push_back(best_band); used.insert(best_band); }
    }

    // 5. 按波长排序
    std::sort(top3.begin(), top3.end(),
              [&](int a, int b) { return wavelengths[a] < wavelengths[b]; });

    return {top3[0], top3[1], top3[2]};
}
```

### 8.4 完整 colorize 调用链

```cpp
RgbResult colorize(const HsiImage& img,
                   const std::string& method = "falsecolored",
                   bool contrast_stretching = false,
                   std::optional<std::array<int,3>> band = std::nullopt)
{
    std::array<int,3> bands;
    if (band.has_value()) {
        bands = {(*band)[0]-1, (*band)[1]-1, (*band)[2]-1};  // 1-based→0-based
    } else if (method == "rgb") {
        bands = {find_nearest(wl,650), find_nearest(wl,550), find_nearest(wl,480)};
    } else if (method == "cir") {
        bands = {find_nearest(wl,850), find_nearest(wl,650), find_nearest(wl,550)};
    } else {  // falsecolored
        bands = select_bands_pca(img);
    }
    return get_rgb(img, bands,
                   contrast_stretching ? 0.02 : 0.0,
                   contrast_stretching ? 0.98 : 1.0,
                   false);
}
```

---

## 9. NDVI 算法

```
NDVI = (NIR - Red) / (NIR + Red)

NIR: 默认 850nm 最近波段
Red: 默认 650nm 最近波段
输出: [-1, 1], NaN 表示分母为 0
```

```cpp
std::vector<float> ndvi(const HsiImage& img,
                         int nir_band = -1, int red_band = -1) {
    if (nir_band < 0) nir_band = find_nearest_band(img.wavelengths(), 850.0);
    if (red_band < 0) red_band = find_nearest_band(img.wavelengths(), 650.0);

    int n = img.rows() * img.cols();
    auto nir = img.read_band<float>(nir_band);
    auto red = img.read_band<float>(red_band);

    std::vector<float> result(n, NAN);
    for (int i = 0; i < n; i++) {
        float denom = nir[i] + red[i];
        if (denom != 0) result[i] = (nir[i] - red[i]) / denom;
    }
    return result;
}
```

---

## 10. 关键注意事项

| 要点 | 说明 |
|---|---|
| **字节序** | 如果 `byte_order != 本机字节序`，需要 swap。ENVI 默认 0 (little-endian) |
| **header_offset** | 数据文件前面可能有不属于图像的字节，要跳过 |
| **mmap vs seek+read** | mmap 性能大幅优于频繁 fseek (OS 页面缓存) |
| **百分位计算** | 对整个通道排序是 O(n log n)，可以用 histogram binning 做 O(n) 近似 |
| **浮点 vs 整型** | `get_rgb` 内部用 float/double 运算，返回 float [0,1] |
| **wavelength 元数据** | 可选字段，不是所有 ENVI 文件都有；若缺失，波段对应关系来自外部知识 |
| **default bands** | ENVI 头可以包含 `default bands = {3, 2, 1}` 指定默认 RGB 映射 |
