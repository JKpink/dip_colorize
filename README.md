# dip_colorize

多光谱/高光谱伪彩色化工具。实现 MATLAB Image Processing Toolbox 兼容的
`colorize` + `ndvi` API。

## 目录

```
├── src/                        # 核心库
│   ├── colorize.py             # MATLAB colorize 兼容 (PCA/rgb/cir)
│   ├── indices.py              # NDVI 植被指数
│   └── envi_io.py              # ENVI / GeoTIFF 读写
├── scripts/                    # 演示脚本
│   ├── demo_spy_view.py        # SPy 原生可视化
│   ├── demo_manual_get_rgb.py  # 手写 get_rgb 流水线 + 9 宫格
│   └── demo_ndvi.py            # Indian Pines NDVI 演示
├── docs/                       # 文档
│   └── spy_cpp_port_report.md  # C++ 移植技术报告
├── data/                       # 数据 (gitignored)
│   ├── landsat8/               # Landsat 8 OLI (7 波段, 30m)
│   ├── paviaU/                 # Pavia University ROSIS (103 波段)
│   ├── indian_pines/           # Indian Pines AVIRIS (200 波段)
│   ├── salinas_a/              # SalinasA AVIRIS (204 波段)
│   └── botswana/               # Botswana EO-1 Hyperion (145 波段)
├── output/                     # 生成图 (gitignored)
└── .venv/                      # Python 虚拟环境
```

## 用法

```bash
source .venv/bin/activate

# 手写 get_rgb 流水线 + 9 宫格参数对比
python scripts/demo_manual_get_rgb.py

# SPy 原生 Landsat 8 可视化
python scripts/demo_spy_view.py

# Indian Pines NDVI
python scripts/demo_ndvi.py

# 全数据集 3 Method 对比
python src/colorize.py
```

## API

```python
from src.colorize import colorize
from src.indices import ndvi

# --- colorize: MATLAB hypercube.colorize 兼容 ---
rgb, indices = colorize(hcube)                                    # 默认 falsecolored (PCA)
rgb, indices = colorize(hcube, Method="rgb")                      # 真彩色 650/550/480nm
rgb, indices = colorize(hcube, Method="cir")                      # 彩红外 NIR→R,Red→G,Grn→B
rgb, indices = colorize(hcube, band=[4, 3, 2])                    # 手动指定 (1-based, MATLAB 惯例)
rgb, indices = colorize(hcube, ContrastStretching=True)           # 2%-98% 百分位拉伸

# --- ndvi: MATLAB ndvi 兼容 ---
ndvi_img, nir_idx, red_idx = ndvi(hcube)                          # 自动 850/650nm
ndvi_img, nir_idx, red_idx = ndvi(hcube, nir_band=44, red_band=25)  # 手动
```

## MATLAB 参数对照

| MATLAB | Python | 说明 |
|---|---|---|
| `Method="falsecolored"` | `Method="falsecolored"` (默认) | PCA 选 3 个最不冗余波段 |
| `Method="rgb"` | `Method="rgb"` | 波长匹配 R/G/B |
| `Method="cir"` | `Method="cir"` | NIR→R, Red→G, Green→B |
| `ContrastStretching=false` | `ContrastStretching=False` (默认) | 线性 min-max |
| `ContrastStretching=true` | `ContrastStretching=True` | 2%-98% 百分位拉伸 |
| `band=[4,3,2]` | `band=[4,3,2]` (1-based) | 手动指定波段 |
| `ndvi(hcube)` | `ndvi(hcube)` | NIR≈850nm, Red≈650nm |

## `falsecolored` 算法

SPy 的简单 Top3 方差法会把波段全选在 NIR 区（如 B88/B90/B91 ≈ 800nm），
导致接近灰度图。我们采用 PCA 替代：

```
data → SVD → Vt[0:3] (前3个主成分载荷)
     → 每成分取 |loading| 最大且不重复的波段
     → 按波长排序 → R/G/B
```

这样保证选出的 3 个波段在光谱上正交（如 B39/B64/B91 → 590/696/809nm）。

## 数据集

| 名称 | 传感器 | 波段 | 用途 | 来源 |
|---|---|---|---|---|
| Landsat 8 | OLI | 7 | 多光谱 | USGS |
| PaviaU | ROSIS | 103 | 城市 | EHU |
| Indian Pines | AVIRIS | 200 | 农业 | EHU |
| SalinasA | AVIRIS | 204 | 农业 | EHU |
| Botswana | EO-1 Hyperion | 145 | 湿地 | EHU |

下载: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

## 依赖

Python 3.9+, spectral, rasterio, matplotlib, numpy, scipy, wxPython
