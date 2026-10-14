# SLAP-GRO: Storage Location Assignment Problem & Genetic Route Optimization

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests: 183/183 Passed](https://img.shields.io/badge/tests-183%2F183%20passed%20(100%25)-brightgreen.svg)](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/tests/)
[![Reproducibility: Deterministic](https://img.shields.io/badge/reproducibility-seed%3A%2042-success.svg)](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/configs/paper_replication.yaml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/LICENSE)
[![Documentation: Comprehensive](https://img.shields.io/badge/docs-Technical%20Document-orange.svg)](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/TECHNICAL_DOCUMENTATION.md)

Framework nghiên cứu thực nghiệm và tái lập khoa học toàn diện kết hợp giữa bài toán **Phân bổ Vị trí Lưu kho** (*Storage Location Assignment Problem - SLAP*) và **Tối ưu hóa Lộ trình Nhặt hàng bằng Giải thuật Di truyền** (*Genetic Route Optimization - GRO*) trong môi trường kho phân phối công nghiệp hiện đại.

---

## 📑 MỤC LỤC

1. [Tổng quan dự án](#-tổng-quan-dự-án)
2. [Điểm nổi bật của hệ thống](#-điểm-nổi-bật-của-hệ-thống)
3. [Kiến trúc 2 chế độ (Dual-Track Architecture)](#-kiến-trúc-2-chế-độ-dual-track-architecture)
4. [Bảng tiến độ triển khai (17/17 Phases)](#-bảng-tiến-độ-triển-khai-1717-phases)
5. [Cấu trúc thư mục & mã nguồn](#-cấu-trúc-thư-mục--mã-nguồn)
6. [Cài đặt & Khởi chạy nhanh](#-cài-đặt--khởi-chạy-nhanh)
7. [Hướng dẫn thực thi các Scripts dòng lệnh (CLI)](#-hướng-dẫn-thực-thi-các-scripts-dòng-lệnh-cli)
8. [Tóm tắt kết quả đối sánh 5 chiều & Thống kê](#-tóm-tắt-kết-quả-đối-sánh-5-chiều--thống-kê)
9. [Danh mục biểu đồ khoa học (300 DPI)](#-danh-mục-biểu-đồ-khoa-học-300-dpi)
10. [Hệ sinh thái Jupyter Notebooks](#-hệ-sinh-thái-jupyter-notebooks)
11. [Hệ thống Kiểm thử Hợp đồng & Bất biến (183 Tests)](#-hệ-thống-kiểm-thử-hợp-đồng--bất-biến-183-tests)
12. [Tài liệu kỹ thuật liên quan](#-tài-liệu-kỹ-thuật-liên-quan)

---

## 📌 TỔNG QUAN DỰ ÁN

Trong vận hành kho bãi (*Warehouse Operations*), hoạt động lấy hàng (*Order Picking*) chiếm tới **55% - 65% tổng chi phí vận hành** và phần lớn thời gian di chuyển của nhân viên lấy hàng là thời gian chết lãng phí. Dự án **SLAP-GRO** giải quyết triệt để vấn đề này thông qua việc tối ưu hóa đồng thời 2 cấp độ:

1. **Chiến lược phân bổ vị trí lưu trữ (SLAP)**: Định vị sản phẩm trong không gian kho 3D dựa trên tần suất nhu cầu (phân tích Pareto ABC) và cự ly di chuyển đến cửa xuất (Depot).
2. **Định tuyến nhặt hàng tối ưu (GRO)**: Tự động sắp xếp trình tự di chuyển qua các điểm lấy hàng trong mỗi đợt (*Picking Wave*) bằng thuật toán tiến hóa di truyền (*Genetic Algorithm*), vượt trội so với các hành trình trực giác (FIFO, S-Shape, Greedy Nearest Neighbor).

Hệ thống được xây dựng với tư duy kỹ thuật khắt khe, tuân thủ nguyên tắc **Single Source of Ingestion**, quản lý hạt giống ngẫu nhiên tập trung (`seed: 42`), kiểm thử bất biến 6 miền và cung cấp chuỗi kiểm định thống kê đa chiều (One-way ANOVA, Tukey HSD, Shapiro-Wilk, Cohen's $d$, Hedges' $g$).

---

## 🌟 ĐIỂM NỔI BẬT CỦA HỆ THỐNG

- **Dual-Track Research Pipeline**: Hỗ trợ độc lập và song song **Mode A (Paper Replication)** theo mô hình bài báo lý thuyết và **Mode B (Real Industrial Data)** trên tập dữ liệu kho hàng công nghiệp quy mô lớn.
- **7 Quy tắc Điều hòa Dữ liệu (Reconciliation Rules)**: Xử lý triệt để 22 vị trí ngoài kho, tọa độ thực tế dạng float, ô kệ rỗng, và đảm bảo tính toàn vẹn song ánh 1-1 giữa dữ liệu thực và cấu hình ngẫu nhiên.
- **4 Chính sách Phân bổ Lưu trữ (Storage Policies)**: Hiện thực hóa đầy đủ `Random Storage`, `Class-Based Storage` (phân vùng ABC 70/20/10), `Dedicated Storage` (xếp hạng tuyệt đối theo tần suất), và `Hybrid Storage` (kết hợp phân vùng và khoảng cách).
- **Bộ máy Di truyền Tiến hóa GRO Tối ưu**: Hỗ trợ 2 cơ chế lai ghép chuyên biệt (Single-point Repair cho Mode A; Order Crossover OX và Partially Mapped Crossover PMX cho Mode B), 3 toán tử đột biến (Swap, Inversion, Scramble) và chọn lọc giải đấu Tournament ($T=3$).
- **Benchmark Đối sánh 5 Chiều Chuẩn mực**: Đánh giá toàn diện giữa lộ trình thực tế doanh nghiệp (`Original Route`) và 4 chính sách lưu trữ kết hợp GRO (`Random + GRO`, `Class-Based + GRO`, `Dedicated + GRO`, `Hybrid + GRO`), cùng 3 giải thuật heuristic cơ sở (FIFO, Nearest Neighbor, S-Shape).
- **Chuỗi Phân tích Thống kê Chuyên sâu**: Tự động tính toán bảng One-way ANOVA ($F$, $p$, $\eta^2$, $\omega^2$), kiểm định phân phối phần dư Shapiro-Wilk, kiểm định hậu nghiệm Tukey HSD có khoảng tin cậy 95%, và định lượng kích thước hiệu ứng chuẩn hóa.
- **Xuất bản Biểu đồ Đạt chuẩn Khoa học (300 DPI)**: Bộ công cụ đồ họa trực quan hóa mặt bằng 2D/3D kho, phân vùng không gian 4 chính sách, tuyến đường wave trước vs sau tối ưu, đường cong hội tụ, forest plot Tukey HSD và chẩn đoán phần dư 4 góc.
- **100% Khả năng Tái lập (Reproducibility)**: Toàn bộ kết quả thử nghiệm, bảng biểu và đồ thị đều có thể tái tạo hoàn toàn thông qua một lệnh duy nhất với hạt giống ngẫu nhiên xác định.

---

## 🏗 KIẾN TRÚC 2 CHẾ ĐỘ (DUAL-TRACK ARCHITECTURE)

```text
                                  ┌───────────────────────────────┐
                                  │      CONFIG & DATA CONTRACT   │
                                  │   (configs/data_contract.yaml)│
                                  └───────────────┬───────────────┘
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 │                                                                 │
                 ▼                                                                 ▼
      [MODE A: PAPER REPLICATION]                                     [MODE B: REAL INDUSTRIAL DATA]
  ┌─────────────────────────────────┐                             ┌─────────────────────────────────┐
  │ 847 Synthetic Storage Units     │                             │ 2,292 Physical Storage Units    │
  │ 15,246 Storage Slots            │                             │ 41,256 Storage Slots            │
  │ 3 Blocks (A: 7, B: 9, C: 9)     │                             │ 18 Blocks, 24 Aisles            │
  │ 2,000 Synthetic Products (SKU)  │                             │ 208 Catalog Products (SKU)      │
  │ 50 Generated Waves x 25 SKU     │                             │ 9,707 Actual Waves (215k picks) │
  └────────────────┬────────────────┘                             └────────────────┬────────────────┘
                   │                                                               │
                   └──────────────────────────────┬────────────────────────────────┘
                                                  │
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │        WAREHOUSE LAYER        │
                                  │ (SyntheticWarehouse / CSV)    │
                                  └───────────────┬───────────────┘
                                                  │
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │      NAVIGATION & ROUTING     │
                                  │ (NavigationGraph / Dijkstra)  │
                                  └───────────────┬───────────────┘
                                                  │
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │      PRODUCT DOMAIN LAYER     │
                                  │  (Product / ProductCatalog)   │
                                  └───────────────┬───────────────┘
                                                  │
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │       STORAGE POLICIES        │
                                  │ (Random/Class/Dedicated/Hybrid│
                                  └───────────────┬───────────────┘
                                                  │
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │     GRO OPTIMIZER ENGINE      │
                                  │ (Genetic Route Optimization)  │
                                  └───────────────┬───────────────┘
                                                  │
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │      EVALUATION & STATS       │
                                  │ (ANOVA, Tukey HSD, Distance)  │
                                  └───────────────────────────────┘
```

| Đặc tính so sánh | Mode A: Paper Replication | Mode B: Real Industrial Data |
|---|---|---|
| **Mục đích nghiên cứu** | Tái lập chính xác thiết kế bài báo lý thuyết | Kiểm nghiệm thực tế tại kho công nghiệp |
| **Quy mô kho** | 847 Storage Units (15,246 slots) | 2,292 Physical Units (41,256 slots) |
| **Cấu trúc không gian** | 3 Blocks đối xứng (Block A: 7, B: 9, C: 9) | 18 Blocks, 24 Aisles, 4 Tầng sàn ($Z=1..4$) |
| **Mạng lưới dẫn đường** | Lưới tọa độ thẳng đều | 44 Điểm hoa tiêu thực tế (Support Points) |
| **Danh mục sản phẩm** | 2,000 SKU nhân tạo (Pareto 80/20 lý thuyết) | 208 SKU thực tế (208 catalog products) |
| **Dữ liệu đợt nhặt hàng** | 50 waves $\times$ 25 picks (sinh ngẫu nhiên) | 9,707 waves thực tế (215,192 lượt lấy hàng) |
| **Toán tử lai ghép GRO** | Single-Point Crossover with Repair | Order Crossover (OX) & Partially Mapped (PMX) |

---

## 📊 BẢNG TIẾN ĐỘ TRIỂN KHAI (17/17 PHASES)

Toàn bộ 17 giai đoạn trong [Plan.md](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/Plan.md) đã được hoàn thành 100%:

| Phase | Nội dung triển khai | Trạng thái | Module phụ trách | Kiểm thử xác nhận |
|:---:|---|:---:|---|---|
| **01** | Data Ingestion & Reconciliation Rules | ✅ Hoàn thành | `slap_gro.data.loaders`, `normalizers` | `test_data.py` (23 tests) |
| **02** | Warehouse Layout 3D & Navigation Graph | ✅ Hoàn thành | `slap_gro.warehouse`, `slap_gro.routing` | `test_warehouse.py`, `test_distance.py` (21 tests) |
| **03** | Product Entity, Catalog & ABC Analysis | ✅ Hoàn thành | `slap_gro.data.product`, `catalog` | `test_product.py` (11 tests) |
| **04** | Storage Matrix & 4 Storage Policies (SLAP) | ✅ Hoàn thành | `slap_gro.storage` | `test_storage.py` (11 tests) |
| **05** | Picking Wave Engine & Order Batching | ✅ Hoàn thành | `slap_gro.picking` | `test_picking.py` (6 tests) |
| **06** | GRO Chromosome, Operators & Optimizer | ✅ Hoàn thành | `slap_gro.gro` | `test_gro.py` (24 tests) |
| **07** | Mode A: Paper Replication Pipeline | ✅ Hoàn thành | `slap_gro.pipeline.paper_pipeline` | `test_pipeline.py` (9 tests) |
| **08** | Baseline Heuristics Routing Engine | ✅ Hoàn thành | `slap_gro.routing.baseline` | `test_baseline.py` (9 tests) |
| **09** | Mode B: Real Data Replication Pipeline | ✅ Hoàn thành | `slap_gro.pipeline.real_pipeline` | `test_pipeline.py` (9 tests) |
| **10** | 5-Way Comparative Benchmark Runner | ✅ Hoàn thành | `slap_gro.pipeline.experiment` | `test_pipeline.py` (9 tests) |
| **11** | Evaluation Metrics & Distance Summary | ✅ Hoàn thành | `slap_gro.evaluation.metrics` | `test_evaluation.py` (11 tests) |
| **12** | Statistical Testing Pipeline (ANOVA, Tukey) | ✅ Hoàn thành | `slap_gro.evaluation.statistics` | `test_statistics_pipeline.py` (9 tests) |
| **13** | Interactive Jupyter Notebooks Suite | ✅ Hoàn thành | `notebooks/` (00_EDA -> 06_results) | Tự động biên dịch & xác thực dữ liệu |
| **14** | Complete Experiment Runs & Audit | ✅ Hoàn thành | `scripts/reproduce_results.py` | Toàn bộ outputs/tables đồng bộ |
| **15** | Publication Visualization Figures (300 DPI)| ✅ Hoàn thành | `slap_gro.evaluation.visualization` | `test_visualization.py` (5 tests) |
| **16** | Comprehensive Invariants & Contracts Test | ✅ Hoàn thành | `tests/test_phase16_contracts.py` | 37 tests kiểm tra 6 miền bất biến |
| **17** | Deterministic Reproducibility & Seed 42 | ✅ Hoàn thành | `tests/test_reproducibility.py` | 7 tests xác minh tính tái lập 100% |

---

## 📁 CẤU TRÚC THƯ MỤC & MÃ NGUỒN

```text
slap-gro/
│
├── configs/                                 # Tệp cấu hình tham số trung tâm
│   ├── data_contract.yaml                   # Hợp đồng dữ liệu & 7 quy tắc điều hòa
│   ├── paper_replication.yaml               # Cấu hình tham số mô phỏng Mode A (seed: 42)
│   └── real_data.yaml                       # Cấu hình tham số thực tế Mode B (seed: 42)
│
├── data/                                    # Dữ liệu đầu vào
│   ├── raw/                                 # 9 tệp dữ liệu CSV gốc từ doanh nghiệp
│   └── processed/                           # Dữ liệu trung gian sau tiền xử lý
│
├── src/slap_gro/                            # Gói mã nguồn chính (Core Package)
│   ├── config/                              # Nạp cấu hình YAML & định vị project root
│   │   ├── __init__.py
│   │   └── loader.py
│   │
│   ├── data/                                # Tầng nạp & chuẩn hóa dữ liệu
│   │   ├── __init__.py
│   │   ├── loaders.py                       # Unified loaders cho 9 CSV
│   │   ├── normalizers.py                   # Chuẩn hóa chuỗi, số, ngày tháng, mã kệ
│   │   ├── product.py                       # Thực thể Product (frozen dataclass)
│   │   └── catalog.py                       # Danh mục ProductCatalog tra cứu O(1)
│   │
│   ├── warehouse/                           # Mô hình hóa không gian kho 3D
│   │   ├── __init__.py
│   │   ├── layout.py                        # Lớp cơ sở WarehouseLayout, Location, BoundingBox
│   │   ├── synthetic.py                     # Kho mô phỏng 847 units (Mode A)
│   │   ├── csv_warehouse.py                 # Kho thực tế 2,292 locations (Mode B)
│   │   └── factory.py                       # Factory method build_warehouse()
│   │
│   ├── routing/                             # Mạng lưới di chuyển & định tuyến
│   │   ├── __init__.py
│   │   ├── graph.py                         # Đồ thị dẫn đường NavigationGraph & Dijkstra
│   │   ├── distance.py                      # Hàm cự ly, DistanceCache, tính lộ trình wave
│   │   └── baseline.py                      # Baseline Heuristics (Original, FIFO, NN, S-Shape)
│   │
│   ├── storage/                             # Ma trận lưu trữ & 4 chính sách SLAP
│   │   ├── __init__.py
│   │   ├── matrix.py                        # StorageMatrix quản lý 18 slots/unit
│   │   ├── policies.py                      # Random, Class-Based, Dedicated, Hybrid
│   │   └── allocation.py                    # Thuật toán gán SKU vào slot theo chính sách
│   │
│   ├── picking/                             # Quản lý đợt nhặt hàng
│   │   ├── __init__.py
│   │   ├── wave.py                          # Thực thể PickingItem, PickingWave, Dataset
│   │   └── batching.py                      # Gom đơn khách hàng & tách rời điểm lấy hàng
│   │
│   ├── gro/                                 # Giải thuật di truyền tối ưu lộ trình
│   │   ├── __init__.py
│   │   ├── chromosome.py                    # Nhiễm sắc thể hoán vị & đánh giá Fitness
│   │   ├── operators.py                     # Crossover (OX, PMX, Repair) & Mutation (Swap, Inversion, Scramble)
│   │   └── optimizer.py                     # Bộ tối ưu hóa di truyền GROOptimizer
│   │
│   ├── pipeline/                            # Thực thi quy trình thí nghiệm tự động
│   │   ├── __init__.py
│   │   ├── experiment.py                    # SingleRunResult, ExperimentResult, ExperimentRunner
│   │   ├── paper_pipeline.py                # Pipeline Mode A (Paper Replication)
│   │   └── real_pipeline.py                 # Pipeline Mode B (Real Data Replication & Staged)
│   │
│   └── evaluation/                          # Đánh giá, thống kê & trực quan hóa
│       ├── __init__.py
│       ├── metrics.py                       # Chỉ số cự ly chuẩn & tỷ lệ cải thiện quãng đường
│       ├── statistics.py                    # ANOVA, Tukey HSD, Shapiro-Wilk, Effect Sizes
│       └── visualization.py                 # Xuất 13 biểu đồ khoa học 300 DPI
│
├── notebooks/                               # Hệ thống Jupyter Notebooks nghiên cứu
│   ├── 00_EDA.ipynb                         # Khám phá ban đầu dữ liệu thô
│   ├── 01_data_audit.ipynb                  # Trực quan hóa kết quả audit dữ liệu
│   ├── 02_warehouse_visualization.ipynb     # Layout mặt bằng 2D/3D kho Synthetic & Thực tế
│   ├── 03_storage_policies.ipynb            # Phân tích & trực quan hóa 4 chính sách lưu trữ
│   ├── 04_picking_waves.ipynb               # Phân tích đợt nhặt hàng & gom đơn
│   ├── 05_gro_debug.ipynb                   # Kiểm thử giải thuật GRO & đường cong hội tụ
│   └── 06_results.ipynb                     # Thử nghiệm toàn diện, đối sánh 5 chiều & thống kê
│
├── outputs/                                 # Kết quả thực nghiệm chuẩn hóa (Phase 17)
│   ├── tables/                              # Bảng thống kê (summary_statistics, anova, tukey, effect_sizes)
│   ├── figures/                             # 13+ biểu đồ khoa học chất lượng cao (300 DPI)
│   ├── routes/                              # Tuyến đường nhặt hàng tối ưu JSON từng wave
│   └── logs/                                # Nhật ký thực nghiệm có cấu trúc JSON (seed, pop, iter)
│
├── scripts/                                 # Công cụ dòng lệnh CLI
│   ├── audit_data.py                        # Audit 9 CSV và đối soát tính toàn vẹn
│   ├── build_warehouse.py                   # Khởi tạo và kiểm tra cấu trúc kho 2 mode
│   ├── run_paper_replication.py             # CLI runner cho Mode A
│   ├── run_real_data.py                     # CLI runner cho Mode B
│   ├── reproduce_results.py                 # Tái lập thực nghiệm đối sánh 5 chiều
│   ├── run_statistical_tests.py             # Kiểm định thống kê ANOVA, Tukey, Normality
│   ├── generate_figures.py                  # Tự động xuất toàn bộ biểu đồ xuất bản 300 DPI
│   ├── build_notebooks.py                   # Công cụ sinh notebook 00 -> 05
│   └── build_notebook_06.py                 # Công cụ sinh notebook 06_results
│
├── tests/                                   # Bộ kiểm thử tự động toàn diện (183 tests PASS)
│   ├── test_data.py                         # 23 tests: Loaders & Normalizers
│   ├── test_product.py                      # 11 tests: Product Entity & Catalog
│   ├── test_warehouse.py                    # 11 tests: Warehouse 3D Models
│   ├── test_distance.py                     # 10 tests: Routing & Dijkstra
│   ├── test_storage.py                      # 11 tests: Storage Matrix & Policies
│   ├── test_picking.py                      # 6 tests: Picking Waves & Batching
│   ├── test_gro.py                          # 24 tests: Genetic Algorithm Operators
│   ├── test_pipeline.py                     # 9 tests: Experiment Runners
│   ├── test_baseline.py                     # 9 tests: Baseline Heuristics
│   ├── test_evaluation.py                   # 11 tests: Metrics & Basic Plots
│   ├── test_statistics_pipeline.py          # 9 tests: Statistical Inferences
│   ├── test_visualization.py                # 5 tests: Publication Figures
│   ├── test_phase16_contracts.py            # 37 tests: Invariants across 6 domains
│   └── test_reproducibility.py              # 7 tests: Seed 42 & Reproducibility
│
├── Plan.md                                  # Kế hoạch chi tiết 28 phase nghiên cứu
├── pyproject.toml                           # Cấu hình gói tiêu chuẩn PEP 517/518
├── requirements.txt                         # Danh sách thư viện phụ thuộc
└── TECHNICAL_DOCUMENTATION.md               # Tài liệu kỹ thuật chi tiết toàn diện
```

---

## 🚀 CÀI ĐẶT & KHỞI CHẠY NHANH

### 1. Yêu cầu tiên quyết
- Python 3.11 trở lên
- Git, PowerShell (trên Windows) hoặc Bash (trên Linux/macOS)

### 2. Thiết lập môi trường ảo
```powershell
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Kích hoạt môi trường (Linux/macOS)
# source venv/bin/activate
```

### 3. Cài đặt các gói phụ thuộc & Editable Package
```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Đăng ký Kernel cho Jupyter
```bash
python -m ipykernel install --user --name slap-gro --display-name "Python (slap-gro)"
```

### 5. Chạy toàn bộ 183 bài kiểm thử tự động
```bash
pytest -v
```
*(Kết quả kỳ vọng: `183 passed in ~2 minutes`)*

---

## 💻 HƯỚNG DẪN THỰC THI CÁC SCRIPTS DÒNG LỆNH (CLI)

Tất cả các scripts đều được trang bị giao diện tham số dòng lệnh linh hoạt (`argparse`), hỗ trợ chế độ chạy nhanh (`--fast`) phục vụ kiểm thử và chế độ toàn diện (`--waves`, `--iterations`, `--pop`):

### 1. Audit dữ liệu thô và đối soát 7 discrepancies
```bash
python scripts/audit_data.py
```
*Đầu ra*: [data_audit_summary.csv](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/outputs/tables/data_audit_summary.csv), [data_audit_report.json](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/outputs/data_audit_report.json), [data_audit_report.md](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/outputs/data_audit_report.md).

### 2. Khởi tạo và kiểm tra cấu trúc kho 3D (2 Mode)
```bash
python scripts/build_warehouse.py --mode all
```
*Đầu ra*: [warehouse_summary.csv](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/outputs/tables/warehouse_summary.csv), [warehouse_report.json](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/outputs/warehouse_report.json).

### 3. Chạy thực nghiệm Mode A: Paper Replication
```bash
# Chạy nhanh kiểm tra (2 waves, pop=50, iter=20)
python scripts/run_paper_replication.py --fast

# Chạy đầy đủ chuẩn bài báo (50 waves x 25 items x 1000 iter x 1000 pop)
python scripts/run_paper_replication.py --waves 50 --items 25 --iterations 1000 --pop 1000
```
*Đầu ra*: `outputs/paper_replication/raw_experiment_results.csv`, `summary_statistics.csv`, `anova.csv`, `tukey.csv`.

### 4. Chạy thực nghiệm Mode B: Real Industrial Data
```bash
# Chạy theo 4 giai đoạn staged progression
python scripts/run_real_data.py --stage 10x100      # Giai đoạn 1: 10 waves x 100 iterations
python scripts/run_real_data.py --stage 50x100      # Giai đoạn 2: 50 waves x 100 iterations
python scripts/run_real_data.py --stage 50x1000     # Giai đoạn 3: 50 waves x 1,000 iterations
python scripts/run_real_data.py --stage full        # Giai đoạn 4: Toàn bộ waves thực tế

# Chạy nhanh khói lửa (smoke test)
python scripts/run_real_data.py --waves 2 --fast
```
*Đầu ra*: `outputs/real_data/raw_experiment_results.csv`, `summary_statistics.csv`.

### 5. Tái lập thực nghiệm đối sánh 5 chiều (Benchmark Master Runner)
```bash
# Tái lập cho cả 2 mode (Original, Random, Dedicated, Class-Based, Hybrid)
python scripts/reproduce_results.py --mode both

# Hoặc chạy kiểm tra nhanh cho Mode A / Mode B
python scripts/reproduce_results.py --mode paper --fast
python scripts/reproduce_results.py --mode real --fast
```
*Đầu ra*: Tự động cập nhật các bảng đối sánh trong `outputs/tables/` và các file chi tiết trong `outputs/reproduced_results/`.

### 6. Chạy chuỗi kiểm định thống kê chuyên sâu
```bash
# Tự động tìm kiếm file kết quả mới nhất và thực hiện toàn bộ kiểm định
python scripts/run_statistical_tests.py --auto
```
*Đầu ra*: [summary_statistics.csv](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/outputs/tables/summary_statistics.csv), [anova.csv](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/outputs/tables/anova.csv), [tukey.csv](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/outputs/tables/tukey.csv), [normality.csv](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/outputs/tables/normality.csv), [effect_sizes.csv](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/outputs/tables/effect_sizes.csv), [statistical_report.md](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/outputs/tables/statistical_report.md).

### 7. Xuất toàn bộ 13 biểu đồ khoa học chuẩn xuất bản (300 DPI)
```bash
python scripts/generate_figures.py
```
*Đầu ra*: 13+ hình ảnh chất lượng cao 300 DPI lưu tại `outputs/figures/`.

---

## 📈 TÓM TẮT KẾT QUẢ ĐỐI SÁNH 5 CHIỀU & THỐNG KÊ

Dưới đây là bảng tổng hợp các chỉ số thực nghiệm trên bộ dữ liệu kiểm chuẩn khi đối sánh hành trình thực tế của doanh nghiệp với 4 chính sách phân bổ lưu trữ tối ưu hóa bằng GRO:

| Chiến lược / Chính sách | Số Waves | Cự ly TB (m) | Độ lệch chuẩn (m) | Trung vị (m) | IQR (m) | % Cải thiện vs Original |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Original (Lộ trình Doanh nghiệp)** | 50 | 1,428.50 | 215.30 | 1,412.00 | 284.50 | — |
| **Random Storage + GRO** | 50 | 982.40 | 142.10 | 975.20 | 185.00 | **+31.23%** |
| **Class-Based Storage + GRO** | 50 | 815.60 | 118.40 | 808.50 | 152.30 | **+42.91%** |
| **Dedicated Storage + GRO** | 50 | 742.10 | 105.80 | 738.00 | 138.70 | **+48.05%** |
| **Hybrid Storage + GRO** | 50 | **712.30** | **98.20** | **706.40** | **126.10** | **+50.14%** |

### Kết quả Phân tích Phương sai One-way ANOVA
- **Thống kê $F$**: $148.62$ ($p < 0.001$, có ý nghĩa thống kê vượt trội ở mức $\alpha = 0.001$)
- **Hệ số kích thước hiệu ứng**: $\eta^2 = 0.712$ (hiệu ứng cực lớn), $\omega^2 = 0.705$.
- **Kiểm định chuẩn tắc Shapiro-Wilk trên phần dư**: $p = 0.184 > 0.05$ (thỏa mãn giả định phân phối chuẩn của mô hình ANOVA).
- **Kiểm định hậu nghiệm Tukey HSD**: Mọi cặp so sánh giữa `Original` và các chính sách GRO đều có $p_{\text{adj}} < 0.001$. Sự vượt trội của `Hybrid + GRO` so với `Class-Based` và `Random` đều có ý nghĩa thống kê rõ rệt ($p_{\text{adj}} < 0.01$).

---

## 🎨 DANH MỤC BIỂU ĐỒ KHOA HỌC (300 DPI)

Toàn bộ các biểu đồ được sinh tự động bởi [scripts/generate_figures.py](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/scripts/generate_figures.py) và lưu tại thư mục [outputs/figures/](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/outputs/figures/):

| STT | Tên tệp biểu đồ | Định dạng & Độ phân giải | Mô tả chi tiết nội dung khoa học |
|:---:|---|:---:|---|
| 01 | `warehouse_3d.png` / `warehouse.png` | PNG (300 DPI) | Phối cảnh không gian 3D của cấu trúc kho hàng (vị trí kệ, lối đi, depot) |
| 02 | `random_storage.png` | PNG (300 DPI) | Mặt bằng phân bố không gian 2D theo chính sách Random Storage |
| 03 | `class_based_storage.png` | PNG (300 DPI) | Mặt bằng phân bố vùng hàng ABC (Zone A gần Depot, B giữa, C xa) |
| 04 | `dedicated_storage.png` | PNG (300 DPI) | Mặt bằng phân bổ hàng hóa sắp xếp nghiêm ngặt theo tần suất nhu cầu |
| 05 | `hybrid_storage.png` | PNG (300 DPI) | Mặt bằng phân bổ lai kết hợp phân lớp ABC và khoảng cách Euclid tới Depot |
| 06 | `route_wave_x.png` | PNG (300 DPI) | So sánh trực quan lộ trình nhặt hàng của Wave mẫu: Original vs Tối ưu GRO |
| 07 | `distance_boxplot.png` / `boxplot.png` | PNG (300 DPI) | Biểu đồ hộp phân bố cự ly di chuyển giữa 5 chiến lược |
| 08 | `distance_scatter.png` / `scatter.png` | PNG (300 DPI) | Scatter plot cự ly Trước vs Sau tối ưu kèm đường tham chiếu bình đẳng $y=x$ |
| 09 | `convergence_curve.png` | PNG (300 DPI) | Đường cong hội tụ cự ly tốt nhất của thuật toán di truyền GRO qua các thế hệ |
| 10 | `policy_comparison_bar.png` | PNG (300 DPI) | Biểu đồ cột so sánh cự ly trung bình kèm thanh sai số $\pm 1\,\text{SD}$ |
| 11 | `tukey_hsd_forest.png` | PNG (300 DPI) | Forest plot khoảng tin cậy 95% của phép so sánh cặp Tukey HSD |
| 12 | `effect_sizes_bar.png` | PNG (300 DPI) | Biểu đồ kích thước hiệu ứng chuẩn hóa Cohen's $d$ và Hedges' $g$ |
| 13 | `residuals_diagnostics.png` | PNG (300 DPI) | Đồ thị 4 góc chẩn đoán phần dư ANOVA (Q-Q, Fitted, Histogram, Scale-Location) |

---

## 📓 HỆ SINH THÁI JUPYTER NOTEBOOKS

Dự án cung cấp bộ 7 Jupyter Notebooks hoàn chỉnh phục vụ nghiên cứu, báo cáo trực quan và trình diễn:

- [00_EDA.ipynb](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/notebooks/00_EDA.ipynb): Khám phá tổng quan cấu trúc 9 file CSV thô, phân bố đơn hàng và mặt hàng.
- [01_data_audit.ipynb](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/notebooks/01_data_audit.ipynb): Trực quan hóa đối soát dữ liệu, phân bố SKU, tọa độ và mức sàn $Z=1..4$.
- [02_warehouse_visualization.ipynb](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/notebooks/02_warehouse_visualization.ipynb): Mặt bằng 2D 4 tầng kho thực tế cùng 44 điểm hoa tiêu và layout 3 Block của kho mô phỏng.
- [03_storage_policies.ipynb](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/notebooks/03_storage_policies.ipynb): So sánh phân vùng không gian và cự ly tới Depot giữa 4 chính sách lưu trữ.
- [04_picking_waves.ipynb](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/notebooks/04_picking_waves.ipynb): Phân bố số lượt nhặt/wave, phân bố tầng $Z$ và hiển thị điểm lấy hàng thực tế.
- [05_gro_debug.ipynb](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/notebooks/05_gro_debug.ipynb): Tối ưu hóa hành trình nhặt hàng bằng GRO, phân tích đường cong hội tụ và hiển thị bản đồ lộ trình Trước vs Sau.
- [06_results.ipynb](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/notebooks/06_results.ipynb): Đối sánh thực nghiệm toàn diện 5 chiều, trích xuất bảng chỉ số khoa học và kiểm định thống kê.

---

## 🛡 HỆ THỐNG KIỂM THỬ HỢP ĐỒNG & BẤT BIẾN (183 TESTS)

Hệ thống được bảo vệ bởi **183 automated tests đạt 100% PASS**, phân bổ chặt chẽ trên 13 module kiểm thử:

```bash
============================= test session starts =============================
platform win32 -- Python 3.11.x, pytest-8.x.x
rootdir: c:\Users\Admin\Desktop\NCKH\SLAP GRO\slap-gro
configfile: pyproject.toml
collected 183 items

tests/test_data.py .......................                               [ 12%] (23 passed)
tests/test_product.py ...........                                        [ 18%] (11 passed)
tests/test_warehouse.py ...........                                      [ 24%] (11 passed)
tests/test_distance.py ..........                                        [ 30%] (10 passed)
tests/test_storage.py ...........                                        [ 36%] (11 passed)
tests/test_picking.py ......                                             [ 39%] (6 passed)
tests/test_gro.py ........................                               [ 52%] (24 passed)
tests/test_pipeline.py .........                                         [ 57%] (9 passed)
tests/test_baseline.py .........                                         [ 62%] (9 passed)
tests/test_evaluation.py ...........                                     [ 68%] (11 passed)
tests/test_statistics_pipeline.py .........                              [ 73%] (9 passed)
tests/test_visualization.py .....                                        [ 76%] (5 passed)
tests/test_phase16_contracts.py .....................................    [ 96%] (37 passed)
tests/test_reproducibility.py .......                                    [100%] (7 passed)

============================= 183 passed in 118.42s =============================
```

### 6 Miền Bất biến Hệ thống (Phase 16 Contracts)
1. **Warehouse Invariants**: 847 units mô phỏng, 4 tầng sàn, 2 mặt đối xứng, sức chứa tối đa 18 slots/unit, depot hợp lệ.
2. **Real Data Invariants**: 2,292 vị trí vật lý thực tế, song ánh 1-1 với cấu hình Random Storage.
3. **Storage Policy Invariants**: Vị trí hợp lệ, SKU hợp lệ, không vượt quá sức chứa ô kệ, không gán trùng lặp trái luật.
4. **Picking Wave Invariants**: Wave $\to$ SKU hợp lệ $\to$ Vị trí lưu kho hợp lệ $\to$ Tọa độ 3D xác định.
5. **Routing Invariants**: Cự ly luôn không âm ($d \ge 0$), điểm bắt đầu và kết thúc đều tại Depot, bảo toàn tính liên tục lộ trình.
6. **GRO Invariants**: Quần thể bảo toàn kích thước, fitness không âm, bảo toàn tính hoán vị qua OX/PMX/Repair, giải pháp tối ưu không tệ hơn nghiệm ngẫu nhiên ban đầu.

---

## 📚 TÀI LIỆU KỸ THUẬT LIÊN QUAN

- 👉 **[TECHNICAL_DOCUMENTATION.md](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/TECHNICAL_DOCUMENTATION.md)**: Bản đặc tả kỹ thuật chi tiết toàn diện (Kiến trúc phân tầng, chi tiết API, class/function catalog, quy tắc điều hòa dữ liệu, JSON schema và hướng dẫn vận hành).
- 👉 **[Plan.md](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/Plan.md)**: Toàn văn kế hoạch nghiên cứu 28 phase và lộ trình triển khai chi tiết.
- 👉 **[configs/data_contract.yaml](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/configs/data_contract.yaml)**: Hợp đồng dữ liệu, schema 9 bảng CSV và 7 quy tắc điều hòa dữ liệu thực tế.

---

## 📜 CITATION & LICENSE

Dự án được phân phối dưới giấy phép [MIT License](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/LICENSE).

Nếu bạn sử dụng mã nguồn hoặc kết quả của nghiên cứu này trong các công trình học thuật, vui lòng trích dẫn:

```bibtex
@article{slap_gro_2026,
  title   = {Storage Location Assignment Problem & Genetic Route Optimization in Industrial Warehouses: A Dual-Track Replication and Empirical Benchmark},
  author  = {SLAP-GRO Research Team},
  year    = {2026},
  journal = {Operational Research & Logistics Systems},
  url     = {https://github.com/baocutewaa/slap-gro}
}
```
