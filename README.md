# SLAP-GRO

**Storage Location Assignment Problem & Genetic Route Optimization**

Pipeline nghiên cứu và tái lập thực nghiệm giải thuật tối ưu hóa vị trí lưu kho kết hợp định tuyến nhặt hàng trong kho công nghiệp.

---

## 📌 Tổng quan dự án

Dự án hiện thực hóa kiến trúc pipeline nghiên cứu 2 chế độ:
- **Mode A (`paper_replication`)**: Tái lập theo bài báo gốc với kho tổng hợp 847 units (15,246 slots), 3 blocks (A, B, C), 25 aisles, 2,000 SKU và 50 picking waves.
- **Mode B (`real_data_replication`)**: Kiểm chuẩn trên bộ dữ liệu công nghiệp thực tế với 2,292 vị trí kho, 41,256 slots, 208 SKU, 9,707 waves (215,192 lượt nhặt) và 44 điểm hoa tiêu dẫn đường.

---

## 📖 Tài liệu kỹ thuật

Toàn bộ thông tin chi tiết về kiến trúc, cấu trúc module, luồng xử lý và đặc tả API được trình bày tại:
👉 **[TECHNICAL_DOCUMENTATION.md](file:///c:/Users/Admin/Desktop/NCKH/SLAP%20GRO/slap-gro/TECHNICAL_DOCUMENTATION.md)**

Nội dung tài liệu bao gồm:
1. **Architecture** – Kiến trúc hệ thống & pipeline 2 chế độ
2. **Project Structure** – Cấu trúc thư mục & phân tầng mã nguồn
3. **Modules & Functions** – Chi tiết các package `config`, `data`, `warehouse`, `routing`
4. **Data Flow** – Luồng xử lý dữ liệu từ file thô đến đánh giá tuyến đường
5. **API & Database** – Đặc tả hợp đồng dữ liệu, 7 quy tắc điều hòa (reconciliation) và bất biến toán học
6. **Setup & Configuration** – Hướng dẫn cài đặt môi trường và kernel Jupyter
7. **Testing & Deployment** – Kiểm thử tự động với Pytest (36/36 tests passed) và hướng dẫn thực thi

---

## 🚀 Khởi chạy nhanh

### 1. Kích hoạt môi trường và kiểm tra test suite:
```bash
.\venv\Scripts\activate
pytest -v
```

### 2. Chạy audit dữ liệu:
```bash
python scripts/audit_data.py
```

### 3. Xây dựng và kiểm tra mô hình kho:
```bash
python scripts/build_warehouse.py --mode all
```
