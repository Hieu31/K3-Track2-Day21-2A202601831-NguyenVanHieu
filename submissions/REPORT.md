# BÁO CÁO KẾT QUẢ TRIỂN KHAI MLOPS PIPELINE

**Họ và tên:** Nguyễn Văn Hiếu  
**Mã sinh viên:** 2A202601831  
**Môn học / Đề tài:** Quản lý và Tự động hóa Pipeline MLOps - Dự án Phân loại Chất lượng Rượu (Wine Quality)  

---

### 1. Bộ Siêu Tham Số Đã Chọn Và Phương Pháp Tìm Kiếm

Để tìm ra cấu hình mô hình tốt nhất, em đã xây dựng script **`find_best_params.py`** thực hiện **Randomized Search** tự động quét qua **500 thử nghiệm (trials)** trên không gian siêu tham số của thuật toán `RandomForestClassifier` (bao gồm `n_estimators`, `criterion`, `max_depth`, `max_features`, `class_weight`, `bootstrap`, `min_samples_split`, `ccp_alpha`, v.v.).

Dựa trên kết quả thử nghiệm và theo dõi trên MLflow, bộ siêu tham số tốt nhất được chọn và ghi tự động vào file `params.yaml` bao gồm:
* `n_estimators`: 50
* `criterion`: `'entropy'`
* `max_depth`: 50
* `min_samples_split`: 6
* `min_samples_leaf`: 1
* `max_features`: 5
* `class_weight`: `'balanced_subsample'`
* `bootstrap`: `False`

**Lý do lựa chọn:**
* Script `find_best_params.py` giúp xác định `criterion='entropy'` kết hợp `max_features=5` tối ưu hóa độ tương tin (Information Gain) trên các thuộc tính hóa học của rượu.
* Cấu hình `class_weight='balanced_subsample'` giúp xử lý hiệu quả tình trạng mất cân bằng phân bố chất lượng giữa các phân lớp.
* Bộ thông số này đạt Accuracy/F1 tối ưu nhất trên MLflow mà không bị hiện tượng overfitting nặng.

---

### 2. So Sánh Hiệu Năng Giữa 2 Lần Chạy Dữ Liệu (Số liệu chuẩn xác từ MLflow)

| Lần chạy | Số lượng mẫu | Accuracy | F1-Score (Weighted) | Đánh giá cổng Gate ($\ge 0.70$) |
| :--- | :---: | :---: | :---: | :---: |
| **Lần 1 (Phase 1)** | 2.998 mẫu | **0.6880** | **0.6869** | ❌ Chưa đạt ($0.6880 < 0.7000$) |
| **Lần 2 (Phase 1 + 2)** | 5.996 mẫu | **0.7260** | **0.7240** | ✅ **ĐẠT** ($0.7260 \ge 0.7000 \rightarrow$ Tự động Deploy) |

**Phân tích:** 
Khi mở rộng tập dữ liệu huấn luyện từ 2.998 mẫu lên 5.996 mẫu (tăng gấp 2 lần), Accuracy tăng thêm **+3.80%** (từ 0.6880 lên 0.7260) và F1-Score tăng **+3.71%** (từ 0.6869 lên 0.7240). Việc bổ sung dữ liệu giúp mô hình phân định ranh giới chất lượng rượu chính xác hơn, vượt qua ngưỡng Gate $0.70$ để kích hoạt tiến trình triển khai tự động lên VM.

---

### 3. Khó Khăn Gặp Phải Và Cách Giải Quyết

1. **Quản lý bộ nhớ RAM khi tìm siêu tham số bằng `find_best_params.py`**:
   * *Khó khăn:* Khởi tạo và huấn luyện 500 mô hình liên tục dễ dẫn đến tràn bộ nhớ RAM (Out of Memory).
   * *Giải quyết:* Tích hợp thư viện `gc` (Garbage Collector), giải phóng biến `del model` và gọi `gc.collect()` ở mỗi vòng lặp trial để thu hồi tài nguyên bộ nhớ ngay lập tức.

2. **Lỗi xác thực quyền GCP Storage trên GitHub Actions (`401 Invalid Credentials` / `Anonymous caller`)**:
   * *Khó khăn:* Secret trên GitHub đặt tên là `CLOUD_CREDENTIALS` khác tên cấu hình mặc định, đồng thời lệnh `echo` trong bash làm hỏng định dạng xuống dòng của file JSON chìa khóa.
   * *Giải quyết:* Cập nhật đúng tên Secret `${{ secrets.CLOUD_CREDENTIALS }}`, dùng script Python đọc dòng JSON chuẩn ra `/tmp/sa-key.json` và thiết lập biến môi trường `GOOGLE_APPLICATION_CREDENTIALS` cùng `credentialpath` cho DVC.

3. **Tiến trình Deploy bị hủy do FastAPI Server khởi động chậm**:
   * *Khó khăn:* Server FastAPI cần khoảng 8-10 giây để tải file `model.pkl` từ GCS về khi khởi động lại, khiến lệnh `curl` kiểm tra `/health` mặc định bị timeout ngắt kết nối.
   * *Giải quyết:* Viết vòng lặp thử lại (`retry loop`) 10 lần trong Job Deploy để chờ server hoàn tất quá trình tải mô hình và phản hồi `{"status":"ok"}` thành công.
