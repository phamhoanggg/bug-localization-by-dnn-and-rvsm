# Hướng dẫn chạy dự án

## 1. Tiền xử lý dữ liệu

### Bước 1: Chọn dataset

- Mở file `src/datasets.py`.
- Chỉnh giá trị `DATASET` thành tên repository bạn muốn chạy (ví dụ: `tomcat`, `aspectj`,...).

### Bước 2: Chạy tiền xử lý

- Mở và chạy file `src/preprocessing.py`.

### Bước 3: Kết quả thu được

Sau khi chạy xong, sẽ thu được các file kết quả:

| File                               | Mô tả                                                     |
| ---------------------------------- | --------------------------------------------------------- |
| `data/preprocessed_reports.pickle` | File pickle chứa bug reports sau tiền xử lý.              |
| `data/src.pickle`                  | File pickle chứa toàn bộ source code Java sau tiền xử lý. |
| `src/bug_report.csv`               | File CSV chuyển từ `preprocessed_reports.pickle`.         |
| `src/src_file.csv`                 | File CSV chuyển từ `src.pickle` (chỉ để tham khảo,        |

| **không sử dụng**). |
| ------------------- |

---

## 2. Trích xuất đặc trưng

### Bước 4: Chuẩn bị chạy trích xuất

- Mở file `src/feature_extraction.py`.
- Trong hàm `extract_features()`, tìm dòng:
  ```python
  bug_reports = csv2dict_for_br(bug_report_csv)[:10]
  ```
  - Dòng này sẽ lấy 10 bug reports đầu tiên để **chạy thử**.

### Bước 5: Chạy trích xuất đặc trưng

- Chạy file `src/feature_extraction.py`.

### Bước 6: Kiểm tra kết quả

- Kết quả sẽ lưu vào file `data/feature_{name}.csv`.
- **Tiêu chí kiểm tra:**\
  Với mỗi giá trị `bug_id`, nếu mỗi dòng có `match = 1` kèm theo 50 dòng `match = 0` thì kết quả là hợp lệ.

### Bước 7: Chạy toàn bộ dữ liệu

- Nếu kết quả thử nghiệm ổn, hãy **xóa **`` trong dòng:
  ```python
  bug_reports = csv2dict_for_br(bug_report_csv)[:10]
  ```
  - Sau đó, chạy lại `feature_extraction.py` để xử lý toàn bộ bug reports.

---

## 3. Tính toán top-k

### Bước 8: Chạy đánh giá

- Mở file `src/main.py` và chạy để tính toán chỉ số **top-k**.


