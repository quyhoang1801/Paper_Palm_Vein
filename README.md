# Palm ROI & SSD Training Pipeline

Dự án huấn luyện và đánh giá mô hình phát hiện/nhận diện ROI lòng bàn tay (palm) theo định dạng COCO, có sẵn script huấn luyện, đánh giá, tải dữ liệu từ Roboflow và API trích xuất ROI.

## 1) Cấu trúc thư mục

```
PMT_PAPER/
├─ configs/
│  └─ ssd_palm.yaml             # Cấu hình huấn luyện/đánh giá
├─ output/                       # Checkpoints, logs, kết quả mặc định
├─ output_newdataset/            # (tùy chọn) Kết quả khi dùng bộ dữ liệu mới
├─ plots/                        # Biểu đồ, hình minh họa kết quả
├─ runs/                         # Log huấn luyện (tensorboard, txt, …)
├─ scripts/
│  ├─ eval_ssd_palm.py           # Đánh giá mô hình trên tập val/test
│  ├─ extract_rois_api.py        # API trích xuất ROI từ ảnh/stream
│  └─ train_ssd_palm.py          # Huấn luyện mô hình
├─ src/
│  ├─ dataset/
│  │  ├─ coco_palm.py            # Dataset COCO (reader/transform)
│  │  └─ download_data.py        # Tải dữ liệu từ Roboflow
│  ├─ metrics/
│  │  ├─ coco_eval.py            # Tính mAP theo COCO
│  │  └─ prf.py                  # Precision/Recall/F1, confusion matrix
│  └─ utils/
│     └─ common.py               # Tiện ích chung: logging, seed, v.v.
├─ .env                          # (khuyến nghị) Biến môi trường cục bộ
├─ .gitignore
├─ README.md
└─ requirements.txt
```

## 2) Yêu cầu hệ thống

- Python 3.9+ (khuyến nghị 3.10/3.11)
- PyTorch + TorchVision phù hợp CUDA (nếu dùng GPU)
- Các gói khác trong `requirements.txt`


## 3) Cài đặt nhanh

```bash
# 1) Tạo môi trường ảo (ví dụ venv)
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# Linux/Mac:
source .venv/bin/activate

# 2) Cài PyTorch (chọn bản phù hợp từ pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # ví dụ CUDA 12.1

# 3) Cài các phụ thuộc khác
pip install -r requirements.txt
```

## 4) Dữ liệu

```bash
python src/dataset/download_data.py
```

Dữ liệu sẽ được tải về thư mục (ví dụ) `./roboflow_dl1/` hoặc theo cấu hình trong script.

### 4.1 Dữ liệu tự có (COCO)

- Chuẩn bị theo COCO: `annotations/{train,val,test}.json`, thư mục `images/`.
- Cập nhật đường dẫn tương ứng trong `configs/ssd_palm.yaml` (mục `data`).

## 5) Cấu hình (configs/ssd_palm.yaml)

Mở và cập nhật những mục tối thiểu sau (tên khóa có thể khác đôi chút tùy bản code):

- `data.train`, `data.val`, `data.test`: đường dẫn ảnh + annotations
- `model.num_classes`: số lớp (bao gồm background nếu mô hình yêu cầu hoặc không, tùy kiến trúc)
- `train.batch_size`, `train.epochs`, `train.lr`: siêu tham số cơ bản
- `output_dir`, `log_dir`: nơi lưu checkpoint, log

> Gợi ý: luôn **khóa seed** để tái lập kết quả (`seed: 42`) và bật lưu biểu đồ vào `./plots`.

## 6) Huấn luyện

```bash
# Đơn giản nhất: dùng toàn bộ tham số trong YAML
python scripts/train_ssd_palm.py --config configs/ssd_palm.yaml
```

- Checkpoints & log: `./output/` và `./runs/`
- Biểu đồ loss/metrics (nếu có): `./plots/`
- Nếu bạn tạo bộ dữ liệu mới, cấu hình để kết quả rơi vào `./output_newdataset/`

> Trong Windows PowerShell nếu đường dẫn có khoảng trắng, nhớ thêm ngoặc kép:  
> `python "scripts/train_ssd_palm.py" --config "configs/ssd_palm.yaml"`

## 7) Đánh giá

```bash
python scripts/eval_ssd_palm.py --config configs/ssd_palm.yaml --ckpt path/to/checkpoint.pt
```

- mAP (COCO) tính bởi `src/metrics/coco_eval.py`
- P/R/F1, ma trận nhầm lẫn bởi `src/metrics/prf.py`
- Kết quả tổng hợp và hình vẽ nằm trong `./plots/` (và/hoặc in ra console)


```bash
python scripts/extract_rois_api.py
```
API trả về toạ độ ROI/bounding boxes và (tùy phiên bản) ảnh đã vẽ khung.

## 8) Mô-đun chính

- `src/dataset/coco_palm.py`: Reader COCO + augment/transform.
- `src/metrics/coco_eval.py`: mAP theo chuẩn COCO.
- `src/metrics/prf.py`: Precision/Recall/F1, confusion matrix.
- `src/utils/common.py`: đặt seed, logger, lưu/đọc checkpoint, v.v.

## 9) Quy trình khuyến nghị

1. **Chuẩn bị dữ liệu** (Roboflow hoặc COCO của bạn).  
2. **Cập nhật `configs/ssd_palm.yaml`** (đường dẫn + num_classes + siêu tham số).  
3. **Huấn luyện** với `train_ssd_palm.py`.  
4. **Đánh giá** với `eval_ssd_palm.py`, xem mAP, P/R/F1 và biểu đồ trong `plots/`.  
5. **(Tùy chọn) Triển khai** API `extract_rois_api.py` để trích ROI tự động.
.
# Palm ROI & SSD Training Pipeline

Dự án huấn luyện và đánh giá mô hình phát hiện/nhận diện ROI lòng bàn tay (palm) theo định dạng COCO, có sẵn script huấn luyện, đánh giá, tải dữ liệu từ Roboflow và API trích xuất ROI.

## 1) Cấu trúc thư mục

```
PMT_PAPER/
├─ configs/
│  └─ ssd_palm.yaml             # Cấu hình huấn luyện/đánh giá
├─ output/                       # Checkpoints, logs, kết quả mặc định
├─ output_newdataset/            # (tùy chọn) Kết quả khi dùng bộ dữ liệu mới
├─ plots/                        # Biểu đồ, hình minh họa kết quả
├─ runs/                         # Log huấn luyện (tensorboard, txt, …)
├─ scripts/
│  ├─ eval_ssd_palm.py           # Đánh giá mô hình trên tập val/test
│  ├─ extract_rois_api.py        # API trích xuất ROI từ ảnh/stream
│  └─ train_ssd_palm.py          # Huấn luyện mô hình
├─ src/
│  ├─ dataset/
│  │  ├─ coco_palm.py            # Dataset COCO (reader/transform)
│  │  └─ download_data.py        # Tải dữ liệu từ Roboflow
│  ├─ metrics/
│  │  ├─ coco_eval.py            # Tính mAP theo COCO
│  │  └─ prf.py                  # Precision/Recall/F1, confusion matrix
│  └─ utils/
│     └─ common.py               # Tiện ích chung: logging, seed, v.v.
├─ .env                          # (khuyến nghị) Biến môi trường cục bộ
├─ .gitignore
├─ README.md
└─ requirements.txt
```

## 2) Yêu cầu hệ thống

- Python 3.9+ (khuyến nghị 3.10/3.11)
- PyTorch + TorchVision phù hợp CUDA (nếu dùng GPU)
- Các gói khác trong `requirements.txt`


## 3) Cài đặt nhanh

```bash
# 1) Tạo môi trường ảo (ví dụ venv)
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# Linux/Mac:
source .venv/bin/activate

# 2) Cài PyTorch (chọn bản phù hợp từ pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # ví dụ CUDA 12.1

# 3) Cài các phụ thuộc khác
pip install -r requirements.txt
```

## 4) Dữ liệu

```bash
python src/dataset/download_data.py
```

Dữ liệu sẽ được tải về thư mục (ví dụ) `./roboflow_dl1/` hoặc theo cấu hình trong script.

### 4.1 Dữ liệu tự có (COCO)

- Chuẩn bị theo COCO: `annotations/{train,val,test}.json`, thư mục `images/`.
- Cập nhật đường dẫn tương ứng trong `configs/ssd_palm.yaml` (mục `data`).

## 5) Cấu hình (configs/ssd_palm.yaml)

Mở và cập nhật những mục tối thiểu sau (tên khóa có thể khác đôi chút tùy bản code):

- `data.train`, `data.val`, `data.test`: đường dẫn ảnh + annotations
- `model.num_classes`: số lớp (bao gồm background nếu mô hình yêu cầu hoặc không, tùy kiến trúc)
- `train.batch_size`, `train.epochs`, `train.lr`: siêu tham số cơ bản
- `output_dir`, `log_dir`: nơi lưu checkpoint, log

> Gợi ý: luôn **khóa seed** để tái lập kết quả (`seed: 42`) và bật lưu biểu đồ vào `./plots`.

## 6) Huấn luyện

```bash
# Đơn giản nhất: dùng toàn bộ tham số trong YAML
python scripts/train_ssd_palm.py --config configs/ssd_palm.yaml
```

- Checkpoints & log: `./output/` và `./runs/`
- Biểu đồ loss/metrics (nếu có): `./plots/`
- Nếu bạn tạo bộ dữ liệu mới, cấu hình để kết quả rơi vào `./output_newdataset/`

> Trong Windows PowerShell nếu đường dẫn có khoảng trắng, nhớ thêm ngoặc kép:  
> `python "scripts/train_ssd_palm.py" --config "configs/ssd_palm.yaml"`

## 7) Đánh giá

```bash
python scripts/eval_ssd_palm.py --config configs/ssd_palm.yaml --ckpt path/to/checkpoint.pt
```

- mAP (COCO) tính bởi `src/metrics/coco_eval.py`
- P/R/F1, ma trận nhầm lẫn bởi `src/metrics/prf.py`
- Kết quả tổng hợp và hình vẽ nằm trong `./plots/` (và/hoặc in ra console)


```bash
python scripts/extract_rois_api.py
```
API trả về toạ độ ROI/bounding boxes và (tùy phiên bản) ảnh đã vẽ khung.

## 8) Mô-đun chính

- `src/dataset/coco_palm.py`: Reader COCO + augment/transform.
- `src/metrics/coco_eval.py`: mAP theo chuẩn COCO.
- `src/metrics/prf.py`: Precision/Recall/F1, confusion matrix.
- `src/utils/common.py`: đặt seed, logger, lưu/đọc checkpoint, v.v.

## 9) Quy trình khuyến nghị

1. **Chuẩn bị dữ liệu** (Roboflow hoặc COCO của bạn).  
2. **Cập nhật `configs/ssd_palm.yaml`** (đường dẫn + num_classes + siêu tham số).  
3. **Huấn luyện** với `train_ssd_palm.py`.  
4. **Đánh giá** với `eval_ssd_palm.py`, xem mAP, P/R/F1 và biểu đồ trong `plots/`.  
5. **(Tùy chọn) Triển khai** API `extract_rois_api.py` để trích ROI tự động.
.
