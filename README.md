# PMT_PAPER — Palm Vein Verification (ROI Detection + Feature Extraction)

Dự án phục vụ bài báo/đề tài **xác thực tĩnh mạch lòng bàn tay (Palm Vein)** theo pipeline 2 giai đoạn:

1) **Detection_ROI**: Phát hiện vùng ROI (bàn tay/ROI tĩnh mạch) từ ảnh đầu vào (RAW 8-bit, 640×480).  
2) **Feature_Extraction_ROI**: Cắt ROI → chuẩn hoá kích thước → huấn luyện/ suy luận mô hình **EfficientNet-B1 (1 channel)** để **phân loại theo user** hoặc trích xuất đặc trưng phục vụ xác thực.

> Mục tiêu thực tế: đưa pipeline chạy được từ dữ liệu RAW → ROI → model xác thực/nhận dạng.

---

## 1. Cấu trúc thư mục

```text
PMT_PAPER/
├─ Detection_ROI/
│  ├─ configs/
│  │  ├─ fasterrcnn_resnet50_palm.yaml
│  │  └─ ssd_palm.yaml
│  ├─ output/
│  │  ├─ output_faster/
│  │  └─ output_ssd/
│  ├─ output_newdataset/
│  │  ├─ checkpoints/
│  │  ├─ logs/
│  │  ├─ plots/
│  │  └─ pred_map_valid.json
│  ├─ runs/
│  ├─ scripts/
│  │  ├─ faster/
│  │  └─ ssd/
│  └─ src/
│
├─ Feature_Extraction_ROI/
│  ├─ configs/
│  │  ├─ effb1_roi240_train.json
│  │  └─ effb1_roi240_predict.json
│  ├─ models/
│  │  ├─ class_to_idx.json
│  │  └─ efficientnet_b1_1ch_roi240_best.pth
│  ├─ outputs/
│  └─ Scripts/
│     ├─ train.py
│     ├─ predict.py
│     └─ predict_random_10_user.py
│
├─ .env
├─ .gitattributes
├─ .gitignore
├─ requirements.txt
└─ README.md
```

---

## 2. Yêu cầu môi trường

- Python (khuyến nghị 3.9–3.11)
- PyTorch + Torchvision (phù hợp CUDA nếu có GPU)
- PIL/OpenCV, numpy, pandas, matplotlib, tqdm, yaml/json …

Cài đặt nhanh:

```bash
pip install -r requirements.txt
```

> Nếu bạn chạy GPU: cài PyTorch đúng phiên bản CUDA theo máy (khuyến nghị theo trang pytorch.org).

---

## 3. Dữ liệu & format

### 3.1. Dữ liệu Detection (COCO)
Module `Detection_ROI` được thiết kế để train/eval trên dataset theo **COCO format** (images + annotations JSON).

Bạn cần:
- Folder ảnh
- File annotation COCO (train/val/test)

Trong thực nghiệm, thư mục output và log được lưu trong:
- `Detection_ROI/output/` hoặc `Detection_ROI/output_newdataset/`

### 3.2. Dữ liệu Feature Extraction (ROI classification/embedding)
Module `Feature_Extraction_ROI` dùng ROI đã cắt và chuẩn hoá (ví dụ 240×240) để train mô hình EfficientNetB1 (1 channel).

Thông tin tập train/valid/test thường được mô tả trong:
- `Feature_Extraction_ROI/configs/effb1_roi240_train.json`
- `Feature_Extraction_ROI/configs/effb1_roi240_predict.json`

Ngoài ra có:
- `models/class_to_idx.json`: ánh xạ nhãn user ↔ chỉ số lớp  
- `models/efficientnet_b1_1ch_roi240_best.pth`: trọng số tốt nhất

---

## 4. Pipeline tổng thể (khuyến nghị)

### Bước A — RAW → ảnh xử lý được
- RAW của bạn là **8-bit**, kích thước **640×480**.
- Chuyển RAW → ảnh numpy `uint8` → (tuỳ chọn) cân bằng sáng/CLAHE → lưu PNG/JPG hoặc đưa thẳng vào detector.

> Nếu bạn đã có script convert RAW→JPG ở nơi khác, hãy đặt vào `scripts/` hoặc tạo `tools/convert_raw.py` để tái lập pipeline.

### Bước B — Detection ROI
- Chạy mô hình detector (Faster R-CNN hoặc SSD) để lấy bbox ROI.
- Crop ROI theo bbox (có thể padding một chút để ổn định).
- Resize ROI về kích thước chuẩn cho feature model (ví dụ 240×240).

### Bước C — Feature Extraction / Classification
- Dùng EfficientNetB1 (1 channel) để:
  - **Phân loại user** (closed-set) theo `class_to_idx.json`, hoặc
  - Trích xuất embedding (nếu script của bạn hỗ trợ) để làm xác thực (so khớp cosine/ArcFace head…).

---

## 5. Chạy huấn luyện & suy luận

> Lưu ý: tham số CLI phụ thuộc vào script trong `Detection_ROI/scripts/*`. README này cung cấp lệnh mẫu; bạn có thể mở file script để xem đúng tên arguments.

### 5.1. Train / Eval ROI Detection

**Faster R-CNN**
```bash
cd Detection_ROI
python scripts/faster/train.py --config configs/fasterrcnn_resnet50_palm.yaml
```

**SSD**
```bash
cd Detection_ROI
python scripts/ssd/train.py --config configs/ssd_palm.yaml
```

Kết quả thường nằm ở:
- `Detection_ROI/output/*`
- `Detection_ROI/output_newdataset/checkpoints/` (weights)
- `Detection_ROI/output_newdataset/logs/` (log theo epoch)
- `Detection_ROI/output_newdataset/plots/` (biểu đồ)
- `Detection_ROI/output_newdataset/pred_map_valid.json` (map dự đoán/đánh giá)

### 5.2. Train EfficientNetB1 (ROI 240, 1-channel)

```bash
cd Feature_Extraction_ROI
python Scripts/train.py --config configs/effb1_roi240_train.json
```

Output/weights:
- `Feature_Extraction_ROI/models/`
- `Feature_Extraction_ROI/outputs/`

### 5.3. Predict 1 ảnh / 1 thư mục ROI

```bash
cd Feature_Extraction_ROI
python Scripts/predict.py --config configs/effb1_roi240_predict.json
```

### 5.4. Predict ngẫu nhiên 10 user (demo)

```bash
cd Feature_Extraction_ROI
python Scripts/predict_random_10_user.py --config configs/effb1_roi240_predict.json
```

---

## 6. Gợi ý đánh giá (Metrics)

### Detection
- mAP@0.5, mAP@0.5:0.95
- Precision / Recall / F1 theo ngưỡng score + NMS

### Feature / Classification (ROI)
- Accuracy / Precision / Recall / F1 (valid/test)
- Confusion matrix (nếu cần phân tích nhầm lẫn giữa user)
- Nếu làm xác thực theo 1:N bằng embedding:
  - ROC / AUC, FAR/FRR, EER
  - Top-k retrieval accuracy

---

## 7. Ghi chú tái lập (Reproducibility)

- Cố định seed (numpy/torch/random)
- Ghi log theo epoch (loss, metric)
- Lưu best checkpoint theo metric mục tiêu (val_f1 hoặc val_acc hoặc val_map)

---

## 8. Troubleshooting nhanh

- **Loss = NaN / nổ loss**: kiểm tra bbox sai (w/h âm, vượt ảnh), learning rate quá lớn, ảnh lỗi/nhãn lỗi.
- **Sai kênh ảnh (1ch vs 3ch)**: EfficientNet 1-channel cần đảm bảo input là grayscale đúng shape.
- **Không khớp label**: kiểm tra lại `class_to_idx.json` và cách encode label trong train.json.