# -*- coding: utf-8 -*-
# API trích xuất ROI (crop) bằng SSDLite320-MobileNetV3 — chạy trong code, không dùng argparse

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathlib import Path
import cv2
import torch
import numpy as np

# Albumentations: giống pipeline train
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from torchvision.models.detection import ssdlite320_mobilenet_v3_large

class SSDROIExtractor:
    """
    Dùng model SSD để detect bàn tay và crop ROI.
    - Khởi tạo 1 lần (nạp model + weights)
    - Gọi process_image / process_folder trong code
    """
    def __init__(self, weights="output/best_loss.pth", num_classes=2, device=None,
                 resize_hw=(640,480), mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225),
                 class_names=("bg","palm")):
        self.project_root = ROOT
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.class_names = class_names
        self.resize_hw = resize_hw

        # build model
        self.model = ssdlite320_mobilenet_v3_large(weights_backbone="DEFAULT", num_classes=num_classes)
        # load weights
        ckpt = Path(weights)
        if not ckpt.is_absolute():
            ckpt = self.project_root / ckpt
        if not ckpt.is_file():
            raise FileNotFoundError(f"Không thấy weights: {ckpt}")
        state = torch.load(str(ckpt), map_location=self.device)
        try:
            self.model.load_state_dict(state)   # state_dict
            print(f"[INFO] Loaded state_dict: {ckpt}")
        except Exception:
            self.model = state
            print(f"[INFO] Loaded full model: {ckpt}")
        self.model = self.model.to(self.device).eval()

        # transform
        h, w = resize_hw
        self.tf = A.Compose([
            A.Resize(h, w),
            A.Normalize(mean, std),
            ToTensorV2(),
        ])

    @staticmethod
    def _expand_and_clip(x1, y1, x2, y2, W, H, expand=0.0):
        if expand > 0:
            bw, bh = x2 - x1, y2 - y1
            cx, cy = x1 + bw/2, y1 + bh/2
            bw2, bh2 = bw * (1 + expand), bh * (1 + expand)
            x1, y1 = cx - bw2/2, cy - bh2/2
            x2, y2 = cx + bw2/2, cy + bh2/2
        x1 = max(0, int(round(x1))); y1 = max(0, int(round(y1)))
        x2 = min(W, int(round(x2))); y2 = min(H, int(round(y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def process_image(self, image_path, out_dir="runs/rois", conf=0.30, expand=0.10, topk=0):
        """
        Trích xuất ROI từ 1 ảnh.
        Return: list[dict] với keys: image, roi_path, score, x1,y1,x2,y2
        """
        img_path = Path(image_path)
        if not img_path.is_file():
            raise FileNotFoundError(f"Không thấy ảnh: {img_path}")

        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Không đọc được ảnh: {img_path}")
        H, W = img.shape[:2]

        sample = self.tf(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), bboxes=[], cls=[])
        tensor = sample["image"]
        in_h, in_w = tensor.shape[-2], tensor.shape[-1]

        with torch.no_grad():
            out = self.model([tensor.to(self.device)])[0]
            boxes = out["boxes"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()

        # bỏ nền + lọc conf
        keep = (labels != 0) & (scores >= conf)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # sắp xếp theo score
        order = np.argsort(-scores)
        if topk and topk > 0:
            order = order[:topk]
        boxes, scores, labels = boxes[order], scores[order], labels[order]

        # scale về kích thước gốc
        if len(boxes) > 0:
            sx = float(W) / float(in_w)
            sy = float(H) / float(in_h)
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy

        rois = []
        for i, (b, s, l) in enumerate(zip(boxes, scores, labels), 1):
            x1, y1, x2, y2 = b.tolist()
            rect = self._expand_and_clip(x1, y1, x2, y2, W, H, expand=expand)
            if rect is None:
                continue
            x1_, y1_, x2_, y2_ = rect
            crop = img[y1_:y2_, x1_:x2_]
            out_name = f"{img_path.stem}_roi{i}_{s:.3f}.jpg"
            out_file = out_dir / out_name
            cv2.imwrite(str(out_file), crop)
            rois.append({
                "image": str(img_path),
                "roi_path": str(out_file),
                "score": float(s),
                "x1": int(x1_), "y1": int(y1_), "x2": int(x2_), "y2": int(y2_)
            })
        return rois

    def process_folder(self, folder, out_dir="runs/rois", conf=0.30, expand=0.10, topk=0):
        """
        Trích xuất ROI cho tất cả ảnh trong thư mục (đệ quy).
        Return: list tổng hợp các ROI của toàn bộ thư mục.
        """
        folder = Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(f"Không phải thư mục: {folder}")

        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

        all_rois = []
        for p in folder.rglob("*"):
            if p.suffix.lower() in exts:
                try:
                    rois = self.process_image(p, out_dir=out_dir, conf=conf, expand=expand, topk=topk)
                    all_rois.extend(rois)
                except Exception as e:
                    print(f"[WARN] {p.name}: {e}")
        return all_rois


extractor = SSDROIExtractor(
    weights="output/best_loss.pth",   # hoặc output/best_map.pth
    num_classes=2,                     # 0=bg, 1=palm
    class_names=("bg","palm")
)

rois = extractor.process_image(
    image_path=r"D:\PMT_\detect_roi\processed_dataset\autoUser1554\img_9.png",
    out_dir=r"D:\PMT_Paper\runs\rois",
    conf=0.30,
    expand=0.10,
    topk=1,        # 1 = chỉ lấy ROI tốt nhất; 0 = lấy tất cả
)

print(rois) 