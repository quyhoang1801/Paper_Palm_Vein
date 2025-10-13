# -*- coding: utf-8 -*-
# SSDLite320-MobileNetV3 1-class (palm) — Train + Eval + Save best(by loss/map)

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]   # D:\PMT_Paper
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os, json, yaml, warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from torchvision.models.detection import ssdlite320_mobilenet_v3_large

# project modules
from src.utils.common import seed_everything, get_device, amp_ctx
from src.dataset.coco_palm import COCODataset, collate, get_transforms
from src.metrics.coco_eval import coco_map
from src.metrics.prf import detection_prf_cm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def plot_confusion_matrix(cm, class_names, out_png):
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks); ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center")
    ax.set_ylabel("True"); ax.set_xlabel("Pred")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_cfg(path="configs/ssd_palm.yaml"):
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.is_file():
        raise FileNotFoundError(f"Không thấy config: {p}")
    text = p.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    # expand ${root_path}
    root = cfg.get("root_path", "")
    for k in ["train_ann", "val_ann", "test_ann"]:
        if isinstance(cfg.get(k), str):
            cfg[k] = cfg[k].replace("${root_path}", root)
    return cfg


def main():
    cfg = load_cfg()
    seed_everything(cfg.get("seed", 42))
    device = get_device()
    use_amp = (device.type == "cuda")
    print("Device:", device)

    # ==== Datasets & Loaders ====
    train_set = COCODataset(str(Path(cfg["root_path"]) / "train"),
                            cfg["train_ann"], get_transforms(train=True))
    val_set   = COCODataset(str(Path(cfg["root_path"]) / "valid"),
                            cfg["val_ann"],  get_transforms(train=False))

    train_loader = DataLoader(train_set, batch_size=cfg["batch_train"], shuffle=True,
                              collate_fn=collate, num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"])
    val_loader   = DataLoader(val_set, batch_size=cfg["batch_val"], shuffle=False,
                              collate_fn=collate, num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"])

    print("Mapping cat2label (train):", train_set.cat2label)
    print("label2cat (train):", train_set.label2cat)
    print("num_classes:", train_set.num_classes)

    # ==== Model / Optim / Sched ====
    num_classes = train_set.num_classes  # 2: background + palm
    model = ssdlite320_mobilenet_v3_large(weights_backbone="DEFAULT", num_classes=num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["step_size"], gamma=cfg["gamma"])

    # ==== Save dirs ====
    save_dir = PROJECT_ROOT / cfg["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "metrics_log.txt"
    hist_csv = save_dir / "metrics_history.csv"

    # ==== Use GT from val_set to avoid id mismatch ====
    coco_gt_val = val_set.coco_ds.coco
    gt_val_ids  = sorted({a['category_id'] for a in coco_gt_val.anns.values()})
    label2cat_val  = {1: gt_val_ids[0]}

    # ==== Logs init ====
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Training log\n\n")
    if not hist_csv.exists():
        hist_csv.write_text("epoch,train_loss,val_loss,map,mAP03,prec,rec,f1\n", encoding="utf-8")

    # ==== Track best ====
    best_map, best_epoch_map = -1.0, -1
    best_val, best_epoch_val = float("inf"), -1

    # ==== Train loop ====
    for epoch in range(cfg["epochs"]):
        torch.cuda.empty_cache()
        model.train()
        total_train = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [Train]")
        for imgs, tgts in pbar:
            imgs = [i.to(device) for i in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            with amp_ctx(use_amp):
                loss_dict = model(imgs, tgts)  # detection model returns loss dict with targets
                loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train += loss.item()
            pbar.set_postfix(loss=float(loss.item()))

        avg_train = total_train / max(1, len(train_loader))

        # ---- Val Loss (no grad) ----
        model.train()
        total_val = 0.0
        with torch.no_grad():
            for imgs, tgts in tqdm(val_loader, desc=f"Epoch {epoch+1} [ValLoss]", leave=False):
                imgs = [i.to(device) for i in imgs]
                tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
                loss_dict = model(imgs, tgts)
                total_val += sum(loss_dict.values()).item()
        avg_val = total_val / max(1, len(val_loader))

        # ---- Metrics on val ----
        map_std, map_03 = coco_map(model, val_loader, coco_gt=coco_gt_val,
                                   label2cat=label2cat_val, conf_thres=cfg["conf_eval"], device=device)
        prec, rec, f1, cm = detection_prf_cm(model, val_loader, train_set.num_classes,
                                             score_thr=cfg["conf_eval"], device=device)

        # Console
        print(f"Epoch {epoch+1}: TrainLoss={avg_train:.4f}  ValLoss={avg_val:.4f}  "
              f"mAP@0.5:0.95={map_std:.4f}  mAP@0.3={map_03:.4f}  "
              f"Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
        print("Confusion Matrix:\n", cm)

        # Text log
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Epoch {epoch+1}: "
                    f"TrainLoss={avg_train:.4f}  ValLoss={avg_val:.4f}  "
                    f"mAP@0.5:0.95={map_std:.4f}  mAP@0.3={map_03:.4f}  "
                    f"Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n\n")

        # CSV history
        with open(hist_csv, "a", encoding="utf-8") as fcsv:
            fcsv.write(f"{epoch+1},{avg_train:.6f},{avg_val:.6f},{map_std:.6f},{map_03:.6f},{prec:.6f},{rec:.6f},{f1:.6f}\n")

        # ---- Save best by mAP ----
        if map_std >= best_map:
            best_map, best_epoch_map = map_std, epoch + 1
            torch.save(model.state_dict(), save_dir / "best_map.pth")
            print(f"💾 Saved Best-by-mAP at Epoch {best_epoch_map} -> {save_dir/'best_map.pth'} (mAP={best_map:.4f})")

        # ---- Save best by ValLoss ----
        if avg_val <= best_val:
            best_val, best_epoch_val = avg_val, epoch + 1
            torch.save(model.state_dict(), save_dir / "best_loss.pth")
            # Save metrics summary for this best-by-loss
            metrics_best = {
                "epoch": best_epoch_val,
                "best_val_loss": float(best_val),
                "map_0.5_0.95": float(map_std),
                "map_0.3": float(map_03),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
            }
            (save_dir / "best_loss_metrics.json").write_text(
                json.dumps(metrics_best, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            # Save confusion matrix figure
            plot_confusion_matrix(cm, ["bg(0)", "palm(1)"], save_dir / "best_loss_confusion_matrix.png")
            print(f"💾 Saved Best-by-ValLoss at Epoch {best_epoch_val} -> {save_dir/'best_loss.pth'} (ValLoss={best_val:.4f})")

        scheduler.step()

    # ==== Summary ====
    print("\n========== SUMMARY ==========")
    if best_epoch_val != -1:
        print(f"Best by ValLoss: epoch {best_epoch_val}, val_loss={best_val:.4f} -> {save_dir/'best_loss.pth'}")
    else:
        print("No best-by-loss found.")
    if best_epoch_map != -1:
        print(f"Best by mAP:    epoch {best_epoch_map}, mAP={best_map:.4f} -> {save_dir/'best_map.pth'}")
    else:
        print("No best-by-mAP found.")

    # ==== Export pred_map (VALID) ====
    if cfg.get("save_pred_map", True) and cfg.get("save_pred_map_split", "valid") == "valid":
        loader = val_loader
        out_json = save_dir / "pred_map_valid.json"
        coco_gt_for_pred = coco_gt_val
        l2c = label2cat_val

        pred_map = {}
        model.eval()
        with torch.no_grad():
            for imgs, tgts in loader:
                imgs = [img.to(device) for img in imgs]
                outs = model(imgs)
                for out, tgt, img_tensor in zip(outs, tgts, imgs):
                    image_id = int(tgt["image_id"])
                    boxes  = out["boxes"].detach().cpu().clone().numpy()
                    scores = out["scores"].detach().cpu().numpy()
                    labels = out["labels"].detach().cpu().numpy()

                    keep = scores >= cfg["conf_eval"]
                    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

                    # scale về kích thước gốc theo COCO val
                    orig_w = coco_gt_for_pred.imgs[image_id]['width']
                    orig_h = coco_gt_for_pred.imgs[image_id]['height']
                    in_h, in_w = img_tensor.shape[-2], img_tensor.shape[-1]
                    sx, sy = float(orig_w)/float(in_w), float(orig_h)/float(in_h)
                    boxes[:, [0,2]] *= sx
                    boxes[:, [1,3]] *= sy

                    entries = []
                    for b, s, l in zip(boxes, scores, labels):
                        if int(l) == 0:
                            continue
                        x1, y1, x2, y2 = b.tolist()
                        entries.append({
                            "xyxy_orig": [float(x1), float(y1), float(x2), float(y2)],
                            "score": float(s),
                            "label": int(l)
                        })
                    pred_map[image_id] = entries

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(pred_map, f, ensure_ascii=False, indent=2)
        print(f"📝 Saved pred_map to {out_json}")


if __name__ == "__main__":
    main()
