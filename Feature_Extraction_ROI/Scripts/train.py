# train_effb1_1ch_roi_varshape_to240_no_earlystop_metrics.py
import os
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt

# =====================
# CONFIG
# =====================
TRAIN_DIR = r"train"
VALID_DIR = r"valid"
TEST_DIR  = r"test"

TRAIN_JSON = os.path.join(TRAIN_DIR, "train.json")
VALID_JSON = os.path.join(VALID_DIR, "valid.json")
TEST_JSON  = os.path.join(TEST_DIR,  "test.json")

MODEL_SAVE_PATH = r"models/efficientnet_b1_1ch_roi240_best.pth"
PLOT_SAVE_PATH  = r"output/plot_train_effb1_roi240.png"
CLASSMAP_SAVE   = r"models/class_to_idx.json"
CM_SAVE_VALID   = r"output/cm_valid_best.npy"
CM_SAVE_TEST    = r"output/cm_test_best.npy"

RAW_DTYPE = np.uint8   # nếu raw là uint16 -> đổi np.uint16 (code sẽ scale)

SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 200
LR_BACKBONE = 3e-4
LR_HEAD     = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0  # Windows

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True if DEVICE.type == "cuda" else False

IMG_SIZE = 240

USE_NORMALIZE = True
MEAN_1CH = (0.5,)
STD_1CH  = (0.5,)

# =====================
# TRANSFORMS
# =====================
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomApply([transforms.RandomRotation(5)], p=0.3),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
    transforms.ToTensor(),
    transforms.Normalize(MEAN_1CH, STD_1CH) if USE_NORMALIZE else transforms.Lambda(lambda x: x),
])

transform_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN_1CH, STD_1CH) if USE_NORMALIZE else transforms.Lambda(lambda x: x),
])

# =====================
# HELPERS
# =====================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_parent(p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def topk_accuracy_from_logits(logits, labels, k=5):
    k = min(k, logits.shape[1])
    topk = logits.topk(k, dim=1).indices
    correct = topk.eq(labels.view(-1, 1)).any(dim=1).float().sum().item()
    return correct / labels.size(0) * 100.0

def update_confusion_matrix(cm: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int):
    """
    cm: (C,C) int64 on CPU
    y_true,y_pred: 1D on CPU
    """
    idx = y_true * num_classes + y_pred
    bins = torch.bincount(idx, minlength=num_classes * num_classes)
    cm += bins.reshape(num_classes, num_classes)

def metrics_from_confusion_matrix(cm: torch.Tensor, eps: float = 1e-12):
    """
    Returns macro/weighted precision/recall/f1 + balanced acc.
    cm: (C,C) int64
    """
    cm = cm.to(torch.float64)
    tp = torch.diag(cm)
    support = cm.sum(dim=1)      # true count per class
    pred_cnt = cm.sum(dim=0)     # predicted count per class

    recall = tp / torch.clamp(support, min=1.0)
    precision = tp / torch.clamp(pred_cnt, min=1.0)
    f1 = (2 * precision * recall) / torch.clamp(precision + recall, min=eps)

    # macro over classes that appear (support>0)
    mask = support > 0
    macro_precision = precision[mask].mean().item() if mask.any() else 0.0
    macro_recall    = recall[mask].mean().item() if mask.any() else 0.0
    macro_f1        = f1[mask].mean().item() if mask.any() else 0.0

    # weighted by support
    w = support / torch.clamp(support.sum(), min=1.0)
    weighted_f1 = (f1 * w).sum().item()

    balanced_acc = macro_recall

    # extra stats
    recall_np = recall[mask].cpu().numpy() if mask.any() else np.array([])
    stats = {}
    if recall_np.size > 0:
        stats["recall_min"] = float(np.min(recall_np))
        stats["recall_median"] = float(np.median(recall_np))
        stats["recall_zero_pct"] = float((recall_np == 0).mean() * 100.0)
    else:
        stats["recall_min"] = 0.0
        stats["recall_median"] = 0.0
        stats["recall_zero_pct"] = 0.0

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_acc": balanced_acc,
        **stats
    }

# =====================
# DATASET
# shape trong JSON là [h,w]
# =====================
class RawRoiJsonDataset(Dataset):
    def __init__(self, json_path, root_dir, transform=None, class_to_idx=None):
        with open(json_path, "r", encoding="utf-8") as f:
            self.anns = json.load(f)

        self.root_dir = Path(root_dir)
        self.transform = transform

        labels = sorted({a["label"] for a in self.anns})
        if class_to_idx is None:
            self.class_to_idx = {c: i for i, c in enumerate(labels)}
        else:
            self.class_to_idx = class_to_idx

        self.classes = list(self.class_to_idx.keys())

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        rel = ann["filename"].replace("\\", os.sep).replace("/", os.sep)

        h, w = ann["shape"]
        h, w = int(h), int(w)

        label = ann["label"]
        path = self.root_dir / rel
        if not path.is_file():
            raise FileNotFoundError(f"Không thấy file: {path}")

        raw = np.fromfile(str(path), dtype=RAW_DTYPE)
        expected = h * w
        if raw.size != expected:
            raise ValueError(
                f"RAW size mismatch: {path}\n"
                f"got={raw.size}, expected={expected} (h={h}, w={w})"
            )

        img = raw.reshape((h, w))

        if RAW_DTYPE != np.uint8:
            img = np.clip((img.astype(np.float32) / 65535.0) * 255.0, 0, 255).astype(np.uint8)

        pil = Image.fromarray(img, mode="L")

        if self.transform:
            pil = self.transform(pil)

        y = self.class_to_idx[label]
        return pil, y

# =====================
# MODEL
# =====================
def make_efficientnet_b1_1ch(num_classes: int) -> nn.Module:
    weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1
    model = models.efficientnet_b1(weights=weights)

    first_conv = model.features[0][0]
    if not (isinstance(first_conv, nn.Conv2d) and first_conv.in_channels == 3):
        raise RuntimeError("Không tìm thấy conv đầu vào 3 kênh ở EfficientNet-B1.")

    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=(first_conv.bias is not None),
    )

    with torch.no_grad():
        new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
        if first_conv.bias is not None:
            new_conv.bias.copy_(first_conv.bias)

    model.features[0][0] = new_conv
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

@torch.no_grad()
def evaluate(model, loader, criterion, num_classes: int, name="EVAL", return_cm=False):
    model.eval()
    loss_sum = 0.0
    correct, total = 0, 0
    top5_sum = 0.0

    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)  # CPU

    conf_sum = 0.0
    conf_correct_sum = 0.0
    conf_wrong_sum = 0.0
    correct_cnt = 0
    wrong_cnt = 0

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        logits = model(x)
        loss_sum += criterion(logits, y).item()

        pred = logits.argmax(1)
        correct_mask = (pred == y)
        correct += correct_mask.sum().item()
        total += y.size(0)

        top5_sum += topk_accuracy_from_logits(logits, y, k=5) * y.size(0)

        # confidence stats
        prob = torch.softmax(logits, dim=1)
        conf = prob.max(dim=1).values
        conf_sum += conf.sum().item()

        if correct_mask.any():
            conf_correct_sum += conf[correct_mask].sum().item()
            correct_cnt += int(correct_mask.sum().item())
        if (~correct_mask).any():
            conf_wrong_sum += conf[~correct_mask].sum().item()
            wrong_cnt += int((~correct_mask).sum().item())

        # confusion matrix on CPU
        y_cpu = y.detach().to("cpu", non_blocking=False).to(torch.int64)
        p_cpu = pred.detach().to("cpu", non_blocking=False).to(torch.int64)
        update_confusion_matrix(cm, y_cpu, p_cpu, num_classes)

    avg_loss = loss_sum / max(1, len(loader))
    acc = 100.0 * correct / max(1, total)
    top5 = top5_sum / max(1, total)

    extra = metrics_from_confusion_matrix(cm)
    conf_mean = conf_sum / max(1, total)
    conf_mean_correct = conf_correct_sum / max(1, correct_cnt)
    conf_mean_wrong = conf_wrong_sum / max(1, wrong_cnt)

    print(
        f"{name}: loss={avg_loss:.4f} "
        f"acc={acc:.2f}% top5={top5:.2f}% | "
        f"macroF1={extra['macro_f1']*100:.2f}% "
        f"macroP={extra['macro_precision']*100:.2f}% "
        f"macroR={extra['macro_recall']*100:.2f}% "
        f"wF1={extra['weighted_f1']*100:.2f}% "
        f"balAcc={extra['balanced_acc']*100:.2f}% | "
        f"recall0={extra['recall_zero_pct']:.2f}% "
        f"recall_min={extra['recall_min']*100:.2f}% "
        f"recall_med={extra['recall_median']*100:.2f}% | "
        f"conf={conf_mean:.4f} conf_ok={conf_mean_correct:.4f} conf_bad={conf_mean_wrong:.4f}"
    )

    out = {
        "loss": avg_loss,
        "acc": acc,
        "top5": top5,
        "macro_precision": extra["macro_precision"] * 100.0,
        "macro_recall": extra["macro_recall"] * 100.0,
        "macro_f1": extra["macro_f1"] * 100.0,
        "weighted_f1": extra["weighted_f1"] * 100.0,
        "balanced_acc": extra["balanced_acc"] * 100.0,
        "recall_zero_pct": extra["recall_zero_pct"],
        "recall_min": extra["recall_min"] * 100.0,
        "recall_median": extra["recall_median"] * 100.0,
        "conf_mean": conf_mean,
        "conf_mean_correct": conf_mean_correct,
        "conf_mean_wrong": conf_mean_wrong,
    }

    if return_cm:
        return out, cm
    return out

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    seed_everything(SEED)
    ensure_parent(MODEL_SAVE_PATH)
    ensure_parent(PLOT_SAVE_PATH)
    ensure_parent(CLASSMAP_SAVE)
    ensure_parent(CM_SAVE_VALID)
    ensure_parent(CM_SAVE_TEST)

    train_ds = RawRoiJsonDataset(TRAIN_JSON, TRAIN_DIR, transform=transform_train)
    valid_ds = RawRoiJsonDataset(VALID_JSON, VALID_DIR, transform=transform_eval,
                                 class_to_idx=train_ds.class_to_idx)
    test_ds  = RawRoiJsonDataset(TEST_JSON,  TEST_DIR,  transform=transform_eval,
                                 class_to_idx=train_ds.class_to_idx)

    num_classes = len(train_ds.classes)
    print(f"num_classes={num_classes} | train={len(train_ds)} valid={len(valid_ds)} test={len(test_ds)}")

    with open(CLASSMAP_SAVE, "w", encoding="utf-8") as f:
        json.dump(train_ds.class_to_idx, f, ensure_ascii=False, indent=2)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = make_efficientnet_b1_1ch(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    # discriminative LR: backbone nhỏ hơn head
    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        if "classifier" in n:
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": LR_BACKBONE},
         {"params": head_params,     "lr": LR_HEAD}],
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    use_amp = (DEVICE.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # logs
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_top5s, val_top5s = [], []
    val_f1s, val_balaccs = [], []

    best_val_acc = -1.0

    # warmup freeze backbone vài epoch đầu
    FREEZE_EPOCHS = 3
    for p in backbone_params:
        p.requires_grad = False

    for epoch in range(NUM_EPOCHS):
        if epoch == FREEZE_EPOCHS:
            for p in backbone_params:
                p.requires_grad = True
            print("Unfreeze backbone.")

        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        top5_sum = 0.0

        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            top5_sum += topk_accuracy_from_logits(logits, y, k=5) * y.size(0)

        train_loss = running_loss / max(1, len(train_loader))
        train_acc  = 100.0 * correct / max(1, total)
        train_top5 = top5_sum / max(1, total)

        val_metrics = evaluate(model, valid_loader, criterion, num_classes=num_classes, name="VALID")

        train_losses.append(train_loss); val_losses.append(val_metrics["loss"])
        train_accs.append(train_acc);   val_accs.append(val_metrics["acc"])
        train_top5s.append(train_top5); val_top5s.append(val_metrics["top5"])
        val_f1s.append(val_metrics["macro_f1"])
        val_balaccs.append(val_metrics["balanced_acc"])

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch+1:03d}] lr={lr_now:.6f} "
              f"Train: loss={train_loss:.4f} acc={train_acc:.2f}% top5={train_top5:.2f}% | "
              f"Val: acc={val_metrics['acc']:.2f}% top5={val_metrics['top5']:.2f}% macroF1={val_metrics['macro_f1']:.2f}%")

        scheduler.step()

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Save best: {MODEL_SAVE_PATH} (val_acc={best_val_acc:.2f}%)")

    # ====== LOAD BEST & FINAL EVAL ======
    print("\nTraining done. Loading best checkpoint for final evaluation...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))

    valid_best, cm_valid = evaluate(model, valid_loader, criterion, num_classes=num_classes, name="VALID (best)", return_cm=True)
    test_best,  cm_test  = evaluate(model, test_loader,  criterion, num_classes=num_classes, name="TEST  (best)", return_cm=True)

    np.save(CM_SAVE_VALID, cm_valid.numpy())
    np.save(CM_SAVE_TEST,  cm_test.numpy())
    print(f"Saved confusion matrix: {CM_SAVE_VALID}")
    print(f"Saved confusion matrix: {CM_SAVE_TEST}")

    # ---- PLOT ----
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend(); plt.title("Loss per Epoch")

    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.plot(train_top5s, label="Train Top5")
    plt.plot(val_top5s, label="Val Top5")
    plt.legend(); plt.title("Accuracy per Epoch")

    plt.subplot(1, 3, 3)
    plt.plot(val_f1s, label="Val Macro-F1")
    plt.plot(val_balaccs, label="Val Balanced Acc")
    plt.legend(); plt.title("Val Macro-F1 / Balanced Acc")

    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH)
    plt.show()

    print("Done.")
