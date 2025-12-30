# eval_effb1_1ch_roi240_softmax_far_levels.py
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


# =====================
# CONFIG
# =====================
TRAIN_DIR = r"train"
VALID_DIR = r"valid"
TEST_DIR  = r"test"

TRAIN_JSON = os.path.join(TRAIN_DIR, "train.json")
VALID_JSON = os.path.join(VALID_DIR, "valid.json")
TEST_JSON  = os.path.join(TEST_DIR,  "test.json")

MODEL_SAVE_PATH = r"efficientnet_b1_1ch_roi240_best.pth"
CLASSMAP_SAVE   = r"class_to_idx.json"

OUT_DIR         = r"output"
CM_SAVE_VALID   = os.path.join(OUT_DIR, "cm_valid_bestv1.npy")
CM_SAVE_TEST    = os.path.join(OUT_DIR, "cm_test_bestv1.npy")
REPORT_JSON     = os.path.join(OUT_DIR, "eer_softmax_report_with_far_levels.json")

RAW_DTYPE = np.uint8

SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 0  # Windows

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True if DEVICE.type == "cuda" else False

IMG_SIZE = 240
USE_NORMALIZE = True
MEAN_1CH = (0.5,)
STD_1CH  = (0.5,)

# --- verification settings (random_claim) ---
IMPOSTOR_PER_SAMPLE = 20  # khuyến nghị 10~50 cho ổn định

# FAR levels "chuẩn paper" (tỷ lệ, không phải %)
FAR_TARGETS = [
    ("FAR=1%",   0.01),
    ("FAR=0.1%", 0.001),
    ("FAR=0.01%",0.0001),
]


# =====================
# TRANSFORMS
# =====================
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

def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def topk_accuracy_from_logits(logits, labels, k=5):
    k = min(k, logits.shape[1])
    topk = logits.topk(k, dim=1).indices
    correct = topk.eq(labels.view(-1, 1)).any(dim=1).float().sum().item()
    return correct / labels.size(0) * 100.0

def update_confusion_matrix(cm: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int):
    idx = y_true * num_classes + y_pred
    bins = torch.bincount(idx, minlength=num_classes * num_classes)
    cm += bins.reshape(num_classes, num_classes)

def metrics_from_confusion_matrix(cm: torch.Tensor, eps: float = 1e-12):
    cm = cm.to(torch.float64)
    tp = torch.diag(cm)
    support = cm.sum(dim=1)
    pred_cnt = cm.sum(dim=0)

    recall = tp / torch.clamp(support, min=1.0)
    precision = tp / torch.clamp(pred_cnt, min=1.0)
    f1 = (2 * precision * recall) / torch.clamp(precision + recall, min=eps)

    mask = support > 0
    macro_precision = precision[mask].mean().item() if mask.any() else 0.0
    macro_recall    = recall[mask].mean().item() if mask.any() else 0.0
    macro_f1        = f1[mask].mean().item() if mask.any() else 0.0

    w = support / torch.clamp(support.sum(), min=1.0)
    weighted_f1 = (f1 * w).sum().item()

    balanced_acc = macro_recall

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

def far_frr_at_threshold(genuine_scores, impostor_scores, th):
    g = np.asarray(genuine_scores, dtype=np.float64)
    i = np.asarray(impostor_scores, dtype=np.float64)
    far = float((i >= th).mean()) if i.size else 0.0
    frr = float((g <  th).mean()) if g.size else 0.0
    return far, frr

def threshold_for_far_target(impostor_scores: np.ndarray, far_target: float) -> float:
    """
    Chọn threshold sao cho FAR_valid <= far_target (bảo thủ).
    FAR(th) = mean(impostor >= th)

    Dùng quantile upper-tail:
      th = quantile(impostor, 1 - far_target, method='higher')
    """
    i = np.asarray(impostor_scores, dtype=np.float64)
    if i.size == 0:
        return 1.0
    q = 1.0 - far_target
    q = float(np.clip(q, 0.0, 1.0))

    try:
        th = float(np.quantile(i, q, method="higher"))
    except TypeError:
        # numpy cũ
        th = float(np.quantile(i, q, interpolation="higher"))

    # đảm bảo trong [0,1]
    return float(np.clip(th, 0.0, 1.0))


# =====================
# DATASET
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

    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        logits = model(x)
        loss_sum += criterion(logits, y).item()

        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        top5_sum += topk_accuracy_from_logits(logits, y, k=5) * y.size(0)

        y_cpu = y.detach().to("cpu").to(torch.int64)
        p_cpu = pred.detach().to("cpu").to(torch.int64)
        update_confusion_matrix(cm, y_cpu, p_cpu, num_classes)

    avg_loss = loss_sum / max(1, len(loader))
    acc = 100.0 * correct / max(1, total)
    top5 = top5_sum / max(1, total)

    extra = metrics_from_confusion_matrix(cm)

    print(
        f"{name}: loss={avg_loss:.4f} "
        f"acc={acc:.2f}% top5={top5:.2f}% | "
        f"macroF1={extra['macro_f1']*100:.2f}% "
        f"macroP={extra['macro_precision']*100:.2f}% "
        f"macroR={extra['macro_recall']*100:.2f}% "
        f"balAcc={extra['balanced_acc']*100:.2f}%"
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
    }

    if return_cm:
        return out, cm
    return out


def load_classmap_or_build():
    if os.path.isfile(CLASSMAP_SAVE):
        with open(CLASSMAP_SAVE, "r", encoding="utf-8") as f:
            m = json.load(f)
        return {k: int(v) for k, v in m.items()}

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        anns = json.load(f)
    labels = sorted({a["label"] for a in anns})
    return {c: i for i, c in enumerate(labels)}


# =====================
# SCORE COLLECTION (random_claim)
# =====================
@torch.no_grad()
def collect_softmax_scores_random_claim(model, loader, num_classes: int, n_impostor_per_sample: int, seed: int = 42):
    """
    genuine score = P(y_true|x)
    impostor score = P(wrong_claim|x) với wrong_claim random != y_true
    """
    model.eval()
    genuine_scores = []
    impostor_scores = []

    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        g = probs.gather(1, y.view(-1, 1)).squeeze(1)
        genuine_scores.append(g.detach().cpu())

        B = y.size(0)
        wrong = torch.randint(0, num_classes - 1, (B, n_impostor_per_sample), device=DEVICE, generator=gen)
        wrong = wrong + (wrong >= y.view(-1, 1)).long()
        imp = probs.gather(1, wrong).reshape(-1)
        impostor_scores.append(imp.detach().cpu())

    genuine_scores = torch.cat(genuine_scores).numpy()
    impostor_scores = torch.cat(impostor_scores).numpy()
    return genuine_scores, impostor_scores


def print_far_table(rows):
    # in bảng gọn bằng print (không cần pandas)
    print("\n=== THRESHOLD BY FAR ON VALID (random_claim) ===")
    header = f"{'Target':>10} | {'th_valid':>10} | {'FAR_valid%':>9} | {'FRR_valid%':>9} | {'FAR_test%':>9} | {'FRR_test%':>9} | {'TAR_test%':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['target']:>10} | "
            f"{r['threshold']:.8f} | "
            f"{r['far_valid_pct']:9.4f} | {r['frr_valid_pct']:9.4f} | "
            f"{r['far_test_pct']:9.4f} | {r['frr_test_pct']:9.4f} | {r['tar_test_pct']:9.4f}"
        )


if __name__ == "__main__":
    seed_everything(SEED)
    ensure_parent(CM_SAVE_VALID)
    ensure_parent(CM_SAVE_TEST)
    ensure_parent(REPORT_JSON)

    if not os.path.isfile(MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Không thấy checkpoint: {MODEL_SAVE_PATH}")

    class_to_idx = load_classmap_or_build()
    num_classes = len(class_to_idx)
    print(f"Loaded class_to_idx: {num_classes} classes")

    valid_ds = RawRoiJsonDataset(VALID_JSON, VALID_DIR, transform=transform_eval, class_to_idx=class_to_idx)
    test_ds  = RawRoiJsonDataset(TEST_JSON,  TEST_DIR,  transform=transform_eval, class_to_idx=class_to_idx)

    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = make_efficientnet_b1_1ch(num_classes).to(DEVICE)
    state = safe_torch_load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {MODEL_SAVE_PATH}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    # ---- classification + confusion matrix ----
    valid_metrics, cm_valid = evaluate(model, valid_loader, criterion, num_classes, name="VALID (ckpt)", return_cm=True)
    test_metrics,  cm_test  = evaluate(model, test_loader,  criterion, num_classes, name="TEST  (ckpt)", return_cm=True)

    np.save(CM_SAVE_VALID, cm_valid.numpy())
    np.save(CM_SAVE_TEST,  cm_test.numpy())
    print(f"Saved confusion matrix: {CM_SAVE_VALID}")
    print(f"Saved confusion matrix: {CM_SAVE_TEST}")

    # ---- collect scores (random_claim) ----
    print("\nCollect softmax scores (random_claim) ...")
    g_val, i_val = collect_softmax_scores_random_claim(
        model, valid_loader, num_classes=num_classes, n_impostor_per_sample=IMPOSTOR_PER_SAMPLE, seed=SEED
    )
    g_test, i_test = collect_softmax_scores_random_claim(
        model, test_loader, num_classes=num_classes, n_impostor_per_sample=IMPOSTOR_PER_SAMPLE, seed=SEED
    )

    # ---- thresholds by FAR on VALID; evaluate FRR on TEST ----
    rows = []
    for name, far_t in FAR_TARGETS:
        th = threshold_for_far_target(i_val, far_t)

        far_v, frr_v = far_frr_at_threshold(g_val, i_val, th)
        far_te, frr_te = far_frr_at_threshold(g_test, i_test, th)

        rows.append({
            "target": name,
            "far_target": far_t,
            "threshold": float(th),

            "far_valid_pct": float(far_v * 100.0),
            "frr_valid_pct": float(frr_v * 100.0),

            "far_test_pct":  float(far_te * 100.0),
            "frr_test_pct":  float(frr_te * 100.0),
            "tar_test_pct":  float((1.0 - frr_te) * 100.0),

            "num_genuine_valid": int(len(g_val)),
            "num_impostor_valid": int(len(i_val)),
            "num_genuine_test": int(len(g_test)),
            "num_impostor_test": int(len(i_test)),
        })

    print_far_table(rows)

    report = {
        "checkpoint": MODEL_SAVE_PATH,
        "settings": {
            "IMG_SIZE": IMG_SIZE,
            "IMPOSTOR_PER_SAMPLE": IMPOSTOR_PER_SAMPLE,
            "note": "Thresholds are selected on VALID to meet FAR targets (random_claim). Then evaluated on TEST with fixed thresholds.",
        },
        "valid": {"classification": valid_metrics},
        "test":  {"classification": test_metrics},
        "verification_random_claim": {
            "FAR_targets_on_valid": rows
        }
    }

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nSaved report JSON: {REPORT_JSON}")
    print("Done.")
