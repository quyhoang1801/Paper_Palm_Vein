# scripts/predict_samples_savecsv.py
import os
import json
import csv
import random
import argparse
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models


# -------------------------
# Model (same as train)
# -------------------------
def make_efficientnet_b1_1ch(num_classes: int) -> nn.Module:
    weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1
    model = models.efficientnet_b1(weights=weights)

    first_conv = model.features[0][0]
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


# -------------------------
# Helpers
# -------------------------
def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def safe_torch_load_state(path: Path, device):
    try:
        return torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=device)

def load_classmaps(classmap_path: Path):
    with open(classmap_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class

def build_ann_list(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        anns = json.load(f)
    return [a for a in anns if ("filename" in a and "shape" in a and "label" in a)]

def join_json_file(json_dir: Path, rel_in_json: str):
    rel = rel_in_json.replace("\\", os.sep).replace("/", os.sep)
    return (json_dir / rel).resolve()

def read_raw_with_shape(raw_path: Path, h: int, w: int, raw_dtype, tf):
    raw = np.fromfile(str(raw_path), dtype=raw_dtype)
    expected = int(h) * int(w)
    if raw.size != expected:
        raise ValueError(f"RAW mismatch: got={raw.size}, expected={expected} (h={h}, w={w}) | file={raw_path}")

    img = raw.reshape((int(h), int(w)))

    # scale uint16 -> uint8
    if raw_dtype != np.uint8:
        img = np.clip((img.astype(np.float32) / 65535.0) * 255.0, 0, 255).astype(np.uint8)

    pil = Image.fromarray(img, mode="L")
    x = tf(pil).unsqueeze(0)  # (1,1,H,W)
    return x

@torch.no_grad()
def predict_one(model, x, idx_to_class, topk=5):
    model.eval()
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0]

    k = min(topk, prob.numel())
    vals, idxs = torch.topk(prob, k=k)

    top1 = float(vals[0])
    top2 = float(vals[1]) if k > 1 else 0.0
    margin = top1 - top2

    pred_idx = int(idxs[0])
    pred_label = idx_to_class[pred_idx]

    top_list = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        top_list.append((idx_to_class[int(i)], float(v), int(i)))

    return pred_label, top1, pred_idx, margin, top_list

def topk_to_string(top_list):
    return "; ".join([f"{lb}:{p:.6f}" for (lb, p, _) in top_list])

def stable_user_seed(user: str, base_seed: int) -> int:
    # stable across runs (avoid Python hash randomization)
    return base_seed + (zlib.crc32(user.encode("utf-8")) % 100000)

def pick_users_and_samples(anns, max_users=10, samples_per_user=1, seed=42, users_pick=None):
    by_user = {}
    for a in anns:
        by_user.setdefault(a["label"], []).append(a)

    all_users = sorted(by_user.keys())

    if users_pick and len(users_pick) > 0:
        users = [u for u in users_pick if u in by_user]
    else:
        rng = random.Random(seed)
        users = all_users if len(all_users) <= max_users else rng.sample(all_users, k=max_users)

    samples = []
    for u in users:
        rng_u = random.Random(stable_user_seed(u, seed))
        pool = by_user[u]
        k = min(samples_per_user, len(pool))
        chosen = rng_u.sample(pool, k=k) if k > 1 else [rng_u.choice(pool)]
        for ann in chosen:
            samples.append((u, ann))
    return samples


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to best.pth")
    ap.add_argument("--classmap", required=True, help="path to class_to_idx.json")
    ap.add_argument("--json", required=True, help="path to test.json")
    ap.add_argument("--json-dir", required=True, help="dir that contains raw files referenced by json filename")
    ap.add_argument("--out-csv", required=True, help="output csv path")

    ap.add_argument("--img-size", type=int, default=240)
    ap.add_argument("--raw-dtype", choices=["uint8", "uint16"], default="uint8")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max-users", type=int, default=10)
    ap.add_argument("--samples-per-user", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--users-pick", default="", help='comma list: "autoUser1,autoUser2" (optional)')
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_dtype = np.uint8 if args.raw_dtype == "uint8" else np.uint16

    # transform (same as eval)
    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    model_path = Path(args.model)
    classmap_path = Path(args.classmap)
    json_path = Path(args.json)
    json_dir = Path(args.json_dir)
    out_csv = Path(args.out_csv)

    ensure_parent(out_csv)

    class_to_idx, idx_to_class = load_classmaps(classmap_path)
    num_classes = len(class_to_idx)
    print("num_classes =", num_classes)

    anns = build_ann_list(json_path)
    if len(anns) == 0:
        raise RuntimeError(f"JSON rỗng/không đúng format: {json_path}")

    users_pick = [u.strip() for u in args.users_pick.split(",") if u.strip()] if args.users_pick else None

    samples = pick_users_and_samples(
        anns,
        max_users=args.max_users,
        samples_per_user=args.samples_per_user,
        seed=args.seed,
        users_pick=users_pick
    )
    print(f"Will predict {len(samples)} samples from {len(set([u for u,_ in samples]))} users")

    model = make_efficientnet_b1_1ch(num_classes).to(device)
    state = safe_torch_load_state(model_path, device)
    model.load_state_dict(state, strict=True)

    rows = []
    ok_cnt = 0
    total = 0

    for user, ann in samples:
        rel = ann["filename"]
        gt_label = ann["label"]
        h, w = ann["shape"]
        raw_path = join_json_file(json_dir, rel)

        row = {
            "user": user,
            "json_filename": rel,
            "raw_path": str(raw_path),
            "h": int(h),
            "w": int(w),
            "gt_label": gt_label,
            "pred_label": "",
            "conf": "",
            "margin": "",
            "ok": "",
            "topk": "",
            "status": "ok",
            "error": ""
        }

        if not raw_path.is_file():
            row["status"] = "missing"
            row["error"] = "file_not_found"
            rows.append(row)
            print(f"[SKIP] missing file: {raw_path}")
            continue

        try:
            x = read_raw_with_shape(raw_path, int(h), int(w), raw_dtype, tf).to(device)
            pred_label, conf, _, margin, top_list = predict_one(model, x, idx_to_class, topk=args.topk)

            ok = (pred_label == gt_label)
            total += 1
            ok_cnt += int(ok)

            row["pred_label"] = pred_label
            row["conf"] = f"{conf:.6f}"
            row["margin"] = f"{margin:.6f}"
            row["ok"] = int(ok)
            row["topk"] = topk_to_string(top_list)

            print(f"USER={user} | GT={gt_label} | PRED={pred_label} | conf={conf:.6f} | OK={ok}")

        except Exception as e:
            row["status"] = "error"
            row["error"] = str(e)
            print(f"[ERROR] {user} | {raw_path}\n  {e}")

        rows.append(row)

    fieldnames = [
        "user", "json_filename", "raw_path", "h", "w",
        "gt_label", "pred_label", "conf", "margin", "ok",
        "topk", "status", "error"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 70)
    print(f"Saved CSV: {out_csv}")

    if total > 0:
        acc = ok_cnt / total * 100.0
        print(f"Summary: {ok_cnt}/{total} correct -> Top1 acc = {acc:.2f}%")
    else:
        print("Summary: No samples predicted (missing/error).")


if __name__ == "__main__":
    main()
