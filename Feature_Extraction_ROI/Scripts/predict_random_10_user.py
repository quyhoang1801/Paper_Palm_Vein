# predict_10users_savecsv.py
import os
import json
import random
import csv
import argparse
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models


# =====================
# CONFIG LOADER
# =====================
def _abs(p: str, base: Path) -> str:
    p = str(p)
    return p if os.path.isabs(p) else str((base / p).resolve())

def load_cfg(config_path: str):
    cfg_path = Path(config_path).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # ...\training\configs\xxx.json
    TRAINING_ROOT = cfg_path.parent.parent       # ...\training
    PROJECT_ROOT  = TRAINING_ROOT.parent         # ...\axtract_feature

    paths = cfg["paths"]
    model_path    = _abs(paths["model_path"], PROJECT_ROOT)
    classmap_path = _abs(paths["classmap_path"], PROJECT_ROOT)

    json_dir = _abs(paths["json_dir"], PROJECT_ROOT)
    json_path = os.path.join(json_dir, paths["json_name"])

    out_csv = _abs(paths["out_csv"], PROJECT_ROOT)

    dt = cfg["raw"]["dtype"].lower()
    raw_dtype = np.uint8 if dt == "uint8" else np.uint16

    infer = cfg["infer"]
    normalize = cfg["normalize"]

    return {
        "cfg": cfg,
        "PROJECT_ROOT": PROJECT_ROOT,
        "MODEL_PATH": model_path,
        "CLASSMAP_PATH": classmap_path,
        "JSON_DIR": json_dir,
        "JSON_PATH": json_path,
        "OUT_CSV": out_csv,
        "RAW_DTYPE": raw_dtype,
        "IMG_SIZE": int(infer["img_size"]),
        "TOPK": int(infer["topk"]),
        "MAX_USERS": int(infer["max_users"]),
        "SEED": int(infer["seed"]),
        "USERS_PICK": infer.get("users_pick", []),
        "USE_NORMALIZE": bool(normalize["use"]),
        "MEAN_1CH": tuple(normalize["mean_1ch"]),
        "STD_1CH": tuple(normalize["std_1ch"]),
    }


# =====================
# MODEL (GIỐNG TRAIN)
# =====================
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


# =====================
# HELPERS
# =====================
def ensure_parent(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def load_classmaps(classmap_path):
    with open(classmap_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class

def safe_torch_load_state(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)

def build_ann_list(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        anns = json.load(f)
    out = []
    for a in anns:
        if "filename" in a and "shape" in a and "label" in a:
            out.append(a)
    return out

def join_json_file(json_dir, rel_in_json):
    rel = rel_in_json.replace("\\", os.sep).replace("/", os.sep)
    return os.path.normpath(os.path.join(json_dir, rel))

def read_raw_with_shape(raw_path, h, w, raw_dtype, tf):
    raw = np.fromfile(raw_path, dtype=raw_dtype)
    expected = int(h) * int(w)
    if raw.size != expected:
        raise ValueError(f"RAW mismatch: got={raw.size}, expected={expected} (h={h}, w={w}) | file={raw_path}")

    img = raw.reshape((int(h), int(w)))

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

def stable_user_seed(user: str, base_seed: int) -> int:
    # ổn định giữa các lần chạy (tránh Python hash randomization)
    return base_seed + (zlib.crc32(user.encode("utf-8")) % 100000)

def pick_users_and_one_sample_each(anns, max_users=10, seed=42, users_pick=None):
    by_user = {}
    for a in anns:
        u = a["label"]
        by_user.setdefault(u, []).append(a)

    all_users = sorted(by_user.keys())

    if users_pick and len(users_pick) > 0:
        users = [u for u in users_pick if u in by_user]
    else:
        rng = random.Random(seed)
        users = all_users if len(all_users) <= max_users else rng.sample(all_users, k=max_users)

    samples = []
    for u in users:
        rng_u = random.Random(stable_user_seed(u, seed))
        ann = rng_u.choice(by_user[u])
        samples.append((u, ann))
    return samples

def topk_to_string(top_list):
    return "; ".join([f"{lb}:{p:.6f}" for (lb, p, _) in top_list])


# =====================
# MAIN
# =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=r"..\configs\effb1_roi240_predict.json", help="path to predict config json")
    args = parser.parse_args()

    pack = load_cfg(args.config)

    MODEL_PATH    = pack["MODEL_PATH"]
    CLASSMAP_PATH = pack["CLASSMAP_PATH"]
    JSON_DIR      = pack["JSON_DIR"]
    JSON_PATH     = pack["JSON_PATH"]
    OUT_CSV       = pack["OUT_CSV"]

    IMG_SIZE      = pack["IMG_SIZE"]
    RAW_DTYPE     = pack["RAW_DTYPE"]
    TOPK          = pack["TOPK"]
    MAX_USERS     = pack["MAX_USERS"]
    SEED          = pack["SEED"]

    USERS_PICK = pack["USERS_PICK"]
    if isinstance(USERS_PICK, list) and len(USERS_PICK) == 0:
        USERS_PICK = None

    USE_NORMALIZE = pack["USE_NORMALIZE"]
    MEAN_1CH      = pack["MEAN_1CH"]
    STD_1CH       = pack["STD_1CH"]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TRANSFORM (GIỐNG EVAL)
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_1CH, STD_1CH) if USE_NORMALIZE else transforms.Lambda(lambda x: x),
    ])

    ensure_parent(OUT_CSV)

    class_to_idx, idx_to_class = load_classmaps(CLASSMAP_PATH)
    num_classes = len(class_to_idx)
    print("num_classes =", num_classes)

    anns = build_ann_list(JSON_PATH)
    if len(anns) == 0:
        raise RuntimeError(f"JSON rỗng/không đúng format: {JSON_PATH}")

    samples = pick_users_and_one_sample_each(
        anns,
        max_users=MAX_USERS,
        seed=SEED,
        users_pick=USERS_PICK
    )
    print(f"\nWill predict {len(samples)} users (each 1 sample) from: {JSON_PATH}")

    # build model + load weights
    model = make_efficientnet_b1_1ch(num_classes).to(DEVICE)
    state = safe_torch_load_state(MODEL_PATH, DEVICE)
    model.load_state_dict(state, strict=True)

    rows = []
    ok_cnt = 0
    total = 0

    for user, ann in samples:
        rel = ann["filename"]
        gt_label = ann["label"]
        h, w = ann["shape"]
        raw_path = join_json_file(JSON_DIR, rel)

        row = {
            "user": user,
            "json_filename": rel,
            "raw_path": raw_path,
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

        if not os.path.isfile(raw_path):
            row["status"] = "missing"
            row["error"] = "file_not_found"
            rows.append(row)
            print(f"[SKIP] missing file: {raw_path}")
            continue

        try:
            x = read_raw_with_shape(raw_path, h, w, RAW_DTYPE, tf).to(DEVICE)
            pred_label, conf, pred_idx, margin, top_list = predict_one(model, x, idx_to_class, topk=TOPK)

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
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n" + "=" * 70)
    print(f"Saved CSV: {OUT_CSV}")

    if total > 0:
        acc = ok_cnt / total * 100.0
        print(f"Summary: {ok_cnt}/{total} correct -> Top1 user-sample acc = {acc:.2f}%")
    else:
        print("Summary: No samples predicted (missing/error).")
