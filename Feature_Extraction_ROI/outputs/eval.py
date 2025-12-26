# analyze_confusion_matrix.py
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv


# -------------------------
# IO helpers
# -------------------------
def load_classmap(classmap_path: str):
    if not classmap_path or not os.path.isfile(classmap_path):
        return None, None
    with open(classmap_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class


# -------------------------
# Metrics from confusion matrix
# -------------------------
def metrics_from_cm(cm: np.ndarray, eps: float = 1e-12):
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    support = cm.sum(axis=1)      # true count per class
    pred_cnt = cm.sum(axis=0)     # predicted count per class

    recall = tp / np.clip(support, 1, None)
    precision = tp / np.clip(pred_cnt, 1, None)
    f1 = (2 * precision * recall) / np.clip(precision + recall, eps, None)

    mask = support > 0
    macroP = precision[mask].mean() if mask.any() else 0.0
    macroR = recall[mask].mean() if mask.any() else 0.0
    macroF1 = f1[mask].mean() if mask.any() else 0.0

    w = support / np.clip(support.sum(), 1, None)
    wF1 = (f1 * w).sum()

    acc = tp.sum() / np.clip(cm.sum(), 1, None)
    balAcc = macroR  # balanced accuracy ~ macro recall

    recall_masked = recall[mask] if mask.any() else np.array([])
    return {
        "acc": acc * 100.0,
        "macroP": macroP * 100.0,
        "macroR": macroR * 100.0,
        "macroF1": macroF1 * 100.0,
        "wF1": wF1 * 100.0,
        "balAcc": balAcc * 100.0,
        "recall_zero_pct": float((recall_masked == 0).mean() * 100.0) if recall_masked.size else 0.0,
        "recall_min": float(recall_masked.min() * 100.0) if recall_masked.size else 0.0,
        "recall_med": float(np.median(recall_masked) * 100.0) if recall_masked.size else 0.0,
        "per_class": {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    }


# -------------------------
# Ranking helpers
# -------------------------
def topk_worst_recall(per_class, idx_to_class, k=20):
    support = per_class["support"]
    recall = per_class["recall"]
    precision = per_class["precision"]
    f1 = per_class["f1"]

    mask = support > 0
    ids = np.where(mask)[0]
    if ids.size == 0:
        return []

    sort_ids = ids[np.argsort(recall[ids])]  # recall asc
    sort_ids = sort_ids[:min(k, sort_ids.size)]

    rows = []
    for i in sort_ids:
        name = idx_to_class.get(int(i), str(int(i))) if idx_to_class else str(int(i))
        rows.append({
            "idx": int(i),
            "class": name,
            "support": int(support[i]),
            "precision": float(precision[i] * 100.0),
            "recall": float(recall[i] * 100.0),
            "f1": float(f1[i] * 100.0),
        })
    return rows


def topk_confusions(cm: np.ndarray, idx_to_class, k=20):
    cm = cm.copy().astype(np.int64)
    np.fill_diagonal(cm, 0)
    flat = cm.flatten()
    if flat.max() <= 0:
        return []

    top = np.argsort(flat)[::-1][:k]
    C = cm.shape[0]
    out = []
    for t in top:
        v = int(flat[t])
        if v <= 0:
            break
        i, j = divmod(int(t), C)
        true_name = idx_to_class.get(i, str(i)) if idx_to_class else str(i)
        pred_name = idx_to_class.get(j, str(j)) if idx_to_class else str(j)
        out.append({
            "true_idx": i, "pred_idx": j,
            "true": true_name, "pred": pred_name,
            "count": v
        })
    return out


# -------------------------
# Save CSV
# -------------------------
def save_per_class_csv(out_path: str, per_class, idx_to_class):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    support = per_class["support"]
    precision = per_class["precision"]
    recall = per_class["recall"]
    f1 = per_class["f1"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_idx", "class_name", "support", "precision_pct", "recall_pct", "f1_pct"])
        for i in range(len(support)):
            name = idx_to_class.get(int(i), str(int(i))) if idx_to_class else str(int(i))
            w.writerow([int(i), name, int(support[i]),
                        float(precision[i] * 100.0),
                        float(recall[i] * 100.0),
                        float(f1[i] * 100.0)])


def save_rows_csv(out_path: str, header: list, rows: list):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(h, "") for h in header])


def append_summary_csv(out_path: str, tag: str, m: dict):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    file_exists = os.path.isfile(out_path)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["split", "acc", "macroP", "macroR", "macroF1", "wF1", "balAcc",
                        "recall_zero_pct", "recall_min", "recall_med"])
        w.writerow([
            tag,
            f"{m['acc']:.4f}",
            f"{m['macroP']:.4f}",
            f"{m['macroR']:.4f}",
            f"{m['macroF1']:.4f}",
            f"{m['wF1']:.4f}",
            f"{m['balAcc']:.4f}",
            f"{m['recall_zero_pct']:.4f}",
            f"{m['recall_min']:.4f}",
            f"{m['recall_med']:.4f}",
        ])


# -------------------------
# Plots
# -------------------------
def plot_cm(cm: np.ndarray, out_png: str, title: str, normalize: bool = False):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    mat = cm.astype(np.float64)
    if normalize:
        row_sum = mat.sum(axis=1, keepdims=True)
        mat = mat / np.clip(row_sum, 1.0, None)

    plt.figure(figsize=(10, 8))
    plt.imshow(mat, aspect="auto")
    plt.title(title + (" (row-normalized)" if normalize else ""))
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.colorbar()

    # nếu quá nhiều class thì bỏ tick cho đỡ rối
    if mat.shape[0] > 50:
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_topk_worst_recall_bar(worst_rows, out_png: str, title: str):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    if not worst_rows:
        return

    # worst_rows đang là recall tăng dần -> giữ nguyên để thấy class tệ nhất ở trên cùng
    labels = [r["class"] for r in worst_rows]
    recall = [r["recall"] for r in worst_rows]
    support = [r["support"] for r in worst_rows]

    plt.figure(figsize=(12, max(4, 0.35 * len(worst_rows) + 2)))
    y = np.arange(len(worst_rows))
    plt.barh(y, recall)
    plt.yticks(y, labels)
    plt.xlabel("Recall (%)")
    plt.title(title)

    # annotate support
    for i, (rv, sp) in enumerate(zip(recall, support)):
        plt.text(rv + 0.2, i, f"n={sp}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -------------------------
# Printing
# -------------------------
def print_report(tag: str, m: dict, worst_rows, conf_rows):
    print(f"\n===== {tag} =====")
    print(f"Acc      : {m['acc']:.2f}%")
    print(f"Macro-P  : {m['macroP']:.2f}%")
    print(f"Macro-R  : {m['macroR']:.2f}%")
    print(f"Macro-F1 : {m['macroF1']:.2f}%")
    print(f"wF1      : {m['wF1']:.2f}%")
    print(f"BalAcc   : {m['balAcc']:.2f}%")
    print(f"recall0  : {m['recall_zero_pct']:.2f}% | recall_min={m['recall_min']:.2f}% | recall_med={m['recall_med']:.2f}%")

    print("\n-- Worst classes by Recall --")
    for r in worst_rows:
        print(f"[{r['idx']:4d}] {r['class']} | supp={r['support']:4d} | "
              f"P={r['precision']:.2f}% R={r['recall']:.2f}% F1={r['f1']:.2f}%")

    print("\n-- Top confusions (true -> pred) --")
    for c in conf_rows:
        print(f"{c['true']} ({c['true_idx']}) -> {c['pred']} ({c['pred_idx']}): {c['count']}")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cm_valid", default=r"D:\PMT_axtract_feature\axtract_feature\training\output\outputv6\cm_valid_best.npy")
    ap.add_argument("--cm_test",  default=r"D:\PMT_axtract_feature\axtract_feature\training\output\outputv6\cm_test_best.npy")
    ap.add_argument("--classmap", default=r"D:\PMT_axtract_feature\axtract_feature\training\models\modelv6\class_to_idx.json")
    ap.add_argument("--out_dir",  default=r"output/cm_reports")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    _, idx_to_class = load_classmap(args.classmap)

    cm_valid = np.load(args.cm_valid)
    cm_test  = np.load(args.cm_test)

    summary_csv = os.path.join(args.out_dir, "summary_metrics.csv")
    # reset summary csv each run (optional)
    if os.path.isfile(summary_csv):
        os.remove(summary_csv)

    for tag, cm in [("valid", cm_valid), ("test", cm_test)]:
        m = metrics_from_cm(cm)
        worst = topk_worst_recall(m["per_class"], idx_to_class, k=args.topk)
        confs = topk_confusions(cm, idx_to_class, k=args.topk)

        print_report(tag.upper(), m, worst, confs)

        # summary CSV
        append_summary_csv(summary_csv, tag, m)

        # per-class CSV
        per_class_csv = os.path.join(args.out_dir, f"per_class_{tag}.csv")
        save_per_class_csv(per_class_csv, m["per_class"], idx_to_class)

        # worst recall CSV
        worst_csv = os.path.join(args.out_dir, f"topk_worst_recall_{tag}.csv")
        save_rows_csv(
            worst_csv,
            header=["idx", "class", "support", "precision", "recall", "f1"],
            rows=worst
        )

        # top confusions CSV
        conf_csv = os.path.join(args.out_dir, f"topk_confusions_{tag}.csv")
        save_rows_csv(
            conf_csv,
            header=["true_idx", "true", "pred_idx", "pred", "count"],
            rows=confs
        )

        # --- ALWAYS SAVE PLOTS ---
        # confusion matrix (raw)
        cm_png = os.path.join(args.out_dir, f"cm_{tag}.png")
        plot_cm(cm, cm_png, title=f"Confusion Matrix ({tag.upper()})", normalize=False)

        # confusion matrix (row-normalized)
        cm_png_norm = os.path.join(args.out_dir, f"cm_{tag}_norm.png")
        plot_cm(cm, cm_png_norm, title=f"Confusion Matrix ({tag.upper()})", normalize=True)

        # top-k worst recall bar
        bar_png = os.path.join(args.out_dir, f"worst_recall_{tag}.png")
        plot_topk_worst_recall_bar(worst, bar_png, title=f"Top-{len(worst)} Worst Recall Classes ({tag.upper()})")

    print(f"\nSaved reports to: {args.out_dir}")
    print(f"- {summary_csv}")
    print(f"- per_class_valid.csv / per_class_test.csv")
    print(f"- cm_valid.png, cm_valid_norm.png, worst_recall_valid.png")
    print(f"- cm_test.png,  cm_test_norm.png,  worst_recall_test.png")


if __name__ == "__main__":
    main()
