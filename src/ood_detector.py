"""
ood_detector.py
OOD Detection Pipeline — Energy Score from Run 5 model.

Μέθοδος: Energy Score (Liu et al., NeurIPS 2020)
    E(x) = -log Σ_c exp(f_c(x))
    Υψηλό E(x) → το μοντέλο είναι αβέβαιο για ΟΛΑ τα γνωστά labels → OOD

Pipeline:
    1. Φόρτωση Run 5 checkpoint (6-class, Water/Bridge excluded from training)
    2. Inference σε FRACTAL val+test με ORIGINAL labels (χωρίς remap)
    3. OOD GT: Water(5) ή Bridge(6) στα FRACTAL labels = OOD positive
    4. Calibration: βρες threshold που μεγιστοποιεί F1 στο val set
    5. Evaluation: AUROC / Precision / Recall / F1 στο test set

Αναφορά: Liu et al., "Energy-based Out-of-Distribution Detection", NeurIPS 2020

Χρήση:
    python src/ood_detector.py

Outputs:
    outputs/ood_results.json       ← final OOD metrics για Poster
    outputs/ood_energy_dist.csv    ← energy distributions για plotting
"""

import sys
import csv
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

# ── Project imports ─────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.dataset     import FractalPatchDataset, NUM_CLASSES, CLASS_NAMES
from models.pointnet2 import PointNet2Mini

# Water=5, Bridge=6 στο FRACTAL original labeling → OOD positives
OOD_CLASS_IDS = frozenset({5, 6})   # Water, Bridge
OOD_NAMES     = {5: "Water", 6: "Bridge"}


# ═══════════════════════════════════════════════════════════════════════════
# Inference helpers
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_energy_scores(
    model:  torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """
    Τρέχει inference σε όλα τα patches και συλλέγει:
        energy  : (M,) float32 — per-point Energy Score (υψηλό = OOD)
        ood_gt  : (M,) bool    — True αν original label ∈ {Water, Bridge}
        class_gt: (M,) int     — original FRACTAL class index (για ανάλυση)

    Αγνοεί points με label=-1 (ήδη invalid στο FRACTAL dataset).

    Energy Score:
        logits: (B, N, C_active)   — εδώ C_active=6 (Run 5)
        energy = -logsumexp(logits, dim=-1)   → (B, N)

    Όταν το μοντέλο είναι σίγουρο για κάποια κλάση:
        logsumexp ≈ max_logit (μεγάλο) → energy ≈ -max_logit (πολύ αρνητικό, χαμηλό)
    Όταν το μοντέλο είναι αβέβαιο (OOD point):
        logsumexp ≈ log(C) (μικρό) → energy ≈ -log(C) (λιγότερο αρνητικό, υψηλό)
    """
    model.eval()
    all_energy   = []
    all_ood_gt   = []
    all_class_gt = []

    for X, y in loader:
        X = X.to(device)              # (B, N, 7)
        # y παραμένει CPU: (B, N), τιμές 0-7 ή -1

        logits = model(X)             # (B, N, C_active)
        energy = -torch.logsumexp(logits, dim=-1)   # (B, N)

        energy_np = energy.cpu().numpy().reshape(-1)   # (B*N,)
        y_np      = y.numpy().reshape(-1)              # (B*N,)

        # Μόνο valid points (αποκλεισμός label=-1)
        valid     = y_np != -1
        energy_np = energy_np[valid]
        y_valid   = y_np[valid]

        ood_gt    = np.isin(y_valid, list(OOD_CLASS_IDS))

        all_energy.append(energy_np)
        all_ood_gt.append(ood_gt)
        all_class_gt.append(y_valid)

    return (
        np.concatenate(all_energy),
        np.concatenate(all_ood_gt),
        np.concatenate(all_class_gt),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Threshold calibration
# ═══════════════════════════════════════════════════════════════════════════

def find_best_threshold(
    energy:       np.ndarray,
    ood_gt:       np.ndarray,
    n_thresholds: int = 300,
) -> tuple:
    """
    Βρίσκει το threshold που μεγιστοποιεί F1 στο val set.

    Λογική: higher energy → more OOD → predict=True αν energy > threshold.
    Σαρώνουμε το εύρος [p1, p99] για να αποφύγουμε ακραίες τιμές.
    """
    p1, p99     = np.percentile(energy, [1, 99])
    thresholds  = np.linspace(p1, p99, n_thresholds)

    best_f1  = -1.0
    best_thr = thresholds[0]
    best_pr  = 0.0
    best_rec = 0.0

    for thr in thresholds:
        pred   = energy > thr
        tp     = (pred  &  ood_gt).sum()
        fp     = (pred  & ~ood_gt).sum()
        fn     = (~pred &  ood_gt).sum()

        prec   = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1     = 2 * prec * recall / (prec + recall + 1e-9)

        if f1 > best_f1:
            best_f1  = f1
            best_thr = thr
            best_pr  = prec
            best_rec = recall

    return float(best_thr), float(best_f1), float(best_pr), float(best_rec)


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_ood_metrics(
    energy:    np.ndarray,
    ood_gt:    np.ndarray,
    threshold: float,
) -> dict:
    """Precision / Recall / F1 @ threshold."""
    pred   = energy > threshold
    tp     = int((pred  &  ood_gt).sum())
    fp     = int((pred  & ~ood_gt).sum())
    fn     = int((~pred &  ood_gt).sum())
    tn     = int((~pred & ~ood_gt).sum())

    prec   = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1     = 2 * prec * recall / (prec + recall + 1e-9)

    return {
        "threshold": round(float(threshold), 4),
        "precision": round(prec, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_ood":   int(ood_gt.sum()),
        "n_known": int((~ood_gt).sum()),
        "n_total": len(ood_gt),
        "ood_pct": round(float(ood_gt.mean()) * 100, 2),
    }


def compute_auroc_aupr(energy: np.ndarray, ood_gt: np.ndarray) -> tuple:
    """
    AUROC και AUPR (Average Precision) χρησιμοποιώντας sklearn αν είναι
    διαθέσιμο, αλλιώς manual AUROC via trapezoidal rule.
    """
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auroc = float(roc_auc_score(ood_gt, energy))
        aupr  = float(average_precision_score(ood_gt, energy))
        return auroc, aupr

    except ImportError:
        # Manual AUROC μέσω ταξινόμησης
        sort_idx   = np.argsort(energy)[::-1]
        ood_sorted = ood_gt[sort_idx]
        n_pos = ood_gt.sum()
        n_neg = (~ood_gt).sum()

        tpr_list, fpr_list = [0.0], [0.0]
        tp = fp = 0
        for label in ood_sorted:
            if label:
                tp += 1
            else:
                fp += 1
            tpr_list.append(tp / (n_pos + 1e-9))
            fpr_list.append(fp / (n_neg + 1e-9))

        tpr_arr = np.array(tpr_list)
        fpr_arr = np.array(fpr_list)
        auroc   = float(np.trapz(tpr_arr, fpr_arr))
        return auroc, None


# ═══════════════════════════════════════════════════════════════════════════
# Visualization helpers
# ═══════════════════════════════════════════════════════════════════════════

def print_energy_stats(energy: np.ndarray, ood_gt: np.ndarray, split: str):
    """Εκτυπώνει στατιστικά κατανομής energy score για known vs OOD."""
    known_e = energy[~ood_gt]
    ood_e   = energy[ ood_gt]

    print(f"\n  [{split}] Energy Score distribution:")
    print(f"  {'':16}  {'mean':>8}  {'std':>7}  {'p5':>7}  {'p50':>7}  {'p95':>7}")
    print(f"  {'─'*58}")
    print(f"  {'Known (6 cls)':<16}  "
          f"{known_e.mean():>8.3f}  "
          f"{known_e.std():>7.3f}  "
          f"{np.percentile(known_e, 5):>7.3f}  "
          f"{np.percentile(known_e,50):>7.3f}  "
          f"{np.percentile(known_e,95):>7.3f}")

    if len(ood_e) > 0:
        print(f"  {'OOD (W+B)':<16}  "
              f"{ood_e.mean():>8.3f}  "
              f"{ood_e.std():>7.3f}  "
              f"{np.percentile(ood_e, 5):>7.3f}  "
              f"{np.percentile(ood_e,50):>7.3f}  "
              f"{np.percentile(ood_e,95):>7.3f}")

        # Separability: πόσες std units χωρίζουν Known και OOD
        delta_mean = ood_e.mean() - known_e.mean()
        pooled_std = np.sqrt((known_e.std()**2 + ood_e.std()**2) / 2)
        d_prime    = delta_mean / (pooled_std + 1e-9)
        quality    = "✓ καλή διαχωρισιμότητα" if d_prime > 1.0 else "⚠ χαμηλή"
        print(f"\n  Separability d' = {d_prime:.2f}  {quality}")
    else:
        print(f"  {'OOD (W+B)':<16}  — δεν βρέθηκαν OOD points σε αυτό το split")


def print_per_class_energy(energy: np.ndarray, class_gt: np.ndarray):
    """Μέσο energy ανά κλάση — για κατανόηση ποιες κλάσεις φαίνονται "OOD"."""
    print(f"\n  Mean energy ανά κλάση (χαμηλό = confident, υψηλό = uncertain):")
    print(f"  {'Κλάση':<18}  {'N points':>10}  {'Mean E':>9}  {'Std E':>8}")
    print(f"  {'─'*50}")

    for c in range(NUM_CLASSES):
        mask = class_gt == c
        if mask.sum() == 0:
            continue
        name = CLASS_NAMES.get(c, f"class_{c}")
        tag  = "  ← OOD target" if c in OOD_CLASS_IDS else ""
        print(f"  {name:<18}  {mask.sum():>10,}  {energy[mask].mean():>9.3f}  "
              f"{energy[mask].std():>8.3f}{tag}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ROOT / "outputs" / "checkpoints" / "best_model.pt"

    print(f"\n{'═'*62}")
    print(f"  OOD Detection — Energy Score")
    print(f"  Model  : Run 5  (6-class, Water/Bridge excluded)")
    print(f"  OOD GT : Water(5) + Bridge(6) from FRACTAL labels")
    print(f"  Ref    : Liu et al., NeurIPS 2020")
    print(f"{'═'*62}\n")

    # ── Load model ─────────────────────────────────────────────────────────
    if not ckpt_path.exists():
        print(f"❌  Checkpoint δεν βρέθηκε: {ckpt_path}")
        print("    Τρέξε πρώτα: python src/train.py --epochs 100 --bs 16 "
              "--cache --loss focal --gamma 2.0 --exclude-classes water bridge")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes  = ckpt.get("num_classes",  6)
    active_names = ckpt.get("active_names", [])

    print(f"► Checkpoint: epoch={ckpt['epoch']}, val mIoU={ckpt['mIoU']:.4f}, "
          f"classes={num_classes} ({', '.join(active_names)})")

    if num_classes == NUM_CLASSES:
        print("  ⚠️  Αυτό φαίνεται να είναι 8-class μοντέλο (Run 4), όχι 6-class Run 5.")
        print("  Για OOD evaluation χρειάζεται Run 5 (--exclude-classes water bridge).")

    model = PointNet2Mini(in_channels=7, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Μοντέλο φορτώθηκε ✓\n")

    # ── Load datasets (ORIGINAL labels — χωρίς RemappedDataset) ───────────
    # Χρειαζόμαστε τα ΑΡΧΙΚΑ labels για να ξέρουμε ποια είναι Water/Bridge
    print("► Φόρτωση datasets (original FRACTAL labels)...")
    nw = 0 if sys.platform == "win32" else 4

    val_ds   = FractalPatchDataset(ROOT, split="val",  num_points=4096, cache=True)
    test_ds  = FractalPatchDataset(ROOT, split="test", num_points=4096, cache=True)

    val_loader  = DataLoader(val_ds,  batch_size=16, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=nw)

    print(f"  Val: {len(val_ds):,} patches | Test: {len(test_ds):,} patches\n")

    # ── Phase 1: Calibration on val ────────────────────────────────────────
    print("► Phase 1 — Val set: υπολογισμός energy scores...")
    val_energy, val_ood_gt, val_class_gt = collect_energy_scores(
        model, val_loader, device
    )

    n_ood_val   = val_ood_gt.sum()
    n_known_val = (~val_ood_gt).sum()
    print(f"  Σύνολο points: {len(val_ood_gt):,}  |  "
          f"OOD (W+B): {n_ood_val:,} ({n_ood_val/len(val_ood_gt)*100:.1f}%)  |  "
          f"Known: {n_known_val:,}")

    print_energy_stats(val_energy, val_ood_gt, "val")
    print_per_class_energy(val_energy, val_class_gt)

    # Calibrate threshold
    print(f"\n  Βελτιστοποίηση threshold (maximize F1)...")
    best_thr, best_f1_cal, best_pr_cal, best_rec_cal = find_best_threshold(
        val_energy, val_ood_gt
    )
    auroc_val, aupr_val = compute_auroc_aupr(val_energy, val_ood_gt)
    val_m = compute_ood_metrics(val_energy, val_ood_gt, best_thr)

    print(f"\n  ── Val Calibration Results ──────────────────────")
    print(f"  Threshold  : {best_thr:+.4f}")
    print(f"  AUROC      : {auroc_val:.4f}  {'✓ excellent' if auroc_val>0.90 else '✓ good' if auroc_val>0.80 else '⚠ moderate'}")
    if aupr_val is not None:
        print(f"  AUPR       : {aupr_val:.4f}")
    print(f"  Precision  : {val_m['precision']:.4f}")
    print(f"  Recall     : {val_m['recall']:.4f}")
    print(f"  F1         : {val_m['f1']:.4f}")

    # ── Phase 2: Evaluation on test ────────────────────────────────────────
    print(f"\n► Phase 2 — Test set: evaluation με threshold={best_thr:+.4f}...")
    test_energy, test_ood_gt, test_class_gt = collect_energy_scores(
        model, test_loader, device
    )

    n_ood_test   = test_ood_gt.sum()
    n_known_test = (~test_ood_gt).sum()
    print(f"  Σύνολο points: {len(test_ood_gt):,}  |  "
          f"OOD (W+B): {n_ood_test:,} ({n_ood_test/len(test_ood_gt)*100:.1f}%)  |  "
          f"Known: {n_known_test:,}")

    print_energy_stats(test_energy, test_ood_gt, "test")

    test_m          = compute_ood_metrics(test_energy, test_ood_gt, best_thr)
    auroc_test, aupr_test = compute_auroc_aupr(test_energy, test_ood_gt)

    print(f"\n{'═'*62}")
    print(f"  OOD DETECTION — ΤΕΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ (TEST SET)")
    print(f"{'═'*62}")
    print(f"  AUROC     : {auroc_test:.4f}  "
          f"{'✓ excellent' if auroc_test>0.90 else '✓ good' if auroc_test>0.80 else '⚠ moderate'}")
    if aupr_test is not None:
        print(f"  AUPR      : {aupr_test:.4f}")
    print(f"  Threshold : {best_thr:+.4f}  (calibrated on val)")
    print(f"  Precision : {test_m['precision']:.4f}")
    print(f"  Recall    : {test_m['recall']:.4f}")
    print(f"  F1        : {test_m['f1']:.4f}")
    print(f"\n  Confusion Matrix (test):")
    print(f"               Predicted")
    print(f"               OOD    Known")
    print(f"  Actual OOD   {test_m['tp']:>6,}  {test_m['fn']:>6,}")
    print(f"  Actual Known {test_m['fp']:>6,}  {test_m['tn']:>6,}")
    print(f"{'═'*62}\n")

    # ── Save JSON ──────────────────────────────────────────────────────────
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": {
            "checkpoint_epoch": int(ckpt["epoch"]),
            "val_miou":   round(float(ckpt["mIoU"]), 4),
            "num_classes": num_classes,
            "active_names": active_names,
        },
        "ood_setup": {
            "ood_class_ids":   list(OOD_CLASS_IDS),
            "ood_class_names": [OOD_NAMES[i] for i in OOD_CLASS_IDS],
            "method":          "Energy Score: E(x) = -logsumexp(f(x))",
            "reference":       "Liu et al., NeurIPS 2020",
        },
        "calibration_val": {
            "threshold":  round(float(best_thr), 4),
            "auroc":      round(auroc_val,  4),
            "aupr":       round(aupr_val,   4) if aupr_val else None,
            "precision":  val_m["precision"],
            "recall":     val_m["recall"],
            "f1":         val_m["f1"],
            "tp": val_m["tp"], "fp": val_m["fp"],
            "fn": val_m["fn"], "tn": val_m["tn"],
            "n_ood":   int(n_ood_val),
            "n_known": int(n_known_val),
        },
        "test_results": {
            "threshold":  round(float(best_thr), 4),
            "auroc":      round(auroc_test, 4),
            "aupr":       round(aupr_test,  4) if aupr_test else None,
            "precision":  test_m["precision"],
            "recall":     test_m["recall"],
            "f1":         test_m["f1"],
            "tp": test_m["tp"], "fp": test_m["fp"],
            "fn": test_m["fn"], "tn": test_m["tn"],
            "n_ood":   int(n_ood_test),
            "n_known": int(n_known_test),
        },
        "energy_stats": {
            "val_known_mean":  round(float(val_energy[~val_ood_gt].mean()),  4),
            "val_known_p95":   round(float(np.percentile(val_energy[~val_ood_gt], 95)), 4),
            "val_ood_mean":    round(float(val_energy[val_ood_gt].mean()),   4) if n_ood_val > 0 else None,
            "val_ood_p5":      round(float(np.percentile(val_energy[val_ood_gt], 5)), 4) if n_ood_val > 0 else None,
            "test_known_mean": round(float(test_energy[~test_ood_gt].mean()), 4),
            "test_ood_mean":   round(float(test_energy[test_ood_gt].mean()),  4) if n_ood_test > 0 else None,
        }
    }

    json_path = out_dir / "ood_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"► JSON αποθηκεύτηκε : {json_path}")

    # ── Save energy distributions CSV (για notebook plotting) ─────────────
    csv_path = out_dir / "ood_energy_dist.csv"
    rng      = np.random.default_rng(42)

    def sample_rows(energy, ood_gt, class_gt, split, max_per_class=5000):
        """Sample για να κρατήσουμε το CSV manageable."""
        rows = []
        for c in list(range(NUM_CLASSES)):
            mask    = class_gt == c
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue
            chosen = rng.choice(indices, min(len(indices), max_per_class), replace=False)
            for idx in chosen:
                rows.append((split, float(energy[idx]), int(ood_gt[idx]), int(c)))
        return rows

    rows_val  = sample_rows(val_energy,  val_ood_gt,  val_class_gt,  "val")
    rows_test = sample_rows(test_energy, test_ood_gt, test_class_gt, "test")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "energy", "ood_gt", "class_id"])
        writer.writerows(rows_val)
        writer.writerows(rows_test)

    print(f"► CSV αποθηκεύτηκε  : {csv_path}  "
          f"({len(rows_val)+len(rows_test):,} rows)")

    # Summary line for Poster
    print(f"\n  ── POSTER NUMBERS ──────────────────────────────────")
    print(f"  OOD AUROC = {auroc_test:.4f}  |  "
          f"Precision = {test_m['precision']:.4f}  |  "
          f"Recall = {test_m['recall']:.4f}  |  "
          f"F1 = {test_m['f1']:.4f}")
    print(f"  Threshold = {best_thr:+.4f}  (calibrated on val, applied to test)\n")


if __name__ == "__main__":
    main()
