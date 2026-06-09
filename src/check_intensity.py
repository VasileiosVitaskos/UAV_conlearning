"""
check_intensity.py
Feature-level analysis: Water vs Ground στα raw LiDAR features.

Υπόθεση: το Water στο ALS LiDAR έχει χαρακτηριστικό χαμηλό intensity
λόγω specular reflection (IR laser αντανακλάται εκτός beam).
Αν αυτό ισχύει στα FRACTAL data, ένα απλό feature threshold μπορεί να
ανιχνεύσει το Water καλύτερα από το Energy Score ή Mahalanobis.

Features (index):
    0: x_rel
    1: y_rel
    2: z_rel
    3: intensity         <-- κύριο ενδιαφέρον
    4: return_number
    5: number_of_returns
    6: scan_angle

Χρήση:
    python src/check_intensity.py
    python src/check_intensity.py --split val --max-patches 200

Output:
    Εκτυπώνει per-feature stats για Water vs Ground
    Εκτιμά AUROC ανά raw feature
    Εκτυπώνει threshold που μεγιστοποιεί F1
"""

import sys
import argparse
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import FractalPatchDataset, NUM_CLASSES, CLASS_NAMES

# Classes of interest
TARGET_CLASS  = 5   # Water (OOD)
COMPARE_CLASS = 0   # Ground (most similar)

FEATURE_NAMES = [
    "x_rel", "y_rel", "z_rel",
    "intensity",
    "return_number",
    "number_of_returns",
    "scan_angle",
]


# ============================================================================
# Data collection
# ============================================================================

def collect_features(loader, max_patches: int = 0):
    """
    Συλλέγει raw features για Water και Ground points.

    Returns:
        water_feats  : (N_water, 7)
        ground_feats : (N_ground, 7)
        ood_gt       : (M,) bool  -- True = Water or Bridge
        all_feats    : (M, 7)     -- όλα τα valid points
        all_labels   : (M,) int
    """
    water_feats  = []
    ground_feats = []
    all_feats    = []
    all_labels   = []
    done         = 0

    for X, y in loader:
        if max_patches > 0 and done >= max_patches:
            break

        X_np = X.numpy()   # (B, N, 7)
        y_np = y.numpy()   # (B, N)

        for b in range(X_np.shape[0]):
            feats  = X_np[b]   # (N, 7)
            labels = y_np[b]   # (N,)

            valid = labels != -1
            feats  = feats[valid]
            labels = labels[valid]

            water_mask  = labels == TARGET_CLASS
            ground_mask = labels == COMPARE_CLASS

            if water_mask.sum() > 0:
                water_feats.append(feats[water_mask])
            if ground_mask.sum() > 0:
                ground_feats.append(feats[ground_mask])

            all_feats.append(feats)
            all_labels.append(labels)

        done += X_np.shape[0]
        if done % 64 == 0:
            print(f"  {done} patches...", end="\r")

    print()
    water_feats  = np.concatenate(water_feats,  axis=0) if water_feats  else np.empty((0, 7))
    ground_feats = np.concatenate(ground_feats, axis=0) if ground_feats else np.empty((0, 7))
    all_feats    = np.concatenate(all_feats,    axis=0)
    all_labels   = np.concatenate(all_labels,   axis=0)

    # OOD GT: Water(5) or Bridge(6) = positive
    ood_gt = np.isin(all_labels, [5, 6])

    return water_feats, ground_feats, ood_gt, all_feats, all_labels


# ============================================================================
# Analysis
# ============================================================================

def separability_d_prime(a: np.ndarray, b: np.ndarray) -> float:
    """d' = (mean_b - mean_a) / pooled_std"""
    delta  = b.mean() - a.mean()
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2 + 1e-9)
    return float(delta / pooled)


def auroc_1d(scores: np.ndarray, ood_gt: np.ndarray) -> float:
    """Quick AUROC for a 1D OOD score array."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(ood_gt, scores))
    except ImportError:
        # Manual
        sort_idx   = np.argsort(scores)[::-1]
        ood_sorted = ood_gt[sort_idx]
        n_pos = ood_gt.sum()
        n_neg = (~ood_gt).sum()
        tp = fp = 0
        tpr = [0.0]
        fpr = [0.0]
        for lbl in ood_sorted:
            if lbl:
                tp += 1
            else:
                fp += 1
            tpr.append(tp / (n_pos + 1e-9))
            fpr.append(fp / (n_neg + 1e-9))
        return float(np.trapz(tpr, fpr))


def best_threshold_f1(scores: np.ndarray, ood_gt: np.ndarray,
                      n_thr: int = 300) -> tuple:
    """Threshold @ max F1."""
    lo, hi = np.percentile(scores, [1, 99])
    thresholds = np.linspace(lo, hi, n_thr)
    best_f1 = best_thr = best_pr = best_rec = 0.0
    for thr in thresholds:
        pred = scores > thr
        tp   = (pred &  ood_gt).sum()
        fp   = (pred & ~ood_gt).sum()
        fn   = (~pred & ood_gt).sum()
        p    = tp / (tp + fp + 1e-9)
        r    = tp / (tp + fn + 1e-9)
        f1   = 2 * p * r / (p + r + 1e-9)
        if f1 > best_f1:
            best_f1  = f1
            best_thr = thr
            best_pr  = p
            best_rec = r
    return float(best_thr), float(best_f1), float(best_pr), float(best_rec)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",        default="val",
                   choices=["train", "val", "test"])
    p.add_argument("--max-patches",  type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*64}")
    print(f"  Feature Analysis: Water vs Ground (raw LiDAR features)")
    print(f"  Split: {args.split}  max_patches={args.max_patches}")
    print(f"{'='*64}\n")

    ds     = FractalPatchDataset(ROOT, split=args.split, num_points=4096, cache=True)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    print("  Collecting features...")
    water_f, ground_f, ood_gt, all_f, all_labels = collect_features(
        loader, max_patches=args.max_patches
    )

    print(f"  Water  points: {len(water_f):,}")
    print(f"  Ground points: {len(ground_f):,}")
    print(f"  Total  valid : {len(all_f):,}  OOD (W+B): {ood_gt.sum():,} ({ood_gt.mean()*100:.1f}%)\n")

    if len(water_f) == 0:
        print("  No Water points found in this split/patch subset.")
        return

    # ── Per-feature comparison ─────────────────────────────────────────────
    print(f"  {'Feature':<20}  {'Water mean':>12}  {'Ground mean':>12}  "
          f"{'d_prime':>9}  {'AUROC (feat)':>13}")
    print(f"  {'-'*72}")

    best_auroc_feat  = -1.0
    best_feat_name   = ""
    best_feat_scores = None

    for fi, fname in enumerate(FEATURE_NAMES):
        w_vals = water_f[:, fi]
        g_vals = ground_f[:, fi]
        a_vals = all_f[:, fi]

        d_prime = separability_d_prime(g_vals, w_vals)

        # AUROC using this feature as OOD score (try both directions)
        auroc_pos = auroc_1d( a_vals, ood_gt)   # high feature = OOD
        auroc_neg = auroc_1d(-a_vals, ood_gt)   # low  feature = OOD
        auroc, direction = (auroc_pos, "high") if auroc_pos >= auroc_neg else (auroc_neg, "low")

        flag = "  <-- best!" if auroc > 0.55 else ("  <-- inverted!" if auroc < 0.45 else "")

        print(f"  {fname:<20}  {w_vals.mean():>12.4f}  {g_vals.mean():>12.4f}  "
              f"{d_prime:>9.3f}  {auroc:>12.4f}  ({direction}){flag}")

        if auroc > best_auroc_feat:
            best_auroc_feat  = auroc
            best_feat_name   = fname
            best_feat_scores = a_vals if direction == "high" else -a_vals

    print(f"\n  Best single-feature AUROC: {best_feat_name} = {best_auroc_feat:.4f}")

    # ── Intensity deep-dive ────────────────────────────────────────────────
    fi_intensity = 3
    w_int = water_f[:, fi_intensity]
    g_int = ground_f[:, fi_intensity]

    print(f"\n  {'='*64}")
    print(f"  Intensity deep-dive: Water vs Ground")
    print(f"  {'='*64}")
    print(f"  {'':20}  {'Water':>12}  {'Ground':>12}")
    print(f"  {'-'*47}")
    for pct in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
        pw = np.percentile(w_int, pct)
        pg = np.percentile(g_int, pct)
        marker = " <--" if abs(pw - pg) > 0.2 * max(abs(pw), abs(pg), 1e-9) else ""
        print(f"  {'p' + str(pct):<20}  {pw:>12.4f}  {pg:>12.4f}{marker}")

    # ── Return number ──────────────────────────────────────────────────────
    fi_ret = 4
    fi_nret = 5
    w_ret  = water_f[:, fi_ret]
    g_ret  = ground_f[:, fi_ret]
    w_nret = water_f[:, fi_nret]
    g_nret = ground_f[:, fi_nret]

    print(f"\n  Return characteristics:")
    print(f"  {'Feature':<20}  {'Water mean':>12}  {'Ground mean':>12}")
    print(f"  {'-'*47}")
    print(f"  {'return_number':<20}  {w_ret.mean():>12.3f}  {g_ret.mean():>12.3f}")
    print(f"  {'num_of_returns':<20}  {w_nret.mean():>12.3f}  {g_nret.mean():>12.3f}")
    print(f"  % single-return: Water {(w_nret == 1).mean()*100:.1f}%  "
          f"Ground {(g_nret == 1).mean()*100:.1f}%")

    # ── AUROC comparison table ─────────────────────────────────────────────
    print(f"\n  {'='*64}")
    print(f"  AUROC comparison: model-based vs feature-based OOD")
    print(f"  {'='*64}")
    print(f"  {'Method':<35}  {'AUROC':>8}")
    print(f"  {'-'*45}")
    print(f"  {'Energy Score (Run 5)':<35}  {'0.4321':>8}")
    print(f"  {'Mahalanobis (32-dim emb)':<35}  {'0.4104':>8}")
    for fi, fname in enumerate(FEATURE_NAMES):
        a_vals   = all_f[:, fi]
        auroc_p  = auroc_1d( a_vals, ood_gt)
        auroc_n  = auroc_1d(-a_vals, ood_gt)
        auroc    = max(auroc_p, auroc_n)
        print(f"  {'Feature: ' + fname:<35}  {auroc:>8.4f}")

    # ── Best feature threshold @ max F1 ───────────────────────────────────
    if best_auroc_feat > 0.52:
        print(f"\n  Best feature ({best_feat_name}) threshold @ max F1:")
        thr, f1, prec, rec = best_threshold_f1(best_feat_scores, ood_gt)
        print(f"  Threshold = {thr:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
        print(f"\n  --> A simple '{best_feat_name} > {thr:.3f}' rule gives AUROC={best_auroc_feat:.4f}")
        print(f"      This is {'BETTER' if best_auroc_feat > 0.5 else 'worse'} than the model-based methods!")
    else:
        print(f"\n  No single raw feature achieves AUROC > 0.52.")
        print(f"  --> Water and Ground are indistinguishable at BOTH the feature level")
        print(f"      AND the embedding level. The ALS sensor cannot separate them.")
        print(f"      This is a FUNDAMENTAL sensor limitation, not a model limitation.")


if __name__ == "__main__":
    main()
