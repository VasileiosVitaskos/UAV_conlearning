"""
ood_mahalanobis.py
OOD Detection via Mahalanobis Distance in Embedding Space.

Αναφορά: Lee et al., "A Simple Unified Framework for Detecting Out-of-Distribution
Samples and Adversarial Attacks", NeurIPS 2018.

Γιατί Mahalanobis αντί για Energy Score:
  Energy Score αποτυγχάνει για Water (AUROC=0.43) γιατί το μοντέλο ταξινομεί
  Water ως Ground με υψηλή εμπιστοσύνη (d'=0.05). Τα logits είναι
  1D projection 32-dim -> 6-dim -> scalar. Πληροφορία χάνεται.

  Η Mahalanobis distance χρησιμοποιεί το ΠΛΗΡΕΣ 32-dim embedding space:
      score(z) = min_c sqrt[(z - mu_c)^T Sigma^-1 (z - mu_c)]

  Υψηλή score -> η embedding είναι μακριά από ΟΛΕΣ τις known classes -> OOD.
  Αν το Water έχει έστω μικρές συστηματικές διαφορές από το Ground στον
  32-dim χώρο, η Mahalanobis θα τις ανιχνεύσει.

Pipeline:
  1. Φόρτωση Run 5 checkpoint (6-class)
  2. Συλλογή embeddings από train set (subset ~300 patches)
     -> Εκτίμηση class means mu_c (32-dim) και tied covariance Sigma (32x32)
  3. Val set: scores + threshold calibration (max F1)
  4. Test set: AUROC / Precision / Recall / F1
  5. Σύγκριση με Energy Score αποτελέσματα

Χρήση:
    python src/ood_mahalanobis.py
    python src/ood_mahalanobis.py --fit-patches 50 --max-eval-patches 50   # quick

Outputs:
    outputs/ood_mahal_results.json     Mahalanobis OOD metrics
    outputs/ood_comparison.json        Συγκριση Energy vs Mahalanobis
"""

import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── Project imports ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.dataset     import FractalPatchDataset, NUM_CLASSES, CLASS_NAMES
from models.pointnet2 import PointNet2Mini
from ood_detector     import (
    OOD_CLASS_IDS, OOD_NAMES,
    find_best_threshold, compute_ood_metrics, compute_auroc_aupr,
)

SEP_HEAVY = "=" * 66
SEP_LIGHT = "-" * 66


# ============================================================================
# Embedding collection
# ============================================================================

@torch.no_grad()
def collect_train_embeddings(
    model:         torch.nn.Module,
    loader:        DataLoader,
    device:        torch.device,
    known_classes: set,
    max_patches:   int = 300,
) -> dict:
    """
    Συλλέγει embeddings για κάθε γνωστή κλάση από το training set.

    Χρησιμοποιείται για να εκτιμηθούν οι class means mu_c και η tied Sigma.

    max_patches: 300 patches x 4096 pts x 32 dims ≈ 1.5 GB float32,
    μειώνεται με subsampling 500 pts/class/patch.

    Returns: {class_id: np.ndarray (N_c, 32)}
    """
    model.eval()
    emb_by_class  = {c: [] for c in known_classes}
    patches_done  = 0

    for X, y in loader:
        if patches_done >= max_patches:
            break

        X      = X.to(device)
        emb    = model.get_embeddings(X)      # (B, N, 32)
        emb_np = emb.cpu().numpy()            # (B, N, 32)
        y_np   = y.numpy()                    # (B, N)

        B = emb_np.shape[0]
        for b in range(B):
            emb_b = emb_np[b]   # (N, 32)
            y_b   = y_np[b]     # (N,)
            for c in known_classes:
                idx = np.where(y_b == c)[0]
                if len(idx) == 0:
                    continue
                if len(idx) > 500:
                    idx = np.random.choice(idx, 500, replace=False)
                emb_by_class[c].append(emb_b[idx])

        patches_done += B
        if patches_done % 32 == 0:
            print(f"    {patches_done}/{max_patches} patches processed...", end="\r")

    print()
    result = {}
    for c in known_classes:
        parts = emb_by_class[c]
        if parts:
            result[c] = np.concatenate(parts, axis=0)
    return result


@torch.no_grad()
def collect_eval_embeddings(
    model:       torch.nn.Module,
    loader:      DataLoader,
    device:      torch.device,
    max_patches: int = 0,
) -> tuple:
    """
    Συλλέγει embeddings + original labels για val/test set.

    Returns:
        embeddings : (M, 32) float32
        ood_gt     : (M,)   bool   -- True αν label in {Water, Bridge}
        class_gt   : (M,)   int    -- original FRACTAL label
    """
    model.eval()
    all_emb      = []
    all_ood_gt   = []
    all_class_gt = []
    patches_done = 0

    for X, y in loader:
        if max_patches > 0 and patches_done >= max_patches:
            break

        X      = X.to(device)
        emb    = model.get_embeddings(X)      # (B, N, 32)
        emb_np = emb.cpu().numpy().reshape(-1, 32)
        y_flat = y.numpy().reshape(-1)

        valid    = y_flat != -1
        emb_np   = emb_np[valid]
        y_flat   = y_flat[valid]
        ood_gt   = np.isin(y_flat, list(OOD_CLASS_IDS))

        all_emb.append(emb_np)
        all_ood_gt.append(ood_gt)
        all_class_gt.append(y_flat)
        patches_done += X.shape[0]

    return (
        np.concatenate(all_emb,      axis=0),
        np.concatenate(all_ood_gt,   axis=0),
        np.concatenate(all_class_gt, axis=0),
    )


# ============================================================================
# Gaussian fitting
# ============================================================================

def fit_gaussian_params(
    embeddings_by_class: dict,
    reg: float = 1e-3,
) -> tuple:
    """
    Εκτιμά class means mu_c και tied covariance Sigma (shared across classes).

    Returns:
        means     : {class_id: (D,) ndarray}
        Sigma_inv : (D, D) ndarray
    """
    D     = next(iter(embeddings_by_class.values())).shape[1]
    means = {}
    all_centered = []
    total_n      = 0

    for c, emb in embeddings_by_class.items():
        mu       = emb.mean(axis=0)
        means[c] = mu
        centered = emb - mu
        all_centered.append(centered)
        total_n += len(centered)
        print(f"    Class {CLASS_NAMES.get(c, f'cls_{c}'):<16}: "
              f"{len(emb):>8,} pts  mu_L2={np.linalg.norm(mu):.3f}  "
              f"mu_range=[{mu.min():.3f}, {mu.max():.3f}]")

    all_c = np.concatenate(all_centered, axis=0)
    Sigma = (all_c.T @ all_c) / total_n
    Sigma += np.eye(D) * reg   # regularize for invertibility

    print(f"\n  Tied Sigma ({D}x{D}):  "
          f"trace={np.trace(Sigma):.3f}  "
          f"cond={np.linalg.cond(Sigma):.2e}  "
          f"reg={reg}")

    Sigma_inv = np.linalg.inv(Sigma)
    return means, Sigma_inv


# ============================================================================
# Mahalanobis scoring  (CPU numpy + GPU torch variants)
# ============================================================================

def mahalanobis_scores_cpu(
    embeddings: np.ndarray,
    means:      dict,
    Sigma_inv:  np.ndarray,
    batch_size: int = 50_000,
) -> np.ndarray:
    """CPU numpy implementation (fallback)."""
    M          = len(embeddings)
    min_scores = np.full(M, np.inf, dtype=np.float32)
    class_ids  = sorted(means.keys())

    for i, c in enumerate(class_ids):
        mu       = means[c]
        scores_c = np.empty(M, dtype=np.float32)
        for start in range(0, M, batch_size):
            end  = min(start + batch_size, M)
            diff = embeddings[start:end] - mu
            tmp  = diff @ Sigma_inv
            m2   = (tmp * diff).sum(axis=1)
            scores_c[start:end] = np.sqrt(np.maximum(m2, 0.0))
        min_scores = np.minimum(min_scores, scores_c)
        print(f"    [{i+1}/{len(class_ids)}] {CLASS_NAMES.get(c, f'cls_{c}'):<14}: "
              f"mean={scores_c.mean():.3f}")

    return min_scores


def mahalanobis_scores_gpu(
    embeddings: np.ndarray,
    means:      dict,
    Sigma_inv:  np.ndarray,
    device:     torch.device,
    batch_size: int = 200_000,
) -> np.ndarray:
    """
    GPU-accelerated Mahalanobis scoring (RTX 3060 optimized).

    Βήμα A: μεταφέρουμε Sigma^-1 στη GPU μία φορά (32x32 = ασήμαντο VRAM).
    Βήμα B: για κάθε κλάση c, κάνουμε batch forward pass:
        diff   = embeddings - mu_c          (B, 32)
        tmp    = diff @ Sigma_inv           (B, 32)
        m2     = (tmp * diff).sum(dim=1)    (B,)   <- Mahalanobis^2
        score  = sqrt(m2)                   (B,)

    RTX 3060 (12GB): batch_size=200K -> 200K x 32 x 4 bytes = 25MB per batch.
    Αναμενόμενος χρόνος: ~2-4 δευτερόλεπτα για 4M σημεία.
    """
    M          = len(embeddings)
    Sinv_t     = torch.from_numpy(Sigma_inv.astype(np.float32)).to(device)
    class_ids  = sorted(means.keys())
    n_cls      = len(class_ids)

    min_scores = torch.full((M,), float("inf"), device=device, dtype=torch.float32)

    for i, c in enumerate(class_ids):
        mu_t     = torch.from_numpy(means[c].astype(np.float32)).to(device)
        scores_c = torch.empty(M, device=device, dtype=torch.float32)

        for start in range(0, M, batch_size):
            end   = min(start + batch_size, M)
            emb_t = torch.from_numpy(
                embeddings[start:end].astype(np.float32)
            ).to(device)
            diff  = emb_t - mu_t
            tmp   = diff @ Sinv_t
            m2    = (tmp * diff).sum(dim=1)
            scores_c[start:end] = m2.clamp(min=0.0).sqrt()

        min_scores = torch.minimum(min_scores, scores_c)
        print(f"    [{i+1}/{n_cls}] {CLASS_NAMES.get(c, f'cls_{c}'):<14}: "
              f"mean={scores_c.mean().item():.3f}  "
              f"min={scores_c.min().item():.3f}  "
              f"max={scores_c.max().item():.3f}")

    return min_scores.cpu().numpy()


# ============================================================================
# Stats printing
# ============================================================================

def print_mahal_stats(scores: np.ndarray, ood_gt: np.ndarray,
                      class_gt: np.ndarray, split: str):
    """Στατιστικά Mahalanobis score ανά original κλάση."""
    known_s = scores[~ood_gt]
    ood_s   = scores[ ood_gt]

    print(f"\n  [{split}] Mahalanobis score distribution:")
    print(f"  {'':16}  {'mean':>8}  {'std':>7}  {'p5':>7}  {'p50':>7}  {'p95':>7}")
    print(f"  {'-'*58}")
    print(f"  {'Known (6 cls)':<16}  "
          f"{known_s.mean():>8.3f}  {known_s.std():>7.3f}  "
          f"{np.percentile(known_s,  5):>7.3f}  "
          f"{np.percentile(known_s, 50):>7.3f}  "
          f"{np.percentile(known_s, 95):>7.3f}")

    if len(ood_s) > 0:
        print(f"  {'OOD (W+B)':<16}  "
              f"{ood_s.mean():>8.3f}  {ood_s.std():>7.3f}  "
              f"{np.percentile(ood_s,  5):>7.3f}  "
              f"{np.percentile(ood_s, 50):>7.3f}  "
              f"{np.percentile(ood_s, 95):>7.3f}")

        delta     = ood_s.mean() - known_s.mean()
        pooled    = np.sqrt((known_s.std()**2 + ood_s.std()**2) / 2)
        d_prime   = delta / (pooled + 1e-9)
        quality   = "OK separability" if d_prime > 1.0 else "low separability"
        print(f"\n  Separability d' = {d_prime:.2f}  ({quality})")
    else:
        print("  OOD (W+B)        -- no OOD points found")

    # Per-class table
    print(f"\n  Mean Mahalanobis score per class:")
    print(f"  {'Class':<18}  {'N points':>10}  {'Mean':>8}  {'Std':>7}")
    print(f"  {'-'*50}")
    for c in range(NUM_CLASSES):
        mask = class_gt == c
        if mask.sum() == 0:
            continue
        name = CLASS_NAMES.get(c, f"class_{c}")
        tag  = "  <- OOD target" if c in OOD_CLASS_IDS else ""
        print(f"  {name:<18}  {mask.sum():>10,}  "
              f"{scores[mask].mean():>8.3f}  {scores[mask].std():>7.3f}{tag}")


# ============================================================================
# Args
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Mahalanobis OOD detection")
    p.add_argument("--fit-patches",      type=int,   default=300,
                   help="train patches for fitting Sigma (default: 300)")
    p.add_argument("--max-eval-patches", type=int,   default=0,
                   help="max val/test patches (0=all)")
    p.add_argument("--reg",              type=float, default=1e-3,
                   help="Sigma regularization (default: 1e-3)")
    p.add_argument("--batch-size",       type=int,   default=0,
                   help="DataLoader batch size (0=auto: 32 GPU / 8 CPU)")
    p.add_argument("--seed",             type=int,   default=42)
    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # RTX 3060 (12 GB) -> batch=32 σε άνεση για forward pass
    bs = (32 if device.type == "cuda" else 8) if args.batch_size == 0 else args.batch_size

    ckpt_path = ROOT / "outputs" / "checkpoints" / "best_model.pt"

    print(f"\n{SEP_HEAVY}")
    print(f"  OOD Detection -- Mahalanobis Distance in Embedding Space")
    print(f"  Model  : Run 5  (6-class, Water/Bridge excluded)")
    print(f"  Method : Lee et al., NeurIPS 2018")
    print(f"  OOD GT : Water(5) + Bridge(6) from FRACTAL labels")
    print(f"  Device : {device}  (DataLoader bs={bs})")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU    : {torch.cuda.get_device_name(0)}"
              f"  ({props.total_memory // 1024**2:,} MB VRAM)")
    print(f"{SEP_HEAVY}\n")

    # -- Load model -----------------------------------------------------------
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt         = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes  = ckpt.get("num_classes",  6)
    active_names = ckpt.get("active_names", [])
    remap_lut    = ckpt.get("remap_lut",    None)

    print(f"Checkpoint: epoch={ckpt['epoch']}, "
          f"val mIoU={ckpt['mIoU']:.4f}, "
          f"classes={num_classes} ({', '.join(active_names)})")

    if num_classes == 8:
        print("ERROR: Need Run 5 (6-class) checkpoint. Got 8-class (Run 4).")
        sys.exit(1)

    model = PointNet2Mini(in_channels=7, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}\n")

    # Known class IDs (original FRACTAL IDs excluding Water/Bridge)
    if remap_lut is not None:
        known_class_ids = {int(i) for i in range(8) if remap_lut[i].item() != -1}
    else:
        known_class_ids = set(range(8)) - OOD_CLASS_IDS
    print(f"  Known class IDs : {sorted(known_class_ids)}")
    print(f"  OOD   class IDs : {sorted(OOD_CLASS_IDS)}\n")

    # -- DataLoaders (no RemappedDataset -- need original labels) -------------
    nw  = 0 if sys.platform == "win32" else 4
    pin = device.type == "cuda"

    def make_loader(split: str, shuffle: bool = False) -> DataLoader:
        ds = FractalPatchDataset(ROOT, split=split, num_points=4096, cache=True)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                          num_workers=nw, pin_memory=pin)

    # =========================================================================
    # Step 1: Fit class means & covariance from train set
    # =========================================================================
    print(SEP_LIGHT)
    print(f"  Step 1: Fitting Gaussian parameters from train set")
    print(f"  max_patches={args.fit_patches}  "
          f"(~{args.fit_patches * 4096 * 500 // 1_000_000}M embedding points)")
    print(SEP_LIGHT)

    t_fit = time.perf_counter()
    train_loader = make_loader("train", shuffle=True)
    emb_by_cls   = collect_train_embeddings(
        model, train_loader, device,
        known_classes=known_class_ids,
        max_patches=args.fit_patches,
    )

    n_fit = sum(len(v) for v in emb_by_cls.values())
    print(f"\n  Total fitting points: {n_fit:,}")

    print(f"\n  Fitting class means & tied Sigma:")
    means, Sigma_inv = fit_gaussian_params(emb_by_cls, reg=args.reg)
    print(f"  Fitting time: {time.perf_counter()-t_fit:.1f}s")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # =========================================================================
    # Step 2: Val set -- calibrate threshold
    # =========================================================================
    print(f"\n{SEP_LIGHT}")
    print(f"  Step 2: Val set -- scores & threshold calibration")
    print(SEP_LIGHT)

    val_loader = make_loader("val")
    print("  Collecting val embeddings...")
    t0 = time.perf_counter()
    val_emb, val_ood_gt, val_class_gt = collect_eval_embeddings(
        model, val_loader, device, max_patches=args.max_eval_patches)
    print(f"  Val: {len(val_emb):,} points  "
          f"(OOD: {val_ood_gt.sum():,} = {val_ood_gt.mean()*100:.1f}%)  "
          f"[{time.perf_counter()-t0:.1f}s]\n")

    print("  Computing Mahalanobis scores (val)...")
    t0 = time.perf_counter()
    val_scores = (mahalanobis_scores_gpu(val_emb, means, Sigma_inv, device)
                  if device.type == "cuda"
                  else mahalanobis_scores_cpu(val_emb, means, Sigma_inv))
    print(f"  Scoring time: {time.perf_counter()-t0:.1f}s")

    print_mahal_stats(val_scores, val_ood_gt, val_class_gt, "VAL")

    val_auroc, val_aupr = compute_auroc_aupr(val_scores, val_ood_gt)
    print(f"\n  Val AUROC = {val_auroc:.4f}   AUPR = {val_aupr:.4f}")

    threshold, val_f1, val_prec, val_rec = find_best_threshold(val_scores, val_ood_gt)
    print(f"  Best threshold = {threshold:.4f}  "
          f"F1={val_f1:.4f}  Prec={val_prec:.4f}  Rec={val_rec:.4f}")

    # =========================================================================
    # Step 3: Test set
    # =========================================================================
    print(f"\n{SEP_LIGHT}")
    print(f"  Step 3: Test set -- final evaluation")
    print(SEP_LIGHT)

    test_loader = make_loader("test")
    print("  Collecting test embeddings...")
    t0 = time.perf_counter()
    test_emb, test_ood_gt, test_class_gt = collect_eval_embeddings(
        model, test_loader, device, max_patches=args.max_eval_patches)
    print(f"  Test: {len(test_emb):,} points  "
          f"(OOD: {test_ood_gt.sum():,} = {test_ood_gt.mean()*100:.1f}%)  "
          f"[{time.perf_counter()-t0:.1f}s]\n")

    print("  Computing Mahalanobis scores (test)...")
    t0 = time.perf_counter()
    test_scores = (mahalanobis_scores_gpu(test_emb, means, Sigma_inv, device)
                   if device.type == "cuda"
                   else mahalanobis_scores_cpu(test_emb, means, Sigma_inv))
    print(f"  Scoring time: {time.perf_counter()-t0:.1f}s")

    print_mahal_stats(test_scores, test_ood_gt, test_class_gt, "TEST")

    test_auroc, test_aupr = compute_auroc_aupr(test_scores, test_ood_gt)
    test_m                = compute_ood_metrics(test_scores, test_ood_gt, threshold)

    # =========================================================================
    # Summary comparison
    # =========================================================================
    ENERGY_AUROC = 0.4321
    ENERGY_AUPR  = 0.1621
    ENERGY_F1    = 0.2951

    delta = test_auroc - ENERGY_AUROC
    if delta > 0.05:
        verdict = f"BETTER  AUROC +{delta:.4f} -- Mahalanobis wins"
    elif delta > 0.0:
        verdict = f"MARGINAL  AUROC +{delta:.4f} -- similar performance"
    else:
        verdict = f"WORSE  AUROC {delta:.4f} -- Water/Ground inseparable in 32-dim space"

    print(f"\n{SEP_HEAVY}")
    print(f"  RESULTS -- Mahalanobis vs Energy Score")
    print(SEP_HEAVY)
    print(f"  {'Metric':<26}  {'Mahalanobis':>13}  {'Energy Score':>13}")
    print(f"  {'-'*55}")
    print(f"  {'AUROC (test)':<26}  {test_auroc:>13.4f}  {ENERGY_AUROC:>13.4f}")
    print(f"  {'AUPR (test)':<26}  {test_aupr or 0.0:>13.4f}  {ENERGY_AUPR:>13.4f}")
    print(f"  {'F1 @ best thr (test)':<26}  {test_m['f1']:>13.4f}  {ENERGY_F1:>13.4f}")
    print(f"  {'Precision':<26}  {test_m['precision']:>13.4f}  {'--':>13}")
    print(f"  {'Recall':<26}  {test_m['recall']:>13.4f}  {'--':>13}")
    print(f"  {'-'*55}")
    print(f"\n  Verdict: {verdict}")
    print(SEP_HEAVY + "\n")

    # =========================================================================
    # Save results
    # =========================================================================
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    mahal_results = {
        "timestamp":     datetime.now().isoformat(),
        "method":        "Mahalanobis Distance (Lee et al., NeurIPS 2018)",
        "fit_patches":   args.fit_patches,
        "fit_points":    n_fit,
        "reg":           args.reg,
        "embedding_dim": 32,
        "val": {
            "auroc":     round(val_auroc, 4),
            "aupr":      round(val_aupr, 4) if val_aupr else None,
            "best_f1":   round(val_f1,   4),
            "threshold": round(float(threshold), 4),
        },
        "test": {
            "auroc":     round(test_auroc,       4),
            "aupr":      round(test_aupr, 4) if test_aupr else None,
            "f1":        round(test_m["f1"],         4),
            "precision": round(test_m["precision"],  4),
            "recall":    round(test_m["recall"],     4),
            "threshold": round(float(threshold),     4),
            "tp": test_m["tp"], "fp": test_m["fp"],
            "fn": test_m["fn"], "tn": test_m["tn"],
        },
    }

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "energy_score": {
            "test_auroc":      ENERGY_AUROC,
            "test_aupr":       ENERGY_AUPR,
            "test_f1":         ENERGY_F1,
            "water_d_prime":   0.05,
            "bridge_d_prime":  1.66,
        },
        "mahalanobis": {
            "test_auroc": round(test_auroc, 4),
            "test_aupr":  round(test_aupr, 4) if test_aupr else None,
            "test_f1":    round(test_m["f1"], 4),
        },
        "verdict": verdict,
    }

    mahal_path = out_dir / "ood_mahal_results.json"
    comp_path  = out_dir / "ood_comparison.json"
    mahal_path.write_text(
        json.dumps(mahal_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    comp_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Results saved:")
    print(f"  {mahal_path}")
    print(f"  {comp_path}\n")


if __name__ == "__main__":
    main()
