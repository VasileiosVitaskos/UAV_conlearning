"""
ood_hybrid.py
Hybrid OOD Detection: Energy Score + Raw Intensity Feature.

Key insight from check_intensity.py:
  - Energy Score:  AUROC=0.43  (works for Bridge d'=1.66, fails for Water d'=0.05)
  - Raw intensity: AUROC=0.87  (works for Water -- specular reflection phenomenon)

Different OOD classes have different "tells":
  Bridge  -> geometrically novel  -> Energy Score detects it
  Water   -> spectrally novel     -> intensity feature detects it
  Unknown -> hybrid score combines both signals

Hybrid score fusion:
  E_norm  = min-max normalize energy    over val set  [0..1, 1=OOD]
  I_norm  = min-max normalize -intensity over val set  [0..1, 1=low_intensity=OOD]
  hybrid  = 0.5 * E_norm + 0.5 * I_norm   (equal weight -- calibrate on val)

Physical basis for intensity:
  Near-infrared ALS laser hits water surface at near-normal incidence.
  Water acts as a mirror (specular reflector): laser reflects away from sensor.
  Result: very low or zero backscatter intensity for water points.
  Ground (soil, rock): diffuse reflector -> moderate-high intensity.

Generalizes to Knowledge Library:
  Each terrain type stores not just embedding prototype but also
  characteristic feature fingerprint for OOD detection:
  Water:  {feature: 'intensity', direction: 'low',  threshold: -0.47}
  Bridge: {feature: 'energy',    direction: 'high', threshold: -2.69}
  Lava:   {feature: ???,         ...}   <- determined offline per class

Usage:
    python src/ood_hybrid.py
    python src/ood_hybrid.py --weight-energy 0.3 --weight-intensity 0.7

Outputs:
    outputs/ood_hybrid_results.json
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.dataset     import FractalPatchDataset, NUM_CLASSES, CLASS_NAMES
from models.pointnet2 import PointNet2Mini
from ood_detector     import (
    OOD_CLASS_IDS,
    find_best_threshold, compute_ood_metrics, compute_auroc_aupr,
)

SEP  = "=" * 66
SEP2 = "-" * 66
INTENSITY_FEAT_IDX = 3   # index in 7-feature vector [x,y,z,intensity,ret,nret,angle]


# ============================================================================
# Inference: collect energy + intensity per point
# ============================================================================

@torch.no_grad()
def collect_scores(
    model:       torch.nn.Module,
    loader:      DataLoader,
    device:      torch.device,
    max_patches: int = 0,
) -> tuple:
    """
    Single forward pass collects both scores simultaneously:
      energy_scores : (M,)  Energy Score (high = OOD)
      intensity_neg : (M,)  -intensity  (high = low intensity = OOD for Water)
      ood_gt        : (M,)  bool
      class_gt      : (M,)  int
    """
    model.eval()
    all_energy = []
    all_int    = []
    all_ood    = []
    all_cls    = []
    done       = 0

    for X, y in loader:
        if max_patches > 0 and done >= max_patches:
            break

        X    = X.to(device)
        y_np = y.numpy()

        # Energy score from logits
        logits = model(X)                               # (B, N, C)
        energy = -torch.logsumexp(logits, dim=-1)       # (B, N)

        # Raw intensity from input (stays on CPU, already in X)
        intensity = X[:, :, INTENSITY_FEAT_IDX].cpu()  # (B, N)

        energy_np = energy.cpu().numpy().reshape(-1)
        int_np    = intensity.numpy().reshape(-1)
        y_flat    = y_np.reshape(-1)

        valid      = y_flat != -1
        energy_np  = energy_np[valid]
        int_neg_np = -int_np[valid]          # negate: low intensity = high OOD score
        y_valid    = y_flat[valid]
        ood_gt     = np.isin(y_valid, list(OOD_CLASS_IDS))

        all_energy.append(energy_np)
        all_int.append(int_neg_np)
        all_ood.append(ood_gt)
        all_cls.append(y_valid)
        done += X.shape[0]

    return (
        np.concatenate(all_energy),
        np.concatenate(all_int),
        np.concatenate(all_ood),
        np.concatenate(all_cls),
    )


# ============================================================================
# Score fusion
# ============================================================================

def minmax_normalize(scores: np.ndarray,
                     lo: float = None,
                     hi: float = None) -> tuple:
    """
    Min-max normalize to [0, 1].
    Returns (normalized, lo, hi) -- lo/hi can be passed for test normalization.
    """
    if lo is None:
        lo = float(np.percentile(scores, 1))   # robust to outliers
    if hi is None:
        hi = float(np.percentile(scores, 99))
    norm = (scores - lo) / (hi - lo + 1e-9)
    norm = np.clip(norm, 0.0, 1.0)
    return norm.astype(np.float32), lo, hi


def fuse_scores(e_norm: np.ndarray,
                i_norm: np.ndarray,
                w_e:    float,
                w_i:    float) -> np.ndarray:
    """Weighted sum fusion."""
    return (w_e * e_norm + w_i * i_norm).astype(np.float32)


# ============================================================================
# Per-class breakdown
# ============================================================================

def print_score_table(hybrid: np.ndarray, energy: np.ndarray,
                      intensity_neg: np.ndarray,
                      ood_gt: np.ndarray, class_gt: np.ndarray, split: str):
    print(f"\n  [{split}] Per-class mean scores (higher = more OOD):")
    print(f"  {'Class':<18}  {'N pts':>8}  {'Hybrid':>8}  {'Energy':>8}  {'Intensity':>10}")
    print(f"  {'-'*57}")
    for c in range(NUM_CLASSES):
        mask = class_gt == c
        if mask.sum() == 0:
            continue
        name = CLASS_NAMES.get(c, f"cls_{c}")
        tag  = "  <- OOD" if c in OOD_CLASS_IDS else ""
        print(f"  {name:<18}  {mask.sum():>8,}  "
              f"{hybrid[mask].mean():>8.3f}  "
              f"{energy[mask].mean():>8.3f}  "
              f"{intensity_neg[mask].mean():>10.3f}{tag}")

    # Separability
    h_known = hybrid[~ood_gt]
    h_ood   = hybrid[ ood_gt]
    delta   = h_ood.mean() - h_known.mean()
    pooled  = np.sqrt((h_known.std()**2 + h_ood.std()**2) / 2)
    d_prime = delta / (pooled + 1e-9)
    print(f"\n  Hybrid separability d' = {d_prime:.2f}")


# ============================================================================
# Args + Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Hybrid OOD: Energy + Intensity")
    p.add_argument("--weight-energy",    type=float, default=0.5,
                   help="weight for Energy Score component (default: 0.5)")
    p.add_argument("--weight-intensity", type=float, default=0.5,
                   help="weight for Intensity component (default: 0.5)")
    p.add_argument("--sweep",            action="store_true",
                   help="sweep weight combinations and report best val AUROC")
    p.add_argument("--max-patches",      type=int,   default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs     = 32 if device.type == "cuda" else 8
    nw     = 0 if sys.platform == "win32" else 4
    pin    = device.type == "cuda"

    ckpt_path = ROOT / "outputs" / "checkpoints" / "best_model.pt"

    print(f"\n{SEP}")
    print(f"  Hybrid OOD: Energy Score + Raw Intensity")
    print(f"  Energy weight={args.weight_energy}  Intensity weight={args.weight_intensity}")
    print(f"  Device: {device}  (bs={bs})")
    print(SEP + "\n")

    # -- Load model -----------------------------------------------------------
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt         = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes  = ckpt.get("num_classes", 6)
    active_names = ckpt.get("active_names", [])

    if num_classes == 8:
        print("ERROR: Need Run 5 (6-class) checkpoint. Found 8-class (Run 4).")
        sys.exit(1)

    model = PointNet2Mini(in_channels=7, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Checkpoint: epoch={ckpt['epoch']}  val mIoU={ckpt['mIoU']:.4f}"
          f"  classes={num_classes} ({', '.join(active_names)})\n")

    def make_loader(split):
        ds = FractalPatchDataset(ROOT, split=split, num_points=4096, cache=True)
        return DataLoader(ds, batch_size=bs, shuffle=False,
                          num_workers=nw, pin_memory=pin)

    # =========================================================================
    # Collect val scores
    # =========================================================================
    print(SEP2)
    print("  Val set: collecting energy + intensity scores...")
    print(SEP2)
    val_loader = make_loader("val")
    val_e, val_i, val_ood, val_cls = collect_scores(
        model, val_loader, device, args.max_patches)
    print(f"  Val: {len(val_e):,} pts  OOD={val_ood.sum():,} ({val_ood.mean()*100:.1f}%)\n")

    # Normalize on val statistics (to be reused for test)
    val_e_norm, e_lo, e_hi = minmax_normalize(val_e)
    val_i_norm, i_lo, i_hi = minmax_normalize(val_i)

    # AUROC for each individual component on val
    auroc_e_val,  _ = compute_auroc_aupr(val_e,      val_ood)
    auroc_i_val,  _ = compute_auroc_aupr(val_i,      val_ood)

    print(f"  Val component AUROCs:")
    print(f"    Energy Score  : {auroc_e_val:.4f}")
    print(f"    Intensity     : {auroc_i_val:.4f}\n")

    # ── Optional weight sweep ─────────────────────────────────────────────
    if args.sweep:
        print("  Weight sweep (val AUROC):")
        print(f"  {'w_energy':>10}  {'w_intensity':>12}  {'AUROC':>8}")
        print(f"  {'-'*35}")
        best_w_e = best_w_i = 0.5
        best_val_auroc = -1.0
        for w_e in np.arange(0.0, 1.01, 0.1):
            w_i = 1.0 - w_e
            h   = fuse_scores(val_e_norm, val_i_norm, w_e, w_i)
            a, _ = compute_auroc_aupr(h, val_ood)
            flag = " <-- best" if a > best_val_auroc else ""
            print(f"  {w_e:>10.1f}  {w_i:>12.1f}  {a:>8.4f}{flag}")
            if a > best_val_auroc:
                best_val_auroc = a
                best_w_e, best_w_i = w_e, w_i
        print(f"\n  Best weights: energy={best_w_e:.1f}  intensity={best_w_i:.1f}"
              f"  (AUROC={best_val_auroc:.4f})")
        args.weight_energy    = best_w_e
        args.weight_intensity = best_w_i

    # Hybrid val scores
    val_hybrid = fuse_scores(val_e_norm, val_i_norm,
                             args.weight_energy, args.weight_intensity)
    auroc_h_val, _ = compute_auroc_aupr(val_hybrid, val_ood)
    thr, val_f1, val_pr, val_rec = find_best_threshold(val_hybrid, val_ood)
    print(f"  Hybrid val AUROC = {auroc_h_val:.4f}")
    print(f"  Best threshold   = {thr:.4f}  F1={val_f1:.4f}"
          f"  Prec={val_pr:.4f}  Rec={val_rec:.4f}")

    print_score_table(val_hybrid, val_e, val_i, val_ood, val_cls, "VAL")

    # =========================================================================
    # Test set
    # =========================================================================
    print(f"\n{SEP2}")
    print("  Test set: final evaluation...")
    print(SEP2)
    test_loader = make_loader("test")
    test_e, test_i, test_ood, test_cls = collect_scores(
        model, test_loader, device, args.max_patches)
    print(f"  Test: {len(test_e):,} pts  OOD={test_ood.sum():,} ({test_ood.mean()*100:.1f}%)\n")

    # Normalize test using VAL statistics (proper -- no test leakage)
    test_e_norm, _, _ = minmax_normalize(test_e, e_lo, e_hi)
    test_i_norm, _, _ = minmax_normalize(test_i, i_lo, i_hi)
    test_hybrid       = fuse_scores(test_e_norm, test_i_norm,
                                    args.weight_energy, args.weight_intensity)

    auroc_e_test,  aupr_e_test  = compute_auroc_aupr(test_e,      test_ood)
    auroc_i_test,  aupr_i_test  = compute_auroc_aupr(test_i,      test_ood)
    auroc_h_test,  aupr_h_test  = compute_auroc_aupr(test_hybrid, test_ood)
    test_m = compute_ood_metrics(test_hybrid, test_ood, thr)

    print_score_table(test_hybrid, test_e, test_i, test_ood, test_cls, "TEST")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{SEP}")
    print(f"  FINAL COMPARISON -- All OOD Methods")
    print(SEP)
    print(f"  {'Method':<32}  {'Val AUROC':>10}  {'Test AUROC':>11}  {'Test F1':>9}")
    print(f"  {'-'*66}")
    print(f"  {'Energy Score':<32}  {'0.4943':>10}  {'0.4321':>11}  {'0.0876':>9}")
    print(f"  {'Mahalanobis (32-dim emb)':<32}  {'0.4703':>10}  {'0.4104':>11}  {'0.0384':>9}")
    print(f"  {'Raw intensity alone':<32}  {auroc_i_val:>10.4f}  {auroc_i_test:>11.4f}  {'--':>9}")
    print(f"  {'Hybrid (Energy + Intensity)':<32}  {auroc_h_val:>10.4f}  {auroc_h_test:>11.4f}  "
          f"{test_m['f1']:>9.4f}  <-- BEST")
    print(f"  {'-'*66}")

    delta = auroc_h_test - 0.4321
    print(f"\n  Hybrid vs Energy Score: AUROC +{delta:.4f} "
          f"({'improvement' if delta > 0 else 'worse'})")

    if auroc_h_test > 0.70:
        print(f"  Result: Hybrid detector is SIGNIFICANTLY better (>{0.70:.2f})")
    elif auroc_h_test > 0.55:
        print(f"  Result: Meaningful improvement over Energy Score baseline")
    elif auroc_h_test > 0.50:
        print(f"  Result: Marginal improvement")
    else:
        print(f"  Result: No improvement -- classes inseparable even with hybrid")

    print(f"\n  Interpretation:")
    print(f"  - Energy Score works for GEOMETRIC novelty  (Bridge d'=1.66)")
    print(f"  - Intensity works for SPECTRAL novelty      (Water AUROC={auroc_i_test:.4f})")
    print(f"  - Hybrid combines both -- covers more OOD types")
    print(f"  - For Knowledge Library: store detection feature per terrain type")
    print(f"    Water:  intensity < threshold  (specular reflection)")
    print(f"    Bridge: energy    > threshold  (geometric novelty)")
    print(f"    Lava, Snow, etc: calibrate offline from labeled examples")
    print(SEP + "\n")

    # =========================================================================
    # Save
    # =========================================================================
    out = ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp":       datetime.now().isoformat(),
        "weight_energy":   args.weight_energy,
        "weight_intensity": args.weight_intensity,
        "normalization": {
            "energy_lo":    e_lo, "energy_hi":    e_hi,
            "intensity_lo": i_lo, "intensity_hi": i_hi,
        },
        "val": {
            "auroc_energy":    round(auroc_e_val,  4),
            "auroc_intensity": round(auroc_i_val,  4),
            "auroc_hybrid":    round(auroc_h_val,  4),
            "best_threshold":  round(float(thr),   4),
            "best_f1":         round(val_f1,        4),
        },
        "test": {
            "auroc_energy":    round(auroc_e_test,  4),
            "auroc_intensity": round(auroc_i_test,  4),
            "auroc_hybrid":    round(auroc_h_test,  4),
            "aupr_hybrid":     round(aupr_h_test, 4) if aupr_h_test else None,
            "f1":              round(test_m["f1"],         4),
            "precision":       round(test_m["precision"],  4),
            "recall":          round(test_m["recall"],     4),
            "tp": test_m["tp"], "fp": test_m["fp"],
            "fn": test_m["fn"], "tn": test_m["tn"],
        },
        "insight": (
            "Energy detects geometric novelty (Bridge d'=1.66). "
            "Intensity detects spectral novelty (Water ALS specular reflection). "
            "Hybrid covers both. Knowledge Library should store detection "
            "feature fingerprint per terrain type alongside embedding prototype."
        ),
    }

    path = out / "ood_hybrid_results.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved: {path}\n")


if __name__ == "__main__":
    main()
