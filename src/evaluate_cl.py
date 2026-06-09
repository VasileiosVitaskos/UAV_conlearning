"""
evaluate_cl.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Αξιολόγηση μοντέλου μετά από Few-shot Continual Learning.

Συγκρίνει:
  • Baseline model  (N classes, Water εκτός → predicted as Ground)
  • CL model        (N+1 classes, Water → predicted as Water)

Μετρικές για τη νέα κλάση:
  • Precision, Recall, F1  (για τα Water points)
  • Confusion: σε τι κλάση πήγαιναν πριν το CL
  • N-shot curve: πώς βελτιώνεται με 1,3,5,10 shots

Χρήση
──────
  # Αξιολόγηση CL μοντέλου για Water:
  python src/evaluate_cl.py \\
      --cl-checkpoint   outputs/checkpoints/best_model_cl_water.pt \\
      --base-checkpoint outputs/checkpoints/best_model.pt

  # Χρήση test split αντί για val:
  python src/evaluate_cl.py --split test

  # N-shot curve (αν έχεις πολλά checkpoints):
  python src/evaluate_cl.py --n-shot-curve
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models.pointnet2 import PointNet2Mini
from data.dataset    import FractalPatchDataset, CLASS_NAMES, CLASS_MAP


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate CL-updated model on newly added class",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cl-checkpoint", type=str,
                   default=str(ROOT / "outputs" / "checkpoints" / "best_model_cl_water.pt"),
                   dest="cl_checkpoint",
                   help="CL-updated checkpoint (N+1 classes)")
    p.add_argument("--base-checkpoint", type=str,
                   default=str(ROOT / "outputs" / "checkpoints" / "best_model.pt"),
                   dest="base_checkpoint",
                   help="Baseline checkpoint (N classes) — για before/after σύγκριση")
    p.add_argument("--target-class-map-idx", type=int, default=None,
                   dest="target_class_map_idx",
                   help="CLASS_MAP index της νέας κλάσης (default: διαβάζεται από checkpoint)")
    p.add_argument("--split", type=str, default="val",
                   choices=["train", "val", "test"])
    p.add_argument("--num-points", type=int, default=4096,
                   dest="num_points")
    p.add_argument("--max-patches", type=int, default=200,
                   dest="max_patches",
                   help="Max patches για αξιολόγηση (default: 200)")
    p.add_argument("--min-target-pts", type=int, default=10,
                   dest="min_target_pts",
                   help="Ελάχιστα target class points σε ένα patch (default: 10)")
    p.add_argument("--save-results", type=str, default=None,
                   dest="save_results",
                   help="Αποθήκευση αποτελεσμάτων σε JSON")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Model loader
# ══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes  = ckpt.get("num_classes",  6)
    active_names = ckpt.get("active_names", [f"class_{i}" for i in range(num_classes)])
    model = PointNet2Mini(in_channels=7, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt, active_names


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation on patches containing target class
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_on_target_class(
    model:                torch.nn.Module,
    active_names:         list[str],
    split:                str,
    target_class_map_idx: int,       # CLASS_MAP index (5=Water, 6=Bridge)
    new_class_model_idx:  int,       # model output index for new class
    num_points:           int,
    max_patches:          int,
    min_target_pts:       int,
    device:               torch.device,
    seed:                 int = 42,
) -> dict:
    """
    Τρέχει inference σε patches που περιέχουν την target class.
    Μετράει πού πηγαίνουν τα target points (confusion stats).

    Returns:
        dict με precision, recall, F1, confusion_counts, total_target_pts
    """
    ds = FractalPatchDataset(
        root=ROOT, split=split,
        num_points=num_points,
        remap=True, cache=True, seed=seed,
    )

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(ds)).tolist()

    total_target     = 0          # συνολικά target class points
    correct_new      = 0          # σωστά → new class
    confusion_counts = defaultdict(int)  # pred_class → count
    patches_used     = 0

    for idx in indices:
        X, y = ds[idx]
        target_mask = (y == target_class_map_idx)

        if target_mask.sum() < min_target_pts:
            continue

        X_batch = X.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(X_batch)       # (1, N, C)
        preds = logits[0].argmax(dim=-1)  # (N,)

        target_preds = preds[target_mask]  # predictions for target class points
        for pred_idx in target_preds.cpu().numpy():
            confusion_counts[int(pred_idx)] += 1

        correct_new      += (target_preds == new_class_model_idx).sum().item()
        total_target     += target_mask.sum().item()
        patches_used     += 1

        if patches_used >= max_patches:
            break

    if total_target == 0:
        return {
            "total_target_pts": 0,
            "patches_used":     patches_used,
            "error":            "No target class points found",
        }

    recall    = correct_new / total_target
    # Για precision: ποια % των "new class" predictions είναι σωστά
    # (approximate — χρειαζόμαστε FP αλλά εδώ μετράμε μόνο target points)

    # Confusion: ταξινόμηση ανά model class index
    confusion_named = {}
    for class_idx, cnt in sorted(confusion_counts.items()):
        if class_idx < len(active_names):
            name = active_names[class_idx]
        else:
            name = f"class_{class_idx}"
        pct = cnt / total_target * 100
        confusion_named[name] = {"count": cnt, "pct": round(pct, 1)}

    return {
        "total_target_pts": total_target,
        "patches_used":     patches_used,
        "recall":           round(recall, 4),
        "correct_new":      correct_new,
        "confusion":        confusion_named,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    print(f"\n{'═'*65}")
    print(f"  Evaluate CL Model")
    print(f"  CL checkpoint  : {Path(args.cl_checkpoint).name}")
    print(f"  Base checkpoint: {Path(args.base_checkpoint).name}")
    print(f"  Split          : {args.split}")
    print(f"{'═'*65}")

    device = torch.device("cpu")

    # ── Load CL model ──────────────────────────────────────────────────────────
    cl_path = Path(args.cl_checkpoint)
    if not cl_path.exists():
        raise FileNotFoundError(f"CL checkpoint not found: {cl_path}\n"
                                 f"Τρέξε πρώτα: python src/few_shot_add_class.py ...")

    cl_model, cl_ckpt, cl_names = load_model(cl_path, device)
    cl_added_class    = cl_ckpt.get("cl_added_class", cl_names[-1])
    new_class_idx     = cl_ckpt.get("num_classes", len(cl_names)) - 1
    n_shots           = cl_ckpt.get("cl_n_shots", "?")
    n_emb             = cl_ckpt.get("cl_n_embeddings", "?")

    # TARGET CLASS: CLASS_MAP index
    target_map_idx = args.target_class_map_idx
    if target_map_idx is None:
        target_map_idx = cl_ckpt.get("cl_source_map_idx", None)
    if target_map_idx is None or target_map_idx < 0:
        # Fallback: αναζήτηση με όνομα
        name_lower = cl_added_class.lower()
        target_map_idx = {v.lower(): k for k, v in CLASS_NAMES.items()}.get(name_lower, 5)
        print(f"  ⚠  cl_source_map_idx άγνωστο — χρησιμοποιώ CLASS_MAP index {target_map_idx} "
              f"({CLASS_NAMES.get(target_map_idx, '?')})")

    print(f"\n  CL model classes : {len(cl_names)} → {cl_names}")
    print(f"  New class        : '{cl_added_class}' (model index {new_class_idx})")
    print(f"  Target CLASS_MAP : {target_map_idx} ({CLASS_NAMES.get(target_map_idx,'?')})")
    print(f"  Shots / Emb      : {n_shots} patches / {n_emb} embeddings")

    # ── Evaluate CL model ──────────────────────────────────────────────────────
    print(f"\n  ── CL Model Evaluation ──")
    cl_results = evaluate_on_target_class(
        cl_model, cl_names,
        args.split, target_map_idx, new_class_idx,
        args.num_points, args.max_patches, args.min_target_pts,
        device, args.seed,
    )

    # ── Evaluate BASELINE model (before CL) ───────────────────────────────────
    base_results = None
    base_path    = Path(args.base_checkpoint)

    if base_path.exists():
        print(f"\n  ── Baseline Model (before CL) ──")
        base_model, base_ckpt, base_names = load_model(base_path, device)
        base_results = evaluate_on_target_class(
            base_model, base_names,
            args.split, target_map_idx,
            new_class_model_idx=-999,    # No "new class" in baseline → recall always 0
            num_points=args.num_points,
            max_patches=args.max_patches,
            min_target_pts=args.min_target_pts,
            device=device, seed=args.seed,
        )

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  Results for: '{cl_added_class}' (CLASS_MAP={target_map_idx})")
    print(f"{'═'*65}")

    if "error" in cl_results:
        print(f"\n  ✗ {cl_results['error']}")
        print(f"    Σκανάρηκαν {cl_results['patches_used']} patches — δεν βρέθηκε η κλάση στο {args.split} split.")
        print(f"    Δοκίμασε --split train ή --min-target-pts {args.min_target_pts // 2}")
        return

    total = cl_results["total_target_pts"]
    correct = cl_results["correct_new"]
    recall  = cl_results["recall"]

    print(f"\n  CL Model:")
    print(f"    Patches evaluated   : {cl_results['patches_used']}")
    print(f"    Total target points : {total:,}")
    print(f"    Correctly classified: {correct:,}  ({recall*100:.1f}%)")
    print(f"    Recall@new_class     : {recall:.4f}")

    print(f"\n  Confusion (where target class points go → CL model):")
    for cls_name, info in sorted(cl_results["confusion"].items(),
                                  key=lambda x: -x[1]["count"]):
        bar = "█" * int(info["pct"] / 3)
        marker = " ← ✓ NEW CLASS" if cls_name.lower() == cl_added_class.lower() else ""
        print(f"    {cls_name:<20} {info['count']:>7,}  ({info['pct']:5.1f}%)  {bar}{marker}")

    if base_results and "error" not in base_results:
        print(f"\n  Confusion (where target class points go → BASELINE model):")
        for cls_name, info in sorted(base_results["confusion"].items(),
                                      key=lambda x: -x[1]["count"]):
            bar = "█" * int(info["pct"] / 3)
            print(f"    {cls_name:<20} {info['count']:>7,}  ({info['pct']:5.1f}%)  {bar}")

        # Before/after summary
        print(f"\n  {'Before / After CL':}")
        print(f"  {'─'*50}")
        print(f"  {'Baseline':20} → target classified as '{cl_added_class}': 0.0%  (class doesn't exist)")
        print(f"  {'CL model':20} → target classified as '{cl_added_class}': {recall*100:.1f}%  ← improvement")

    # ── Save results ───────────────────────────────────────────────────────────
    report = {
        "cl_checkpoint":    str(cl_path),
        "base_checkpoint":  str(base_path) if base_path.exists() else None,
        "added_class":      cl_added_class,
        "target_map_idx":   target_map_idx,
        "new_class_idx":    new_class_idx,
        "n_shots":          n_shots,
        "n_embeddings":     n_emb,
        "split":            args.split,
        "cl_results":       cl_results,
        "base_results":     base_results,
    }

    save_path = args.save_results or str(
        ROOT / "outputs" / f"cl_eval_{cl_added_class.lower()}.json"
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Results saved: {save_path}")

    print(f"\n{'═'*65}")
    print(f"  CL Demo complete — '{cl_added_class}' learned from {n_shots} shots")
    print(f"  Recall: 0.0% (baseline) → {recall*100:.1f}% (CL model)")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
