"""
few_shot_add_class.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Few-shot Continual Learning: προσθήκη νέας κλάσης στο CosineClassifier
χωρίς αναεκπαίδευση του backbone (frozen encoder).

Αρχή λειτουργίας
────────────────
  1. Φορτώνει frozen backbone (PointNet++ encoder)
  2. Τρέχει get_embeddings() σε N few-shot patches
  3. Υπολογίζει mean prototype για τη νέα κλάση
  4. Καλεί model.classifier.add_class(prototype)  ← μόνο 32 νέοι αριθμοί
  5. Αποθηκεύει ενημερωμένο checkpoint

Modes
──────
  fractal : labeled FRACTAL patches — για quantitative demo (Water / Bridge)
  basket  : αποθηκευμένα OOD embeddings — για YellowScan (χωρίς labels)

Pipeline (basket mode):
  inference.py --save-basket outputs/ood_basket  [Pi5 ή dev machine]
  → accumulates .npy files in ood_basket/
  → few_shot_add_class.py --mode basket --class-name WaterYS
  → re-export ONNX automatically

Χρήση
──────
  # FRACTAL Water (10-shot):
  python src/few_shot_add_class.py --mode fractal --class-name Water --n-shots 10

  # FRACTAL Bridge (5-shot από val):
  python src/few_shot_add_class.py --mode fractal --class-name Bridge --n-shots 5 --split val

  # YellowScan basket (unsupervised):
  python src/few_shot_add_class.py --mode basket --class-name Unknown_Water \\
      --basket-dir outputs/ood_basket

  # N-shot curve (1,2,3,5,10):
  python src/few_shot_add_class.py --mode fractal --class-name Water --n-shots 10 --n-shot-curve

Αρχεία εξόδου
──────────────
  outputs/checkpoints/best_model_cl_<class_name>.pt   ← updated checkpoint
  outputs/model_cl_<class_name>.onnx                  ← re-exported ONNX (αν --export-onnx)
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models.pointnet2 import PointNet2Mini
from data.dataset import FractalPatchDataset, CLASS_MAP, CLASS_NAMES


# CLASS_MAP αντίστροφα: όνομα → CLASS_MAP index
_NAME_TO_IDX = {name.lower(): idx for idx, name in CLASS_NAMES.items()}
# Ποιες κλάσεις ήταν excluded (Water=5, Bridge=6 στη CLASS_MAP)
_TYPICAL_OOD = {"water": 5, "bridge": 6}


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Few-shot Continual Learning: add new class to CosineClassifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--mode", type=str, default="fractal",
                   choices=["fractal", "basket"],
                   help="fractal: labeled FRACTAL patches | basket: OOD embeddings (YellowScan)")
    p.add_argument("--class-name", type=str, default="Water",
                   dest="class_name",
                   help="Όνομα νέας κλάσης (π.χ. Water, Bridge, Unknown_1). Default: Water")
    p.add_argument("--n-shots", type=int, default=10,
                   dest="n_shots",
                   help="Πλήθος patches για few-shot (fractal mode). Default: 10")
    p.add_argument("--split", type=str, default="val",
                   choices=["train", "val", "test"],
                   help="FRACTAL split για αναζήτηση (default: val)")
    p.add_argument("--basket-dir", type=str,
                   default=str(ROOT / "outputs" / "ood_basket"),
                   dest="basket_dir",
                   help="Φάκελος με .npy embedding files (basket mode)")
    p.add_argument("--checkpoint", type=str,
                   default=str(ROOT / "outputs" / "checkpoints" / "best_model.pt"),
                   help="Βασικό checkpoint (default: best_model.pt)")
    p.add_argument("--output", type=str, default=None,
                   help="Output checkpoint path (default: best_model_cl_<class_name>.pt)")
    p.add_argument("--num-points", type=int, default=4096,
                   dest="num_points",
                   help="Points per patch για embedding extraction (default: 4096)")
    p.add_argument("--min-target-pts", type=int, default=20,
                   dest="min_target_pts",
                   help="Ελάχιστα target points σε sampled patch (default: 20)")
    p.add_argument("--export-onnx", action="store_true",
                   dest="export_onnx",
                   help="Re-export ONNX μετά από add_class")
    p.add_argument("--n-shot-curve", action="store_true",
                   dest="n_shot_curve",
                   help="Υπολογισμός prototype για 1,2,3,5,10 shots (για θέση)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# FRACTAL mode: embedding collection από labeled patches
# ══════════════════════════════════════════════════════════════════════════════

def collect_fractal_embeddings(
    model:               torch.nn.Module,
    split:               str,
    target_class_idx:    int,          # CLASS_MAP index (5=Water, 6=Bridge)
    n_shots:             int,
    num_points:          int,
    min_target_pts:      int,
    device:              torch.device,
    seed:                int = 42,
) -> tuple[np.ndarray, int]:
    """
    Σκανάρει FRACTAL patches, εξάγει embeddings για τη target class.

    Επιστρέφει:
        embeddings  : (M_total, 32) — pooled embeddings όλων των target points
        n_patches   : int — αριθμός patches που χρησιμοποιήθηκαν
    """
    ds = FractalPatchDataset(
        root=ROOT, split=split,
        num_points=num_points,
        remap=True,              # y: CLASS_MAP indices (0-7)
        cache=True,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(ds)).tolist()

    all_emb     = []
    patches_used = 0
    scanned      = 0

    print(f"\n  Σκανάρισμα {split} split για class {target_class_idx} "
          f"({CLASS_NAMES.get(target_class_idx,'?')})...")
    print(f"  {'Patch':>6}  {'Target pts':>10}  {'Emb shape':>12}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*12}")

    for idx in indices:
        scanned += 1
        X, y = ds[idx]                        # (N,7), (N,) CLASS_MAP indices
        target_mask = (y == target_class_idx) # Water/Bridge points

        if target_mask.sum() < min_target_pts:
            continue

        X_batch = X.unsqueeze(0).to(device)   # (1, N, 7)
        with torch.no_grad():
            emb = model.get_embeddings(X_batch)  # (1, N, 32)
        emb_np = emb[0].cpu().numpy()         # (N, 32)

        # Κρατάμε ΜΟΝΟ embeddings των target points
        target_emb = emb_np[target_mask.numpy()]  # (M, 32)
        all_emb.append(target_emb)
        patches_used += 1

        print(f"  {patches_used:>6}  {target_mask.sum():>10}  "
              f"  {str(target_emb.shape):>12}  (scanned {scanned})")

        if patches_used >= n_shots:
            break

    if not all_emb:
        raise RuntimeError(
            f"Δεν βρέθηκαν patches με ≥{min_target_pts} points κλάσης "
            f"{CLASS_NAMES.get(target_class_idx,'?')} στο {split} split.\n"
            f"Σκανάρηκαν {scanned} patches. "
            f"Δοκίμασε --min-target-pts {min_target_pts // 2} ή --split train."
        )

    return np.concatenate(all_emb, axis=0), patches_used


# ══════════════════════════════════════════════════════════════════════════════
# Basket mode: embedding collection από OOD inference (YellowScan)
# ══════════════════════════════════════════════════════════════════════════════

def collect_basket_embeddings(basket_dir: Path) -> np.ndarray:
    """
    Φορτώνει αποθηκευμένα OOD embeddings από το basket directory.

    Κάθε .npy αρχείο = (M, 32) embeddings από ένα OOD batch
    (αποθηκεύτηκε από inference.py --save-basket).

    Επιστρέφει: (M_total, 32) stacked embeddings
    """
    emb_files = sorted(basket_dir.glob("*.npy"))
    if not emb_files:
        raise FileNotFoundError(
            f"Κανένα .npy αρχείο στο basket: {basket_dir}\n"
            f"Τρέξε πρώτα: python src/inference.py <patch.laz> --save-basket {basket_dir}"
        )

    all_emb = []
    total_pts = 0
    print(f"\n  Φόρτωση basket από {basket_dir} ({len(emb_files)} files)...")
    for f in emb_files:
        emb = np.load(str(f))               # (M, 32)
        all_emb.append(emb)
        total_pts += len(emb)
        print(f"    {f.name}: {len(emb)} OOD embeddings")

    result = np.concatenate(all_emb, axis=0)
    print(f"  Σύνολο: {total_pts} OOD embeddings από {len(emb_files)} batches")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Prototype computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_prototype(embeddings: np.ndarray) -> torch.Tensor:
    """
    Mean embedding → CosineClassifier prototype.

    Το CosineClassifier.add_class() κάνει L2 normalization εσωτερικά,
    οπότε επιστρέφουμε un-normalized mean.

    Parameters
    ----------
    embeddings : (M, 32)  per-point embeddings για τη νέα κλάση

    Returns
    -------
    (32,) float32 Tensor — prototype
    """
    mean_emb  = embeddings.mean(axis=0).astype(np.float32)   # (32,)
    prototype = torch.from_numpy(mean_emb)
    return prototype


# ══════════════════════════════════════════════════════════════════════════════
# N-shot curve: δείχνει πώς βελτιώνεται το prototype με περισσότερα shots
# ══════════════════════════════════════════════════════════════════════════════

def compute_nshot_curve(
    embeddings_by_shot: list[np.ndarray],   # list[array(M_i, 32)] — ανά shot
    all_test_emb:       Optional[np.ndarray],
    all_test_labels:    Optional[np.ndarray],
) -> list[dict]:
    """
    Υπολογίζει prototype ποιότητα για κάθε αριθμό shots.

    Εκτιμά με intra-class cosine similarity (χωρίς test set):
    → high similarity = compact prototype = καλό για CL

    Επιστρέφει λίστα από dicts με στατιστικά ανά n_shots.
    """
    results = []
    acc_emb = []
    shot_counts = [1, 2, 3, 5, 10]

    for n in range(1, len(embeddings_by_shot) + 1):
        acc_emb.append(embeddings_by_shot[n - 1])
        all_so_far = np.concatenate(acc_emb, axis=0)
        proto = compute_prototype(all_so_far)
        proto_norm = F.normalize(proto.unsqueeze(0), dim=-1).numpy()[0]  # (32,)

        # Intra-class cosine similarity: πόσο "κοντά" είναι τα embeddings στο prototype
        emb_norm = all_so_far / (np.linalg.norm(all_so_far, axis=-1, keepdims=True) + 1e-8)
        cos_sim  = emb_norm @ proto_norm                # (M,)
        mean_cos = float(cos_sim.mean())
        std_cos  = float(cos_sim.std())

        if n in shot_counts or n == len(embeddings_by_shot):
            results.append({
                "n_shots":        n,
                "total_pts":      len(all_so_far),
                "mean_cos_sim":   round(mean_cos, 4),
                "std_cos_sim":    round(std_cos, 4),
                "prototype_norm": round(float(np.linalg.norm(proto.numpy())), 4),
            })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ONNX re-export
# ══════════════════════════════════════════════════════════════════════════════

def reexport_onnx(
    model:      torch.nn.Module,
    out_path:   Path,
    num_points: int,
    num_classes: int,
) -> None:
    """Re-exports updated model to ONNX (opset 14, static shapes)."""
    print(f"\n  Re-exporting ONNX ({num_classes} classes) → {out_path.name}...")
    dummy = torch.randn(1, num_points, 7)
    try:
        torch.onnx.export(
            model, dummy, str(out_path),
            opset_version=14,
            input_names=["point_cloud"],
            output_names=["logits"],
            dynamic_axes=None,
            export_params=True,
            do_constant_folding=True,
            verbose=False,
        )
        size_kb = out_path.stat().st_size / 1024
        print(f"  ✓ ONNX saved: {out_path}  ({size_kb:.0f} KB)")

        # Quick verify
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out_path),
                                    providers=["CPUExecutionProvider"])
        ort_out = sess.run(["logits"], {"point_cloud": dummy.numpy()})[0]
        with torch.no_grad():
            pt_out = model(dummy).numpy()
        agree = (pt_out.argmax(-1) == ort_out.argmax(-1)).mean() * 100
        print(f"  ✓ Argmax agreement: {agree:.1f}%")
    except ImportError:
        print("  ⚠  onnxruntime not installed — skipping ONNX verification")
    except Exception as e:
        print(f"  ✗ ONNX export failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    print(f"\n{'═'*65}")
    print(f"  Few-shot Continual Learning — add class '{args.class_name}'")
    print(f"  Mode      : {args.mode}")
    print(f"  Checkpoint: {Path(args.checkpoint).name}")
    if args.mode == "fractal":
        print(f"  N-shots   : {args.n_shots}  (split={args.split})")
    else:
        print(f"  Basket    : {args.basket_dir}")
    print(f"{'═'*65}")

    device = torch.device("cpu")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    num_classes  = ckpt.get("num_classes",  6)
    active_names = ckpt.get("active_names", [f"class_{i}" for i in range(num_classes)])
    epoch        = ckpt.get("epoch", "?")
    miou         = ckpt.get("mIoU", 0.0)

    print(f"\n  Base model : {num_classes} classes → {active_names}")
    print(f"  Val mIoU   : {miou:.4f}  (epoch {epoch})")

    # Βεβαιωνόμαστε ότι η νέα κλάση δεν υπάρχει ήδη
    if args.class_name.lower() in [n.lower() for n in active_names]:
        raise ValueError(
            f"Η κλάση '{args.class_name}' υπάρχει ήδη στο μοντέλο: {active_names}"
        )

    # ── Build model ───────────────────────────────────────────────────────────
    model = PointNet2Mini(in_channels=7, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.eval()   # Freeze BatchNorm, no xyz_dropout

    # ── Collect embeddings ───────────────────────────────────────────────────
    if args.mode == "fractal":
        # Αναζήτηση CLASS_MAP index για target class name
        class_lower = args.class_name.lower()
        if class_lower in _NAME_TO_IDX:
            target_class_idx = _NAME_TO_IDX[class_lower]
        elif class_lower in _TYPICAL_OOD:
            target_class_idx = _TYPICAL_OOD[class_lower]
        else:
            print(f"  ⚠  Άγνωστο class name '{args.class_name}'. "
                  f"Γνωστά: {list(_NAME_TO_IDX.keys())}")
            print(f"     Χρησιμοποιώ CLASS_MAP index 5 (Water) ως default.")
            target_class_idx = 5

        print(f"\n  Target: CLASS_MAP[{target_class_idx}] = "
              f"{CLASS_NAMES.get(target_class_idx, '?')} (raw LAS = "
              f"{[k for k,v in CLASS_MAP.items() if v == target_class_idx]})")

        if args.n_shot_curve:
            # Μαζεύουμε embeddings patch-by-patch για την καμπύλη
            all_shot_embs = []
            temp_n = min(args.n_shots, 10)
            for shot_i in range(temp_n):
                emb_i, _ = collect_fractal_embeddings(
                    model, args.split, target_class_idx,
                    n_shots=1, num_points=args.num_points,
                    min_target_pts=args.min_target_pts,
                    device=device, seed=args.seed + shot_i,
                )
                all_shot_embs.append(emb_i)
            embeddings = np.concatenate(all_shot_embs)
            n_patches  = temp_n
            # N-shot curve statistics
            curve = compute_nshot_curve(all_shot_embs, None, None)
            print(f"\n  ── N-shot learning curve ──")
            print(f"  {'Shots':>6}  {'Total pts':>10}  {'Mean cos-sim':>13}  {'±':>8}  {'Proto norm':>11}")
            print(f"  {'─'*6}  {'─'*10}  {'─'*13}  {'─'*8}  {'─'*11}")
            for r in curve:
                print(f"  {r['n_shots']:>6}  {r['total_pts']:>10}  "
                      f"  {r['mean_cos_sim']:>13.4f}  {r['std_cos_sim']:>8.4f}  "
                      f"  {r['prototype_norm']:>11.4f}")
        else:
            embeddings, n_patches = collect_fractal_embeddings(
                model, args.split, target_class_idx,
                n_shots=args.n_shots,
                num_points=args.num_points,
                min_target_pts=args.min_target_pts,
                device=device, seed=args.seed,
            )
        src_class_map_idx = target_class_idx

    else:  # basket mode
        embeddings = collect_basket_embeddings(Path(args.basket_dir))
        n_patches  = -1          # unknown (YellowScan, no patch count)
        src_class_map_idx = -1   # no CLASS_MAP mapping (unknown class)

    # ── Compute prototype ─────────────────────────────────────────────────────
    prototype = compute_prototype(embeddings)
    proto_norm = F.normalize(prototype.unsqueeze(0), dim=-1)[0]

    print(f"\n  Embeddings collected : {len(embeddings):,} points  "
          f"(shape {embeddings.shape})")
    print(f"  Prototype norm       : {prototype.norm():.4f}")
    print(f"  Intra-class cos-sim  : "
          f"{float((F.normalize(torch.from_numpy(embeddings), dim=-1) @ proto_norm).mean()):.4f}")

    # ── Add class to CosineClassifier ─────────────────────────────────────────
    new_class_idx = model.classifier.num_classes   # index ΠΡΙΝ add
    model.classifier.add_class(prototype)
    assert model.classifier.num_classes == new_class_idx + 1

    print(f"\n  ✓ add_class('{args.class_name}') → class index {new_class_idx}")
    print(f"  Νέο μοντέλο: {model.classifier.num_classes} classes: "
          f"{active_names + [args.class_name]}")

    # ── Save updated checkpoint ───────────────────────────────────────────────
    out_name = args.output or str(
        ROOT / "outputs" / "checkpoints" /
        f"best_model_cl_{args.class_name.lower()}.pt"
    )
    out_path = Path(out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    new_ckpt = {
        **ckpt,                                         # κρατάμε όλα τα υπόλοιπα
        "model_state":        model.state_dict(),
        "num_classes":        model.classifier.num_classes,
        "active_names":       active_names + [args.class_name],
        "cl_added_class":     args.class_name,
        "cl_source_map_idx":  src_class_map_idx,        # CLASS_MAP index ή -1
        "cl_n_shots":         n_patches,
        "cl_n_embeddings":    len(embeddings),
        "cl_mode":            args.mode,
    }
    torch.save(new_ckpt, out_path)
    print(f"\n  ✓ Checkpoint saved: {out_path}")

    # ── ONNX re-export (optional) ─────────────────────────────────────────────
    if args.export_onnx:
        onnx_path = out_path.with_suffix("").parent.parent / \
                    f"model_cl_{args.class_name.lower()}.onnx"
        reexport_onnx(model, onnx_path, args.num_points,
                      model.classifier.num_classes)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  CL Update Complete")
    print(f"  Base classes : {num_classes} → {model.classifier.num_classes}")
    print(f"  New class    : '{args.class_name}' (index {new_class_idx})")
    print(f"  Shots used   : {n_patches}")
    print(f"  Embeddings   : {len(embeddings):,} points")
    print(f"  Saved        : {out_path}")
    print(f"\n  NEXT: python src/evaluate_cl.py \\")
    print(f"         --checkpoint {out_path} \\")
    print(f"         --target-class-map-idx {src_class_map_idx}")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
