"""
evaluate_test.py
Τελική αξιολόγηση στο TEST SET — αγγίζεται μόνο μία φορά.

Φορτώνει το best checkpoint (best_model.pt), τρέχει inference στο test set,
εκτυπώνει per-class IoU / F1 / OA και αποθηκεύει JSON με τα αποτελέσματα.

Χρήση:
    python src/evaluate_test.py

Output:
    outputs/test_results.json   ← final numbers για Poster

ΣΗΜΑΝΤΙΚΟ: μην τρέξεις αυτό πολλές φορές (test set contamination).
Χρησιμοποίησε το val set για tuning. Αυτό τρέχει μόνο μία φορά στο τέλος.
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

# ── Project imports ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.dataset    import FractalPatchDataset, NUM_CLASSES, CLASS_NAMES
from models.pointnet2 import PointNet2Mini
from train           import FocalLoss, RemappedDataset, evaluate


def main():
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ROOT / "outputs" / "checkpoints" / "best_model.pt"

    if not ckpt_path.exists():
        print(f"❌  Checkpoint δεν βρέθηκε: {ckpt_path}")
        print("    Τρέξε πρώτα: python src/train.py --epochs 100 --bs 16 --cache --loss focal")
        sys.exit(1)

    print(f"\n{'═'*58}")
    print(f"  TEST SET EVALUATION — Final Poster Numbers")
    print(f"  ⚠️  Αυτό τρέχει μόνο μία φορά (test set contamination)")
    print(f"{'═'*58}\n")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    print(f"► Φόρτωση checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    saved_epoch    = ckpt.get("epoch",        "?")
    saved_val_miou = ckpt.get("mIoU",         float("nan"))
    num_classes    = ckpt.get("num_classes",  NUM_CLASSES)
    active_names   = ckpt.get("active_names", list(CLASS_NAMES.values()))
    remap_lut      = ckpt.get("remap_lut",    None)
    saved_args     = ckpt.get("args",         {})

    print(f"  Epoch saved    : {saved_epoch}")
    print(f"  Val mIoU (best): {saved_val_miou:.4f}")
    print(f"  Num classes    : {num_classes}  ({', '.join(active_names)})")
    if remap_lut is not None:
        excluded = [CLASS_NAMES[i] for i in range(NUM_CLASSES)
                    if remap_lut[i] == -1]
        print(f"  Excluded (OOD) : {', '.join(excluded)}")

    # ── Reconstruct model ──────────────────────────────────────────────────────
    model = PointNet2Mini(in_channels=7, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Model loaded   ✓  ({sum(p.numel() for p in model.parameters()):,} params)\n")

    # ── Test dataset ───────────────────────────────────────────────────────────
    print("► Φόρτωση test set...")
    num_points = saved_args.get("num_points", 4096)
    test_ds = FractalPatchDataset(ROOT, split="test",
                                  num_points=num_points,
                                  cache=True)

    # Εφάρμοσε το ίδιο remap που χρησιμοποιήθηκε στο training
    if remap_lut is not None:
        test_ds = RemappedDataset(test_ds, remap_lut)

    nw          = 0 if sys.platform == "win32" else 4
    test_loader = DataLoader(test_ds, batch_size=16,
                             shuffle=False, num_workers=nw, pin_memory=True)
    print(f"  Test patches: {len(test_ds)}\n")

    # ── Criterion (ίδιο με training) ───────────────────────────────────────────
    gamma     = saved_args.get("gamma", 2.0)
    loss_type = saved_args.get("loss",  "focal")

    if loss_type == "focal":
        criterion = FocalLoss(gamma=gamma, weight=None, ignore_index=-1)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print("► Αξιολόγηση στο test set...")
    metrics = evaluate(model, test_loader, criterion, device, num_classes)

    # ── Print results ──────────────────────────────────────────────────────────
    print(f"\n{'═'*58}")
    print(f"  ΤΕΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ — TEST SET")
    print(f"{'═'*58}")
    print(f"  mIoU  : {metrics['mIoU']:.4f}")
    print(f"  F1    : {metrics['macro_F1']:.4f}")
    print(f"  OA    : {metrics['OA']:.4f}")
    print(f"  Loss  : {metrics['loss']:.4f}")
    print(f"\n  Per-class IoU:")
    print(f"  {'─'*40}")

    per_class = {}
    for c, iou_c in enumerate(metrics["iou_per_class"]):
        name     = active_names[c] if c < len(active_names) else f"class_{c}"
        iou_val  = float(iou_c) if not np.isnan(iou_c) else None
        bar      = "█" * int(iou_c * 20) if (iou_val is not None) else "—"
        flag     = "⚠️" if (iou_val is not None and iou_val < 0.15) else ""
        print(f"  {name:<18} {iou_c:>6.3f}  {bar} {flag}")
        per_class[name] = round(iou_val, 4) if iou_val is not None else None

    print(f"  {'─'*40}")
    print(f"  {'mIoU':<18} {metrics['mIoU']:>6.4f}")
    print(f"  {'macro F1':<18} {metrics['macro_F1']:>6.4f}")
    print(f"  {'OA':<18} {metrics['OA']:>6.4f}")
    print(f"{'═'*58}\n")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_path = ROOT / "outputs" / "test_results.json"
    results  = {
        "timestamp":    datetime.now().isoformat(),
        "checkpoint":   str(ckpt_path),
        "checkpoint_epoch": saved_epoch,
        "val_miou":     round(saved_val_miou, 4),
        "num_classes":  num_classes,
        "active_names": active_names,
        "test_miou":    round(metrics["mIoU"],      4),
        "test_f1":      round(metrics["macro_F1"],  4),
        "test_oa":      round(metrics["OA"],        4),
        "test_loss":    round(metrics["loss"],      4),
        "per_class_iou": per_class,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"► Αποτελέσματα αποθηκεύτηκαν: {out_path}")
    print(f"\n  POSTER: test mIoU = {metrics['mIoU']:.4f}  |  "
          f"F1 = {metrics['macro_F1']:.4f}  |  OA = {metrics['OA']:.4f}\n")


if __name__ == "__main__":
    import sys as _sys
    # Προειδοποίηση — ο χρήστης επιβεβαιώνει πριν τρέξει
    print("\n⚠️  ΠΡΟΣΟΧΗ: Το test set αγγίζεται μόνο μία φορά.")
    print("   Βεβαιώσου ότι έχεις τελειώσει ΟΛΕΣ τις αποφάσεις (hyperparameters,")
    print("   architecture, loss function) πριν τρέξεις αυτό το script.")
    ans = input("\n   Συνέχεια; (y/N): ").strip().lower()
    if ans != "y":
        print("   Ακυρώθηκε.")
        _sys.exit(0)
    main()
