"""
train.py
Training pipeline για PointNet++ Mini on FRACTAL dataset.

Χρήση:
  python src/train.py                        # default settings
  python src/train.py --epochs 50 --bs 16    # custom
  python src/train.py --dry-run              # 2 batches για έλεγχο

Outputs (αποθηκεύονται στο outputs/):
  checkpoints/best_model.pt    ← καλύτερο μοντέλο βάσει val mIoU
  logs/run_<timestamp>.csv     ← metrics ανά epoch (για plots)

Metrics:
  mIoU   — mean Intersection over Union (κύριο metric)
  F1     — per-class, macro average
  OA     — Overall Accuracy
"""

import sys
import csv
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── Project imports ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.dataset   import FractalPatchDataset, NUM_CLASSES, CLASS_NAMES
from models.pointnet2 import PointNet2Mini


# ══════════════════════════════════════════════════════════════════════════════
# Focal Loss  (Lin et al., IEEE TPAMI 2020)
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss για point-wise segmentation σε imbalanced datasets.

    FL(p_t) = -(1 - p_t)^γ · log(p_t)

    Γιατί βοηθάει στο Bridge/LowVegetation:
    • Όταν το μοντέλο κάνει σωστή πρόβλεψη με confidence p=0.9 (π.χ. Ground):
        focal weight = (1-0.9)^2 = 0.01  → το loss σχεδόν εξαφανίζεται
    • Όταν κάνει λάθος με confidence p=0.1 (π.χ. Bridge):
        focal weight = (1-0.1)^2 = 0.81  → το loss παραμένει υψηλό

    Αποτέλεσμα: τα "easy" Ground/HighVeg examples δεν κατακλύζουν το gradient.
    Τα "hard" Bridge/LowVeg examples παίρνουν πίσω τον έλεγχο.

    Μπορεί να συνδυαστεί με class weights (weight= για επιπλέον βαρύτητα).

    Αναφορά (για Poster): Lin et al., "Focal Loss for Dense Object Detection",
                           IEEE TPAMI 2020. (Διαφάνειες Τσουμάκα, Class Imbalance)

    inputs: (B, C, N)  — ίδιο format με nn.CrossEntropyLoss
    targets: (B, N)    — class indices, -1 = ignore
    """

    def __init__(
        self,
        gamma:        float = 2.0,
        weight:       torch.Tensor = None,
        ignore_index: int   = -1,
    ):
        super().__init__()
        self.gamma        = gamma
        self.weight       = weight
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CE χωρίς reduction → (B, N), αγνοεί ignore_index positions
        ce = F.cross_entropy(
            inputs, targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        # p_t: πιθανότητα της σωστής κλάσης
        # exp(-CE) είναι ισοδύναμο αλλά numerically stable
        p_t = torch.exp(-ce)

        # Focal weight
        focal = ((1.0 - p_t) ** self.gamma) * ce

        # Μάσκα για valid positions (ignore_index → 0 loss)
        valid = targets != self.ignore_index
        if valid.sum() == 0:
            return focal.sum() * 0.0

        return focal[valid].mean()


# ══════════════════════════════════════════════════════════════════════════════
# Label remapping  (για --exclude-classes, π.χ. Run 5)
# ══════════════════════════════════════════════════════════════════════════════

class RemappedDataset(torch.utils.data.Dataset):
    """
    Wrapper dataset που εφαρμόζει label remapping.

    Excluded classes γίνονται -1 (ignore_index).
    Remaining classes re-indexed: 0 … (N_active − 1)

    Παράδειγμα: --exclude-classes water bridge
        Original: Ground=0, LowVeg=1, MedVeg=2, HighVeg=3,
                  Building=4, Water=5, Bridge=6, Noise=7
        Remapped: Ground=0, LowVeg=1, MedVeg=2, HighVeg=3,
                  Building=4, Noise=5   (Water/Bridge → -1)
    """

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        remap_lut:    torch.Tensor,      # (NUM_CLASSES_ORIGINAL,) dtype=long
    ):
        self.base      = base_dataset
        self.remap_lut = remap_lut

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        # y: (N,) int64, τιμές [0, NUM_CLASSES-1] ∪ {-1}
        y_new            = torch.full_like(y, -1)   # default: ignore
        mask             = y >= 0                    # valid (non-ignored)
        y_new[mask]      = self.remap_lut[y[mask]]
        return x, y_new


def build_label_remap(
    exclude_names: list,
    device:        torch.device,
) -> tuple:
    """
    Δημιουργεί LUT για label remapping.

    Returns:
        remap_lut    : Tensor (NUM_CLASSES,) — new index ή -1 αν excluded
                       None αν exclude_names είναι κενό
        active_names : list[str] — class names στη νέα σειρά
        num_active   : int — πλήθος active classes
    """
    if not exclude_names:
        return None, list(CLASS_NAMES.values()), NUM_CLASSES

    excl_lower  = {n.lower() for n in exclude_names}
    exclude_ids = {i for i, name in CLASS_NAMES.items()
                   if name.lower() in excl_lower}

    unknown = excl_lower - {n.lower() for n in CLASS_NAMES.values()}
    if unknown:
        print(f"  ⚠️  Άγνωστες κλάσεις για αποκλεισμό: {unknown}")

    remap_lut    = torch.full((NUM_CLASSES,), -1, dtype=torch.long)
    active_names = []
    new_idx      = 0
    for i in range(NUM_CLASSES):
        if i not in exclude_ids:
            remap_lut[i] = new_idx
            active_names.append(CLASS_NAMES.get(i, f"class_{i}"))
            new_idx += 1

    # remap_lut παραμένει σε CPU — χρησιμοποιείται στο __getitem__ (CPU tensors)
    # Τα batch tensors μεταφέρονται στο device μέσα στο training loop
    return remap_lut, active_names, new_idx


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    all_preds:   torch.Tensor,   # (M,)  int64
    all_targets: torch.Tensor,   # (M,)  int64
    num_classes: int,
) -> dict:
    """
    Υπολογίζει mIoU, per-class IoU, per-class F1, Overall Accuracy
    από confusion matrix.

    Γιατί IoU και όχι απλά accuracy;
    → Accuracy: "79% σωστά" — αρκεί να πεις "όλα Ground" σε imbalanced dataset
    → IoU: μετράει TP/(TP+FP+FN) ανά κλάση — δεν εξαπατάται από imbalance
    """
    C    = num_classes
    conf = torch.zeros(C, C, dtype=torch.long)

    for t, p in zip(all_targets.tolist(), all_preds.tolist()):
        if 0 <= t < C and 0 <= p < C:
            conf[t, p] += 1

    tp = conf.diag().float()
    fp = (conf.sum(0) - tp).float()
    fn = (conf.sum(1) - tp).float()

    # IoU ανά κλάση (ignore κλάσεις χωρίς ground truth)
    denom    = tp + fp + fn
    iou      = torch.where(denom > 0, tp / (denom + 1e-8), torch.tensor(float("nan")))
    miou     = iou[~iou.isnan()].mean().item()

    # F1 ανά κλάση
    f1_denom = 2 * tp + fp + fn
    f1       = torch.where(f1_denom > 0, 2 * tp / (f1_denom + 1e-8), torch.tensor(float("nan")))
    macro_f1 = f1[~f1.isnan()].mean().item()

    # Overall Accuracy
    oa = (tp.sum() / conf.sum().clamp(min=1)).item()

    return {
        "mIoU":         miou,
        "macro_F1":     macro_f1,
        "OA":           oa,
        "iou_per_class": iou.tolist(),
        "f1_per_class":  f1.tolist(),
        "conf_matrix":   conf,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Class weights (για imbalanced dataset)
# ══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(
    dataset:     FractalPatchDataset,
    num_samples: int = 0,
    device:      torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Υπολογίζει inverse-frequency class weights.

    w_c = total_points / (C × points_in_class_c)
    Normalize: mean(w) = 1

    num_samples:
        0  → ΟΛΑ τα patches (σταθερά, αναπαραγώγιμα weights — συνίσταται)
        >0 → τυχαίο sample N patches (γρηγορότερο αλλά με διακύμανση)

    Πρόβλημα με num_samples=50 (προηγούμενοι runs):
        Bridge είχε 54 points σε run1 (weight=6.79) vs 467 σε run2 (weight=0.63)
        → αστάθεια εκπαίδευσης. Με num_samples=0 τα weights είναι σταθερά.
    """
    n = len(dataset) if num_samples == 0 else min(num_samples, len(dataset))
    label = f"ΟΛΑ ({n})" if num_samples == 0 else str(n)
    print(f"  Υπολογισμός class weights από {label} patches...")

    counts = torch.zeros(NUM_CLASSES)

    indices = list(range(n)) if num_samples == 0 else \
              np.random.choice(len(dataset), size=n, replace=False).tolist()

    for step, i in enumerate(indices):
        _, y = dataset[i]
        for c in range(NUM_CLASSES):
            counts[c] += (y == c).sum().item()
        if (step + 1) % 500 == 0:
            print(f"    {step+1}/{n} patches σκαναρισμένα...")

    # Κλάσεις χωρίς points παίρνουν weight=0
    weights = torch.zeros(NUM_CLASSES)
    total   = counts.sum()
    present = counts > 0
    weights[present] = total / (NUM_CLASSES * counts[present])

    # Normalize ώστε mean weight = 1
    weights[present] /= weights[present].mean()

    # Cap ακραία weights — κυρίως για Noise/Bridge όταν χρησιμοποιείται με Focal Loss.
    # Focal Loss ήδη χειρίζεται hard examples· ακραία weights (>3) αποσταθεροποιούν training.
    # Τιμή 3.0: επιτρέπει Bridge=1.5x, Noise≤3x έναντι Ground — λογική διαφορά χωρίς ακραία κλίση.
    max_w = 3.0
    weights.clamp_(max=max_w)

    print(f"  {'Class':<18} {'Points':>12} {'Weight':>8} (cap={max_w})")
    print(f"  {'─'*42}")
    for c in range(NUM_CLASSES):
        name = CLASS_NAMES.get(c, f"class_{c}")
        print(f"  {name:<18} {int(counts[c]):>12,} {weights[c]:>8.3f}")

    return weights.to(device)


# ══════════════════════════════════════════════════════════════════════════════
# Train / Eval loops
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   nn.Module,
    optimizer:   torch.optim.Optimizer,
    device:      torch.device,
    dry_run:     bool = False,
    xyz_dropout: float = 0.0,
) -> float:
    """Ένας epoch training. Επιστρέφει mean loss."""
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch_idx, (X, y) in enumerate(loader):
        X = X.to(device)               # (B, N, 7)
        y = y.to(device)               # (B, N)

        optimizer.zero_grad()

        logits = model(X, xyz_dropout=xyz_dropout)   # (B, N, C)

        # CrossEntropyLoss περιμένει (B, C, N) αντί για (B, N, C)
        loss = criterion(logits.permute(0, 2, 1), y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

        if dry_run and batch_idx >= 1:
            break

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    device:     torch.device,
    num_classes: int,
    dry_run:    bool = False,
) -> dict:
    """Αξιολόγηση σε val/test set. Επιστρέφει loss + metrics."""
    model.eval()
    total_loss  = 0.0
    n_batches   = 0
    all_preds   = []
    all_targets = []

    for batch_idx, (X, y) in enumerate(loader):
        X = X.to(device)
        y = y.to(device)

        logits = model(X)                           # (B, N, C)
        loss   = criterion(logits.permute(0, 2, 1), y)

        preds  = logits.argmax(dim=-1)              # (B, N)

        all_preds.append(preds.cpu().reshape(-1))
        all_targets.append(y.cpu().reshape(-1))

        total_loss += loss.item()
        n_batches  += 1

        if dry_run and batch_idx >= 1:
            break

    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metrics         = compute_metrics(all_preds, all_targets, num_classes)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════

class CSVLogger:
    """
    Αποθηκεύει metrics ανά epoch σε CSV.
    Χρήσιμο για matplotlib plots στο Poster.
    """
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = False

    def log(self, row: dict) -> None:
        write_header = not self._header_written
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


# ══════════════════════════════════════════════════════════════════════════════
# Main training function
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═'*60}")
    print(f"  PointNet++ Mini — Training")
    print(f"  Device       : {device}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch        : {args.bs}")
    print(f"  Loss         : {args.loss.upper()}" +
          (f"  (γ={args.gamma})" if args.loss == "focal" else ""))
    print(f"  WeightSamples: {'ALL' if args.weight_samples == 0 else args.weight_samples}")
    print(f"  Cache        : {args.cache}")
    print(f"  Dry-run      : {args.dry_run}")
    if args.exclude_classes:
        print(f"  Exclude      : {', '.join(args.exclude_classes)}  (→ ignore_index=-1)")
    if args.xyz_dropout > 0.0:
        print(f"  XYZ Dropout  : {args.xyz_dropout:.2f}  (feature dropout, spatial xyz intact)")
    if args.resume_from:
        print(f"  Resume From  : {args.resume_from}")
    print(f"{'═'*60}\n")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("► Φόρτωση datasets...")
    train_ds = FractalPatchDataset(ROOT, split="train",
                                   num_points=args.num_points,
                                   cache=args.cache)
    val_ds   = FractalPatchDataset(ROOT, split="val",
                                   num_points=args.num_points,
                                   cache=args.cache)

    # ── Label remapping (για --exclude-classes, π.χ. Run 5) ──────────────────
    remap_lut, active_names, num_active = build_label_remap(
        getattr(args, "exclude_classes", []), device
    )
    if remap_lut is not None:
        print(f"  Active classes ({num_active}): {', '.join(active_names)}")
        train_ds = RemappedDataset(train_ds, remap_lut)
        val_ds   = RemappedDataset(val_ds,   remap_lut)

    # Windows: num_workers=0 για αποφυγή multiprocessing issues
    nw = 0 if sys.platform == "win32" else 4

    train_loader = DataLoader(train_ds, batch_size=args.bs,
                              shuffle=True,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.bs,
                              shuffle=False, num_workers=nw, pin_memory=True)

    print(f"  Train: {len(train_ds)} patches | "
          f"Val: {len(val_ds)} patches | "
          f"Batches/epoch: {len(train_loader)}")

    # ── Class weights ─────────────────────────────────────────────────────────
    print("\n► Class weights:")
    class_weights = compute_class_weights(
        train_ds, num_samples=args.weight_samples, device=device
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n► Μοντέλο:")
    model = PointNet2Mini(in_channels=7, num_classes=num_active).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,} | Size: {total_params*4/1024/1024:.2f}MB")

    # ── Resume from checkpoint (Run 7: fine-tune από Run 6) ──────────────────
    if args.resume_from is not None:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume-from: {resume_path} δεν βρέθηκε")
        ckpt = torch.load(resume_path, map_location=device)
        # Φορτώνουμε ΜΟΝΟ weights — όχι optimizer state (νέος LR, νέα training phase)
        model.load_state_dict(ckpt["model_state"])
        print(f"  ✓ Φορτώθηκε checkpoint: {resume_path.name}  "
              f"(epoch={ckpt.get('epoch','?')}, mIoU={ckpt.get('mIoU',0):.4f})")
        print(f"  → Fine-tuning με xyz_dropout={args.xyz_dropout}")
    elif args.xyz_dropout > 0.0:
        print(f"  xyz_dropout={args.xyz_dropout} (training from scratch)")

    # ── Loss & Optimizer ──────────────────────────────────────────────────────
    if args.loss == "focal":
        # Focal Loss ΔΕΝ συνδυάζεται με inverse-frequency weights σε ακραία imbalance (>1000:1).
        # Το (1-p_t)^γ term ήδη κάνει downweight τα easy examples — προσθήκη weights
        # αποδυναμώνει Ground/HighVeg (w≈0.005) μέχρι να εξαφανιστούν από το gradient.
        # Ακολουθούμε Lin et al. (TPAMI 2020): Focal Loss χωρίς class weights.
        criterion = FocalLoss(gamma=args.gamma, weight=None, ignore_index=-1)
        print(f"\n► Loss: FocalLoss (γ={args.gamma})  [χωρίς class weights — FL handles imbalance]")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        print(f"\n► Loss: CrossEntropyLoss + class weights")
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    ckpt_dir = ROOT / "outputs" / "checkpoints"
    log_dir  = ROOT / "outputs" / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True,  exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = log_dir / f"run_{timestamp}.csv"
    ckpt_path = ckpt_dir / "best_model.pt"
    logger    = CSVLogger(log_path)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n► Training... (logs → {log_path.name})\n")
    print(f"  {'Epoch':>5} {'Train Loss':>11} {'Val Loss':>9} "
          f"{'mIoU':>7} {'F1':>7} {'OA':>7} {'LR':>9} {'Time':>7}")
    print(f"  {'─'*65}")

    best_miou = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            args.dry_run, xyz_dropout=args.xyz_dropout,
        )

        # Validate
        val_metrics = evaluate(
            model, val_loader, criterion, device, num_active, args.dry_run
        )

        scheduler.step()

        elapsed = time.perf_counter() - t0
        lr      = optimizer.param_groups[0]["lr"]
        miou    = val_metrics["mIoU"]
        f1      = val_metrics["macro_F1"]
        oa      = val_metrics["OA"]
        val_loss = val_metrics["loss"]

        # Print
        print(f"  {epoch:>5} {train_loss:>11.4f} {val_loss:>9.4f} "
              f"{miou:>7.4f} {f1:>7.4f} {oa:>7.4f} "
              f"{lr:>9.2e} {elapsed:>6.1f}s")

        # Log to CSV
        row = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "mIoU":       round(miou, 6),
            "macro_F1":   round(f1, 6),
            "OA":         round(oa, 6),
            "lr":         round(lr, 8),
        }
        # Προσθέτουμε per-class IoU στο CSV
        for c, iou_c in enumerate(val_metrics["iou_per_class"]):
            name = CLASS_NAMES.get(c, f"class_{c}")
            row[f"iou_{name}"] = round(iou_c, 6) if not np.isnan(iou_c) else ""
        logger.log(row)

        # Checkpoint (best mIoU)
        if miou > best_miou:
            best_miou = miou
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "optim_state":  optimizer.state_dict(),
                "mIoU":         miou,
                "args":         vars(args),
                "num_classes":  num_active,      # για evaluate_test.py & export_onnx.py
                "active_names": active_names,    # class names σε νέα σειρά
                "remap_lut":    remap_lut,        # None αν δεν χρησιμοποιήθηκε exclude
            }, ckpt_path)
            print(f"  {'':>5} ✓ best mIoU={miou:.4f} → αποθηκεύτηκε")

        if args.dry_run:
            print("\n  [dry-run] Σταματάμε μετά από 1 epoch.")
            break

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  Best val mIoU : {best_miou:.4f}")
    print(f"  Checkpoint    : {ckpt_path}")
    print(f"  Log           : {log_path}")
    print(f"{'═'*55}\n")

    # Per-class IoU του best checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print("► Per-class IoU (best checkpoint, val set):")
    val_final = evaluate(model, val_loader, criterion, device, num_active)
    for c, iou_c in enumerate(val_final["iou_per_class"]):
        name = active_names[c] if c < len(active_names) else f"class_{c}"
        bar  = "█" * int(iou_c * 20) if not np.isnan(iou_c) else "—"
        print(f"  {name:<18} {iou_c:>6.3f}  {bar}")
    print(f"\n  mIoU = {val_final['mIoU']:.4f} | "
          f"F1 = {val_final['macro_F1']:.4f} | "
          f"OA = {val_final['OA']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train PointNet++ Mini")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--bs",         type=int,   default=8,
                   help="batch size (default 8 — μείωσε αν OOM)")
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--num-points", type=int,   default=4096,
                   dest="num_points")
    p.add_argument("--cache",      action="store_true",
                   help="χρήση .npz cache για γρήγορο loading")
    p.add_argument("--dry-run",    action="store_true",
                   dest="dry_run",
                   help="τρέχει 2 batches για έλεγχο pipeline")

    # ── Imbalance techniques ───────────────────────────────────────────────
    p.add_argument("--loss",    type=str, default="focal",
                   choices=["ce", "focal"],
                   help="loss function: 'ce' = CrossEntropy, 'focal' = FocalLoss (default)")
    p.add_argument("--gamma",   type=float, default=2.0,
                   help="Focal Loss γ parameter (default 2.0, ignored if --loss ce)")
    p.add_argument("--weight-samples", type=int, default=0,
                   dest="weight_samples",
                   help="patches για class weight computation (0=ALL, default)")
    p.add_argument("--exclude-classes", type=str, nargs="*", default=[],
                   dest="exclude_classes",
                   metavar="CLASS",
                   help="κλάσεις για αποκλεισμό από training (→ ignore_index=-1, "
                        "π.χ. --exclude-classes water bridge) — για Run 5 / OOD base model")

    # ── Run 7: XYZ feature dropout + fine-tuning ──────────────────────────────
    p.add_argument("--xyz-dropout", type=float, default=0.0,
                   dest="xyz_dropout",
                   help="Probability of zeroing xyz feature dims (0-2) per batch "
                        "during training. Spatial xyz for FPS/KNN stays intact. "
                        "Forces model to use intensity/returns for OOD-relevant classes. "
                        "(default 0.0 = disabled, Run 7: use 0.20)")
    p.add_argument("--resume-from", type=str, default=None,
                   dest="resume_from",
                   help="Path to checkpoint to fine-tune from "
                        "(e.g. outputs/checkpoints/best_model_run6.pt). "
                        "Loads model weights only (not optimizer state). "
                        "Use with lower --lr (e.g. 1e-4) for fine-tuning.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
