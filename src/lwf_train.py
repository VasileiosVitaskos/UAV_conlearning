"""
lwf_train.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Learning without Forgetting (LwF) — Continual Learning update.

Αντί για απλή prototype προσθήκη (k-NN level), αυτό το script κάνει
ΠΡΑΓΜΑΤΙΚΟ gradient descent με Knowledge Distillation:

  Loss = CE(νέα κλάση) + λ·KL(teacher_old || student_old[:N])
         ───────────────  ──────────────────────────────────────
         μαθαίνει Water    δεν ξεχνάει Ground/Veg/Building

Backbone: FROZEN — μόνο ο classifier αλλάζει (32 → 7 weights).
Αυτό το κάνει feasible στο Pi5 (4-6 λεπτά, χωρίς GPU).

Πηγή: Li & Hoiem, "Learning without Forgetting", ECCV 2016 / TPAMI 2018
      https://arxiv.org/abs/1606.09282

Pipeline:
  1. few_shot_add_class.py  → best_model_cl_water.pt (prototype init, 7 κλάσεις)
  2. lwf_train.py            → best_model_lwf_water.pt (gradient update)
  3. export_onnx.py          → model_lwf_water.onnx  (Pi5 deployment)

Χρήση:
  # Water (από FRACTAL basket):
  python src/lwf_train.py \\
      --class-name Water \\
      --basket-dir data/basket/water \\
      --cl-checkpoint outputs/checkpoints/best_model_cl_water.pt

  # Custom hyperparameters:
  python src/lwf_train.py --class-name Water --lambda-kd 2.0 --epochs 5 --lr 1e-3

  # Χωρίς ONNX re-export:
  python src/lwf_train.py --class-name Water --no-export
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models.pointnet2 import PointNet2Mini
from data.dataset     import FractalPatchDataset, CLASS_MAP, CLASS_NAMES


# ── Remap: CLASS_MAP index → model output index ────────────────────────────
# Base model (6 classes): Water(CLASS_MAP=5)→-1, Bridge(6)→-1, Noise(7)→5
# Αυτό φορτώνεται αυτόματα από το checkpoint (remap_lut)
_DEFAULT_REMAP_LUT = [0, 1, 2, 3, 4, -1, -1, 5]  # fallback αν δεν υπάρχει στο ckpt


def parse_args():
    p = argparse.ArgumentParser(
        description="LwF Continual Learning update για νέα κλάση",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--class-name", type=str, default="Water",
                   dest="class_name",
                   help="Νέα κλάση (Water ή Bridge). Default: Water")
    p.add_argument("--basket-dir", type=str, default=None,
                   dest="basket_dir",
                   help="Φάκελος με labeled basket patches (.laz). "
                        "Default: data/basket/<class_name.lower()>")
    p.add_argument("--cl-checkpoint", type=str, default=None,
                   dest="cl_checkpoint",
                   help="Prototype-initialized checkpoint από few_shot_add_class.py. "
                        "Default: outputs/checkpoints/best_model_cl_<class_name>.pt")
    p.add_argument("--base-checkpoint", type=str, default=None,
                   dest="base_checkpoint",
                   help="Βασικό checkpoint (teacher). Default: outputs/checkpoints/best_model.pt")
    p.add_argument("--output", type=str, default=None,
                   help="Output path. Default: outputs/checkpoints/best_model_lwf_<class>.pt")
    p.add_argument("--lambda-kd", type=float, default=1.0, dest="lambda_kd",
                   help="Βάρος Knowledge Distillation loss. Default: 1.0")
    p.add_argument("--temperature", type=float, default=2.0,
                   help="KD temperature T (softens distributions). Default: 2.0")
    p.add_argument("--epochs", type=int, default=10,
                   help="Εποχές gradient descent. Default: 10")
    p.add_argument("--lr", type=float, default=5e-3,
                   help="Learning rate (AdamW). Default: 5e-3")
    p.add_argument("--num-points", type=int, default=4096, dest="num_points",
                   help="Points per patch. Default: 4096")
    p.add_argument("--batch-size", type=int, default=4, dest="batch_size",
                   help="Batch size (Pi5: 2-4). Default: 4")
    p.add_argument("--min-target-pts", type=int, default=20, dest="min_target_pts",
                   help="Ελάχιστα new-class points ανά patch. Default: 20")
    p.add_argument("--fractal-split", type=str, default="val",
                   dest="fractal_split",
                   choices=["train", "val"],
                   help="FRACTAL split για training (αν δεν έχεις basket). Default: val")
    p.add_argument("--n-fractal-shots", type=int, default=10,
                   dest="n_fractal_shots",
                   help="Patches από FRACTAL split (αν δεν έχεις basket). Default: 10")
    p.add_argument("--no-export", action="store_true", dest="no_export",
                   help="Μην κάνεις ONNX re-export μετά το training")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Dataset for LwF training
# ══════════════════════════════════════════════════════════════════════════════

class BasketDataset(Dataset):
    """
    Dataset για LwF: φορτώνει .laz patches από το basket dir.

    Κάθε patch επιστρέφει:
        X      : (N, 7) float32  — normalized features
        y_map  : (N,)   int64    — CLASS_MAP indices (5=Water, 6=Bridge, κ.λπ.)
    """
    def __init__(
        self,
        laz_files:  list,
        num_points: int  = 4096,
        seed:       int  = 42,
    ):
        self.files      = laz_files
        self.num_points = num_points
        self.rng        = np.random.default_rng(seed)

        # Φόρτωσε normalizer stats
        stats_path = ROOT / "outputs" / "normalizer_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                self.stats = json.load(f)
        else:
            raise FileNotFoundError(f"normalizer_stats.json not found: {stats_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        from preprocessing import extract_features
        import laspy

        laz_path = self.files[idx]
        las      = laspy.read(str(laz_path))
        X, y     = extract_features(las, is_fractal=True,
                                     valid_classes={2,3,4,5,6,9,17,64},
                                     stats=self.stats)

        # Sample to fixed N
        n = len(X)
        if n == 0:
            X = np.zeros((self.num_points, 7), dtype=np.float32)
            y = np.zeros(self.num_points, dtype=np.int32)
        else:
            replace = n < self.num_points
            sel = self.rng.choice(n, size=self.num_points, replace=replace)
            X, y = X[sel], y[sel]

        # Convert raw LAS labels → CLASS_MAP indices
        # extract_features returns raw LAS values (2,3,4,5,6,9,17,64).
        # lwf_loss expects CLASS_MAP indices (0,1,2,3,4,5,6,7).
        # Without this, Water (raw=9) gets clamped to full_lut[8]=-1 and is ignored.
        y_map_arr = np.full(len(y), -1, dtype=np.int64)
        for raw_cls, map_idx in CLASS_MAP.items():
            y_map_arr[y == raw_cls] = map_idx

        return (
            torch.from_numpy(X),
            torch.from_numpy(y_map_arr),  # CLASS_MAP indices (0-7)
        )


class FractalCLDataset(Dataset):
    """
    Wrapper γύρω από FractalPatchDataset που επιστρέφει μόνο patches
    με αρκετά new-class points.
    """
    def __init__(
        self,
        split:          str,
        target_map_idx: int,
        n_patches:      int,
        num_points:     int = 4096,
        min_target:     int = 20,
        seed:           int = 42,
    ):
        base_ds = FractalPatchDataset(
            root=ROOT, split=split, num_points=num_points,
            remap=True, cache=True, seed=seed,
        )
        rng     = np.random.default_rng(seed)
        indices = rng.permutation(len(base_ds)).tolist()

        self.samples = []
        for idx in indices:
            X, y = base_ds[idx]
            if (y == target_map_idx).sum() >= min_target:
                self.samples.append((X, y))
            if len(self.samples) >= n_patches:
                break

        print(f"  FractalCLDataset: {len(self.samples)} patches με ≥{min_target} "
              f"{CLASS_NAMES.get(target_map_idx,'?')} points (σκανάρηκαν {len(indices)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ══════════════════════════════════════════════════════════════════════════════
# LwF Loss
# ══════════════════════════════════════════════════════════════════════════════

def lwf_loss(
    student_logits: torch.Tensor,   # (B, N, C_new)  C_new = N+1
    teacher_logits: torch.Tensor,   # (B, N, C_old)  C_old = N
    y_map:          torch.Tensor,   # (B, N) CLASS_MAP indices
    remap_lut:      list,           # CLASS_MAP → model index (-1 = ignored)
    new_class_idx:  int,            # model output index for new class (C_old)
    new_class_map:  int,            # CLASS_MAP index for new class (5=Water)
    lambda_kd:      float,
    temperature:    float,
) -> tuple:
    """
    LwF loss:
      L = CE_new  +  λ · KL_old

    CE_new:  Cross-entropy για τα new-class points
             (points με y_map == new_class_map → target = new_class_idx)

    KL_old:  Knowledge Distillation για τα non-new-class points
             KL(softmax(teacher/T) || softmax(student[:C_old]/T))
             Μόνο για points που έχουν valid label στο παλιό μοντέλο.

    Returns:
        total_loss, ce_loss, kd_loss  (scalars)
    """
    B, N, C_new = student_logits.shape
    C_old = teacher_logits.shape[-1]

    # ── Build target labels για CE ─────────────────────────────────────────
    # remap: CLASS_MAP index → student model index (-1 = ignore)
    lut = torch.tensor(remap_lut + [new_class_idx], dtype=torch.long)
    # lut[new_class_map] = new_class_idx (προσθέτουμε τη νέα κλάση στο LUT)
    full_lut = torch.full((max(CLASS_MAP.values()) + 2,), fill_value=-1, dtype=torch.long)
    for raw_cls, map_idx in CLASS_MAP.items():
        if map_idx < len(remap_lut) and remap_lut[map_idx] >= 0:
            full_lut[map_idx] = remap_lut[map_idx]
    full_lut[new_class_map] = new_class_idx

    # CLASS_MAP indices → student model indices
    y_map_clipped = y_map.clamp(0, len(full_lut) - 1)
    y_student = full_lut[y_map_clipped]  # (B, N)  — -1 = ignore

    # ── CE για νέα κλάση ───────────────────────────────────────────────────
    new_mask = (y_student == new_class_idx)  # (B, N) — Water points
    if new_mask.any():
        student_flat = student_logits.reshape(-1, C_new)   # (B*N, C_new)
        target_flat  = y_student.reshape(-1)               # (B*N,)
        ce_loss = F.cross_entropy(
            student_flat, target_flat, ignore_index=-1, reduction="mean"
        )
    else:
        ce_loss = torch.tensor(0.0, device=student_logits.device)

    # ── KD για παλιές κλάσεις ─────────────────────────────────────────────
    # Μόνο σε points με valid παλιό label (όχι -1, όχι νέα κλάση)
    old_mask = (y_student >= 0) & (y_student < C_old)   # (B, N)

    if old_mask.any() and lambda_kd > 0:
        # Logits παλιών κλάσεων από student (πρώτες C_old εξόδους)
        student_old = student_logits[:, :, :C_old]    # (B, N, C_old)

        T = temperature
        teacher_soft = F.softmax(teacher_logits[old_mask] / T, dim=-1)   # (M, C_old)
        student_soft = F.log_softmax(student_old[old_mask] / T, dim=-1)  # (M, C_old)

        kd_loss = F.kl_div(student_soft, teacher_soft,
                           reduction="batchmean") * (T * T)
    else:
        kd_loss = torch.tensor(0.0, device=student_logits.device)

    total = ce_loss + lambda_kd * kd_loss
    return total, ce_loss.item(), kd_loss.item()


# ══════════════════════════════════════════════════════════════════════════════
# ONNX re-export
# ══════════════════════════════════════════════════════════════════════════════

def export_onnx_after_lwf(model, out_path: Path, num_points: int, num_classes: int) -> None:
    print(f"\n  Re-exporting ONNX ({num_classes} classes) → {out_path.name}...")
    dummy = torch.zeros(1, num_points, 7)   # deterministic (torch.zeros — ONNX fix)
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
        print(f"  ✓ ONNX: {out_path}  ({size_kb:.0f} KB)")
    except Exception as e:
        print(f"  ✗ ONNX export failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu")   # Pi5: always CPU

    # ── Paths ─────────────────────────────────────────────────────────────────
    class_name_lower = args.class_name.lower()
    basket_dir = Path(args.basket_dir) if args.basket_dir else \
                 ROOT / "data" / "basket" / class_name_lower

    cl_ckpt_path = Path(args.cl_checkpoint) if args.cl_checkpoint else \
                   ROOT / "outputs" / "checkpoints" / f"best_model_cl_{class_name_lower}.pt"

    base_ckpt_path = Path(args.base_checkpoint) if args.base_checkpoint else \
                     ROOT / "outputs" / "checkpoints" / "best_model.pt"

    out_path = Path(args.output) if args.output else \
               ROOT / "outputs" / "checkpoints" / f"best_model_lwf_{class_name_lower}.pt"

    print(f"\n{'═'*65}")
    print(f"  LwF Continual Learning — '{args.class_name}'")
    print(f"  Teacher (base)  : {base_ckpt_path.name}")
    print(f"  Student (proto) : {cl_ckpt_path.name}")
    print(f"  Basket          : {basket_dir}")
    print(f"  λ_KD            : {args.lambda_kd}   T={args.temperature}")
    print(f"  Epochs / LR     : {args.epochs} / {args.lr}")
    print(f"{'═'*65}")

    # ── Validation ────────────────────────────────────────────────────────────
    if not base_ckpt_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {base_ckpt_path}")
    if not cl_ckpt_path.exists():
        print(f"\n  ⚠  CL checkpoint not found: {cl_ckpt_path}")
        print(f"  Τρέξε πρώτα: python src/few_shot_add_class.py "
              f"--class-name {args.class_name}")
        sys.exit(1)

    # ── Load teacher ──────────────────────────────────────────────────────────
    base_ckpt = torch.load(str(base_ckpt_path), map_location=device, weights_only=False)
    n_old     = base_ckpt.get("num_classes", 6)
    remap_lut = base_ckpt.get("remap_lut", _DEFAULT_REMAP_LUT)
    # fix: checkpoint saved on CUDA → remap_lut is Tensor, not list
    if isinstance(remap_lut, torch.Tensor):
        remap_lut = remap_lut.cpu().tolist()

    teacher = PointNet2Mini(in_channels=7, num_classes=n_old).to(device)
    teacher.load_state_dict(base_ckpt["model_state"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print(f"\n  Teacher: {n_old} classes → {base_ckpt.get('active_names','?')}")

    # ── Load student ──────────────────────────────────────────────────────────
    cl_ckpt      = torch.load(str(cl_ckpt_path), map_location=device, weights_only=False)
    n_new        = cl_ckpt.get("num_classes", n_old + 1)
    active_names = cl_ckpt.get("active_names", [f"class_{i}" for i in range(n_new)])
    new_class_map_idx = cl_ckpt.get("cl_source_map_idx", 5)  # CLASS_MAP idx (5=Water)
    new_class_model_idx = n_old                               # model output idx

    student = PointNet2Mini(in_channels=7, num_classes=n_new).to(device)
    student.load_state_dict(cl_ckpt["model_state"])

    # Freeze backbone (encoder + feature propagation), ΜΟΝΟ classifier trainable
    for name, param in student.named_parameters():
        if "classifier" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in student.parameters())
    print(f"  Student: {n_new} classes → {active_names}")
    print(f"  Trainable params: {trainable:,} / {total:,}  (classifier only)")
    print(f"  New class: '{args.class_name}'  "
          f"(CLASS_MAP idx={new_class_map_idx}, model idx={new_class_model_idx})")

    # ── Build dataset ─────────────────────────────────────────────────────────
    laz_files = []
    if basket_dir.exists():
        laz_files = sorted(basket_dir.glob("*.laz"))
        print(f"\n  Basket: {len(laz_files)} .laz files από {basket_dir}")

    if laz_files:
        train_ds = BasketDataset(laz_files, num_points=args.num_points, seed=args.seed)
    else:
        print(f"\n  Basket dir δεν υπάρχει ή άδειο. "
              f"Χρησιμοποιώ FRACTAL {args.fractal_split} split ({args.n_fractal_shots} shots).")
        train_ds = FractalCLDataset(
            split=args.fractal_split,
            target_map_idx=new_class_map_idx,
            n_patches=args.n_fractal_shots,
            num_points=args.num_points,
            min_target=args.min_target_pts,
            seed=args.seed,
        )

    if len(train_ds) == 0:
        print("\n  ✗ Κανένα patch για training! "
              "Τρέξε: python src/make_basket.py")
        sys.exit(1)

    nw     = 0   # Pi5: num_workers=0 (no fork overhead)
    loader = DataLoader(train_ds, batch_size=args.batch_size,
                        shuffle=True, num_workers=nw)
    print(f"  Batches/epoch: {len(loader)}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n  ── Training ({args.epochs} epochs) ──")
    print(f"  {'Epoch':>6}  {'Loss':>8}  {'CE':>8}  {'KD':>8}  {'LR':>10}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}")

    best_loss = float("inf")
    best_state = None

    student.train()
    # Forza eval mode su BatchNorm del backbone (frozen)
    for m in student.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.eval()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_ce   = 0.0
        epoch_kd   = 0.0
        n_batches  = 0

        for X_batch, y_map_batch in loader:
            X_batch     = X_batch.to(device)      # (B, N, 7)
            y_map_batch = y_map_batch.to(device)   # (B, N) — CLASS_MAP indices

            # Teacher forward (frozen, no grad)
            with torch.no_grad():
                t_logits = teacher(X_batch)        # (B, N, C_old)

            # Student forward
            s_logits = student(X_batch)            # (B, N, C_new)

            loss, ce, kd = lwf_loss(
                student_logits=s_logits,
                teacher_logits=t_logits,
                y_map=y_map_batch,
                remap_lut=remap_lut,
                new_class_idx=new_class_model_idx,
                new_class_map=new_class_map_idx,
                lambda_kd=args.lambda_kd,
                temperature=args.temperature,
            )

            optimizer.zero_grad()
            loss.backward()
            # ΚΡΙΣΙΜΟ: μόνο η νέα κλάση (γραμμή new_class_model_idx) ενημερώνεται.
            # Μηδενίζουμε τα gradients των παλιών κλάσεων ώστε τα prototypes τους
            # να παραμείνουν ακριβώς όπως ήταν (δεν χρειάζεται KD για αυτό).
            if student.classifier.weight.grad is not None:
                student.classifier.weight.grad[:new_class_model_idx] = 0.0
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_ce   += ce
            epoch_kd   += kd
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_ce   = epoch_ce   / max(n_batches, 1)
        avg_kd   = epoch_kd   / max(n_batches, 1)
        cur_lr   = scheduler.get_last_lr()[0]

        print(f"  {epoch:>6}  {avg_loss:>8.4f}  {avg_ce:>8.4f}  "
              f"{avg_kd:>8.4f}  {cur_lr:>10.2e}")

        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = {k: v.clone() for k, v in student.state_dict().items()}

    # ── Save checkpoint ───────────────────────────────────────────────────────
    student.load_state_dict(best_state)
    student.eval()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_ckpt = {
        **cl_ckpt,
        "model_state":  student.state_dict(),
        "lwf_epochs":   args.epochs,
        "lwf_lr":       args.lr,
        "lwf_lambda_kd": args.lambda_kd,
        "lwf_temperature": args.temperature,
        "lwf_best_loss": round(best_loss, 4),
        "lwf_n_train_patches": len(train_ds),
    }
    torch.save(new_ckpt, out_path)
    print(f"\n  ✓ LwF checkpoint: {out_path}")

    # ── ONNX re-export ────────────────────────────────────────────────────────
    if not args.no_export:
        onnx_path = ROOT / "outputs" / f"model_lwf_{class_name_lower}.onnx"
        export_onnx_after_lwf(student, onnx_path, args.num_points, n_new)

        # Update model.json metadata
        meta_path = ROOT / "outputs" / f"model_lwf_{class_name_lower}.json"
        with open(ROOT / "outputs" / "model.json") as f:
            meta = json.load(f)
        meta["num_classes"]    = n_new
        meta["active_classes"] = acti