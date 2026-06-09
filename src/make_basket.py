"""
make_basket.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Δημιουργεί το "καλάθι" (basket) με labeled patches για Few-shot CL.

Σκανάρει το FRACTAL dataset, βρίσκει patches που περιέχουν Water ή
Bridge points, και τα αντιγράφει σε data/basket/<class>/.

Αυτά τα patches είναι:
  • Labeled (από FRACTAL ground truth)
  • Μικρά (50×50m, ~150K points)
  • Αρκετά ώστε να φτιαχτεί prototype (5-10 patches = αρκετό)

Χρήση:
  # Default: 10 Water + 10 Bridge patches από val split
  python src/make_basket.py

  # Περισσότερα patches:
  python src/make_basket.py --n-water 15 --n-bridge 15 --split train

  # Μόνο Water:
  python src/make_basket.py --classes water --n-water 10

  # Έλεγχος τι υπάρχει ήδη:
  python src/make_basket.py --info-only

Output:
  data/basket/
    water/
      patch_val_0042.laz    ← .laz αρχείο από FRACTAL val
      patch_val_0107.laz
      ...
    bridge/
      patch_val_0055.laz
      ...
    basket_index.json       ← metadata (ποια patches, πόσα points, CLASS_MAP idx)
"""

import sys
import shutil
import json
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import FractalPatchDataset, CLASS_NAMES, CLASS_MAP


# CLASS_MAP ανεστραμμένα: όνομα → CLASS_MAP index
_NAME_TO_MAP_IDX = {v.lower(): k for k, v in CLASS_NAMES.items()}

# Οι κλάσεις που εξαιρέθηκαν από το training (OOD targets)
OOD_CLASSES = {
    "water":  5,   # CLASS_MAP index 5  (raw LAS: 9)
    "bridge": 6,   # CLASS_MAP index 6  (raw LAS: 17)
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Build labeled basket from FRACTAL Water/Bridge patches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--classes", type=str, default="water,bridge",
                   help="Κλάσεις (comma-separated): water,bridge. Default: water,bridge")
    p.add_argument("--n-water", type=int, default=10, dest="n_water",
                   help="Πλήθος Water patches (default: 10)")
    p.add_argument("--n-bridge", type=int, default=10, dest="n_bridge",
                   help="Πλήθος Bridge patches (default: 10)")
    p.add_argument("--split", type=str, default="val",
                   choices=["train", "val", "test"],
                   help="FRACTAL split να σκανάρεις (default: val)")
    p.add_argument("--min-points", type=int, default=50, dest="min_points",
                   help="Ελάχιστα class points σε patch για να θεωρηθεί valid (default: 50)")
    p.add_argument("--output-dir", type=str,
                   default=str(ROOT / "data" / "basket"),
                   dest="output_dir",
                   help="Φάκελος εξόδου (default: data/basket)")
    p.add_argument("--num-points", type=int, default=4096, dest="num_points",
                   help="Sampling per patch για έλεγχο labels (default: 4096)")
    p.add_argument("--info-only", action="store_true", dest="info_only",
                   help="Εμφάνισε μόνο τι υπάρχει στο basket (χωρίς scanning)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Basket info
# ══════════════════════════════════════════════════════════════════════════════

def show_basket_info(basket_dir: Path) -> None:
    """Εκτυπώνει περιεχόμενα του basket."""
    if not basket_dir.exists():
        print(f"\n  Basket dir δεν υπάρχει: {basket_dir}")
        print(f"  Τρέξε: python src/make_basket.py")
        return

    index_path = basket_dir / "basket_index.json"
    if index_path.exists():
        with open(index_path) as f:
            idx = json.load(f)
        print(f"\n  Basket index ({index_path}):")
        for cls_name, entries in idx.get("classes", {}).items():
            print(f"    {cls_name}: {len(entries)} patches")
            for e in entries[:3]:
                print(f"      {e['file']}  ({e['n_target_pts']} pts)")
            if len(entries) > 3:
                print(f"      ... (+{len(entries)-3} more)")
    else:
        print(f"\n  Basket: {basket_dir}")
        for sub in sorted(basket_dir.iterdir()):
            if sub.is_dir():
                laz_files = list(sub.glob("*.laz"))
                print(f"    {sub.name}/  → {len(laz_files)} patches")


# ══════════════════════════════════════════════════════════════════════════════
# Patch selection
# ══════════════════════════════════════════════════════════════════════════════

def collect_class_patches(
    split:          str,
    class_name:     str,
    class_map_idx:  int,
    n_patches:      int,
    num_points:     int,
    min_points:     int,
    seed:           int = 42,
) -> list[dict]:
    """
    Σκανάρει το FRACTAL split και βρίσκει patches με αρκετά class points.

    Returns:
        list of dicts: [{file_path, n_target_pts, pct_target, patch_idx}, ...]
    """
    ds = FractalPatchDataset(
        root=ROOT, split=split,
        num_points=num_points,
        remap=True,
        cache=True,
        seed=seed,
    )

    rng     = np.random.default_rng(seed)
    indices = rng.permutation(len(ds)).tolist()
    found   = []
    scanned = 0

    print(f"\n  Σκανάρισμα {split} split για '{class_name}' "
          f"(CLASS_MAP idx={class_map_idx})...")
    print(f"  {'#':>4}  {'Patch idx':>10}  {'Target pts':>11}  {'%':>6}  File")
    print(f"  {'─'*4}  {'─'*10}  {'─'*11}  {'─'*6}  {'─'*30}")

    for idx in indices:
        scanned += 1

        X, y = ds[idx]
        target_mask = (y == class_map_idx)
        n_target    = int(target_mask.sum())

        if n_target < min_points:
            continue

        pct = n_target / len(y) * 100
        file_path = ds.file_path(idx)

        entry = {
            "file":          str(file_path),
            "file_name":     file_path.name,
            "split":         split,
            "patch_idx":     idx,
            "n_target_pts":  n_target,
            "pct_target":    round(pct, 2),
            "class_name":    class_name,
            "class_map_idx": class_map_idx,
        }
        found.append(entry)
        print(f"  {len(found):>4}  {idx:>10}  {n_target:>11,}  {pct:>5.1f}%  {file_path.name}")

        if len(found) >= n_patches:
            break

    if not found:
        print(f"  ✗ Δεν βρέθηκε κανένα patch με ≥{min_points} '{class_name}' points "
              f"στο {split} split (scanned {scanned})")
        print(f"    Δοκίμασε: --split train  ή  --min-points {min_points // 2}")
    else:
        print(f"  ✓ Βρέθηκαν {len(found)} patches (scanned {scanned} total)")

    return found


# ══════════════════════════════════════════════════════════════════════════════
# Copy patches to basket
# ══════════════════════════════════════════════════════════════════════════════

def copy_to_basket(
    entries:     list[dict],
    class_name:  str,
    basket_dir:  Path,
) -> list[dict]:
    """Αντιγράφει .laz files στο basket/<class_name>/."""
    class_dir = basket_dir / class_name.lower()
    class_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for i, e in enumerate(entries):
        src  = Path(e["file"])
        dst  = class_dir / f"basket_{class_name.lower()}_{i:04d}_{src.stem}.laz"
        shutil.copy2(str(src), str(dst))
        e_copy = {**e, "basket_file": str(dst), "basket_filename": dst.name}
        copied.append(e_copy)
        print(f"    Copied: {src.name}  →  {dst.name}")

    print(f"  ✓ {len(copied)} {class_name} patches στο {class_dir}")
    return copied


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    basket_dir = Path(args.output_dir)

    print(f"\n{'═'*65}")
    print(f"  Make Basket — Labeled Patches για Few-shot CL")
    print(f"  Split  : {args.split}")
    print(f"  Output : {basket_dir}")
    print(f"{'═'*65}")

    # ── Info only ─────────────────────────────────────────────────────────────
    if args.info_only:
        show_basket_info(basket_dir)
        return

    # ── Parse requested classes ───────────────────────────────────────────────
    requested_classes = [c.strip().lower() for c in args.classes.split(",")]
    class_n_patches = {
        "water":  args.n_water,
        "bridge": args.n_bridge,
    }

    basket_index = {"split": args.split, "classes": {}}

    # ── Scan + copy ───────────────────────────────────────────────────────────
    for cls_name in requested_classes:
        if cls_name not in OOD_CLASSES:
            print(f"\n  ⚠  Άγνωστη κλάση '{cls_name}' — παράλειψη. "
                  f"Γνωστές: {list(OOD_CLASSES.keys())}")
            continue

        class_map_idx = OOD_CLASSES[cls_name]
        n_patches     = class_n_patches.get(cls_name, 10)

        entries = collect_class_patches(
            split=args.split,
            class_name=cls_name.capitalize(),
            class_map_idx=class_map_idx,
            n_patches=n_patches,
            num_points=args.num_points,
            min_points=args.min_points,
            seed=args.seed,
        )

        if not entries:
            continue

        print(f"\n  Αντιγραφή {len(entries)} {cls_name} patches...")
        copied = copy_to_basket(entries, cls_name, basket_dir)
        basket_index["classes"][cls_name] = copied

    # ── Save index ────────────────────────────────────────────────────────────
    if basket_index["classes"]:
        index_path = basket_dir / "basket_index.json"
        with open(index_path, "w") as f:
            json.dump(basket_index, f, indent=2)
        print(f"\n  Index saved: {index_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  Basket complete:")
    for cls_name, entries in basket_index["classes"].items():
        total_pts = sum(e["n_target_pts"] for e in entries)
        print(f"    {cls_name:<12} {len(entries):>3} patches  "
              f"~{total_pts:,} labeled points")
    print(f"\n  NEXT:")
    print(f"    # Προσθήκη Water στο μοντέλο:")
    print(f"    python src/few_shot_add_class.py --mode fractal --class-name Water --n-shots 10")
    print(f"    # Ή LwF training:")
    print(f"    python src/lwf_train.py --basket-dir {basket_dir}/water --class-name Water")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
