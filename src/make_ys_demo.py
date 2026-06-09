"""
make_ys_demo.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Επιλέγει N τυχαία YellowScan tiles από το data/yellowscan_patches/
και τα αντιγράφει στο data/basket/yellowscan_demo/ για demo στο Pi5.

Χρήση:
    python src/make_ys_demo.py              # 10 τυχαία tiles (default)
    python src/make_ys_demo.py --n 5        # 5 tiles
    python src/make_ys_demo.py --seed 0     # διαφορετικά tiles

Προϋπόθεση:
    python src/split_yellowscan.py          # πρώτα split το raw .laz
"""

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n",    type=int, default=10, help="Αριθμός tiles (default: 10)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--src",  type=str,
        default=str(ROOT / "data" / "yellowscan_patches"),
        help="Φάκελος με YellowScan tiles (default: data/yellowscan_patches)",
    )
    p.add_argument(
        "--dst",  type=str,
        default=str(ROOT / "data" / "basket" / "yellowscan_demo"),
        help="Output φάκελος (default: data/basket/yellowscan_demo)",
    )
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    # ── Έλεγχος ──────────────────────────────────────────────────────────────
    if not src.exists():
        print(f"\n✗ Φάκελος tiles δεν βρέθηκε: {src}")
        print("  Τρέξε πρώτα: python src/split_yellowscan.py")
        return

    tiles = sorted(src.glob("ys_tile_*.laz"))
    if not tiles:
        print(f"\n✗ Κανένα tile βρέθηκε στο {src}")
        print("  Τρέξε πρώτα: python src/split_yellowscan.py")
        return

    if len(tiles) < args.n:
        print(f"  ⚠  Μόνο {len(tiles)} tiles διαθέσιμα — επιλέγω όλα.")
        selected = tiles
    else:
        random.seed(args.seed)
        selected = random.sample(tiles, args.n)
        selected.sort()

    # ── Αντιγραφή ─────────────────────────────────────────────────────────────
    dst.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*55}")
    print(f"  YellowScan Demo Subset")
    print(f"  Tiles διαθέσιμα : {len(tiles)}")
    print(f"  Επιλέχθηκαν     : {len(selected)}")
    print(f"  Destination     : {dst}")
    print(f"{'═'*55}")

    total_size = 0
    for tile in selected:
        dst_file = dst / tile.name
        shutil.copy2(tile, dst_file)
        size_kb = dst_file.stat().st_size / 1024
        total_size += size_kb
        print(f"  ✓  {tile.name}  ({size_kb:.0f} KB)")

    print(f"{'─'*55}")
    print(f"  Σύνολο: {len(selected)} tiles  ({total_size/1024:.1f} MB)")
    print(f"\n  NEXT — στείλε στο Pi5:")
    print(f"    scp -r {dst}  pi@<ip>:~/Drone_cont_Learing/data/basket/")
    print(f"\n  Inference στο Pi5:")
    print(f"    for f in data/basket/yellowscan_demo/*.laz; do")
    print(f"        python src/inference.py \"$f\" --sensor yellowscan --verbose")
    print(f"    done")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()
