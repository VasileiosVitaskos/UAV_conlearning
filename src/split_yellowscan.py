"""
split_yellowscan.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Τεμαχίζει το μεγάλο YellowScan .laz αρχείο σε 50×50m patches,
ίδιο format με το FRACTAL dataset.

Γιατί χρειάζεται:
  Το YellowScan αρχείο είναι ένα full-flight survey (~1.6GB, >100M points).
  Το μοντέλο δέχεται 50×50m patches (4096 sampled points).
  Κάθε tile αποθηκεύεται σαν ξεχωριστό .laz αρχείο.

Χρήση:
  # Default (50×50m, min 500 points/tile):
  python src/split_yellowscan.py

  # Custom tile size και input:
  python src/split_yellowscan.py \\
      --input "data/real UAV/VA50-SC20-M600-120mAGL-10ms-SACOCU-Pipeline_survey(1).laz" \\
      --output data/yellowscan_patches \\
      --tile-size 50 --min-points 500

  # Μόνο info (χωρίς να γράψει tiles):
  python src/split_yellowscan.py --info-only

Output:
  data/yellowscan_patches/
    ys_tile_x000500_y004200.laz    ← tile στο (500, 4200) origin
    ys_tile_x000500_y004250.laz
    ...
    tiling_summary.json            ← metadata (n_tiles, bbox, stats)
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent


def parse_args():
    p = argparse.ArgumentParser(
        description="Split YellowScan .laz into 50×50m patches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", type=str,
        default=str(ROOT / "data" / "real UAV" /
                    "VA50-SC20-M600-120mAGL-10ms-SACOCU-Pipeline_survey(1).laz"),
        help="Input .laz file (default: data/real UAV/*.laz)",
    )
    p.add_argument(
        "--output", type=str,
        default=str(ROOT / "data" / "yellowscan_patches"),
        help="Output directory for tiles (default: data/yellowscan_patches)",
    )
    p.add_argument(
        "--tile-size", type=float, default=50.0, dest="tile_size",
        help="Tile size in meters (default: 50)",
    )
    p.add_argument(
        "--min-points", type=int, default=500, dest="min_points",
        help="Minimum points per tile to save (default: 500)",
    )
    p.add_argument(
        "--info-only", action="store_true", dest="info_only",
        help="Print cloud info without writing tiles",
    )
    p.add_argument(
        "--max-tiles", type=int, default=None, dest="max_tiles",
        help="Max tiles to write (debug / quick test)",
    )
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Core tiling
# ══════════════════════════════════════════════════════════════════════════════

def inspect_cloud(las_path: Path) -> dict:
    """Εκτυπώνει βασικές πληροφορίες για το point cloud χωρίς να φορτώσει όλα τα points."""
    import laspy
    las = laspy.read(str(las_path))

    x = np.array(las.x, dtype=np.float64)
    y = np.array(las.y, dtype=np.float64)
    z = np.array(las.z, dtype=np.float64)

    info = {
        "n_points":     len(x),
        "point_format": int(las.point_format.id),
        "x_min": float(x.min()), "x_max": float(x.max()),
        "y_min": float(y.min()), "y_max": float(y.max()),
        "z_min": float(z.min()), "z_max": float(z.max()),
        "x_range": float(x.max() - x.min()),
        "y_range": float(y.max() - y.min()),
        "dimensions": list(las.point_format.dimension_names),
        "has_intensity":      hasattr(las, "intensity"),
        "has_return_number":  hasattr(las, "return_number"),
        "has_n_returns":      hasattr(las, "number_of_returns"),
        "has_scan_angle":     hasattr(las, "scan_angle"),
        "has_classification": hasattr(las, "classification"),
    }

    # Se il file ha classificazioni
    if info["has_classification"]:
        cls_vals, cls_counts = np.unique(las.classification, return_counts=True)
        info["classification_counts"] = {
            int(c): int(n) for c, n in zip(cls_vals, cls_counts)
        }

    return info, las


def tile_cloud(
    las,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    tile_size: float,
    min_points: int,
    out_dir: Path,
    max_tiles: int = None,
) -> dict:
    """
    Τεμαχίζει το cloud σε τετράγωνα tiles.

    Returns:
        summary dict με στατιστικά (n_tiles, skipped, etc.)
    """
    import laspy

    out_dir.mkdir(parents=True, exist_ok=True)

    x_min = x.min()
    y_min = y.min()

    # Grid cell indices για κάθε point
    ix = np.floor((x - x_min) / tile_size).astype(np.int32)
    iy = np.floor((y - y_min) / tile_size).astype(np.int32)

    # Unique tiles
    tile_keys = np.unique(np.stack([ix, iy], axis=1), axis=0)
    n_total_tiles = len(tile_keys)

    print(f"\n  Grid: {ix.max()+1} × {iy.max()+1} cells = {n_total_tiles} potential tiles")
    print(f"  Min points/tile threshold: {min_points}")
    print(f"  Writing to: {out_dir}\n")

    saved       = 0
    skipped     = 0
    total_saved_pts = 0
    tile_stats  = []

    # Pre-compute για ταχύτητα: sort by tile key
    combined_key = ix.astype(np.int64) * 100000 + iy.astype(np.int64)
    sort_idx     = np.argsort(combined_key)
    combined_sorted = combined_key[sort_idx]

    t0 = time.perf_counter()

    for tile_ix, tile_iy in tile_keys:
        # Fast boolean mask
        mask = (ix == tile_ix) & (iy == tile_iy)
        n_pts = int(mask.sum())

        if n_pts < min_points:
            skipped += 1
            continue

        # Tile origin in absolute coords
        tx_origin = x_min + tile_ix * tile_size
        ty_origin = y_min + tile_iy * tile_size

        # Tile filename (use integer meters of origin)
        fname = f"ys_tile_x{int(tx_origin):+08d}_y{int(ty_origin):+08d}.laz"
        out_path = out_dir / fname

        # Build new LAS file for this tile
        new_las = laspy.LasData(header=laspy.LasHeader(
            point_format=las.point_format.id,
            version=las.header.version,
        ))

        # Copy all scalar fields for masked points
        for dim in las.point_format.dimension_names:
            try:
                arr = np.array(getattr(las, dim))[mask]
                setattr(new_las, dim, arr)
            except Exception:
                pass  # ignora dimensioni non copiabili

        new_las.write(str(out_path))

        z_tile = z[mask]
        tile_stats.append({
            "file":     fname,
            "n_points": n_pts,
            "x_origin": round(float(tx_origin), 2),
            "y_origin": round(float(ty_origin), 2),
            "z_mean":   round(float(z_tile.mean()), 2),
            "z_range":  round(float(z_tile.max() - z_tile.min()), 2),
        })
        saved           += 1
        total_saved_pts += n_pts

        # Progress
        if saved % 20 == 0 or saved <= 5:
            elapsed = time.perf_counter() - t0
            rate    = saved / elapsed if elapsed > 0 else 0
            print(f"  [{saved:4d}] {fname}  ({n_pts:,} pts)  "
                  f"  rate: {rate:.1f} tiles/s", flush=True)

        if max_tiles and saved >= max_tiles:
            print(f"\n  --max-tiles {max_tiles} reached, stopping early.")
            break

    elapsed = time.perf_counter() - t0
    summary = {
        "input":            str(las.header),
        "tile_size_m":      tile_size,
        "min_points":       min_points,
        "n_potential_tiles": n_total_tiles,
        "n_saved_tiles":    saved,
        "n_skipped_tiles":  skipped,
        "total_saved_pts":  total_saved_pts,
        "elapsed_s":        round(elapsed, 1),
        "tiles":            tile_stats,
    }
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    try:
        import laspy
    except ImportError:
        print("ERROR: laspy not installed.\n  pip install laspy lazrs-python")
        sys.exit(1)

    in_path  = Path(args.input)
    out_dir  = Path(args.output)

    if not in_path.exists():
        print(f"ERROR: Input file not found: {in_path}")
        sys.exit(1)

    file_mb = in_path.stat().st_size / 1024 / 1024
    print(f"\n{'═'*65}")
    print(f"  YellowScan Tiling")
    print(f"  Input  : {in_path.name}  ({file_mb:.0f} MB)")
    print(f"  Tile   : {args.tile_size}×{args.tile_size} m")
    print(f"  Output : {out_dir}")
    print(f"{'═'*65}")

    # ── Load full cloud ────────────────────────────────────────────────────────
    print("\n  Φόρτωση point cloud (μπορεί να πάρει 30-60 sec για 1.6GB)...")
    t0 = time.perf_counter()
    info, las = inspect_cloud(in_path)
    load_s = time.perf_counter() - t0

    print(f"\n  Point cloud info:")
    print(f"    Points      : {info['n_points']:,}")
    print(f"    Point format: LAS {info['point_format']}")
    print(f"    X range     : {info['x_min']:.2f} → {info['x_max']:.2f}  "
          f"({info['x_range']:.1f} m)")
    print(f"    Y range     : {info['y_min']:.2f} → {info['y_max']:.2f}  "
          f"({info['y_range']:.1f} m)")
    print(f"    Z range     : {info['z_min']:.2f} → {info['z_max']:.2f} m")
    print(f"    Intensity   : {'✓' if info['has_intensity'] else '✗'}")
    print(f"    Return num  : {'✓' if info['has_return_number'] else '✗'}")
    print(f"    Scan angle  : {'✓' if info['has_scan_angle'] else '✗'}")
    print(f"    Labels      : {'✓' if info['has_classification'] else '✗ (unlabeled — expected)' }")
    if info.get("classification_counts"):
        print(f"    Classes     : {info['classification_counts']}")
    print(f"    Load time   : {load_s:.1f}s")

    est_tiles = int((info["x_range"] / args.tile_size + 1) *
                    (info["y_range"] / args.tile_size + 1))
    print(f"\n  Estimated tiles (grid) : ~{est_tiles}")

    if args.info_only:
        # Save info JSON
        info_path = out_dir.parent / "yellowscan_info.json"
        info_path.parent.mkdir(parents=True, exist_ok=True)
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"\n  Info saved: {info_path}")
        print(f"  (Χρησιμοποίησε χωρίς --info-only για να γράψεις τα tiles)")
        return

    # ── Tile ──────────────────────────────────────────────────────────────────
    x = np.array(las.x, dtype=np.float64)
    y = np.array(las.y, dtype=np.float64)
    z = np.array(las.z, dtype=np.float64)

    summary = tile_cloud(
        las, x, y, z,
        tile_size=args.tile_size,
        min_points=args.min_points,
        out_dir=out_dir,
        max_tiles=args.max_tiles,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_path = out_dir / "tiling_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'═'*65}")
    print(f"  Tiling complete")
    print(f"  Tiles saved  : {summary['n_saved_tiles']}")
    print(f"  Tiles skipped: {summary['n_skipped_tiles']}  "
          f"(< {args.min_points} pts)")
    print(f"  Total points : {summary['total_saved_pts']:,}")
    print(f"  Time         : {summary['elapsed_s']:.1f}s")
    print(f"  Summary JSON : {summary_path}")
    print(f"\n  NEXT:")
    print(f"    # Τρέξε inference σε ένα tile:")
    print(f"    python src/inference.py {out_dir}/ys_tile_*.laz --sensor yellowscan --verbose")
    print(f"    # Batch inference + OOD basket:")
    print(f"    for f in {out_dir}/*.laz; do")
    print(f"        python src/inference.py \"$f\" --sensor yellowscan --save-basket outputs/ood_basket")
    print(f"    done")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
