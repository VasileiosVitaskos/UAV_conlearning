"""
compute_normalizer_stats.py
Υπολογίζει normalizer stats για το FRACTAL dataset από το TRAIN set μόνο.

Αλλαγή vs. αρχικά stats (Run 1-5):
  ΠΡΙΝ: intensity → z-score(raw)          mean=7340.9, std=14285.3
  ΤΩΡΑ: intensity → z-score(log1p(raw))   mean=log_mean, std=log_std

Γιατί log1p:
  - Το raw intensity είναι heavily right-skewed (std/mean = 1.95 στο train set)
  - Το z-score σε right-skewed distribution συμπιέζει τα χαμηλά values
  - Water έχει raw intensity ~568, Ground ~2277 (4x διαφορά στο raw)
  - Μετά z-score: διαφορά = 0.12 normalized units (αόρατη στο μοντέλο)
  - Μετά log1p + z-score: διαφορά = 1.39 log units (11.6x μεγαλύτερη)
  - log1p είναι standard practice για intensity σε remote sensing LiDAR

Methodology note:
  Η απόφαση για log transform βασίζεται ΜΟΝΟ σε:
  (α) train set statistics (skewness)
  (β) γνωστή φυσική (specular reflection → low intensity for Water)
  (γ) βιβλιογραφία remote sensing
  ΔΕΝ χρησιμοποιείται καμία πληροφορία από val/test set.

Χρήση:
    python src/compute_normalizer_stats.py
    python src/compute_normalizer_stats.py --max-patches 500  # γρήγορο test

Output:
    outputs/normalizer_stats.json  (αντικαθιστά το παλιό αρχείο)

ΠΡΟΣΟΧΗ:
    Μετά από αυτό το script πρέπει να σβηστούν όλα τα .npz cache files:
        find data/ -name "*.npz" -delete
    και να ξαναχτιστεί το cache πριν το Run 6 training.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

import laspy

ROOT = Path(__file__).parent.parent

SEP  = "=" * 64
SEP2 = "-" * 64


def parse_args():
    p = argparse.ArgumentParser(description="Recompute normalizer stats with log1p intensity")
    p.add_argument("--max-patches", type=int, default=0,
                   help="Limit to N patches for speed (0 = all, default)")
    p.add_argument("--no-backup", action="store_true",
                   help="Skip saving a backup of the old stats file")
    return p.parse_args()


def collect_train_stats(max_patches: int = 0) -> dict:
    """
    Περνά από όλα τα .laz αρχεία του train set και συλλέγει running statistics.
    Χρησιμοποιεί online (incremental) Welford algorithm για mean/std
    ώστε να μην χρειάζεται να φορτώσει όλα τα data στη μνήμη ταυτόχρονα.
    """
    train_dir = ROOT / "data" / "train"
    laz_files = sorted(train_dir.rglob("*.laz"))
    if len(laz_files) == 0:
        raise RuntimeError(f"No .laz files found in {train_dir}")

    if max_patches > 0:
        laz_files = laz_files[:max_patches]

    print(f"  Scanning {len(laz_files)} train patches...")

    VALID_CLASSES = {2, 3, 4, 5, 6, 9, 17, 64}

    # Welford online algorithm για κάθε feature
    # features: x_rel, y_rel, z_rel, log1p(intensity)
    counts  = np.zeros(4, dtype=np.float64)
    means   = np.zeros(4, dtype=np.float64)
    M2s     = np.zeros(4, dtype=np.float64)

    for i, laz_path in enumerate(laz_files):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{len(laz_files)}] {laz_path.name}...", end="\r")

        try:
            las    = laspy.read(str(laz_path))
            labels = np.array(las.classification, dtype=np.int32)
            mask   = np.isin(labels, list(VALID_CLASSES))

            x = np.array(las.x)[mask]
            y = np.array(las.y)[mask]
            z = np.array(las.z)[mask]
            intensity = np.array(las.intensity)[mask].astype(np.float64)

            # Center coordinates (relative to patch center)
            x_rel = x - x.mean()
            y_rel = y - y.mean()
            z_rel = z - z.mean()

            # log1p transform for intensity
            log_int = np.log1p(intensity)

            for fi, vals in enumerate([x_rel, y_rel, z_rel, log_int]):
                for v in vals:
                    counts[fi] += 1
                    delta        = v - means[fi]
                    means[fi]   += delta / counts[fi]
                    delta2       = v - means[fi]
                    M2s[fi]     += delta * delta2

        except Exception as e:
            print(f"\n  WARNING: Could not read {laz_path.name}: {e}")
            continue

    print(f"\n  Done. Total points: {counts[0]:,.0f}")

    # Welford variance → std
    stds = np.sqrt(M2s / (counts - 1))

    return {
        "x_rel": {
            "mean": float(means[0]),
            "std":  float(stds[0]),
        },
        "y_rel": {
            "mean": float(means[1]),
            "std":  float(stds[1]),
        },
        "z_rel": {
            "mean": float(means[2]),
            "std":  float(stds[2]),
        },
        "intensity": {
            "transform": "log1p",
            "mean":      float(means[3]),
            "std":       float(stds[3]),
            "note":      "z-score applied to log1p(raw_intensity). Run 6+.",
        },
        "_meta": {
            "n_points":     int(counts[0]),
            "n_patches":    len(laz_files),
            "method":       "Welford online algorithm, train set only",
            "intensity_transform": "log1p",
            "version":      "v2_log1p",
        },
    }


def main():
    args = parse_args()

    print(f"\n{SEP}")
    print(f"  Compute Normalizer Stats — log1p intensity (Run 6)")
    print(f"  Train-set only, no val/test leakage")
    print(SEP + "\n")

    out_path = ROOT / "outputs" / "normalizer_stats.json"

    # Backup of old stats
    if not args.no_backup and out_path.exists():
        backup = out_path.with_name("normalizer_stats_v1_zscoreraw.json")
        backup.write_text(out_path.read_text())
        print(f"  Backup saved: {backup.name}")

    # Compute new stats
    print(f"  Computing log1p intensity stats from train set...")
    if args.max_patches > 0:
        print(f"  (limited to {args.max_patches} patches for speed)\n")

    stats = collect_train_stats(max_patches=args.max_patches)

    # Show results
    print(f"\n{SEP2}")
    print(f"  New normalizer stats:")
    print(SEP2)
    feat_names = ["x_rel", "y_rel", "z_rel", "intensity (log1p)"]
    keys       = ["x_rel", "y_rel", "z_rel", "intensity"]
    for name, key in zip(feat_names, keys):
        m = stats[key]["mean"]
        s = stats[key]["std"]
        print(f"  {name:<22}  mean={m:>10.4f}  std={s:>10.4f}")

    print(f"\n  Patches scanned: {stats['_meta']['n_patches']}")
    print(f"  Points total:    {stats['_meta']['n_points']:,}")

    # Compare: intensity separation before/after
    print(f"\n{SEP2}")
    print(f"  Intensity separation: Water vs Ground (estimated)")
    water_log  = np.log1p(568.0)    # approx raw Water intensity
    ground_log = np.log1p(2277.0)   # approx raw Ground intensity
    m, s       = stats["intensity"]["mean"], stats["intensity"]["std"]
    water_norm  = (water_log  - m) / s
    ground_norm = (ground_log - m) / s
    print(f"  OLD (z-score raw):    Water≈-0.474  Ground≈-0.355  diff=0.119")
    print(f"  NEW (z-score log1p):  Water≈{water_norm:.3f}  Ground≈{ground_norm:.3f}  diff={abs(water_norm-ground_norm):.3f}")
    print(f"  Improvement: ~{abs(water_norm-ground_norm)/0.119:.1f}x larger gap")
    print(SEP2)

    # Save
    out_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\n  Saved: {out_path}")

    print(f"\n{SEP}")
    print(f"  NEXT STEPS:")
    print(f"  1. Delete old .npz cache files:")
    print(f"       find data/ -name '*.npz' -delete")
    print(f"  2. Verify preprocessing.py uses log1p for intensity")
    print(f"  3. Run training (Run 6):")
    print(f"       python src/train.py --run-name run6_log1p_intensity \\")
    print(f"           --exclude-classes Water Bridge --epochs 100")
    print(f"  4. Evaluate OOD: python src/ood_mahalanobis.py")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
