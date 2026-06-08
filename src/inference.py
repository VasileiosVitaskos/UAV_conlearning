"""
inference.py
Pi5 inference script — terrain classification + OOD detection από .laz patch.

Χρήση:
    python src/inference.py data/test/00/patch_001.laz
    python src/inference.py data/test/00/patch_001.laz --model outputs/model.onnx
    python src/inference.py data/test/00/patch_001.laz --verbose

Output (stdout JSON):
    {
        "patch":       "patch_001.laz",
        "n_points":    4096,
        "classes":     {"Ground": 2310, "HighVegetation": 1200, ...},
        "ood_points":  142,
        "ood_pct":     3.47,
        "ood_method":  "hybrid",
        "latency_ms":  234.5
    }

Απαιτήσεις (Pi5):
    pip install onnxruntime laspy numpy
    (δεν χρειάζεται PyTorch!)

OOD Method (hybrid):
    - Energy Score: -logsumexp(logits)  [για Bridge-like geometric anomalies]
    - Intensity threshold: log1p(intensity) < θ_water  [για Water specular reflection]
    - Fusion: flagged αν energy > thr_e OR intensity < thr_i
    - Thresholds βαθμονομημένα στο val set (F1-optimal)
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ── Default paths ──────────────────────────────────────────────────────────────
DEFAULT_MODEL    = ROOT / "outputs" / "model.onnx"
DEFAULT_STATS    = ROOT / "outputs" / "normalizer_stats.json"
DEFAULT_OOD_CFG  = ROOT / "outputs" / "ood_hybrid_results.json"


def parse_args():
    p = argparse.ArgumentParser(
        description="Terrain classification + OOD detection on .laz patch (Pi5)"
    )
    p.add_argument("patch", type=str,
                   help="Path to input .laz or .las file")
    p.add_argument("--model", type=str,
                   default=str(DEFAULT_MODEL),
                   help="Path to ONNX model (default: outputs/model.onnx)")
    p.add_argument("--stats", type=str,
                   default=str(DEFAULT_STATS),
                   help="Path to normalizer_stats.json")
    p.add_argument("--ood-cfg", type=str,
                   default=str(DEFAULT_OOD_CFG),
                   dest="ood_cfg",
                   help="Path to ood_hybrid_results.json (thresholds)")
    p.add_argument("--num-points", type=int, default=4096,
                   dest="num_points",
                   help="Points to sample per patch (default: 4096)")
    p.add_argument("--sensor", type=str, default="fractal",
                   choices=["fractal", "yellowscan"],
                   help="Sensor type: 'fractal' (ALS, LAS 1.4 scan angle units) "
                        "or 'yellowscan' (UAV LiDAR, scan angle already in degrees). "
                        "Affects scan angle normalization. Default: fractal")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-class breakdown to stderr")
    p.add_argument("--no-ood", action="store_true",
                   dest="no_ood",
                   help="Skip OOD detection (faster)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing (mirrors preprocessing.py — no dependency on PyTorch)
# ══════════════════════════════════════════════════════════════════════════════

def normalize_patch(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    intensity:  np.ndarray,
    return_num: np.ndarray,
    n_returns:  np.ndarray,
    scan_angle: np.ndarray,
    stats: dict,
    is_fractal: bool = True,
) -> np.ndarray:
    """
    Αντίγραφο της normalize() χωρίς PyTorch/εξωτερικές εξαρτήσεις.
    Χρησιμοποιείται στο Pi5 όπου δεν έχουμε src/preprocessing.py available.
    """
    N   = len(x)
    out = np.zeros((N, 7), dtype=np.float32)

    out[:, 0] = (x - stats["x_rel"]["mean"])     / stats["x_rel"]["std"]
    out[:, 1] = (y - stats["y_rel"]["mean"])     / stats["y_rel"]["std"]
    out[:, 2] = (z - stats["z_rel"]["mean"])     / stats["z_rel"]["std"]
    # Run 6+: log1p intensity (right-skewed → log normalizes, 8.5× Water/Ground gap)
    out[:, 3] = (np.log1p(intensity.astype(np.float32)) - stats["intensity"]["mean"]) \
                / stats["intensity"]["std"]
    out[:, 4] = return_num / 6.0
    out[:, 5] = n_returns  / 6.0
    if is_fractal:
        out[:, 6] = scan_angle * 0.006 / 60.0
    else:
        out[:, 6] = scan_angle / 60.0
    return out


def load_patch(
    laz_path: Path,
    num_points: int,
    stats: dict,
    is_fractal: bool = True,
) -> tuple:
    """
    Φορτώνει .laz, κάνει random sample num_points, κανονικοποιεί.

    Returns:
        X         : (1, N, 7) float32 — ONNX input
        raw_intensity : (N,) float32 — για OOD intensity threshold
        indices   : (N,) int — sample indices (για αντιστοίχιση με αρχικά points)
        n_total   : int — αρχικός αριθμός points
    """
    import laspy
    las = laspy.read(str(laz_path))

    n_total = len(las.x)

    if n_total >= num_points:
        idx = np.random.choice(n_total, size=num_points, replace=False)
    else:
        idx = np.random.choice(n_total, size=num_points, replace=True)

    x = np.array(las.x)[idx]
    y = np.array(las.y)[idx]
    z = np.array(las.z)[idx]
    intensity  = np.array(las.intensity)[idx].astype(np.float32)
    return_num = np.array(las.return_number)[idx].astype(np.float32)
    n_ret      = np.array(las.number_of_returns)[idx].astype(np.float32)
    scan_ang   = np.array(las.scan_angle)[idx].astype(np.float32)

    # Center coordinates
    X_feat = normalize_patch(
        x - x.mean(), y - y.mean(), z - z.mean(),
        intensity, return_num, n_ret, scan_ang,
        stats=stats, is_fractal=is_fractal,
    )

    # ONNX expects (B, N, 7) — add batch dim
    X_batch = X_feat[np.newaxis, :, :]   # (1, N, 7)

    return X_batch, intensity, idx, n_total


# ══════════════════════════════════════════════════════════════════════════════
# OOD Detection (hybrid: energy + intensity)
# ══════════════════════════════════════════════════════════════════════════════

def detect_ood_hybrid(
    logits:        np.ndarray,   # (N, C) float32
    raw_intensity: np.ndarray,   # (N,)   float32
    ood_cfg:       dict,
) -> np.ndarray:
    """
    Hybrid OOD detection:
    1. Energy score: E = -log(sum(exp(logits)))  — detects structural anomalies (Bridge)
    2. Intensity threshold: log1p(I) < θ_i  — detects Water (specular reflection)

    Returns boolean mask (N,) — True = OOD point
    """
    # Energy score (higher = more OOD for geometric anomalies)
    max_logit  = logits.max(axis=-1, keepdims=True)
    energy     = -(max_logit[:, 0] + np.log(
        np.exp(logits - max_logit).sum(axis=-1)
    ))   # numerically stable logsumexp

    # Get thresholds from val-calibrated config
    # ood_cfg["val_stats"] has percentile-based thresholds
    # Fallback: use conservative defaults if config missing keys
    e_thr = ood_cfg.get("best_energy_threshold", -3.0)
    i_thr = ood_cfg.get("best_intensity_threshold", None)

    # Energy-based flag (Bridge, geometric anomalies)
    energy_flag = energy > e_thr

    # Intensity-based flag (Water: specular reflection → very low raw intensity)
    # Threshold: if log1p(intensity) < log1p(1500) ≈ 7.3 → Water candidate
    if i_thr is not None:
        intensity_flag = np.log1p(raw_intensity) < i_thr
    else:
        # Fallback: percentile-based — bottom 5% of intensity
        intensity_flag = raw_intensity < np.percentile(raw_intensity, 5)

    return energy_flag | intensity_flag


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    t_total = time.perf_counter()

    patch_path = Path(args.patch)
    if not patch_path.exists():
        print(json.dumps({"error": f"File not found: {patch_path}"}))
        sys.exit(1)

    # ── Load configs ───────────────────────────────────────────────────────────
    with open(args.stats) as f:
        stats = json.load(f)

    ood_cfg = {}
    if not args.no_ood:
        ood_cfg_path = Path(args.ood_cfg)
        if ood_cfg_path.exists():
            with open(ood_cfg_path) as f:
                ood_cfg = json.load(f)
        else:
            if args.verbose:
                print(f"[WARN] OOD config not found: {ood_cfg_path}, using defaults",
                      file=sys.stderr)

    # ── Load ONNX model ────────────────────────────────────────────────────────
    try:
        import onnxruntime as ort
    except ImportError:
        print(json.dumps({"error": "onnxruntime not installed. Run: pip install onnxruntime"}))
        sys.exit(1)

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4       # Pi5: 4 ARM Cortex-A76 cores
    sess_opts.inter_op_num_threads = 1
    sess_opts.log_severity_level   = 3       # suppress warnings

    model_path = Path(args.model)
    if not model_path.exists():
        print(json.dumps({"error": f"ONNX model not found: {model_path}. Run: python src/export_onnx.py"}))
        sys.exit(1)

    sess     = ort.InferenceSession(str(model_path), sess_opts,
                                    providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    # Class names from ONNX metadata (saved in model.json)
    meta_path = model_path.with_suffix(".json")
    class_names = [f"class_{i}" for i in range(6)]  # fallback
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        class_names = meta.get("active_classes", class_names)

    # ── Load & preprocess patch ────────────────────────────────────────────────
    is_fractal = (args.sensor == "fractal")
    if args.verbose and not is_fractal:
        print(f"[INFO] YellowScan mode: scan angle in degrees (no ×0.006 conversion)",
              file=sys.stderr)
        print(f"[WARN] Intensity stats from FRACTAL — Water OOD threshold may not transfer.",
              file=sys.stderr)

    t_prep = time.perf_counter()
    try:
        X_batch, raw_intensity, _, n_total = load_patch(
            patch_path, args.num_points, stats, is_fractal=is_fractal
        )
    except Exception as e:
        print(json.dumps({"error": f"Failed to load patch: {e}"}))
        sys.exit(1)

    # Propagate sensor type to result metadata
    result_sensor = args.sensor
    prep_ms = (time.perf_counter() - t_prep) * 1000

    # ── Inference ─────────────────────────────────────────────────────────────
    t_inf = time.perf_counter()
    logits_batch = sess.run([out_name], {inp_name: X_batch})[0]  # (1, N, C)
    inf_ms = (time.perf_counter() - t_inf) * 1000

    logits = logits_batch[0]           # (N, C)
    preds  = logits.argmax(axis=-1)    # (N,)

    # ── Class counts ──────────────────────────────────────────────────────────
    class_counts = {}
    for c, name in enumerate(class_names):
        cnt = int((preds == c).sum())
        if cnt > 0:
            class_counts[name] = cnt

    # ── OOD Detection ─────────────────────────────────────────────────────────
    ood_mask   = np.zeros(args.num_points, dtype=bool)
    ood_method = "none"

    if not args.no_ood:
        ood_mask   = detect_ood_hybrid(logits, raw_intensity, ood_cfg)
        ood_method = "hybrid (energy + intensity)"

    n_ood   = int(ood_mask.sum())
    ood_pct = n_ood / args.num_points * 100

    total_ms = (time.perf_counter() - t_total) * 1000

    # ── Output JSON ───────────────────────────────────────────────────────────
    result = {
        "patch":      patch_path.name,
        "sensor":     args.sensor,
        "n_points":   args.num_points,
        "n_total":    n_total,
        "classes":    class_counts,
        "ood_points": n_ood,
        "ood_pct":    round(ood_pct, 2),
        "ood_method": ood_method,
        "domain_shift_warning": None if is_fractal else
            "YellowScan intensity stats differ from FRACTAL — Water OOD threshold unreliable",
        "latency_ms": {
            "preprocessing": round(prep_ms, 1),
            "inference":     round(inf_ms, 1),
            "total":         round(total_ms, 1),
        },
    }
    print(json.dumps(result, indent=2))

    # ── Verbose breakdown ──────────────────────────────────────────────────────
    if args.verbose:
        print("\n─── Classification breakdown ───", file=sys.stderr)
        for name, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            pct = cnt / args.num_points * 100
            bar = "█" * int(pct / 2)
            print(f"  {name:<20} {cnt:>6,}  ({pct:5.1f}%)  {bar}",
                  file=sys.stderr)
        print(f"\n  OOD: {n_ood} points ({ood_pct:.1f}%)", file=sys.stderr)
        print(f"\n─── Latency ───", file=sys.stderr)
        print(f"  Preprocessing : {prep_ms:.1f} ms", file=sys.stderr)
        print(f"  Inference     : {inf_ms:.1f} ms", file=sys.stderr)
        print(f"  Total         : {total_ms:.1f} ms", file=sys.stderr)

        # Pi5 estimate
        est_pi5 = total_ms * 12   # conservative 12× slowdown
        print(f"\n  Pi5 estimate  : ~{est_pi5:.0f} ms  ({est_pi5/1000:.1f}s)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
