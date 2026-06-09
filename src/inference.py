"""
inference.py
Pi5 inference script — terrain classification + OOD detection από .laz patch.

Χρήση:
    python src/inference.py data/test/00/patch_001.laz
    python src/inference.py data/test/00/patch_001.laz --model outputs/model.onnx
    python src/inference.py data/test/00/patch_001.laz --verbose
    python src/inference.py yellowscan_patch.laz --sensor yellowscan --save-basket outputs/ood_basket

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
    pip install onnxruntime laspy numpy         # για normal inference
    pip install torch                           # μόνο αν χρησιμοποιείς --save-basket

OOD Method (hybrid — calibrated στο val set):
    - Intensity score: (intensity_hi - z_intensity) / (intensity_hi - intensity_lo)
      -> high score = low intensity = Water (specular reflection, near-IR absorbs)
    - Energy score: (z_energy - energy_lo) / (energy_hi - energy_lo)
      -> high score = high energy = Bridge (geometric novelty)
    - Hybrid: weight_energy*E_score + weight_intensity*I_score > best_threshold
    - Params βαθμονομημένα στο val set (ood_hybrid_results.json)
      weight_energy=0.0, weight_intensity=1.0  ->  intensity-only  (AUROC=0.8232)

--save-basket:
    Όταν OOD ανιχνευτεί, αποθηκεύει τα 32-dim embeddings των OOD points σε .npy.
    Αυτά χρησιμοποιούνται από few_shot_add_class.py --mode basket για CL update.
    ΑΠΑΙΤΕΙ PyTorch + best_model.pt (εκτός από ONNX).
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


# -- Default paths --------------------------------------------------------------
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
    p.add_argument("--save-basket", type=str, default=None,
                   dest="save_basket",
                   metavar="DIR",
                   help="Αν OOD ανιχνευτεί, αποθήκευσε embeddings (.npy) στον DIR "
                        "για few-shot CL. Απαιτεί --pt-model ή αυτόματη εύρεση best_model.pt. "
                        "Παράδειγμα: --save-basket outputs/ood_basket")
    p.add_argument("--pt-model", type=str, default=None,
                   dest="pt_model",
                   help="PyTorch checkpoint για embedding extraction (--save-basket). "
                        "Default: outputs/checkpoints/best_model.pt")
    return p.parse_args()


# ==============================================================================
# Preprocessing (mirrors preprocessing.py -- no dependency on PyTorch)
# ==============================================================================

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
    # Run 6+: log1p intensity (right-skewed -> log normalizes, 8.5x Water/Ground gap)
    out[:, 3] = (np.log1p(intensity.astype(np.float32)) - stats["intensity"]["mean"]) \
                / stats["intensity"]["std"]
    out[:, 4] = return_num / 6.0
    out[:, 5] = n_returns  / 6.0
    if is_fractal:
        out[:, 6] = scan_angle * 0.006 / 60.0
    else:
        # YellowScan: scan_angle_rank -- already in degrees, no 0.006 conversion
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
        X         : (1, N, 7) float32 -- ONNX input
        raw_intensity : (N,) float32 -- για OOD intensity threshold
        indices   : (N,) int -- sample indices (για αντιστοίχιση με αρχικά points)
        n_total   : int -- αρχικός αριθμός points
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

    # YellowScan uses scan_angle_rank (degrees); FRACTAL uses scan_angle (LAS 1.4 units)
    if is_fractal:
        scan_ang = np.array(las.scan_angle)[idx].astype(np.float32)
    else:
        try:
            scan_ang = np.array(las.scan_angle_rank)[idx].astype(np.float32)
        except AttributeError:
            # Fallback: some YellowScan files use scan_angle
            scan_ang = np.array(las.scan_angle)[idx].astype(np.float32)

    # Center coordinates
    X_feat = normalize_patch(
        x - x.mean(), y - y.mean(), z - z.mean(),
        intensity, return_num, n_ret, scan_ang,
        stats=stats, is_fractal=is_fractal,
    )

    # ONNX expects (B, N, 7) -- add batch dim
    X_batch = X_feat[np.newaxis, :, :]   # (1, N, 7)

    return X_batch, intensity, idx, n_total


# ==============================================================================
# OOD Detection (hybrid: energy + intensity)
# ==============================================================================

def detect_ood_hybrid(
    logits:        np.ndarray,   # (N, C) float32
    raw_intensity: np.ndarray,   # (N,)   float32
    stats:         dict,          # normalizer_stats.json
    ood_cfg:       dict,
) -> tuple:
    """
    Hybrid OOD detection -- χρησιμοποιεί calibrated thresholds από ood_hybrid_results.json.

    Αλγόριθμος (βαθμονομημένος στο val set):
      1. z_intensity = (log1p(I) - mean_I) / std_I   [από normalizer_stats.json]
      2. I_score = (intensity_hi - z_intensity) / (intensity_hi - intensity_lo)
         -> low intensity -> high score (Water = specular reflection)
      3. E_score = (z_energy - energy_lo) / (energy_hi - energy_lo)
         -> high energy -> high score (Bridge = geometric novelty)
      4. hybrid = weight_energy * E_score + weight_intensity * I_score
      5. OOD if hybrid > best_threshold

    Returns:
      ood_mask   : (N,) bool   -- True = OOD point
      hybrid_scores : (N,) float32 -- continuous OOD score
    """
    N = len(raw_intensity)

    # -- Normalizer stats για intensity ----------------------------------------
    i_mean = stats.get("intensity", {}).get("mean", 6.5)
    i_std  = stats.get("intensity", {}).get("std",  1.2)
    z_intensity = (np.log1p(raw_intensity.astype(np.float32)) - i_mean) / i_std

    # -- Energy score ----------------------------------------------------------
    max_logit = logits.max(axis=-1, keepdims=True)
    energy    = -(max_logit[:, 0] + np.log(
        np.exp(logits - max_logit).sum(axis=-1)
    ))  # (N,) -- numerically stable

    # -- Normalization params από ood_hybrid_results.json ----------------------
    norm       = ood_cfg.get("normalization", {})
    energy_lo  = norm.get("energy_lo",    -6.18)
    energy_hi  = norm.get("energy_hi",    -1.76)
    intensity_lo = norm.get("intensity_lo", -3.05)
    intensity_hi = norm.get("intensity_hi",  0.50)

    # -- Scores (clipped σε [0, 1]) --------------------------------------------
    i_range = intensity_hi - intensity_lo  # > 0
    e_range = energy_hi - energy_lo        # > 0

    # Intensity score: lower z -> higher OOD score
    i_score = np.clip((intensity_hi - z_intensity) / (i_range + 1e-8), 0.0, 1.0)

    # Energy score: higher energy -> higher OOD score
    e_score = np.clip((energy - energy_lo) / (e_range + 1e-8), 0.0, 1.0)

    # -- Hybrid ----------------------------------------------------------------
    w_e = float(ood_cfg.get("weight_energy",    0.0))
    w_i = float(ood_cfg.get("weight_intensity", 1.0))
    hybrid = w_e * e_score + w_i * i_score

    # -- Threshold -------------------------------------------------------------
    # best_threshold από ood_hybrid_results.json["val"]["best_threshold"]
    threshold = ood_cfg.get("val", {}).get("best_threshold", 0.9967)

    ood_mask = hybrid > threshold

    return ood_mask.astype(bool), hybrid.astype(np.float32)


# ==============================================================================
# Embedding extraction via PyTorch (για --save-basket)
# ==============================================================================

def load_pytorch_model(pt_path: Path):
    """Φορτώνει το PyTorch μοντέλο για embedding extraction."""
    try:
        import torch
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "src"))
        from models.pointnet2 import PointNet2Mini

        ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        num_classes = ckpt.get("num_classes", 6)
        model = PointNet2Mini(in_channels=7, num_classes=num_classes)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model
    except ImportError:
        return None
    except Exception as e:
        print(f"[WARN] Failed to load PyTorch model: {e}", file=sys.stderr)
        return None


def extract_embeddings_pytorch(
    X_batch: np.ndarray,    # (1, N, 7) float32
    ood_mask: np.ndarray,   # (N,) bool
    pt_model,               # PointNet2Mini instance
) -> np.ndarray:
    """
    Εξάγει 32-dim embeddings για τα OOD points χρησιμοποιώντας PyTorch.
    Χρησιμοποιείται για τη δημιουργία basket προτύπων.

    Returns:
        (M, 32) float32 -- embeddings των OOD points
    """
    import torch
    X_t = torch.from_numpy(X_batch)       # (1, N, 7)
    with torch.no_grad():
        emb = pt_model.get_embeddings(X_t)  # (1, N, 32)
    emb_np = emb[0].cpu().numpy()          # (N, 32)
    return emb_np[ood_mask]                 # (M, 32)


def save_basket_embeddings(
    embeddings: np.ndarray,   # (M, 32)
    basket_dir: Path,
    patch_name: str,
) -> Path:
    """Αποθηκεύει OOD embeddings σε .npy αρχείο για το basket."""
    import time as _time
    basket_dir.mkdir(parents=True, exist_ok=True)
    ts    = int(_time.time() * 1000) % 1_000_000   # ms timestamp (compact)
    fname = f"ood_{patch_name}_{ts}.npy"
    fpath = basket_dir / fname
    np.save(str(fpath), embeddings.astype(np.float32))
    return fpath


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()
    t_total = time.perf_counter()

    patch_path = Path(args.patch)
    if not patch_path.exists():
        print(json.dumps({"error": f"File not found: {patch_path}"}))
        sys.exit(1)

    # -- --save-basket: φόρτωσε PyTorch μοντέλο νωρίς --------------------------
    pt_model = None
    basket_dir = None
    if args.save_basket:
        basket_dir = Path(args.save_basket)
        pt_path    = Path(args.pt_model) if args.pt_model else \
                     ROOT / "outputs" / "checkpoints" / "best_model.pt"
        if not pt_path.exists():
            print(f"[WARN] --save-basket: PyTorch model not found: {pt_path}",
                  file=sys.stderr)
            print(f"[WARN] Embeddings δεν θα αποθηκευτούν.", file=sys.stderr)
        else:
            pt_model = load_pytorch_model(pt_path)
            if pt_model is None:
                print("[WARN] --save-basket: δεν φορτώθηκε PyTorch (pip install torch?)",
                      file=sys.stderr)
            elif args.verbose:
                print(f"[INFO] PyTorch model loaded for basket: {pt_path.name}",
                      file=sys.stderr)

    # -- Load configs -----------------------------------------------------------
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

    # -- Load ONNX model --------------------------------------------------------
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

    # -- Load & preprocess patch ------------------------------------------------
    is_fractal = (args.sensor == "fractal")
    if args.verbose and not is_fractal:
        print(f"[INFO] YellowScan mode: scan_angle_rank in degrees (no x0.006 conversion)",
              file=sys.stderr)
        print(f"[WARN] Intensity stats from FRACTAL -- Water OOD threshold may not transfer.",
              file=sys.stderr)

    t_prep = time.perf_counter()
    try:
        X_batch, raw_intensity, _, n_total = load_patch(
            patch_path, args.num_points, stats, is_fractal=is_fractal
        )
    except Exception as e:
        print(json.dumps({"error": f"Failed to load patch: {e}"}))
        sys.exit(1)

    result_sensor = args.sensor
    prep_ms = (time.perf_counter() - t_prep) * 1000

    # -- Inference --------------------------------------------------------------
    t_inf = time.perf_counter()
    logits_batch = sess.run([out_name], {inp_name: X_batch})[0]  # (1, N, C)
    inf_ms = (time.perf_counter() - t_inf) * 1000

    logits = logits_batch[0]           # (N, C)
    preds  = logits.argmax(axis=-1)    # (N,)

    # -- Class counts ----------------------------------------------------------
    class_counts = {}
    for c, name in enumerate(class_names):
        cnt = int((preds == c).sum())
        if cnt > 0:
            class_counts[name] = cnt

    # -- OOD Detection ---------------------------------------------------------
    ood_mask      = np.zeros(args.num_points, dtype=bool)
    ood_scores    = np.zeros(args.num_points, dtype=np.float32)
    ood_method    = "none"

    if not args.no_ood:
        ood_mask, ood_scores = detect_ood_hybrid(
            logits, raw_intensity, stats, ood_cfg
        )
        w_e = ood_cfg.get("weight_energy", 0.0)
        w_i = ood_cfg.get("weight_intensity", 1.0)
        ood_method = (
            "intensity-only (calibrated)"   if w_e == 0.0 else
            "energy-only (calibrated)"      if w_i == 0.0 else
            "hybrid (energy + intensity)"
        )

    n_ood   = int(ood_mask.sum())
    ood_pct = n_ood / args.num_points * 100

    # -- Save OOD embeddings για basket (--save-basket) -------------------------
    basket_file = None
    if args.save_basket and n_ood > 0:
        if pt_model is not None:
            try:
                ood_emb = extract_embeddings_pytorch(X_batch, ood_mask, pt_model)
                stem    = Path(args.patch).stem[:40]   # safe filename
                basket_file = save_basket_embeddings(ood_emb, basket_dir, stem)
                if args.verbose:
                    print(f"[INFO] Basket: {len(ood_emb)} OOD embeddings -> {basket_file.name}",
                          file=sys.stderr)
            except Exception as e:
                print(f"[WARN] Basket save failed: {e}", file=sys.stderr)
        elif args.verbose:
            print("[WARN] --save-basket: PyTorch model not available -- skipping",
                  file=sys.stderr)

    total_ms = (time.perf_counter() - t_total) * 1000

    # -- Output JSON -----------------------------------------------------------
    result = {
        "patch":      patch_path.name,
        "sensor":     args.sensor,
        "n_points":   args.num_points,
        "n_total":    n_total,
        "classes":    class_counts,
        "ood_points": n_ood,
        "ood_pct":    round(ood_pct, 2),
        "ood_method": ood_method,
        "ood_score_mean": round(float(ood_scores[ood_mask].mean()), 4) if n_ood > 0 else None,
        "basket_file": str(basket_file) if basket_file else None,
        "domain_shift_warning": None if is_fractal else
            "YellowScan intensity stats differ from FRACTAL -- Water OOD threshold unreliable",
        "latency_ms": {
            "preprocessing": round(prep_ms, 1),
            "inference":     round(inf_ms, 1),
            "total":         round(total_ms, 1),
        },
    }
    print(json.dumps(result, indent=2))

    # -- Verbose breakdown -----------------------------------------------------
    if args.verbose:
        print("\n--- Classification breakdown ---", file=sys.stderr)
        for name, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            pct = cnt / args.num_points * 100
            bar = "X" * int(pct / 2)
            print(f"  {name:<20} {cnt:>6,}  ({pct:5.1f}%)  {bar}",
                  file=sys.stderr)
        print(f"\n  OOD: {n_ood} points ({ood_pct:.1f}%)", file=sys.stderr)
        print(f"\n--- Latency ---", file=sys.stderr)
        print(f"  Preprocessing : {prep_ms:.1f} ms", file=sys.stderr)
        print(f"  Inference     : {inf_ms:.1f} ms", file=sys.stderr)
        print(f"  Total         : {total_ms:.1f} ms", file=sys.stderr)

        # Pi5 estimate
        est_pi5 = total_ms * 12   # conservative 12x slowdown
        print(f"\n  Pi5 estimate  : ~{est_pi5:.0f} ms  ({est_pi5/1000:.1f}s)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
