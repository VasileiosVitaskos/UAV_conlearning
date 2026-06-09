"""
export_onnx.py
Εξαγωγή Run 7 checkpoint → ONNX για deployment σε Raspberry Pi 5.

Χρήση:
    python src/export_onnx.py
    python src/export_onnx.py --checkpoint outputs/checkpoints/best_model.pt
    python src/export_onnx.py --batch-size 1 --num-points 4096

Γιατί ONNX:
    - Το onnxruntime τρέχει σε ARM64 (Pi5) χωρίς PyTorch dependency
    - Static shapes → γνωστό memory footprint
    - ~30K param μοντέλο → <1MB .onnx file

Output:
    outputs/model.onnx   ← deployment artifact
"""

import sys
import argparse
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models.pointnet2 import PointNet2Mini


def parse_args():
    p = argparse.ArgumentParser(description="Export PointNet++ Mini → ONNX")
    p.add_argument("--checkpoint", type=str,
                   default="outputs/checkpoints/best_model.pt",
                   help="Path to .pt checkpoint (default: best_model.pt)")
    p.add_argument("--output", type=str,
                   default="outputs/model.onnx",
                   help="Output .onnx path (default: outputs/model.onnx)")
    p.add_argument("--batch-size", type=int, default=1,
                   dest="batch_size",
                   help="Static batch size for ONNX (default: 1 for Pi5)")
    p.add_argument("--num-points", type=int, default=4096,
                   dest="num_points",
                   help="Points per patch (default: 4096)")
    p.add_argument("--opset", type=int, default=17,
                   help="ONNX opset version (default: 17)")
    return p.parse_args()


def main():
    args = parse_args()

    ckpt_path = ROOT / args.checkpoint
    out_path  = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  PointNet++ Mini → ONNX Export")
    print(f"  Checkpoint : {ckpt_path.name}")
    print(f"  Output     : {out_path}")
    print(f"  Batch size : {args.batch_size}  (static)")
    print(f"  Num points : {args.num_points}  (static)")
    print(f"  ONNX opset : {args.opset}")
    print(f"{'═'*60}\n")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cpu")   # Export on CPU — Pi5 έχει μόνο CPU
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)

    num_classes  = ckpt.get("num_classes", 6)
    active_names = ckpt.get("active_names", [f"class_{i}" for i in range(num_classes)])
    epoch        = ckpt.get("epoch", "?")
    miou         = ckpt.get("mIoU", 0.0)

    print(f"  Checkpoint info:")
    print(f"    Epoch      : {epoch}")
    print(f"    Val mIoU   : {miou:.4f}")
    print(f"    Classes    : {num_classes} ({', '.join(active_names)})")

    # ── Build model ────────────────────────────────────────────────────────────
    model = PointNet2Mini(in_channels=7, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Params     : {total_params:,}  ({total_params*4/1024:.1f} KB)")

    # ── Dummy input ────────────────────────────────────────────────────────────
    # Input shape: (B, N, 7) — batch, points, features
    B, N = args.batch_size, args.num_points
    dummy_input = torch.randn(B, N, 7, dtype=torch.float32)

    # ── Test forward pass ──────────────────────────────────────────────────────
    print(f"\n  Testing forward pass on CPU...")
    with torch.no_grad():
        out = model(dummy_input)
    print(f"  ✓ Input:  {tuple(dummy_input.shape)}  (float32)")
    print(f"  ✓ Output: {tuple(out.shape)}  (logits, float32)")
    print(f"  ✓ Output range: [{out.min():.3f}, {out.max():.3f}]")

    # ── Export to ONNX ─────────────────────────────────────────────────────────
    print(f"\n  Exporting to ONNX (opset {args.opset})...")

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        opset_version=args.opset,
        input_names=["point_cloud"],       # (B, N, 7)
        output_names=["logits"],            # (B, N, C)
        dynamic_axes=None,                  # Static shapes — Pi5 deployment
        export_params=True,
        do_constant_folding=True,
        verbose=False,
    )

    size_kb = out_path.stat().st_size / 1024
    print(f"  ✓ Saved: {out_path}  ({size_kb:.1f} KB)")

    # ── Verify with onnxruntime ────────────────────────────────────────────────
    print(f"\n  Verifying with onnxruntime...")
    try:
        import onnxruntime as ort

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4   # Pi5 has 4 cores
        sess_opts.log_severity_level  = 3    # suppress INFO/WARNING

        sess = ort.InferenceSession(str(out_path), sess_opts,
                                    providers=["CPUExecutionProvider"])

        inp_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name

        ort_out = sess.run([out_name], {inp_name: dummy_input.numpy()})
        ort_logits = ort_out[0]  # (B, N, C)

        # Compare PyTorch vs ONNX output
        pt_np  = out.numpy()
        max_diff = np.abs(pt_np - ort_logits).max()

        print(f"  ✓ ONNX output shape: {ort_logits.shape}")
        print(f"  ✓ Max diff PyTorch vs ONNX: {max_diff:.2e}  ", end="")
        print("(OK)" if max_diff < 1e-4 else f"(WARNING: {max_diff:.4f})")

        # Argmax agreement — class predictions correct even if logits differ?
        pt_classes  = pt_np.argmax(axis=-1)      # (B, N)
        ort_classes = ort_logits.argmax(axis=-1)  # (B, N)
        agree_pct   = (pt_classes == ort_classes).mean() * 100
        print(f"  ✓ Argmax agreement (class predictions): {agree_pct:.2f}%  ", end="")
        if agree_pct >= 99.9:
            print("(OK — predictions match despite logit diff)")
        elif agree_pct >= 95.0:
            print("(WARN — small prediction mismatch, check on real data)")
        else:
            print("(ERROR — predictions diverge! Try --opset 14 or 11)")

        # Latency estimate (10 warm-up + 20 timed runs)
        import time
        print(f"\n  Latency benchmark ({B}×{N} points, CPU, 20 runs)...")
        for _ in range(10):
            sess.run([out_name], {inp_name: dummy_input.numpy()})  # warm-up
        t0 = time.perf_counter()
        for _ in range(20):
            sess.run([out_name], {inp_name: dummy_input.numpy()})
        avg_ms = (time.perf_counter() - t0) / 20 * 1000

        print(f"  ✓ Avg latency (this machine): {avg_ms:.1f} ms")
        print(f"  ℹ  Pi5 expected: ~{avg_ms * 10:.0f}–{avg_ms * 15:.0f} ms  "
              f"(Pi5 CPU ≈ 10–15× slower than desktop)")

    except ImportError:
        print("  ⚠  onnxruntime not installed — skipping verification")
        print("     pip install onnxruntime")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Export complete!")
    print(f"  File     : {out_path}")
    print(f"  Size     : {size_kb:.1f} KB")
    print(f"  Input    : float32 ({B}, {N}, 7)  — [batch, points, features]")
    print(f"  Output   : float32 ({B}, {N}, {num_classes})  — [batch, points, logits]")
    print(f"  Classes  : {', '.join(active_names)}")
    print(f"\n  NEXT: copy outputs/model.onnx to Raspberry Pi 5")
    print(f"        and run: python src/inference.py <patch.laz>")
    print(f"{'═'*60}\n")

    # ── Save metadata alongside onnx ──────────────────────────────────────────
    import json
    meta = {
        "model":        "PointNet2Mini",
        "checkpoint":   ckpt_path.name,
        "epoch":        epoch,
        "val_mIoU":     float(miou),
        "num_classes":  num_classes,
        "active_classes": active_names,
        "input_shape":  [B, N, 7],
        "output_shape": [B, N, num_classes],
        "opset":        args.opset,
        "features":     ["x_rel", "y_rel", "z_rel", "intensity_log1p_zscore",
                         "return_number_div6", "n_returns_div6", "scan_angle_deg60"],
        "ood_note":     "Use hybrid OOD detector: energy score + raw intensity threshold. "
                        "Water AUROC=0.8232 with intensity-only method.",
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Metadata : {meta_path}")


if __name__ == "__main__":
    main()
