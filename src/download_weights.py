"""
download_weights.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Κατεβάζει τα pre-trained model weights (best_model.pt) που εξαιρούνται
από το git λόγω μεγέθους.

Χρήση:
    python src/download_weights.py

Αν δεν έχεις ακόμα κάπου hosted τα weights, μπορείς:
  • Google Drive: μοιράσου το best_model.pt και βάλε το link παρακάτω
  • HuggingFace Hub: huggingface-cli upload <repo> outputs/checkpoints/best_model.pt
  • OneDrive / Sharepoint (ΑΠΘ)

ΤΩΡΑ: Ζήτησε το best_model.pt απευθείας από τους συγγραφείς.
"""

import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
CKPT_DIR = ROOT / "outputs" / "checkpoints"

# ── Ρύθμισε το URL εδώ αφού ανεβάσεις τα weights ─────────────────────────────
# Παράδειγμα Google Drive: https://drive.google.com/uc?id=<FILE_ID>
# Παράδειγμα HuggingFace:  https://huggingface.co/<user>/<repo>/resolve/main/best_model.pt
WEIGHTS_URL = "TODO: replace with actual URL after uploading"

FILES = {
    "best_model.pt": WEIGHTS_URL,
}


def download_file(url: str, dest: Path) -> bool:
    """Κατεβάζει ένα αρχείο από URL."""
    try:
        import urllib.request
        print(f"  Downloading {dest.name} ...")
        urllib.request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"  ✓ {dest.name}  ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False


def main():
    print(f"\n{'═'*60}")
    print(f"  Download Model Weights — Edge CL Project")
    print(f"{'═'*60}\n")

    if WEIGHTS_URL.startswith("TODO"):
        print("  ⚠  Δεν έχει οριστεί URL για τα weights.\n")
        print("  Επίλογος ε΅σωτερικής χρήσης:")
        print("  1. Ζήτησε το best_model.pt από: vvitaskos@gmail.com")
        print("  2. Αντέγραψέ το στο: outputs/checkpoints/best_model.pt")
        print("\n  Ή εκπαίδευσε από την αρχή:")
        print("  python src/train.py --epochs 40 --bs 16 --cache --loss focal \\")
        print("         --exclude-classes Water Bridge --xyz-dropout 0.20")
        sys.exit(0)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    for fname, url in FILES.items():
        dest = CKPT_DIR / fname
        if dest.exists():
            size_mb = dest.stat().st_size / 1024 / 1024
            print(f"  ✓ {fname} already exists ({size_mb:.1f} MB) — skip")
            continue
        download_file(url, dest)

    print(f"\n  ── Verify ──")
    for fname in FILES:
        dest = CKPT_DIR / fname
        if dest.exists():
            print(f"  ✓ {dest}")
        else:
            print(f"  ✗ MISSING: {dest}")

    print(f"\n  NEXT:")
    print(f"    python src/evaluate_test.py")
    print(f"    python src/export_onnx.py")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
