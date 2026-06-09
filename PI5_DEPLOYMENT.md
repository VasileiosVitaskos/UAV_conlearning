# Pi5 Deployment Guide
## Edge-Aware Terrain Classification — Raspberry Pi 5

**Αυτό το αρχείο:** Βήμα-βήμα οδηγίες για να τρέξεις το trained μοντέλο στο Raspberry Pi 5.

---

## Τι χρειάζεσαι

- Raspberry Pi 5 (4GB ή 8GB RAM)
- microSD ή SSD με Raspberry Pi OS (64-bit, Bookworm)
- Σύνδεση στο internet για installation
- Τα αρχεία από αυτό το repo

---

## Βήμα 1 — Κλωνοποίηση του repo στο Pi5

```bash
git clone <repo_url> ~/Drone_cont_Learing
cd ~/Drone_cont_Learing
```

Ή αν δεν έχεις git:
```bash
# Κάνε αντιγραφή τα αρχεία μέσω USB/SCP:
# (το model.onnx παράγεται στον dev machine με: python src/export_onnx.py)
scp outputs/model.onnx               pi@<pi_ip>:~/Drone_cont_Learing/outputs/
scp outputs/model.json               pi@<pi_ip>:~/Drone_cont_Learing/outputs/
scp outputs/normalizer_stats.json    pi@<pi_ip>:~/Drone_cont_Learing/outputs/
scp outputs/ood_hybrid_results.json  pi@<pi_ip>:~/Drone_cont_Learing/outputs/
scp src/inference.py                 pi@<pi_ip>:~/Drone_cont_Learing/src/
```

> **Σημείωση:** Για ONNX deployment δεν χρειάζεσαι PyTorch ή τον source κώδικα του μοντέλου. Μόνο τα 5 αρχεία παραπάνω.

---

## Βήμα 2 — Εγκατάσταση Python dependencies

```bash
# Δημιούργησε virtual environment
python3 -m venv ~/.venv_drone
source ~/.venv_drone/bin/activate

# Βασικές εξαρτήσεις (fast ONNX inference + OOD detection)
pip install onnxruntime laspy numpy

# PyTorch — για Continual Learning (LwF) + --save-basket embedding extraction
# ~200MB ARM64 wheel. Απαιτείται για: lwf_train.py, few_shot_add_class.py
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> **Σημείωση:**
> - Για **ONNX inference μόνο**: αρκεί `onnxruntime laspy numpy` (~50MB)
> - Για **CL pipeline** (LwF update + basket): χρειάζεσαι και `torch` (~200MB)
> - Το Pi5 ARM64 **δεν** έχει GPU — το PyTorch τρέχει αποκλειστικά σε CPU (εντάξει για inference + μικρό CL update)

---

## Βήμα 3 — Δοκιμαστικό τρέξιμο (dry-run)

```bash
cd ~/Drone_cont_Learing
source ~/.venv_drone/bin/activate

# Τρέξε inference σε ένα test patch:
python src/inference.py data/test/00/<οποιοδήποτε_αρχείο>.laz --verbose
```

Αναμενόμενο output (stdout):
```json
{
  "patch": "patch_001.laz",
  "sensor": "fractal",
  "n_points": 4096,
  "n_total": 8213,
  "classes": {
    "Ground": 2100,
    "HighVegetation": 1800,
    "MedVegetation": 200,
    "Building": 0
  },
  "ood_points": 45,
  "ood_pct": 1.10,
  "ood_method": "hybrid (energy + intensity)",
  "latency_ms": {
    "preprocessing": 12.3,
    "inference": 1800.0,
    "total": 1815.0
  }
}
```

---

## Βήμα 4 — Inference σε δικά σου δεδομένα

### FRACTAL patches (default):
```bash
python src/inference.py /path/to/patch.laz
```

### YellowScan UAV data:
```bash
python src/inference.py /path/to/yellowscan.laz --sensor yellowscan --verbose
```

> **Σημείωση για YellowScan:** Το μοντέλο εκπαιδεύτηκε σε FRACTAL (airborne ALS). Το YellowScan έχει διαφορετικό sensor profile — η ταξινόμηση γεωμετρικών κλάσεων (HighVeg, Building) θα δουλέψει αρκετά καλά, αλλά η ανίχνευση Water μέσω intensity threshold μπορεί να χρειαστεί re-calibration.

### Batch processing (πολλά patches):
```bash
for f in data/test/**/*.laz; do
    python src/inference.py "$f" >> results.jsonl
done
```

---

## Τι σημαίνουν τα αποτελέσματα

| Πεδίο | Τιμή | Ερμηνεία |
|-------|------|----------|
| `classes` | `{"Ground": 2310, ...}` | Points ανά terrain class |
| `ood_points` | `142` | Points που ανιχνεύτηκαν ως OOD |
| `ood_pct` | `3.47` | % OOD — αν >10% ύποπτο patch |
| `ood_method` | `"hybrid"` | Energy score + intensity threshold |
| `latency_ms.inference` | `~1800` | ~1.8 δευτερόλεπτα για 4096 points |

### Κλάσεις:
| ID | Όνομα | Περιγραφή |
|----|-------|-----------|
| 0 | Ground | Έδαφος |
| 1 | LowVegetation | Χαμηλή βλάστηση (<0.5m) |
| 2 | MedVegetation | Μεσαία βλάστηση |
| 3 | HighVegetation | Υψηλή βλάστηση / δέντρα |
| 4 | Building | Κτίρια |
| 5 | Noise | Θόρυβος / artifacts |

### OOD flag:
- **Water** ανιχνεύεται μέσω intensity threshold (specular reflection του near-IR laser — Water επιστρέφει πολύ λιγότερο σήμα)
- **Bridge** ανιχνεύεται μέσω energy score (γεωμετρικά "ύποπτο" για το μοντέλο)
- AUROC = 0.8232 στο test set

---

## Αντιμετώπιση προβλημάτων

### `ModuleNotFoundError: No module named 'torch'`
```bash
source ~/.venv_drone/bin/activate
# Ξαναπάτα το inference command
```

### `FileNotFoundError: outputs/model.onnx not found`
Το inference.py τρέχει με PyTorch, όχι ONNX. Το αρχείο που χρειάζεσαι είναι:
```
outputs/checkpoints/best_model.pt  ← αυτό χρησιμοποιείται
```

### `OOM / Killed`
Αν το Pi5 δεν έχει αρκετή RAM (4GB model):
```bash
# Μείωσε τα points ανά patch:
python src/inference.py patch.laz --num-points 2048
```

### Αργό inference (>5 δευτερόλεπτα/patch)
Αναμενόμενο — το Pi5 Cortex-A76 είναι ~10-15× πιο αργό από desktop GPU. Για 4096 points εκτιμούμε ~1.5-2 sec. Αν θες ταχύτητα, χρησιμοποίησε `--num-points 1024`.

---

## Αρχεία που χρειάζεσαι (minimum)

```
Drone_cont_Learing/
├── outputs/
│   ├── model.onnx                 ← ONNX μοντέλο (~4MB, παράγεται με export_onnx.py)
│   ├── model.json                 ← metadata (class names, input shape)
│   ├── normalizer_stats.json      ← preprocessing stats (mean/std ανά feature)
│   └── ood_hybrid_results.json    ← OOD thresholds (energy + intensity)
└── src/
    └── inference.py               ← κύριο script (standalone, μόνο onnxruntime + laspy)
```

> **Πλεονέκτημα ONNX:** Δεν χρειάζεσαι PyTorch, `pointnet2.py`, ή άλλο source κώδικα στο Pi5.

---

## Σημειώσεις για thesis presentation

- Το μοντέλο έχει **30,36