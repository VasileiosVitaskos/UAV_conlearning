# Οδηγίες Pi5 — Edge Continual Learning Pipeline
**Συγγραφέας:** Βασίλειος Βιτάσκος (ΑΕΜ 235)  
**Μοντέλο:** PointNet++ Mini | test mIoU = 0.6196 | OA = 0.9110

---

## Επισκόπηση

Αυτός ο οδηγός καλύπτει δύο ξεχωριστά σενάρια:

| Σενάριο | Τι κάνεις | Χρόνος |
|---------|-----------|--------|
| **A — Inference μόνο** | Τρέχεις το trained μοντέλο σε νέα δεδομένα | ~5 λεπτά setup |
| **B — Πλήρης CL pipeline** | Inference + OOD detection + LwF update με νέα κλάση | ~30 λεπτά |

---

## ΜΕΡΟΣ 1 — Προετοιμασία (dev machine, πριν πας στο Pi5)

> Αυτά τρέχουν **στον υπολογιστή σου**, όχι στο Pi5.
> Το FRACTAL dataset (~5GB) δεν το μεταφέρεις ολόκληρο.

### Βήμα 1.1 — Φτιάξε το labeled basket (Water + Bridge examples)

```bash
# Από το root του repo, στον dev machine:
python src/make_basket.py --n-water 10 --n-bridge 10 --split val
```

Αποτέλεσμα:
```
data/basket/
  water/    ← 10 .laz patches με νερό (labeled, από FRACTAL val set)
  bridge/   ← 10 .laz patches με γέφυρες
```

> Αν θέλεις περισσότερα examples (καλύτερη ακρίβεια CL): `--n-water 20 --n-bridge 20`

### Βήμα 1.2 — Σπάσε το YellowScan σε tiles (προαιρετικό)

**Επιλογή Α** — Σπάσε εδώ, στείλε μόνο τα tiles στο Pi5 (~200-400 MB αντί για 1.6 GB):

```bash
python src/split_yellowscan.py
# → data/yellowscan_patches/ (~100-300 tiles των 50×50m)
```

**Επιλογή Β** — Στείλε ολόκληρο το .laz στο Pi5 και σπάσε το εκεί.  
Χρειάζεται ≥2GB ελεύθερο χώρο στο Pi5.

### Βήμα 1.3 — Αποθήκευσε μέγεθος αρχείων που θα στείλεις

```
Ελάχιστα αρχεία για Pi5 (Σενάριο A):
  outputs/checkpoints/best_model.pt          ~2 MB
  outputs/model.onnx                         ~4 MB (συνήθως ήδη στο git)
  outputs/model.json                         <1 KB
  outputs/normalizer_stats.json              <1 KB
  outputs/ood_hybrid_results.json            <1 KB

Επιπλέον για Σενάριο B (CL):
  data/basket/water/   (10 .laz)             ~5-10 MB
  data/basket/bridge/  (10 .laz)             ~5-10 MB
  data/yellowscan_patches/  ή  data/real UAV/*.laz
```

---

## ΜΕΡΟΣ 2 — Μεταφορά δεδομένων στο Pi5

### Μέθοδος Α — SCP (SSH, στο ίδιο δίκτυο)

```bash
# Αντικατάστησε PI=pi@<ip_διεύθυνση_του_pi>
PI=pi@192.168.1.xxx

# 1. Κάνε git clone στο Pi5 πρώτα (δες Βήμα 3.1 παρακάτω)
#    Μετά στείλε μόνο τα αρχεία που δεν είναι στο git:

# Model weights (ΔΕΝ είναι στο git — πολύ μεγάλο)
scp outputs/checkpoints/best_model.pt \
    $PI:~/Drone_cont_Learing/outputs/checkpoints/

# Basket (από make_basket.py — Βήμα 1.1)
scp -r data/basket/ \
    $PI:~/Drone_cont_Learing/data/

# YellowScan tiles (αν έτρεξες split_yellowscan.py — Βήμα 1.2 Επιλογή Α)
scp -r data/yellowscan_patches/ \
    $PI:~/Drone_cont_Learing/data/

# Ή ολόκληρο το raw .laz (Επιλογή Β — 1.6 GB, αργό)
scp "data/real UAV/VA50-SC20-M600-120mAGL-10ms-SACOCU-Pipeline_survey(1).laz" \
    "$PI:~/Drone_cont_Learing/data/real UAV/"
```

### Μέθοδος Β — USB stick

```bash
# Στον dev machine — αντιγραφή σε USB:
cp outputs/checkpoints/best_model.pt  /Volumes/USB/drone_data/
cp -r data/basket/                    /Volumes/USB/drone_data/
cp -r data/yellowscan_patches/        /Volumes/USB/drone_data/   # ή το .laz

# Στο Pi5 — από USB:
cp /media/pi/USB/drone_data/best_model.pt \
   ~/Drone_cont_Learing/outputs/checkpoints/
cp -r /media/pi/USB/drone_data/basket \
   ~/Drone_cont_Learing/data/
cp -r /media/pi/USB/drone_data/yellowscan_patches \
   ~/Drone_cont_Learing/data/
```

### Μέθοδος Γ — rsync (γρηγορότερο για πολλά μικρά αρχεία)

```bash
PI=pi@192.168.1.xxx

rsync -avz --progress \
  outputs/checkpoints/best_model.pt \
  data/basket/ \
  data/yellowscan_patches/ \
  $PI:~/Drone_cont_Learing/
```

---

## ΜΕΡΟΣ 3 — Setup στο Pi5

### Βήμα 3.1 — Clone repo και εγκατάσταση

```bash
# Στο Pi5 (SSH ή keyboard):
git clone git@github.com:VasileiosVitaskos/UAV_conlearning.git ~/Drone_cont_Learing
cd ~/Drone_cont_Learing

# Virtual environment
python3 -m venv ~/.venv_drone
source ~/.venv_drone/bin/activate

# Εξαρτήσεις — ONNX inference μόνο (Σενάριο A, ~50 MB):
pip install onnxruntime laspy numpy

# Εξαρτήσεις — πλήρης CL pipeline (Σενάριο B, +~200 MB):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> **Σημείωση:** Η εγκατάσταση torch στο Pi5 ARM64 παίρνει ~5-10 λεπτά.  
> Το Pi5 δεν έχει GPU — τρέχει αποκλειστικά CPU (εντάξει για inference + μικρό CL update).

### Βήμα 3.2 — Επαλήθευση αρχείων

```bash
# Βεβαιώσου ότι υπάρχουν τα απαραίτητα αρχεία:
ls -lh outputs/checkpoints/best_model.pt    # ~2 MB
ls -lh outputs/model.onnx                  # ~4 MB
ls    outputs/normalizer_stats.json         # υπάρχει
ls    outputs/ood_hybrid_results.json       # υπάρχει
ls -d data/basket/water/ data/basket/bridge/     # αν έχεις κάνει CL
```

---

## ΜΕΡΟΣ 4 — Εκτέλεση pipeline

### ΣΕΝΑΡΙΟ A — Inference μόνο

```bash
cd ~/Drone_cont_Learing
source ~/.venv_drone/bin/activate

# Inference σε ένα tile:
python src/inference.py data/yellowscan_patches/ys_tile_x000500_y004200.laz \
    --sensor yellowscan --verbose
```

Αναμενόμενο output:
```json
{
  "patch": "ys_tile_x000500_y004200.laz",
  "sensor": "yellowscan",
  "n_points": 4096,
  "classes": {"Ground": 2100, "HighVegetation": 1800, "Building": 50},
  "ood_points": 45,
  "ood_pct": 1.10,
  "latency_ms": {"preprocessing": 12, "inference": 1800, "total": 1815}
}
```

**Batch — όλα τα tiles:**
```bash
mkdir -p outputs/inference_results

for f in data/yellowscan_patches/*.laz; do
    python src/inference.py "$f" --sensor yellowscan \
        >> outputs/inference_results/results.jsonl
done

echo "Τελείωσε. Αποτελέσματα: outputs/inference_results/results.jsonl"
```

**Αν δεν έχεις κάνει tile ακόμα** (Επιλογή Β — raw .laz στο Pi5):
```bash
# Πρώτα split:
python src/split_yellowscan.py
# Μετά inference όπως παραπάνω
```

---

### ΣΕΝΑΡΙΟ B — Πλήρης CL pipeline (LwF update)

> **Προϋπόθεση:** Να έχεις κάνει τα Βήματα 1.1-1.2 (dev machine) και 
> μεταφορά basket + tiles στο Pi5.

#### Βήμα B1 — OOD basket collection (inference με --save-basket)

```bash
# Τρέξε inference σε όλα τα tiles, αποθήκευσε OOD embeddings:
for f in data/yellowscan_patches/*.laz; do
    python src/inference.py "$f" \
        --sensor yellowscan \
        --save-basket outputs/ood_basket \
        --pt-model outputs/checkpoints/best_model.pt \
        --verbose 2>> outputs/inference_log.txt
done

echo "OOD basket: $(ls outputs/ood_basket/ | wc -l) αρχεία"
```

#### Βήμα B2 — Few-shot prototype init

```bash
# Αρχικοποίηση νέας κλάσης Water από τα labeled FRACTAL patches (basket):
python src/few_shot_add_class.py \
    --mode fractal \
    --class-name Water \
    --n-shots 10 \
    --basket-dir data/basket/water

# Και για Bridge (αν θέλεις):
python src/few_shot_add_class.py \
    --mode fractal \
    --class-name Bridge \
    --n-shots 10 \
    --basket-dir data/basket/bridge
```

#### Βήμα B3 — LwF training

```bash
# Water (χρόνος: ~4-6 λεπτά στο Pi5):
python src/lwf_train.py \
    --class-name Water \
    --basket-dir data/basket/water \
    --epochs 10 \
    --lr 5e-3 \
    --lambda-kd 1.0

# → αποθηκεύει: outputs/checkpoints/best_model_lwf_water.pt
```

#### Βήμα B4 — Αξιολόγηση

```bash
python src/evaluate_cl.py \
    --cl-checkpoint  outputs/checkpoints/best_model_lwf_water.pt \
    --base-checkpoint outputs/checkpoints/best_model.pt
```

Αναμενόμενο: Water IoU να ανέβει, παλιές κλάσεις να μείνουν ~σταθερές.

#### Βήμα B5 — Re-export ONNX (για deployment)

```bash
python src/export_onnx.py \
    --checkpoint outputs/checkpoints/best_model_lwf_water.pt \
    --output outputs/model_lwf_water.onnx
# → 7 κλάσεις (6 παλιές + Water)
```

---

## Αντιμετώπιση προβλημάτων

### `ModuleNotFoundError: No module named 'torch'`
```bash
source ~/.venv_drone/bin/activate
# Ξαναπάτα την εντολή
```

### `FileNotFoundError: best_model.pt not found`
```bash
# Επιβεβαίωσε ότι αντέγραψες το checkpoint:
ls -lh outputs/checkpoints/best_model.pt
# Αν λείπει: δες Μέρος 2 (Μεταφορά δεδομένων)
```

### `Killed` ή `OOM` (out of memory)
```bash
# Μείωσε τα points ανά patch:
python src/inference.py tile.laz --num-points 2048 --sensor yellowscan
```

### Αργό inference (>5 sec/patch)
Φυσιολογικό. Το Pi5 Cortex-A76 είναι ~10-15× πιο αργό από desktop GPU.  
Για 4096 points: ~1.5-2 sec | Για 2048 points: ~0.8-1 sec.

### Scan angle error στο YellowScan
```bash
# Το YellowScan χρησιμοποιεί scan_angle_rank, όχι scan_angle
# Βεβαιώσου ότι καλείς με --sensor yellowscan (το χειρίζεται αυτόματα)
python src/inference.py tile.laz --sensor yellowscan
```

---

## Αποτελέσματα για poster/presentation

```
Αρχιτεκτονική   : PointNet++ Mini
Παράμετροι      : 30,368  (0.12 MB)
Εκπαίδευση      : FRACTAL dataset (3000 train patches, 50×50m)

Val  mIoU = 0.6412  (Run 7, epoch 36)
Test mIoU = 0.6196  ← αυτό αναφέρεις στο poster
Test F1   = 0.7166
Test OA   = 0.9110

Per-class IoU:
  Ground          : 0.883  ✓
  LowVegetation   : 0.090  ⚠ (class imbalance)
  MedVegetation   : 0.481
  HighVegetation  : 0.940  ✓
  Building        : 0.795  ✓
  Noise           : 0.528

OOD Detection:
  Μέθοδος  : Hybrid (energy score + intensity)
  AUROC    : 0.8232 (Water + Bridge)
  Threshold: 0.9967 (calibrated on val set)

Latency Pi5 (Cortex-A76):
  4096 points : ~1.5-2.0 sec
  2048 points : ~0.8-1.0 sec
```

---

## Checklist πριν demo

- [ ] `git pull` — τελευταία έκδοση κώδικα
- [ ] `source ~/.venv_drone/bin/activate`
- [ ] `ls outputs/checkpoints/best_model.pt` — υπάρχει;
- [ ] `ls data/yellowscan_patches/ | head` — υπάρχουν tiles;
- [ ] `python src/inference.py data/yellowscan_patches/<οποιοδήποτε>.laz --sensor yellowscan --verbose` — τρέχει;
- [ ] Για CL demo: `ls data/basket/water/ | wc -l` — 10 αρχεία;

---

*Ερωτήσεις: vvitaskos@gmail.com*
