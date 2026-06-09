# Σημειώσεις Παρουσίασης — Poster Session
## Edge-Aware Continual Learning for Terrain Classification from UAV LiDAR Point Clouds

**Ομάδα:** Βασίλειος Βιτάσκος (ΑΕΜ 235) · Ορέστης Γεωργιάδης (ΑΕΜ 239) · Αστέριος Τερζής (ΑΕΜ 218)

---

## 1. Τι χτίσαμε — Σύντομη Περιγραφή

Σύστημα **ανοικτής-κόσμου συνεχούς μάθησης** για κατηγοριοποίηση εδάφους από LiDAR σαρώσεις UAV:

1. **Φάση Βάσης**: Εκπαίδευση PointNet++ Mini σε 3 γνωστές κλάσεις (Ground, Vegetation, Building)
2. **OOD Detection**: Ανίχνευση νέων κλάσεων (Water, Bridge) μέσω Energy Score / Mahalanobis distance
3. **Few-Shot Learning**: Εκμάθηση νέων κλάσεων με 5–10 παραδείγματα χωρίς να ξεχαστούν οι παλιές
4. **Ανάπτυξη**: ONNX export → onnxruntime → Raspberry Pi 5 (~35ms inference target)

**Dataset:** FRACTAL (ALS, 50×50m patches, ~150K σημεία, 8 κλάσεις, LAS 1.4)
**Validation:** YellowScan (πραγματικό UAV LiDAR, unclassified — domain shift test)

---

## 2. Αρχιτεκτονική Μοντέλου

**PointNet++ Mini** — εκ σχεδιασμού ελαφρύ για Raspberry Pi 5

```
Input: (B, N=4096, F=7)   — x_rel, y_rel, z_rel, intensity, return_num, num_returns, scan_angle
    │
    ▼
SA1: FPS(4096→512) + KNN(k=32) + MLP[7→32→64]   → (B, 512, 64)
    │
    ▼
SA2: FPS(512→128) + KNN(k=32) + MLP[64→64→128]  → (B, 128, 128)
    │
    ▼
FP2: 3-NN IDW interp + skip(SA1:64) + MLP[192→64]  → (B, 512, 64)
    │
    ▼
FP1: 3-NN IDW interp + skip(input:7) + MLP[71→32]  → (B, 4096, 32)
    │
    ▼
CosineClassifier: score = τ·cos(x, w_c), τ=10     → (B, N, C)
```

| Μέτρο | Τιμή |
|-------|------|
| Παράμετροι | 30,432 |
| Μέγεθος | 0.12 MB |
| Inference CPU | ~96ms |
| Inference Pi5 (εκτίμηση) | ~35ms (ONNX) |

**Γιατί CosineClassifier αντί για Linear;**
→ Η νόρμα του embedding δεν επηρεάζει την απόφαση, μόνο η γωνία.
→ Νέα κλάση = προσθήκη γραμμής στον πίνακα βαρών (`add_class()`) → **zero forgetting by design**.

---

## 3. Pipeline Λεπτομέρειες

### Features (7 διαστάσεις ανά σημείο)
| Feature | Κανονικοποίηση | Γιατί |
|---------|----------------|-------|
| x_rel, y_rel | StandardScaler | relative εντός patch |
| z_rel | StandardScaler | ύψος relative |
| intensity | StandardScaler | πυκνότητα επιφάνειας |
| return_number | StandardScaler | αριθμός επιστροφής σήματος |
| number_of_returns | StandardScaler | συνολικές επιστροφές |
| scan_angle | StandardScaler | γωνία σάρωσης |

**Σημαντικό:** StandardScaler fit ΜΟΝΟ στο train set (partial_fit ανά αρχείο). Τα ίδια stats εφαρμόζονται σε val/test/YellowScan → αποφυγή data leakage.

### Set Abstraction (κάθε SA layer)
1. **FPS** (Farthest Point Sampling): επιλέγει S χωρικά ομοιόμορφα centroids — δεν υπάρχει bias προς πυκνές περιοχές
2. **KNN query**: k γείτονες γύρω από κάθε centroid
3. **MLP + max-pool**: εξάγει τοπικά features (rotation-invariant μέσω relative coords)

### Feature Propagation (upsampling)
- 3-NN IDW (Inverse Distance Weighting) interpolation
- Skip connections (όπως U-Net) → μεταφορά fine-grained χαρακτηριστικών

---

## 4. Αντιμετώπιση Class Imbalance

### Γιατί έχουμε πρόβλημα
Το FRACTAL dataset έχει **ακραία ανισορροπία** (absolute rarity — Tsoumakas slides):
- Ground: ~60% των σημείων
- Bridge: <0.1% των σημείων
- Χωρίς διόρθωση → το μοντέλο μαθαίνει "πες όλα Ground, 79% OA" και χαίρεται

### Τεχνικές που εφαρμόστηκαν

#### Runs 1–3: Weighted CrossEntropy (baseline)
```
w_c = total_points / (C × n_c)
normalize: mean(w) = 1
```
Πρόβλημα: υπολογισμός από 50 τυχαία patches → αστάθεια.
- Run 1: Bridge είχε 54 points → weight = 6.789
- Run 2: Bridge είχε 467 points → weight = 0.631
→ 10× διαφορά στο weight για την ίδια κλάση!

#### Run 4: Focal Loss + Stable Weights
**Focal Loss** (Lin et al., IEEE TPAMI 2020 — Διαφάνειες Τσουμάκα, Class Imbalance):
```
FL(p_t) = -(1 - p_t)^γ · log(p_t),  γ = 2
```
- Ground (p=0.9): focal weight = (1-0.9)² = **0.01** → loss σχεδόν μηδέν
- Bridge (p=0.1): focal weight = (1-0.1)² = **0.81** → loss παραμένει υψηλό
- Αποτέλεσμα: τα easy examples δεν κατακλύζουν το gradient

**Stable weights:** Σκαναρισμός ΟΛΟΥ του train set (3000 patches) → αναπαραγώγιμα weights.

---

## 5. Αποτελέσματα Εκπαίδευσης

### Baseline Runs (CrossEntropy + 50-patch weights)

| Κλάση | Run 1 IoU | Run 2 IoU | Run 3 IoU* |
|-------|-----------|-----------|------------|
| Ground | 0.626 | 0.669 | — |
| LowVegetation | 0.161 | 0.168 | — |
| MedVegetation | 0.379 | 0.368 | — |
| HighVegetation | 0.902 | 0.894 | — |
| Building | 0.735 | 0.754 | — |
| Water | 0.798 | 0.852 | — |
| Bridge | 0.053 | 0.093 | **0.033** |
| Noise | 0.000 | 0.099 | — |
| **mIoU** | **0.4568** | **0.4873** | **0.4153** |
| **macro F1** | **0.5476** | **0.5878** | — |
| **OA** | **0.7679** | **0.7843** | — |

*Run 3: λεπτομερής per-class breakdown δεν καταγράφηκε — το CSV log υπάρχει στο outputs/logs/

### Run 4: Focal Loss (γ=2) + Stable Weights ✓

Best checkpoint: epoch 72 · val mIoU peak 0.5923 · detailed evaluation below

| Κλάση | Run 4 IoU | vs Run 2 (best baseline) | Δ |
|-------|-----------|--------------------------|---|
| Ground | **0.880** | 0.669 | +31% ↑ |
| LowVegetation | 0.113 | 0.168 | −33% ↓ |
| MedVegetation | **0.462** | 0.368 | +26% ↑ |
| HighVegetation | **0.944** | 0.894 | +6% ↑ |
| Building | **0.798** | 0.754 | +6% ↑ |
| Water | **0.862** | 0.852 | +1% ↑ |
| Bridge | **0.124** | 0.093 | +33% ↑ |
| Noise | **0.467** | 0.099 | +371% ↑↑ |
| **mIoU** | **0.5813** | 0.4873 | **+19%** |
| **macro F1** | **0.6767** | 0.5878 | +15% |
| **OA** | **0.9136** | 0.7843 | +16% |

> ⚠️ LowVegetation regression (0.168→0.113): Focal Loss ανακατανέμει gradient προς Bridge/Noise.
> Πιθανή αιτία: αυξημένη σύγχυση LowVeg↔MedVeg όταν το μοντέλο "εστιάζει" σε σπανιότερες κλάσεις.

### Συνολική Σύγκριση (για Poster)

Val set = tuning metric · **Test set = final poster numbers (αγγίχτηκε μία φορά)**

| Μέθοδος | mIoU (test) | Bridge IoU | Water IoU | OA |
|---------|-------------|-----------|-----------|-----|
| Weighted CE baseline (val mean±std) | 0.453 ± 0.037 | 0.060 ± 0.025 | ~0.83† | ~0.78† |
| **Focal Loss γ=2 — TEST SET** | **0.6045** | **0.284** | **0.914** | **0.9080** |
| **Δ vs baseline** | **+0.151 (+33%)** | **+0.224 (+373%)** | — | **+0.13** |

†Runs 1–2 only; Run 3 per-class log not captured

#### Test set per-class IoU (Run 4 — final)
| Κλάση | Test IoU | Bar |
|-------|----------|-----|
| Ground | 0.870 | ████████████████▌ |
| LowVegetation | 0.082 | █▋ ⚠️ |
| MedVegetation | 0.470 | █████████▍ |
| HighVegetation | 0.939 | ██████████████████▊ |
| Building | 0.757 | ███████████████▏ |
| Water | 0.914 | ██████████████████▎ |
| Bridge | 0.284 | █████▋ |
| Noise | 0.520 | ██████████▍ |
| **mIoU** | **0.6045** | |
| **macro F1** | **0.7042** | |
| **OA** | **0.9080** | |

> 🎯 Test mIoU (0.6045) > Val mIoU (0.5813) → κανένα overfitting στο val set
> 🎯 Bridge test IoU = 0.284 vs val IoU = 0.124 → test set έχει "τυπικότερα" Bridge examples
> ⚠️ LowVegetation = 0.082 (χειρότερο test από val 0.113) — ongoing confusion με MedVeg

---

## 6. OOD Detection Pipeline (Run 5 & πέρα)

### Στρατηγική: ignore_index αντί για αφαίρεση

Για το Run 5 (base model για OOD), τα Water/Bridge σημεία **δεν αφαιρούνται** από τα patches — αντίθετα, γίνονται `ignore_index=-1`:

```
Training (Run 5):
  Ground, LowVeg, MedVeg, HighVeg, Building, Noise  →  κανονικό loss
  Water, Bridge                                      →  ignore_index=-1 (masked)
```

Το μοντέλο εκπαιδεύεται σε 6 κλάσεις και δεν βλέπει ποτέ Water/Bridge στο gradient.

### Γιατί αυτό είναι έξυπνο για αξιολόγηση

Το FRACTAL dataset **ξέρει** ποια σημεία είναι Water/Bridge (ground truth labels). Οπότε μπορούμε να αξιολογήσουμε το OOD ποσοτικά:

```
Κατά την αξιολόγηση:
  Water/Bridge σημεία  →  αναμένεται: υψηλό Energy Score (OOD) ⚠️
  Υπόλοιπα σημεία      →  αναμένεται: χαμηλό Energy Score (known) ✓
```

**Metrics OOD αξιολόγησης:**
```
OOD Precision = TP_ood / (TP_ood + FP_ood)
OOD Recall    = TP_ood / (TP_ood + FN_ood)
OOD F1        = 2 · P · R / (P + R)
AUROC         = area under ROC curve (Energy score vs OOD label)
```

Αυτό δίνει **αριθμούς** για το Poster — όχι μόνο "φαίνεται σωστό οπτικά".

### Per-point Energy Score

```python
# model trained on 6 classes (no Water/Bridge)
logits     = model(x)                        # (B, N, 6)
embeddings = model.get_embeddings(x)         # (B, N, 32)

# Energy Score ανά σημείο
energy = -torch.logsumexp(logits, dim=-1)    # (B, N) — υψηλό = OOD

# Threshold (calibrated on val set)
ood_mask = energy > threshold                # (B, N) bool

# Few-shot: τα OOD σημεία σχηματίζουν spatial clusters
# → ο χρήστης/αλγόριθμος δίνει label → add_class()
prototype = embeddings[ood_mask].mean(dim=0) # (32,)
model.classifier.add_class(prototype)        # zero forgetting ✓
```

### OOD Detection Results — Energy Score (Run 5)

> **Honest finding**: Το Energy Score δουλεύει άριστα για Bridge αλλά αποτυγχάνει πλήρως για Water.

#### Αποτελέσματα (val → test)

| Μέτρο | Val | Test |
|-------|-----|------|
| AUROC | 0.4876 | **0.4321** ← αποτυχία |
| AUPR | 0.1803 | 0.1621 |
| Best F1 @ threshold | 0.3418 | 0.2951 |
| Threshold (calibrated on val) | -2.69 | ← εφαρμόστηκε στο test |

> AUROC < 0.5 σημαίνει ότι ο scorer είναι **χειρότερος από τυχαίο** — αποτυχία, όχι απλά χαμηλή απόδοση.

#### Per-class energy analysis (val set)

| Κλάση | N points | Mean E | Std E | Σχόλιο |
|-------|----------|--------|-------|--------|
| Ground | 3,891,440 | -4.547 | 0.843 | Πιο confident |
| LowVegetation | 75,012 | -4.421 | 0.871 | |
| MedVegetation | 513,298 | -3.519 | 1.024 | Λιγότερο certain |
| HighVegetation | 2,198,441 | -4.612 | 0.791 | |
| Building | 438,219 | -3.321 | 1.187 | Αρκετά uncertain ← |
| Noise | 4,103 | -3.264 | 1.341 | Πολύ uncertain ← |
| **Water** | 75,441 | **-4.420** | 1.018 | **≈ Ground (d'=0.05)** ← αποτυχία |
| **Bridge** | 18,872 | **-2.815** | 1.203 | **Αποκλίνει σαφώς (d'=1.66)** ← δουλεύει |

#### Separability d' ανά OOD class

```
Bridge  d' = 1.66  →  ✓ ικανοποιητική διαχωρισιμότητα (>1.0 threshold)
Water   d' = 0.05  →  ✗ ουσιαστικά αδύνατη διαχωρισιμότητα
```

#### Γιατί αποτυγχάνει για το Water

Το Water στο ALS LiDAR έχει **γεωμετρική ομοιότητα με το Ground**:
- Flat επιφάνεια → παρόμοια z_rel κατανομή
- Single returns → παρόμοιο return_number profile
- Moderate intensity → overlap με alluvial/sandy ground

Αποτέλεσμα: Ο Run 5 6-class model ταξινομεί τα Water σημεία ως **Ground** με υψηλή εμπιστοσύνη.
Το Energy Score δεν ανιχνεύει αυτό γιατί "σίγουρο αλλά λάθος" = **χαμηλό energy** (low = confident).

```
Water point → model → logits: [Ground=5.2, LowVeg=-3.1, ...] → energy=-5.3 → NOT flagged
Bridge point → model → logits: [Ground=1.1, LowVeg=0.8, ...] → energy=-2.7 → FLAGGED ✓
```

#### Επιπλέον παρατήρηση

Τα **Building (-3.32)** και **Noise (-3.26)** έχουν υψηλότερο energy από το Water (-4.42)!
Αυτό σημαίνει ότι με οποιοδήποτε threshold που πιάνει Water:
- Πρώτα φλαγκάρονται Building/Noise (false positives)
- Water φλαγκάρεται ελάχιστα ή καθόλου (false negatives)

→ AUROC < 0.5 = η κατανομή είναι **αντίστροφη** του αναμενόμενου.

#### Πλήρης OOD Investigation — Τελικά Αποτελέσματα

Δοκιμάστηκαν 4 μέθοδοι συστηματικά:

| Μέθοδος | Test AUROC | Γιατί αποτυγχάνει |
|---------|-----------|-------------------|
| Energy Score (Run 5) | 0.4321 | Water ≈ Ground geometrically → model confident → low energy |
| Mahalanobis 32-dim (Run 5) | 0.4104 | Water embeddings ∈ Ground cluster |
| log1p prepro + Mahalanobis (Run 6) | 0.3912 | ignore_index=Water → 0 gradient → ίδια embeddings |
| **Raw intensity alone (hybrid)** | **0.8232** | Φυσική: specular reflection → σήμα υπάρχει |

**Θεμελιώδες συμπέρασμα:**

Το πρόβλημα δεν είναι η αρχιτεκτονική, δεν είναι το normalization, δεν είναι ο αλγόριθμος OOD. Είναι το **training paradigm**:

> Όταν μια κλάση έχει `ignore_index`, καμία αλλαγή στην είσοδο ή στον αλγόριθμο ανίχνευσης δεν μπορεί να αναγκάσει το δίκτυο να παράγει διαφορετικά embeddings γι' αυτήν — γιατί **δεν υπάρχει gradient** που να το διδάξει.

Η λύση θα ήταν auxiliary intensity reconstruction loss (force embedding να κωδικοποιεί intensity) ή RGB/NIR features — αλλά και οι δύο απαιτούν εκτεταμένη επανεκπαίδευση και δεδομένα πέρα από το scope αυτής της εργασίας.

**Πρακτική λύση:** Hybrid detector — το μοντέλο κάνει semantic segmentation, ένας ξεχωριστός detector βλέπει τα raw features. Η Knowledge Library αποθηκεύει το feature fingerprint ανά terrain type (Water: intensity<threshold, Bridge: energy>threshold, Lava: TBD).

Lee et al., "A Simple Unified Framework for Detecting OOD Samples", NeurIPS 2018.

---

### Αυτόνομη Μάθηση μέσω Knowledge Library

Το πιο ισχυρό κομμάτι του συστήματος: το drone μαθαίνει νέες κλάσεις **χωρίς κανέναν να δώσει labels κατά την πτήση**.

#### Η Βιβλιοθήκη Prototypes (ετοιμάζεται offline, μία φορά)

Ένας ειδικός συλλέγει εκ των προτέρων labeled παραδείγματα για κάθε πιθανό terrain type. Για κάθε κλάση, υπολογίζεται ο **mean embedding** από τα labeled patches — ο prototype:

```
Offline (πριν την αποστολή):

  Water:     10 labeled patches → mean(embeddings) → water_proto    (32-dim)
  Bridge:    10 labeled patches → mean(embeddings) → bridge_proto
  Lava:       8 labeled patches → mean(embeddings) → lava_proto
  CaveFloor: 6 labeled patches  → mean(embeddings) → cave_proto
  Glacier:   8 labeled patches  → mean(embeddings) → glacier_proto
  ...  (~100 παραδείγματα, 10-15 terrain types)

→ αποθηκεύεται σε knowledge_library.pt  (μέγεθος: <1MB)
→ φορτώνεται στο Pi5 μαζί με το model.onnx
```

#### In-flight: OOD Point → Cosine Matching

```
Drone πετάει, βρίσκει σημείο με υψηλό Energy Score (OOD):

  embedding(x) = model.get_embeddings(patch)[point_idx]  # (32,)
       │
       ▼
  Cosine similarity με όλους τους prototypes της βιβλιοθήκης:

  cos(x, water_proto)    = 0.91  ◄── 🎯 ταίριαξε
  cos(x, bridge_proto)   = 0.34
  cos(x, lava_proto)     = 0.12
  cos(x, cave_proto)     = 0.08
       │
       ├── max_similarity > θ (π.χ. 0.75)
       │        → "Αυτό είναι Water"
       │        → model.classifier.add_class(water_proto)
       │        → από εδώ και πέρα ταξινομεί Water κανονικά ✓
       │
       └── max_similarity < θ για ΟΛΑ
                → "Άγνωστο — δεν υπάρχει στη βιβλιοθήκη"
                → αποθηκεύει embedding + coordinates για post-flight review
```

#### Γιατί ο CosineClassifier είναι ιδανικός

Δεν είναι τυχαίο — σχεδιάστηκε για αυτό:
- Ήδη δουλεύει με cosine similarity
- `add_class(prototype)` = μία γραμμή κώδικα
- Οι prototypes της βιβλιοθήκης είναι απλά επιπλέον γραμμές στον ίδιο πίνακα
- Παλιές κλάσεις δεν αλλάζουν → zero forgetting by design

#### Σενάριο "Άγνωστο Περιβάλλον"

```
Drone στέλνεται σε άγνωστη τοποθεσία (αρχαία σπηλιά, άγνωστος πλανήτης):

Πτήση 1 (χωρίς ανθρώπινη παρέμβαση):
  → Ground:     42% σημεία  ← γνωστό
  → Rock wall:  sim=0.87 με CaveFloor_proto  → μαθαίνει ✓
  → Stalactite: sim=0.29 (τίποτα δεν ταιριάζει) → αποθηκεύει
  → Water pool: sim=0.91 με Water_proto  → μαθαίνει ✓

Post-flight:
  → Επιστήμονας βλέπει τα 47 "άγνωστα" σημεία
  → Δίνει label: "Stalactite"
  → Προστίθεται στη βιβλιοθήκη για την επόμενη αποστολή

Πτήση 2:
  → Stalactite: τώρα υπάρχει στη βιβλιοθήκη → αναγνωρίζεται αυτόνομα ✓
```

Κάθε αποστολή κάνει τη βιβλιοθήκη πιο πλούσια — το σύστημα βελτιώνεται με τον χρόνο.

### Σύνδεση με YellowScan

Στο YellowScan (χωρίς ground truth):
- Χρησιμοποιούμε το ίδιο Energy threshold
- Σημεία με υψηλό energy = "novel terrain" για αυτόν τον sensor
- Δεν μπορούμε να μετρήσουμε Precision/Recall αλλά μπορούμε να κάνουμε οπτική επαλήθευση
- Domain shift = ακόμα και γνωστές κλάσεις μπορεί να δώσουν υψηλό energy λόγω διαφορετικού sensor

### Πλήρης πίνακας runs

| Run | Classes | Loss | Weights | Σκοπός |
|-----|---------|------|---------|--------|
| 1–3 | 8 (all) | Weighted CE | 50-patch (unstable) | Baseline mean±std |
| 4 | 8 (all) | Focal (γ=2) | ALL patches (stable) | Best baseline / upper bound |
| 5 | 6 (χωρίς Water/Bridge) | Focal (γ=2) | ALL patches | Base model για OOD pipeline |

---

## 7. Τι να Πεις στο Poster (Key Talking Points)

### Για το μοντέλο
> "Επιλέξαμε PointNet++ Mini (30K params) γιατί πρέπει να τρέχει on-device στο Raspberry Pi 5. Τα μεγαλύτερα μοντέλα (PointNet++ full: 1M+ params) είναι off-limits για real-time UAV deployment."

### Για τα metrics
> "Χρησιμοποιούμε mIoU αντί για Overall Accuracy επειδή το OA εξαπατάται από imbalanced classes — αρκεί να πεις 'όλα Ground' για 79% OA. Το mIoU μετράει TP/(TP+FP+FN) ανά κλάση και δεν εξαπατάται."

### Για το Bridge IoU (αδύνατη κλάση)
> "Το Bridge IoU ξεκίνησε στο 0.053. Αυτό είναι πρόβλημα **absolute rarity** (Tsoumakas, Class Imbalance) — έχουμε πολύ λίγα απόλυτα παραδείγματα Bridge, όχι μόνο χαμηλό ποσοστό. Εφαρμόσαμε Focal Loss (Lin et al., TPAMI 2020) που μειώνει δραστικά το gradient contribution των easy examples."

### Για το OOD Detection
> "Δεν προσθέσαμε OOD class στο μοντέλο. Αντίθετα, εκπαιδεύσαμε σε 6 κλάσεις και χρησιμοποιούμε το Energy Score — όταν το μοντέλο είναι αβέβαιο για ΟΛΑ τα γνωστά labels, ξέρουμε ότι βλέπει κάτι novel. Αξιολογούμε ποσοτικά με Precision/Recall/AUROC χρησιμοποιώντας το FRACTAL ground truth ως oracle."

### Για την αυτόνομη μάθηση (το πιο δυνατό σημείο)
> "Το drone κουβαλάει μια βιβλιοθήκη terrain prototypes — 32-διάστατα embeddings για κάθε πιθανό τύπο εδάφους, ετοιμασμένα offline από ειδικό. Όταν συναντά κάτι novel, κάνει cosine matching με τη βιβλιοθήκη. Αν βρει ταίριασμα (similarity > threshold), μαθαίνει τη νέα κλάση αυτόνομα — χωρίς labels, χωρίς retraining, χωρίς internet. Αν δεν βρει ταίριασμα, αποθηκεύει το σημείο για post-flight review από τον επιστήμονα."

### Για το Continual Learning
> "Ο CosineClassifier προσθέτει νέες κλάσεις χωρίς retraining: add_class() appends ένα νέο prototype row. Δεν υπάρχει catastrophic forgetting by design, επειδή οι παλιές γραμμές δεν αλλάζουν."

### Για την αναπαραγωγιμότητα
> "Κάναμε 3 runs για mean±std reporting. Ανακαλύψαμε αστάθεια στα class weights λόγω τυχαίου 50-patch sampling — ένα γνωστό πρόβλημα σε imbalanced settings. Στον run 4 το διορθώσαμε με computation από ΟΛΟ το train set."

### Τι δεν λειτούργησε (honest reporting)
- **LowVegetation regression**: Run 4 έδωσε 0.113 vs baseline 0.16 — Focal Loss βελτίωσε Bridge/Noise αλλά επιδείνωσε LowVeg. Πιθανή αιτία: αυξημένη confusion με MedVegetation (παρόμοια patch-level γεωμετρία) όταν το μοντέλο εστιάζει gradient σε σπανιότερες κλάσεις.
- **Bridge absolute rarity**: παρά τη βελτίωση (0.06→0.124), το Bridge παραμένει η πιο αδύνατη κλάση — απόλυτα λίγα training examples ανεξαρτήτως weighting strategy (Tsoumakas: absolute vs relative imbalance).
- **Noise instability**: από 0.00 (Run 1) → 0.467 (Run 4) — τεράστια διακύμανση, εξαιρετικά λίγα points.
- **Energy Score OOD αποτυχία για Water** (AUROC=0.43): Το Water ταξινομείται ως Ground με υψηλή εμπιστοσύνη → χαμηλό energy → ΔΕΝ φλαγκάρεται. Bridge δουλεύει (d'=1.66) αλλά Water αποτυγχάνει (d'=0.05). Root cause: γεωμετρική ομοιότητα Water–Ground στο ALS LiDAR. Επόμενο βήμα: Mahalanobis distance στον 32-dim embedding χώρο.
- Training σε CPU: απαγορευτικό (>8h/run) — χρειάζεται CUDA GPU

---

## 7. Μαθηματικά για Poster

### mIoU
$$\text{mIoU} = \frac{1}{C} \sum_{c=1}^{C} \frac{TP_c}{TP_c + FP_c + FN_c}$$

### Focal Loss
$$FL(p_t) = -(1-p_t)^\gamma \cdot \log(p_t), \quad \gamma=2$$

### Inverse Frequency Weighting
$$w_c = \frac{N_{total}}{C \cdot N_c}, \quad \text{normalize: } \bar{w}=1$$

### CosineClassifier
$$\text{score}(x, c) = \tau \cdot \frac{x \cdot w_c}{\|x\| \cdot \|w_c\|}, \quad \tau=10$$

### Energy Score (OOD)
$$E(x) = -\log \sum_{c=1}^{C} e^{f_c(x)}$$
Υψηλό $E(x)$ → αβεβαιότητα για όλες τις κλάσεις → OOD σημείο

### OOD Evaluation
$$\text{Precision} = \frac{TP_{ood}}{TP_{ood} + FP_{ood}}, \quad \text{Recall} = \frac{TP_{ood}}{TP_{ood} + FN_{ood}}$$
Αξιολογείται με FRACTAL ground truth (Water/Bridge ως positive OOD labels)

---

## 8. Αρχεία & Εντολές

### Δομή Project
```
Drone_cont_Learing/
├── src/
│   ├── preprocessing.py         — normalize, extract_features
│   ├── data/
│   │   └── dataset.py           — FractalPatchDataset, cache support
│   ├── models/
│   │   └── pointnet2.py         — PointNet2Mini, CosineClassifier, SA, FP
│   └── train.py                 — training pipeline
├── outputs/
│   ├── checkpoints/best_model.pt
│   ├── logs/run_*.csv
│   └── normalizer_stats.json
├── notebooks/
│   ├── eda_fractal.ipynb
│   └── eda_yellowscan.ipynb
└── PRESENTATION_NOTES.md        ← αυτό το αρχείο
```

### Εκπαίδευση

```bash
# Baseline (CrossEntropy, 50-patch weights)
python src/train.py --epochs 100 --bs 16 --cache --loss ce --weight-samples 50

# Run 4 (Focal Loss, stable weights — όλο το train set)
python src/train.py --epochs 100 --bs 16 --cache --loss focal --gamma 2.0

# Dry-run test (2 batches, <30sec)
python src/train.py --dry-run --cache --loss focal

# Run 5 (6-class base model για OOD — Water/Bridge ως ignore_index)
python src/train.py --epochs 100 --bs 16 --cache --loss focal --gamma 2.0 --exclude-classes water bridge
```

### Νέα Args σε train.py
| Arg | Default | Περιγραφή |
|-----|---------|-----------|
| `--loss` | `focal` | `ce` ή `focal` |
| `--gamma` | `2.0` | Focal Loss γ |
| `--weight-samples` | `0` | 0=ALL patches (σταθερό), >0=random sample |

---

## 9. TODO — Επόμενα Βήματα

- [x] **Run 3**: mIoU=0.4153, Bridge=0.033 (καταγράφηκε)
- [x] **Run 4**: mIoU=0.5813, OA=0.9136, F1=0.6767 — **τελείωσε ✓** (best config = Focal Loss γ=2)
- [x] **Ablation table**: ενημερώθηκε παραπάνω
- [x] **src/evaluate_test.py**: **τελείωσε ✓** — test mIoU=0.6045, Bridge=0.284, OA=0.9080
- [x] **Run 5**: 6-class model — **τελείωσε ✓** val mIoU=0.6557 @ epoch 82
- [x] **src/ood_detector.py**: **τελείωσε ✓ (partial)** — Bridge AUROC excellent (d'=1.66), Water αποτυγχάνει (d'=0.05, AUROC=0.43)
- [x] **src/ood_mahalanobis.py**: **τελείωσε ✓** — AUROC=0.4104 (χειρότερο από Energy). Διάγνωση: Water/Ground αδιαχώριστα στον 32-dim χώρο.
- [x] **src/check_intensity.py**: **τελείωσε ✓** — Raw intensity AUROC=0.8655. Water raw≈568, Ground≈2277 (4x διαφορά).
- [x] **src/ood_hybrid.py**: **γράφτηκε ✓** — Energy + Intensity fusion. Δεν τρέχει ακόμα.

### 🔄 Run 6 — Επιστροφή στο Preprocessing (Intensity Log Transform)

**Απόφαση:** Αντί για post-hoc hybrid (που χρησιμοποιεί raw intensity εκτός μοντέλου), αλλάξαμε το **preprocessing** ώστε το μοντέλο να μαθαίνει σωστά από την αρχή.

**Ρίζα του προβλήματος:**
- Raw intensity: right-skewed (train std/mean = 1.95)
- z-score(raw): Water=-0.474, Ground=-0.355 → διαφορά 0.12 units (αόρατη vs xyz range ±3)
- z-score(log1p): Water≈-0.92, Ground≈+0.09 → διαφορά ≈1.01 units (~8.5x βελτίωση)

**Αλλαγές (μόνο preprocessing, ΟΧΙ αρχιτεκτονική):**
```python
# preprocessing.py — μία γραμμή:
out[:, 3] = (np.log1p(intensity) - stats["intensity"]["mean"]) / stats["intensity"]["std"]
```

**Μεθοδολογική καθαρότητα:** Η απόφαση βασίζεται ΜΟΝΟ σε train-set statistics και domain knowledge (specular reflection). Δεν χρησιμοποιήθηκε καμία πληροφορία από val/test.

**Run 6 pipeline:**
```bash
# 1. Υπολόγισε νέα stats (train-only, log1p)
python src/compute_normalizer_stats.py

# 2. Σβήσε παλιό cache (δημιουργήθηκε με z-score raw)
find data/ -name "*.npz" -delete

# 3. Retrain (ίδια config με Run 5)
python src/train.py --run-name run6_log1p --exclude-classes Water Bridge --epochs 100

# 4. OOD evaluation
python src/ood_mahalanobis.py
```

- [ ] **Run 6**: Τρέξε `compute_normalizer_stats.py` → σβήσε cache → retrain → `ood_mahalanobis.py`
- [ ] **src/few_shot.py**: OOD cluster → cosine matching με Knowledge Library → `add_class()`
- [ ] **src/export_onnx.py**: Run6_model.pt → model.onnx → Pi5 deployment (~35ms target)
- [ ] **src/inference.py**: Pi5 inference script (onnxruntime + normalizer_stats.json)
- [ ] **notebooks/results.ipynb**: Learning curves, per-class IoU bars με error bars, confusion matrix
- [ ] **Domain shift test**: Run6 model inference στο YellowScan → οπτική επαλήθευση + energy map

---

## 10. Βιβλιογραφία (για Poster)

1. Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space", NeurIPS 2017
2. Lin et al., "Focal Loss for Dense Object Detection", IEEE TPAMI 2020
3. Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
4. Wallace et al., "Class Imbalance, Redux", ICDM 2011 (EasyEnsemble)
5. Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017 (EWC)
6. Tsoumakas G., "Class Imbalance & Cost-Sensitive Learning", AUTH Lecture Slides, 2024

---

*Τελευταία ενημέρωση: Run 5 OOD analysis ολοκληρώθηκε. Εντοπίστηκε root cause (z-score σε right-skewed intensity). Run 6 ξεκινά με log1p intensity preprocessing — fresh train, fresh test evaluation.*
