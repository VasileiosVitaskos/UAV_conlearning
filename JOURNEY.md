# Engineering Journey
## Edge-Aware Continual Learning for Terrain Classification from UAV LiDAR Point Clouds

**Ομάδα:** Βασίλειος Βιτάσκος (ΑΕΜ 235) · Ορέστης Γεωργιάδης (ΑΕΜ 239) · Αστέριος Τερζής (ΑΕΜ 218)

> Αυτό το αρχείο καταγράφει τις πραγματικές μηχανικές αποφάσεις της εργασίας — τι επιλέξαμε, γιατί, τι δεν δούλεψε και πώς το διορθώσαμε. Η ερευνητική διαδικασία δεν είναι ποτέ γραμμική.

---

## 1. Επιλογή Dataset

### Γιατί FRACTAL
Αξιολογήσαμε πολλά LiDAR datasets για semantic segmentation:

| Dataset | Sensor | Labels | Κλάσεις | Πρόβλημα |
|---------|--------|--------|---------|----------|
| ISPRS Vaihingen | ALS | ✓ | 5 | Πολύ λίγες κλάσεις, παλιό |
| Toronto-3D | MLS (mobile) | ✓ | 8 | Urban only, δεν αφορά UAV |
| SensatUrban | ALS | ✓ | 13 | Πολύ μεγάλο, δύσκολο setup |
| **FRACTAL** | **ALS** | **✓** | **8** | **✓ UAV-σχετικό, καλά documented** |
| YellowScan | UAV LiDAR | ✗ | — | Unclassified — domain shift |

**FRACTAL** επιλέχθηκε γιατί:
1. Έχει **Bridge και Water** — κλάσεις με absolute rarity, ιδανικές για OOD testing
2. **50×50m patches** ευθυγραμμίζονται με πραγματικά UAV scan patterns
3. **LAS 1.4 format** με πλούσια point attributes (return info, scan angle)
4. Αρκετά μεγάλο (5000 patches) για train/val/test split με στατιστική σημασία

**YellowScan** επιλέχθηκε ως **domain shift test**: πραγματικό UAV LiDAR από διαφορετικό sensor, χωρίς labels. Αν το μοντέλο γενικεύει, θα δουλεύει και εκεί.

---

## 2. Χαρακτηριστικά: Τι Κρατήσαμε και Τι Απορρίψαμε

### RGB και NIR — Απόρριψη

Τα αρχεία FRACTAL περιέχουν κανάλια Red, Green, Blue και Near-Infrared (NIR) από co-registered κάμερα. Η αρχική ιδέα ήταν να τα χρησιμοποιήσουμε ως επιπλέον features.

**Γιατί τα αφαιρέσαμε:**

**α) Inconsistency:** Όχι όλα τα patches έχουν RGB/NIR. Σε αρκετά files οι τιμές είναι 0 ή NaN. Ένα μοντέλο που εξαρτάται από RGB θα αποτύγχανε σιωπηρά σε τέτοια patches.

**β) Sensor dependency:** Το RGB εξαρτάται από:
- Ώρα της ημέρας (σκιές, φωτεινότητα)
- Καιρικές συνθήκες (σύννεφα, ομίχλη)
- Κάμερα calibration

Για real-world UAV deployment θέλουμε ένα σύστημα που δουλεύει και τη νύχτα, και σε overcast ουρανό. Το LiDAR εκπέμπει το δικό του φως — δεν εξαρτάται από εξωτερική φωτεινότητα.

**γ) Domain shift:** Το YellowScan δεν έχει RGB/NIR. Αν εκπαιδεύαμε με αυτά τα features, το μοντέλο δεν θα μπορούσε να κάνει inference στο YellowScan — ακυρώνοντας το domain shift test.

**δ) NDVI παγίδα:** Ο Normalized Difference Vegetation Index (NDVI = (NIR-R)/(NIR+R)) είναι χρήσιμος για vegetation classification. Αλλά αν το μοντέλο "μαθαίνει NDVI", γίνεται εξαρτημένο από multispectral sensors — όχι όλα τα UAV LiDAR systems έχουν αυτά.

**Τελική απόφαση:** Κρατάμε μόνο **γεωμετρικά + waveform features** που είναι universally available σε οποιοδήποτε LiDAR:

```
x_rel, y_rel, z_rel      — γεωμετρία (relative εντός patch)
intensity                — ανακλαστικότητα επιφάνειας
return_number            — ποια επιστροφή σήματος είναι αυτή
number_of_returns        — πόσες επιστροφές συνολικά
scan_angle               — γωνία σάρωσης
```

---

### Γιατί Relative Coordinates

Αρχικά χρησιμοποιούσαμε **absolute** coordinates (UTM easting/northing/elevation). Πρόβλημα: ο classifier έμαθε να αναγνωρίζει terrain βάσει **τοποθεσίας**, όχι **γεωμετρίας**.

Παράδειγμα: σε μια περιοχή, τα buildings βρίσκονται πάντα στο [X=500000, Y=4200000]. Σε νέα άγνωστη περιοχή, αυτή η "γνώση" είναι άχρηστη.

**Λύση:** x_rel = x - mean(x_patch), κτλ. Το μοντέλο μαθαίνει **σχήμα** εντός patch, όχι absolute position. Αυτό επιτρέπει generalization σε νέες τοποθεσίες (π.χ. YellowScan).

---

### Γιατί Return Information

Το `return_number` και `number_of_returns` κωδικοποιούν **penetration depth**:

```
Ακτίνα laser → δέντρο → επιστροφή 1 (κορυφή)
                       → επιστροφή 2 (κλαδί)
                       → επιστροφή 3 (έδαφος)
```

- **Ground:** συνήθως last return (laser penetrates vegetation)
- **HighVegetation:** first return
- **Bridge:** τυπικά single return (solid surface)
- **Water:** weak/no return (absorption) ή μόνο specular reflection

Αυτά τα patterns βοηθούν στη διαφοροποίηση κλάσεων που έχουν παρόμοια γεωμετρία αλλά διαφορετικές ιδιότητες επιφάνειας.

---

## 3. Class Handling — Απόφαση για 8 Κλάσεις

### Αρχικό Πρόβλημα: Πολλές Κλάσεις στα Raw Data
Το FRACTAL χρησιμοποιεί LAS classification codes. Βρήκαμε 15+ διαφορετικούς κωδικούς, αλλά πολλοί αντιστοιχούν σε artifacts ή σπάνια terrain types με <100 points total.

**Απόφαση:** Κρατάμε 8 semantically meaningful κλάσεις, merge τα rest ως "Noise" ή αγνοούμε:

```python
VALID_CLASSES = {2, 3, 4, 5, 6, 9, 17, 64}
# 2=Ground, 3=LowVeg, 4=MedVeg, 5=HighVeg, 6=Building,
# 9=Water, 17=Bridge, 64=Noise
```

### Το LowVegetation/MedVegetation Πρόβλημα
Αυτές οι δύο κλάσεις έχουν **ασαφή σύνορα** στο FRACTAL ground truth. Στα patch boundaries, ένα shrub μπορεί να labeleαριστεί ως LowVeg σε ένα patch και MedVeg στο γειτονικό. Αυτό δημιουργεί label noise.

**Αποτέλεσμα:** LowVegetation IoU ~0.16 ακόμα και με καλό training. Δεν είναι failure του μοντέλου — είναι **label inconsistency** στα δεδομένα.

**Honest reporting:** Αναφέρουμε αυτό ρητά στα αποτελέσματα. Ένα confusion matrix δείχνει ότι τα errors είναι κυρίως LowVeg↔MedVeg (γειτονικές κλάσεις), όχι Ground↔Building (ασύνδετες).

---

## 4. Normalization — Data Leakage Παγίδα

### Αρχική Λανθασμένη Προσέγγιση
Πρώτη εκδοχή: fit StandardScaler σε **ΟΛΟ** το dataset (train+val+test) και μετά split.

**Γιατί λάθος:** Ο scaler "βλέπει" statistics από τα test patches κατά το fit. Όταν αξιολογούμε στο test set, το μοντέλο έχει πλεονέκτημα — έχει προσαρμοστεί (μέσω normalization) στα test statistics.

**Λύση:** `partial_fit` μόνο στα 3000 training files. Τα ίδια stats (mean, std) εφαρμόζονται ως **transform** (όχι fit) σε val/test/YellowScan:

```python
# ΣΩΣΤΟ:
scaler.partial_fit(train_data)      # fit μόνο στο train
scaler.transform(val_data)          # apply stats, δεν ξαναμαθαίνει
scaler.transform(yellowscan_data)   # YellowScan: ΊΔΙΑ stats από FRACTAL train
```

**Αντίκτυπος στο YellowScan:** Χρησιμοποιούμε FRACTAL train statistics για normalize το YellowScan — ακόμα και αν οι distributions διαφέρουν. Αυτό **σκόπιμα** αφήνει ορατό το domain shift αντί να το κρύβει.

---

## 5. Class Imbalance — Τα Πραγματικά Προβλήματα

### Ανακάλυψη: Absolute Rarity
Μετά τα baseline runs, το Bridge IoU ήταν 0.033–0.093 με μεγάλη διακύμανση:

| Run | mIoU | Bridge IoU | Bridge pts (50-patch sample) |
|-----|------|-----------|------------------------------|
| 1 | 0.4568 | 0.053 | 54 |
| 2 | 0.4882 | 0.093 | 467 |
| 3 | 0.4153 | 0.033 | 23 |

Αρχικά υποθέσαμε ότι το πρόβλημα ήταν **relative rarity** (Bridge = μικρό ποσοστό). Η ανάλυση έδειξε ότι είναι **absolute rarity** (Tsoumakas, Class Imbalance slides): λίγα απόλυτα παραδείγματα Bridge σε ολόκληρο το dataset → η δειγματοληψία 50 patches δίνει αστάθεια 20× στο Bridge weight.

---

### Αποτυχία 1: Focal Loss + Extreme Class Weights

**Σχέδιο:** Focal Loss (γ=2) + stable weights από ΟΛΑ τα 3000 training patches.

**Αποτέλεσμα:**
```
Epoch 1: OA = 0.006  (αναμενόμενο ~0.20)
Epoch 8: OA = 0.020  (ακόμα catastrophic)
```

**Ανάλυση αιτίας:** Τα weights από ΟΛΑ τα patches:
```
Ground:  5,643,239 pts → weight = 0.005
Noise:       5,044 pts → weight = 6.112
```

Ο συνδυασμός Focal Loss × inverse-frequency weight για Ground:
```
loss_Ground_correct = (1-0.9)² × 0.005 × CE
                    = 0.01 × 0.005 × 0.105
                    = 0.0000053
```
vs Bridge:
```
loss_Bridge_wrong   = (1-0.1)² × 1.555 × CE
                    = 0.81 × 1.555 × 2.303
                    = 2.90
```
**Αναλογία gradient: 548,000:1** — το μοντέλο δεν μαθαίνει τίποτα από το 46% των δεδομένων.

**Ο λανθασμένος συλλογισμός μας:** "Focal Loss + class weights = double protection". Στην πραγματικότητα: **Focal Loss δεν συνδυάζεται με extreme inverse-frequency weights**. Ο Lin et al. (TPAMI 2020) δεν χρησιμοποιεί class weights στο πρωτότυπο paper.

---

### Αποτυχία 2: Cap στο Weight (max=3.0)

**Σχέδιο:** Clamp Noise weight από 6.112 → 3.0.

**Αποτέλεσμα:** Ίδιο πρόβλημα.

**Γιατί δεν αρκεί:**

Το cap επηρέασε μόνο το Noise. Ground παρέμεινε 0.005:
```
ratio = Bridge_gradient / Ground_gradient
      = (0.81 × 1.555) / (0.01 × 0.005)
      = 25,200:1
```

Το Ground εξακολουθεί να μην υπάρχει στο gradient. Ο cap δεν λύνει το θεμελιώδες πρόβλημα: **normalization βασισμένη σε extreme Noise count κάνει το mean να εκτοξευθεί → Ground weight → 0**.

---

### Λύση: Focal Loss Χωρίς Class Weights

Ακολουθούμε τον αρχικό ορισμό: Focal Loss μόνο, weight=None.

**Ακαδημαϊκή αιτιολόγηση:** "The focal mechanism (1-p_t)^γ inherently downweights well-classified majority classes and focuses on hard minority examples. Combining it with inverse-frequency weighting causes gradient collapse at imbalance ratios >1000:1, as the compound effect renders majority class contributions negligible. Following Lin et al. (TPAMI 2020), we use Focal Loss without class reweighting."

**Run 4:** Σε εξέλιξη — αναμένουμε αποτελέσματα.

---

## 6. Αρχιτεκτονικές Αποφάσεις

### FPS vs Random Sampling

Το Farthest Point Sampling (FPS) επιλέγει **χωρικά ομοιόμορφα** centroids — δεν bias-άρει προς πυκνές περιοχές όπου έχει πέσει περισσότερο laser. Θεωρητικά ανώτερο.

**Πρακτικό πρόβλημα:** FPS Python loop: O(N²) → ~96ms για N=4096 σε CPU. Για Pi5 deployment: απαγορευτικό.

**Απόφαση για training:** Κρατάμε FPS (μαθαίνει καλύτερα features).
**Απόφαση για deployment (Pi5):** Random sampling — 3× γρηγορότερο, μικρή απώλεια ακρίβειας.

Αυτό είναι ένα **train/inference mismatch** — γνωστό πρόβλημα σε production ML. Θα αξιολογηθεί πόση απώλεια φέρνει στο final benchmark.

---

### CosineClassifier: Σχεδιαστική Απόφαση για Zero Forgetting

Standard linear classifier: `score = W·x + b`. Για να προστεθεί νέα κλάση, πρέπει να προστεθεί γραμμή στο W και να γίνει retraining — διαφορετικά catastrophic forgetting.

CosineClassifier: `score = τ · cos(x, w_c)`. Το add_class() προσθέτει γραμμή στον πίνακα **χωρίς να αλλάζει τις υπάρχουσες**. Zero forgetting **by mathematical design**, όχι ως heuristic.

**Trade-off:** Το temperature parameter τ=10 χρειάστηκε tuning. Μικρό τ → predictions too soft, μεγάλο τ → overconfident. τ=10 από τη βιβλιογραφία (Prototypical Networks, Snell et al. NeurIPS 2017).

---

## 7. OOD Pipeline — Σχεδιαστική Εξέλιξη

### Πρώτη Ιδέα: Clustering (DBSCAN)
Αυτόνομη ανακάλυψη κλάσεων μέσω clustering των OOD embeddings → κάθε cluster = νέα κλάση.

**Πρόβλημα:** DBSCAN χρειάζεται hyperparameter tuning (ε, min_samples). Σε 32-dim embedding space, η επιλογή ε δεν είναι self-evident. Επίσης: δεν υπάρχει semantic meaning στα clusters — δεν ξέρουμε τι είναι.

### Τελική Προσέγγιση: Knowledge Library Matching
Ο άνθρωπος ετοιμάζει εκ των προτέρων labeled prototypes (offline). In-flight: cosine matching με τη βιβλιοθήκη.

**Γιατί καλύτερα:**
- Semantically meaningful (κάθε prototype έχει label)
- Δεν χρειάζεται hyperparameter tuning
- Scalable: η βιβλιοθήκη μεγαλώνει σε κάθε αποστολή
- Ο άνθρωπος μπαίνει μόνο για naming, όχι για labeling

**Ποσοτική αξιολόγηση:** Χρησιμοποιούμε FRACTAL ground truth ως oracle: Water/Bridge labels ως "ground truth OOD positives" → Precision/Recall/AUROC.

---

## 8. OOD Detection — Honest Post-Mortem

### Energy Score: Γιατί Αποτύχαμε για Water

Υλοποιήσαμε το Energy Score (Liu et al., NeurIPS 2020) στο Run 5 μοντέλο (6-class):

```
E(x) = -log Σ_c exp(f_c(x))     υψηλό E → model αβέβαιο → OOD
```

**Αποτελέσματα val:**
- Bridge: d'=1.66 → excellent separation ✓
- Water: d'=0.05 → completely failed ✗
- Aggregate AUROC test: 0.4321 (χειρότερο από random)

### Root Cause Analysis

Η ανάλυση ανά κλάση αποκάλυψε το πρόβλημα:

```
Mean Energy ανά κλάση (val):
  Ground:        -4.547   ← model πολύ confident
  HighVeg:       -4.612   ← πολύ confident
  Water:         -4.420   ← σχεδόν ίδιο με Ground (d'=0.05)
  Building:      -3.321   ← λιγότερο confident
  Noise:         -3.264   ← αβέβαιο
  Bridge:        -2.815   ← σαφώς αβέβαιο ← OOD ανιχνεύεται
```

Το Water έχει **μικρότερο energy από Building και Noise** — άρα με οποιοδήποτε threshold που πιάνει Water, πιάνουμε πρώτα αυτές τις γνωστές κλάσεις. AUROC < 0.5 = η σειρά κατάταξης είναι αντίστροφη.

**Φυσική εξήγηση:** Στο ALS LiDAR, το Water εμφανίζεται ως flat surface με single returns και moderate intensity — σχεδόν πανομοιότυπο με alluvial/sandy Ground. Το μοντέλο "βλέπει" Ground και είναι σίγουρο γι' αυτό. Ένα εντελώς λογικό λάθος.

**Το Bridge είναι διαφορετικό:** Το Bridge είναι elevated structure με γεωμετρία που δεν μοιάζει με καμία γνωστή κλάση → model αβέβαιο → υψηλό energy → ανιχνεύεται σωστά.

### Τι Μαθαίνουμε

Η αποτυχία έχει μια ακαδημαϊκά ενδιαφέρουσα ερμηνεία:

> **Το Energy Score ανιχνεύει γεωμετρική novel-ness, όχι semantic novel-ness.**

Ένα class που το μοντέλο "βλέπει" ως γνωστή κλάση (ακόμα κι αν είναι λάθος) έχει χαμηλό energy.
Ένα class που το μοντέλο "δεν ξέρει πού να βάλει" έχει υψηλό energy.

Water είναι γεωμετρικά παρόμοιο με Ground → model θεωρεί "ξέρω τι είναι" → χαμηλό energy → ΔΕΝ ανιχνεύεται.

### Επόμενο: Mahalanobis Distance

Αντί να χρησιμοποιούμε τα logits (6-dim projection), χρησιμοποιούμε το **πλήρες 32-dim embedding space**.

```
OOD_score(z) = min_c √[(z - μ_c)ᵀ Σ⁻¹ (z - μ_c)]
```

Η υπόθεση: τα Water embeddings έχουν ίσως παρόμοια **κατεύθυνση** με Ground (ίδιο cosine similarity) αλλά ίσως διαφορετική **θέση/κλίμακα** στον 32-dim χώρο. Αν ναι, η Mahalanobis θα τα ανιχνεύσει ακόμα κι αν το Energy Score αποτυγχάνει.

Αν και αυτό αποτύχει, αποδέχεται ότι Water/Ground δεν είναι χωριστές με αυτό το feature set — και αυτό είναι επίσης σημαντικό εύρημα για το thesis.

---

## 9. Run 6 — Επιστροφή στο Preprocessing (Intensity Log Transform)

### Το Πρόβλημα που Ανακαλύφθηκε

Μετά την OOD post-mortem (Section 8) και το analysis του `check_intensity.py`, είδαμε ότι:

- Water raw intensity ≈ 568, Ground ≈ 2277 — **4x διαφορά στο raw space**
- Μετά το global z-score normalization (mean=7340.9, std=14285.3): διαφορά = **0.12 normalized units**
- Αυτό είναι αόρατο σε σχέση με το XYZ range (±3 units) που κυριαρχεί στο PointNet++ ball query

Η κατανομή του raw intensity είναι **heavily right-skewed** (std/mean = 1.95, υπολογισμένο από train set). Το z-score σε right-skewed distribution συμπιέζει τα χαμηλά values — ακριβώς εκεί που ζει το Water.

### Η Λύση: log1p Transform

**Αλλαγή: μία γραμμή στο `preprocessing.py`**

```python
# Run 1-5 (παλιό):
out[:, 3] = (intensity - stats["intensity"]["mean"]) / stats["intensity"]["std"]

# Run 6+ (νέο):
out[:, 3] = (np.log1p(intensity) - stats["intensity"]["mean"]) / stats["intensity"]["std"]
```

**Αποτέλεσμα στο normalized space:**

| Method | Water norm | Ground norm | Διαφορά |
|--------|-----------|-------------|---------|
| z-score(raw) | -0.474 | -0.355 | 0.119 |
| z-score(log1p) | ≈ -0.92 | ≈ +0.09 | ≈ 1.01 |

**~8.5x μεγαλύτερο gap** — το μοντέλο τώρα "βλέπει" ότι Water είναι spectrally διαφορετικό.

### Γιατί Αυτή η Απόφαση Είναι Μεθοδολογικά Καθαρή

Η επιλογή log transform βασίζεται **αποκλειστικά** σε:
1. **Train set statistics**: skewness std/mean=1.95 — γνωστό ήδη πριν οποιοδήποτε val/test evaluation
2. **Φυσική**: specular reflection → low intensity για Water — domain knowledge, όχι data snooping
3. **Βιβλιογραφία**: log normalization για LiDAR intensity είναι standard practice στο remote sensing

Δεν χρησιμοποιήθηκε καμία πληροφορία από val/test για αυτή την απόφαση.

### Τι Άλλαξε στο Pipeline

1. **`src/compute_normalizer_stats.py`** (νέο): υπολογίζει log1p stats από train set, αποθηκεύει στο `normalizer_stats.json`. Το παλιό αρχείο σώθηκε ως `normalizer_stats_v1_zscoreraw.json`.
2. **`src/preprocessing.py`**: αλλαγή μίας γραμμής (intensity → log1p).
3. **Cache invalidation**: τα 5000 `.npz` cache files δημιουργήθηκαν με τα παλιά stats και πρέπει να σβηστούν: `find data/ -name "*.npz" -delete`
4. **Run 6**: fresh training με νέο normalization, χωρίς να φορτωθεί Run 5 checkpoint (το distribution έχει αλλάξει).

### Αναμενόμενο Αποτέλεσμα

### Run 6 — Αποτέλεσμα (τελική απάντηση)

| | Run 5 | Run 6 (log1p) |
|---|---|---|
| Val mIoU | 0.6557 | 0.6378 |
| Mahalanobis AUROC (test) | 0.4104 | **0.3912** |

Η υπόθεση ήταν **λάθος**. Το log1p δεν βοήθησε — και ο λόγος είναι θεμελιώδης:

```
Training loss = CrossEntropy(known classes, ignore_index=Water)
                                            ↑
                Water → 0 gradient → optimizer ποτέ δεν ενημερώνεται από Water points
                → embeddings Water ≡ Ground ανεξάρτητα από preprocessing
```

Το πρόβλημα δεν είναι η αρχιτεκτονική ούτε το normalization. Είναι το **training paradigm**: όταν μια κλάση έχει `ignore_index`, κανένα preprocessing, κανένα μοντέλο, καμία normalization δεν μπορεί να κάνει το δίκτυο να "μάθει" να την αναγνωρίζει χωρίς explicit training signal.

Θα μπορούσαμε να προσθέσουμε **auxiliary intensity reconstruction loss** — αλλά αυτό είναι μια προσωρινή λύση. Αύριο θα χρειαστεί reconstruction loss για άλλο feature, μεθαύριο για άλλο. Δεν κλιμακώνεται.

Η **πρακτικά σωστή λύση** είναι ο hybrid detector:
- Το μοντέλο κάνει τη δουλειά του: semantic segmentation για γνωστές κλάσεις
- Ο OOD detector βλέπει τα raw features directly, χωρίς να περνά από το μοντέλο
- Η Knowledge Library αποθηκεύει το feature fingerprint ανά τύπο εδάφους

**Πλήρης σύνοψη OOD investigation:**

| Μέθοδος | Test AUROC | Γιατί ναι/όχι |
|---------|-----------|---------------|
| Energy Score | 0.4321 | Water ≈ Ground geometrically |
| Mahalanobis (Run 5) | 0.4104 | Water ∈ Ground cluster στον 32-dim χώρο |
| log1p + Mahalanobis (Run 6) | 0.3912 | ignore_index → 0 gradient → ίδιο αποτέλεσμα |
| Raw intensity (hybrid) | **0.8232** | Φυσική: specular reflection → σήμα υπάρχει, δεν χρειάζεται μοντέλο |

**Τι θα χρειαζόταν για model-based OOD:** RGB/NIR features (specular reflection ορατό σε NIR channel) + YellowScan labeled data. Και τα δύο απαιτούν χρόνο πολύ πέρα από το scope αυτής της εργασίας.

**Συμπέρασμα:** Για edge deployment σε Pi5, ο hybrid detector (Energy + raw intensity) με AUROC=0.82 είναι το **βέλτιστο εφικτό αποτέλεσμα** με τα διαθέσιμα δεδομένα και αρχιτεκτονική.

---

## 10. Run 7 — XYZ Feature Dropout Fine-Tuning

**Ημερομηνία:** Ιούνιος 2026  
**Υπόθεση:** Αν εκπαιδεύσουμε το μοντέλο να λειτουργεί χωρίς xyz features (zeroing xyz dims με p=0.20), θα αναγκαστεί να ενσωματώσει intensity/returns στον encoder — και τότε τα Water embeddings θα ξεχωρίζουν από τα Ground embeddings στον 32-dim χώρο.

**Αρχιτεκτονική αλλαγή:** `pointnet2.py` forward() — δύο ξεχωριστά tensors:
- `xyz` (για FPS/KNN/grouping) — **πάντα άθικτο**
- `x_feat` (για MLP features) — μηδενίζεται 20% per batch κατά training

**Training:**
```
python src/train.py --exclude-classes Water Bridge --epochs 60 --cache --bs 32 \
    --lr 1e-4 --xyz-dropout 0.20 --resume-from outputs/checkpoints/best_model_run6.pt
```
Fine-tuning από Run 6 checkpoint (epoch=91, mIoU=0.6378), όχι από scratch.

**Αποτελέσματα Run 7:**

| Metric | Run 6 | Run 7 | Δ |
|--------|-------|-------|---|
| Best val mIoU | 0.6378 | **0.6412** | +0.34% |
| OA | 0.916 | 0.916 | ≈ ίδιο |
| Water AUROC (Mahalanobis) | 0.3912 | 0.3838 | **-0.74%** |

Per-class IoU (Run 7 val, best checkpoint epoch 36):

| Κλάση | IoU |
|-------|-----|
| Ground | 0.885 |
| LowVegetation | 0.123 |
| MedVegetation | 0.474 |
| HighVegetation | 0.944 |
| Building | 0.812 |
| Noise | 0.528 |

**Τι Συνέβη:**

Η classification απόδοση διατηρήθηκε (εν μέρει και βελτιώθηκε), αλλά το Mahalanobis AUROC για Water έγινε χειρότερο. 

Γιατί: Κοιτώντας τα scores — Water mean=4.201, Ground mean=3.854. Η Water συνεχίζει να ζει μέσα στον Ground cluster.

**Root cause (οριστική επιβεβαίωση):** Το XYZ dropout αλλάζει *πώς* ο encoder δομεί τον embedding space, αλλά δεν αλλάζει ότι Water έχει `ignore_index=-1` → 0 gradient → καμία πληροφορία για το πού πρέπει να τοποθετηθεί η Water στον 32-dim χώρο. Ο encoder βελτίωσε την intensity awareness για τις 6 training classes, αλλά η Water παραμένει unshapen.

**Τελική OOD Σύνοψη (όλες οι μέθοδοι):**

| Μέθοδος | Test AUROC | Γιατί ναι/όχι |
|---------|-----------|---------------|
| Energy Score | 0.4321 | Water ≈ Ground geometrically |
| Mahalanobis (Run 5) | 0.4104 | Water ∈ Ground cluster (32-dim) |
| log1p + Mahalanobis (Run 6) | 0.3912 | ignore_index → 0 gradient, preprocessing irrelevant |
| XYZ dropout + Mahalanobis (Run 7) | 0.3838 | ίδιο root cause, μικρή επιδείνωση |
| **Raw intensity (hybrid)** | **0.8232** | Φυσική: specular reflection → 8.5x gap μετά log1p |

**Συμπέρασμα:** Ο μόνος τρόπος για model-based Water OOD είναι να **συμπεριληφθεί η Water στο training** (few-shot incremental learning με LwF) ή να έχουμε labeled data για αυτή τη κλάση. Με ignore_index approach, καμία architectural αλλαγή δεν λύνει το πρόβλημα.

**Χρήση για deployment:** Run 7 model (mIoU=0.6412) + hybrid OOD (AUROC=0.8232).

---

## 11. Τι Θα Κάναμε Διαφορετικά

1. **Feature selection πρώτα, μοντέλο μετά:** Αφιερώσαμε χρόνο σε architecture πριν επιβεβαιωθεί ότι τα 7 features αρκούν. EDA πρώτα, σχεδιασμός μετά.

2. **Class weights validation:** Πριν οποιοδήποτε training, να υπολογίσουμε την αναλογία max/min weight — αν > 100:1, inverse-frequency δεν είναι κατάλληλο.

3. **Ablation από την αρχή:** Focal Loss ΜΟΝΟ vs CE ΜΟΝΟ vs CE+weights — τρία ξεχωριστά runs από την πρώτη εβδομάδα, όχι αφού υπάρχουν 3 baseline runs.

4. **FRACTAL label quality audit:** Πριν training, να ελέγξουμε τα LowVeg/MedVeg boundaries χειροκίνητα σε ένα sample patches — αν τα labels είναι noisy, mIoU ceiling είναι χαμηλότερα από ό,τι νομίζουμε.

---

*"The failures tell you more about your assumptions than the successes tell you about your method."*
