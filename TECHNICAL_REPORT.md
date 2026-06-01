# Edge-Aware Continual Learning for Terrain Classification from UAV LiDAR Point Clouds
## Πλήρες Τεχνικό Έγγραφο

**Ομάδα:** Βασίλειος Βιτάσκος (235) · Ορέστης Γεωργιάδης (239) · Αστέριος Τερζής (218)

---

## Πίνακας Περιεχομένων

1. [Το Πρόβλημα — Αναλυτικά](#1-το-πρόβλημα--αναλυτικά)
2. [Dataset — FRACTAL](#2-dataset--fractal)
3. [Το Κεντρικό Ερώτημα: Heuristics vs Learning](#3-το-κεντρικό-ερώτημα-heuristics-vs-learning)
4. [Αρχιτεκτονική Μοντέλου](#4-αρχιτεκτονική-μοντέλου)
5. [Open-World Continual Learning Pipeline](#5-open-world-continual-learning-pipeline)
6. [Fourier Prototypes — Η Πρωτότυπη Συνεισφορά](#6-fourier-prototypes--η-πρωτότυπη-συνεισφορά)
7. [Edge Deployment στο Raspberry Pi](#7-edge-deployment-στο-raspberry-pi)
8. [Πλήρης Εκπαίδευση Step-by-Step](#8-πλήρης-εκπαίδευση-step-by-step)
9. [Evaluation Protocol](#9-evaluation-protocol)
10. [Αντικριτική — Τι θα πουν οι κριτές](#10-αντικριτική--τι-θα-πουν-οι-κριτές)
11. [Roadmap Κώδικα](#11-roadmap-κώδικα)

---

## 1. Το Πρόβλημα — Αναλυτικά

### 1.1 Το σενάριο

Ένα αυτόνομο drone με LiDAR πετάει πάνω από άγνωστο έδαφος. Έχει εκπαιδευτεί να αναγνωρίζει **Ground**, **Vegetation**, **Building**. Ξαφνικά συναντά **νερό** — μια κλάση που δεν έχει ξαναδεί.

Τα ερωτήματα που πρέπει να απαντηθούν:

1. **Ανίχνευση:** Πώς ξέρει το μοντέλο ότι αυτό που βλέπει είναι *άγνωστο*;
2. **Μάθηση:** Πώς μαθαίνει τη νέα κλάση με ελάχιστα παραδείγματα;
3. **Διατήρηση:** Πώς δεν "ξεχνάει" τις παλιές κλάσεις;
4. **Edge:** Πώς όλα αυτά τρέχουν σε Raspberry Pi χωρίς να αδειάζει η μπαταρία;

### 1.2 Γιατί είναι δύσκολο

```
Catastrophic Forgetting
    ↑
    │  Νέα κλάση (Water) → update weights → Ground accuracy ↓↓
    │
    └─── Αυτό δεν είναι θεωρητικό: έχει μετρηθεί -40% mIoU σε παλιές κλάσεις
         μετά από naive fine-tuning (McCloskey & Cohen, 1989 · Kirkpatrick et al., 2017)

Open-Set Recognition
    ↑
    │  Softmax πάντα δίνει μια κλάση, ακόμα και σε OOD input
    │  Confidence του Water που κατηγοριοποιείται ως Ground: ~65%  ← ψεύτικη σιγουριά
    │
    └─── Closed-world assumption: το μοντέλο δεν ξέρει ότι "δεν ξέρει"

Few-Shot Learning
    ↑
    │  Στο πεδίο, δεν έχεις 10.000 labeled Water points
    │  Έχεις ίσως 50-200 points από ένα πέρασμα πάνω από λίμνη
    │
    └─── Standard DL χρειάζεται χιλιάδες παραδείγματα
```

### 1.3 Γιατί το LiDAR είναι ιδιαίτερο

Σε αντίθεση με RGB κάμερες, το LiDAR παρέχει **φυσική πληροφορία** για κάθε κλάση:

| Κλάση | Intensity | NIR | Return # | Z variance | Φυσική ερμηνεία |
|-------|-----------|-----|----------|------------|-----------------|
| Ground | μέτριο 2000-4000 | μέτριο | 1 (κυρίως) | χαμηλό | Diffuse reflection |
| Vegetation | μέτριο-υψηλό | **πολύ υψηλό** | **1-5** | **πολύ υψηλό** | Multiple canopy returns |
| Building | **υψηλό** | χαμηλό | 1 | μέτριο | Specular/diffuse roof |
| **Water** | **πολύ χαμηλό** | **πολύ χαμηλό** | **1 μόνο** | **~0** | Specular, absorbs IR |
| Bridge | υψηλό | χαμηλό | 1 | χαμηλό | Metal/concrete |
| Perm. Struct | υψηλό | χαμηλό | 1 | χαμηλό | Rigid material |

Αυτή η φυσική δομή είναι **inductive bias** που μπορούμε να εκμεταλλευτούμε.

---

## 2. Dataset — FRACTAL

### 2.1 Βασικά χαρακτηριστικά

```
Dataset:     FRACTAL (IGNF / Institut Géographique National, France)
Αρχεία:      100,000 patches LAZ 1.4 format
Μέγεθος:     81.6 GB συνολικά
Patch size:  50 × 50 m
Πυκνότητα:   ~40 pts/m² (10 pulses/m²)
Avg points:  ~150,000 points/patch
Split:       80,000 train / 10,000 val / 10,000 test
```

### 2.2 Πεδία ανά point (22 dimensions)

```
Γεωμετρία:
  X, Y, Z                    → συντεταγμένες Lambert 93 (EPSG:2154), float64

LiDAR attributes:
  intensity                  → reflectance 0-65535, uint16
  return_number              → 1η, 2η, ... επιστροφή, uint8
  number_of_returns          → πόσα returns ανά pulse, uint8
  scan_angle                 → γωνία sensor (-180 έως 180°), int16
  gps_time                   → timestamp, float64

LAS metadata (δεν χρησιμοποιούνται για ML):
  synthetic, key_point, withheld, overlap
  scanner_channel, scan_direction_flag
  edge_of_flight_line, user_data, point_source_id

Χρώμα από αεροφωτογραφία ORTHO HR® (0.20m res, IRGB):
  red, green, blue           → uint16
  nir                        → Near-Infrared, uint16  ← πολύτιμο για vegetation

Ground truth:
  classification             → uint8
```

### 2.3 Κλάσεις και κατανομή

```
ID  Κλάση                  Train%   Σημασία για μας
──────────────────────────────────────────────────────
 1  Unclassified            0.6%    αγνοούμε
 2  Ground                 39.0%    BASE CLASS ✓
 5  Vegetation             57.0%    BASE CLASS ✓
 6  Building                2.8%    BASE CLASS ✓
 9  Water                   0.5%    NOVEL CLASS (Task 1)
17  Bridge                  0.1%    NOVEL CLASS (Task 2)
64  Permanent Structure     0.04%   NOVEL CLASS (Task 3)

Σημείωση: Classes 3, 4 (vegetation variants) → merge to 5
          Class 65 (artifacts) → αγνοούμε
```

### 2.4 Γιατί είναι ιδανικό για Continual Learning

Το FRACTAL δεν έχει spatial domain labels στα αρχεία ονόματα (train-00.zip έως train-79.zip), αλλά οι 5 **γεωγραφικές περιοχές** είναι εγγενώς διαφορετικές:

```
Domain 1: Νότια Γαλλία — πεδινά, αγροτικά
Domain 2: Κεντρική Γαλλία — μικτό
Domain 3: Παράκτιο — θάλασσα, λιμάνια  ← Water class κυρίαρχο εδώ
Domain 4: Ορεινό — Άλπεις, Πυρηναία
Domain 5: Αστικό — πυκνή δόμηση

→ Κάθε domain = 1 CL task για Domain-Incremental Learning
→ Water/Bridge/Bridge στα αρχεία train-00..79 ανιχνεύονται από το fractal.gpkg
```

### 2.5 Continual Learning Task Setup

```
Σενάριο Α — Domain-Incremental (εύκολο):
  Task 1: Domain 1 (Ground + Veg + Building)
  Task 2: Domain 2 (ίδιες κλάσεις, νέα γεωγραφία)
  Task 3: Domain 3 (+ Water εμφανίζεται)
  ...
  → Ερώτηση: πόσο forgetting υπάρχει;

Σενάριο Β — Class-Incremental (δύσκολο, το κύριο σενάριο):
  Task 1: Train on {Ground, Vegetation, Building}
  Task 2: Encounter Water → OOD detection → few-shot learn
  Task 3: Encounter Bridge → OOD detection → few-shot learn
  ...
  → Ερώτηση: ανιχνεύει σωστά, μαθαίνει γρήγορα, δεν ξεχνάει;
```

---

## 3. Το Κεντρικό Ερώτημα: Heuristics vs Learning

### 3.1 Η κριτική που θα δεχτείτε

> *"Γιατί δεν βάλατε απλά ένα σύστημα κανόνων; NIR < 500 AND intensity < 300 AND return_number == 1 → Water. Τρέχει σε 1ms, δεν χρειάζεται GPU, ποτέ δεν ξεχνάει."*

Αυτή είναι **νόμιμη** κριτική. Πρέπει να την απαντήσετε πειραματικά.

### 3.2 Η απάντηση — τι αποτυγχάνουν τα heuristics

```
Πρόβλημα 1: Calibration drift
  Ο ίδιος αισθητήρας σε διαφορετική θερμοκρασία ή μετά από σκόνη
  αλλάζει τα thresholds. Το μοντέλο προσαρμόζεται, το heuristic όχι.

Πρόβλημα 2: Novel terrain types
  Παγωμένο νερό (ice): NIR = μέτριο, intensity = υψηλό → heuristic λέει "Building"
  Ξηρό χώμα νερού: flat Z, χαμηλό NIR → heuristic λέει "Water" (false positive)
  Το μοντέλο μαθαίνει τις λεπτές διαφορές.

Πρόβλημα 3: Νέες κλάσεις που δεν φαντάστηκε ο κατασκευαστής
  Π.χ. Solar panels: flat Z + υψηλό intensity + κανονικό NIR
  Κανένα heuristic δεν το καλύπτει. Το μοντέλο ανιχνεύει "unknown" και μαθαίνει.

Πρόβλημα 4: Class overlap στο feature space
  Water σε θυελλώδεις συνθήκες: NIR ≠ πολύ χαμηλό (wave reflections)
  Heuristic: αποτυγχάνει. Μοντέλο: robust λόγω συνδυασμού features.
```

### 3.3 Τι ΠΡΕΠΕΙ να συμπεριλάβετε

Για να αντικρούσετε την κριτική **πειραματικά**, βάλτε στα experiments:

```
Baseline 0: Heuristic rules (NIR + intensity + return thresholds)
Baseline 1: Fine-tuning χωρίς anti-forgetting
Baseline 2: EWC μόνο
Baseline 3: iCaRL (classic, με mean prototype)
Proposed:   iCaRL + Fourier Prototypes + FNO branch + Physics OOD
```

Δείξτε ότι σε known classes το heuristic είναι ανταγωνιστικό, αλλά στα **novel + degraded** scenarios το μοντέλο κερδίζει.

---

## 4. Αρχιτεκτονική Μοντέλου

### 4.1 Γιατί όχι PINN

PINN (Physics-Informed Neural Networks) απαιτεί:
- Γνωστές διαφορικές εξισώσεις που διέπουν το φαινόμενο
- Smooth, differentiable physics
- Πολύπλοκο training (balancing PDE loss + data loss)

Το LiDAR reflection δεν ακολουθεί απλή PDE στο επίπεδο που μελετάμε. Αυτό που έχει νόημα είναι **physics-guided features και physics-informed regularization** — όχι full PINN.

### 4.2 Γιατί FNO — το σωστό framing

Η κλασική FNO (Fourier Neural Operator) λύνει PDEs σε grids. Εμείς δεν λύνουμε PDE. Αυτό που κάνουμε είναι:

> **Χρησιμοποιούμε το Fourier transform ως feature extractor** στο BEV projection του LiDAR patch, εκμεταλλευόμενοι το γεγονός ότι διαφορετικά υλικά εδάφους έχουν διαφορετικές **spatial frequency signatures** στα intensity/NIR/Z fields.

Αυτό είναι πιο σωστό να το αποκαλούμε **FNO-inspired spectral encoder** ή **Spectral BEV Encoder**.

```
Γιατί έχει νόημα φυσικά:
  - Water: πολύ χαμηλές spatial frequencies (ομοιόμορφη επιφάνεια)
  - Vegetation: υψηλές spatial frequencies (τυχαία διάταξη φύλλων)
  - Building: μεσαίες, με ακμές (κανονική γεωμετρία)
  
  Το FFT αποτυπώνει αυτές τις διαφορές άμεσα και αποδοτικά.
```

### 4.3 Η Τελική Αρχιτεκτονική: TinyDualNet

```
┌─────────────────────────────────────────────────────────────────┐
│                        TinyDualNet                              │
│                                                                 │
│  INPUT: LAZ patch (~150K points, 50×50m)                        │
│                                                                 │
│  ┌─────────────────────┐    ┌──────────────────────────────┐    │
│  │   BRANCH A          │    │   BRANCH B                   │    │
│  │   PointNet-lite     │    │   Spectral BEV Encoder       │    │
│  │                     │    │                              │    │
│  │  Subsample 4096 pts │    │  Voxelize → BEV 200×200      │    │
│  │  Shared MLP         │    │  Channels: Z_mean, Z_std,    │    │
│  │  [8,16,32,64]       │    │  intensity, NIR, ret_density,│    │
│  │  MaxPool global     │    │  last_ret_ratio (6 channels) │    │
│  │  → 64-dim           │    │                              │    │
│  │                     │    │  2D FFT per channel          │    │
│  │  Params: ~15K       │    │  Keep top-K freq components  │    │
│  │  Latency Pi4: ~15ms │    │  2 conv layers in freq space │    │
│  │                     │    │  IFFT → 64-dim               │    │
│  └──────────┬──────────┘    │                              │    │
│             │               │  Params: ~25K                │    │
│             │               │  Latency Pi4: ~20ms          │    │
│             └───────┬───────┘                              │    │
│                     │                                       │    │
│              CONCAT [128-dim]                               │    │
│                     ↓                                       │    │
│              Linear(128→64) + BN + ReLU                    │    │
│                     ↓                                       │    │
│         ┌───────────┴────────────┐                         │    │
│         ↓                        ↓                         │    │
│   Cosine Classifier         OOD Detector                   │    │
│   (known classes)           (Mahalanobis)                  │    │
│   Linear → num_classes      threshold τ                    │    │
│                                                             │    │
│  TOTAL PARAMS: ~50K                                         │    │
│  Pi4 INFERENCE: ~40ms/patch                                │    │
│  MODEL SIZE: ~200KB                                         │    │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 Γιατί Cosine Classifier (όχι Linear)

```python
# Linear classifier:
logits = W @ embedding + b
# Πρόβλημα: για νέα κλάση πρέπει να αλλάξεις W (μεγάλο matrix update)
#            → forgetting παλιών κλάσεων

# Cosine classifier:
logits = (embedding / ||embedding||) · (W / ||W||) * scale
# Πλεονέκτημα: νέα κλάση = νέα γραμμή στο W, παλιές ΑΜΕΤΑΒΛΗΤΕΣ
#              → zero forgetting στον classifier
#              → μόνο το backbone μπορεί να αλλάξει (ελεγχόμενα με EWC)
```

### 4.5 BEV Projection — Implementation

```python
def points_to_bev(xyz, intensity, nir, return_num, n_returns,
                  resolution=0.25, grid_size=200):
    """
    50m × 50m patch → 200×200 grid (at 0.25m/pixel)
    Output channels: [Z_mean, Z_std, intensity_mean, NIR_mean,
                      point_density, last_return_ratio]
    """
    # Normalize coordinates to grid
    x_idx = ((xyz[:,0] - xyz[:,0].min()) / resolution).astype(int).clip(0, grid_size-1)
    y_idx = ((xyz[:,1] - xyz[:,1].min()) / resolution).astype(int).clip(0, grid_size-1)
    
    grid = np.zeros((6, grid_size, grid_size), dtype=np.float32)
    count = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    # Accumulate per cell
    np.add.at(grid[0], (x_idx, y_idx), xyz[:,2])           # Z sum
    np.add.at(grid[2], (x_idx, y_idx), intensity)           # intensity sum
    np.add.at(grid[3], (x_idx, y_idx), nir)                 # NIR sum
    np.add.at(count,   (x_idx, y_idx), 1)
    
    # Last return mask
    is_last = return_num == n_returns
    np.add.at(grid[5], (x_idx[is_last], y_idx[is_last]), 1)
    
    # Normalize
    mask = count > 0
    grid[0][mask] /= count[mask]                            # Z_mean
    grid[2][mask] /= count[mask]                            # intensity_mean
    grid[3][mask] /= count[mask]                            # NIR_mean
    grid[4] = count / count.max()                           # density
    grid[5][mask] /= count[mask]                            # last_return_ratio
    
    # Z_std (second pass)
    z_sq = np.zeros((grid_size, grid_size))
    np.add.at(z_sq, (x_idx, y_idx), xyz[:,2]**2)
    grid[1][mask] = np.sqrt(np.maximum(z_sq[mask]/count[mask] - grid[0][mask]**2, 0))
    
    return grid  # (6, 200, 200)
```

### 4.6 Spectral BEV Encoder

```python
import torch
import torch.nn as nn
import torch.fft

class SpectralBEVEncoder(nn.Module):
    def __init__(self, in_channels=6, hidden=32, out_dim=64, top_k_freq=50):
        super().__init__()
        self.top_k = top_k_freq
        
        # Λειτουργεί στο frequency domain
        # Μαθαίνει ποιες συχνότητες είναι σημαντικές ανά κλάση
        self.freq_mixer = nn.Parameter(
            torch.randn(in_channels, top_k_freq, top_k_freq, dtype=torch.cfloat) * 0.02
        )
        
        self.conv1 = nn.Conv2d(in_channels * 2, hidden, 3, padding=1)  # real + imag
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.pool  = nn.AdaptiveAvgPool2d(4)
        self.fc    = nn.Linear(hidden * 16, out_dim)
        self.bn1   = nn.BatchNorm2d(hidden)
        self.bn2   = nn.BatchNorm2d(hidden)
    
    def forward(self, bev):
        # bev: (B, 6, 200, 200)
        
        # 2D FFT
        bev_fft = torch.fft.rfft2(bev)          # (B, 6, 200, 101) complex
        
        # Crop to top-k frequencies
        bev_fft_k = bev_fft[:, :, :self.top_k, :self.top_k]  # (B, 6, k, k)
        
        # Learnable frequency mixing (κύρια innovation)
        bev_fft_mixed = bev_fft_k * self.freq_mixer  # element-wise complex mult
        
        # Concatenate real and imaginary as separate channels
        x = torch.cat([bev_fft_mixed.real, bev_fft_mixed.imag], dim=1)  # (B, 12, k, k)
        
        # Spatial processing
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).flatten(1)              # (B, hidden*16)
        x = self.fc(x)                           # (B, 64)
        return x
```

---

## 5. Open-World Continual Learning Pipeline

### 5.1 Ο Κύκλος Ζωής του Συστήματος

```
╔════════════════════════════════════════════════════╗
║  OFFLINE (πριν deployment)                         ║
║                                                    ║
║  1. Train on Base Classes (Ground, Veg, Building)  ║
║  2. Compute class prototypes + Fisher Information  ║
║  3. Build Fourier Prototype Memory                 ║
║  4. Export to ONNX for Pi deployment               ║
╚════════════════════════════════════════════════════╝
                      ↓ deploy
╔════════════════════════════════════════════════════╗
║  INFERENCE MODE (κατά πτήση)                       ║
║                                                    ║
║  For each LiDAR patch:                             ║
║    1. Extract features (PointNet + SpectralBEV)    ║
║    2. Classify known points                        ║
║    3. OOD detection → flag unknowns               ║
║    4. Store unknowns in ring buffer (RAM)          ║
╚════════════════════════════════════════════════════╝
                      ↓ drone lands
╔════════════════════════════════════════════════════╗
║  LEARNING MODE (σε ηρεμία / φόρτιση)              ║
║                                                    ║
║  1. Cluster unknown buffer                         ║
║  2. If cluster_size > threshold → trigger learning ║
║  3. Few-shot iCaRL update                          ║
║  4. EWC backbone regularization                   ║
║  5. Update Fourier Prototypes                      ║
║  6. Validate on exemplar set                       ║
╚════════════════════════════════════════════════════╝
```

### 5.2 OOD Detection — Τριπλή Στρατηγική

Χρησιμοποιούμε **AND** combination τριών μεθόδων για robustness:

**Μέθοδος 1: Energy Score**
```
E(x) = -log Σ_k exp(f_k(x) / T)

Threshold: τ_E (calibrated on val set)
Αν E(x) > τ_E → candidate unknown
```

**Μέθοδος 2: Mahalanobis Distance**
```
d_k(x) = (z - μ_k)ᵀ Σ⁻¹ (z - μ_k)
d_min = min_k d_k(x)

Αν d_min > τ_M → candidate unknown

Πλεονέκτημα: γεωμετρικά ερμηνεύσιμο στο embedding space
```

**Μέθοδος 3: Physics-Prior Score (lightweight, Pi-friendly)**
```python
def physics_ood_score(intensity, nir, return_num, n_returns, z_std_cell):
    """
    Γρήγορος physics-based scorer για edge pre-filtering.
    Score > 0.7 → πιθανό unknown
    """
    score = 0.0
    n = len(intensity)
    
    # Pattern 1: Specular flat surface (Water candidate)
    is_flat    = z_std_cell < 0.3
    is_low_nir = (nir < 500).mean() > 0.8
    is_single  = (return_num == 1).mean() > 0.95
    is_low_int = (intensity < 600).mean() > 0.8
    score += 0.4 * float(is_flat and is_low_nir and is_single and is_low_int)
    
    # Pattern 2: Rigid elevated structure (Bridge candidate)
    is_elevated = z_std_cell > 2.0  # significant height above ground
    is_high_int = (intensity > 3000).mean() > 0.6
    score += 0.3 * float(is_elevated and is_high_int and is_single)
    
    # Pattern 3: General novelty (entropy-like)
    feat_vector = np.array([
        intensity.mean() / 65535,
        nir.mean() / 65535,
        (return_num == 1).mean(),
        z_std_cell / 10.0
    ])
    # Distance from known class centers (precomputed)
    score += 0.3 * float(np.min([
        np.linalg.norm(feat_vector - KNOWN_CENTERS[c])
        for c in KNOWN_CENTERS
    ]) > TAU_PHYSICS)
    
    return score  # 0.0 = known, 1.0 = definitely unknown
```

**Τελική απόφαση:**
```
is_unknown = (energy_score > τ_E) AND
             (mahal_dist > τ_M) AND
             (physics_score > τ_P)

# Conservative: απαιτεί όλα να συμφωνούν → λιγότερα false positives
# Στο paper: ablation study για κάθε μέθοδο ξεχωριστά + combination
```

### 5.3 Unknown Buffer & Clustering

```python
class UnknownBuffer:
    """
    Ring buffer για αποθήκευση unknown points κατά πτήση.
    Μέγιστο: 5000 points (~20MB RAM)
    """
    def __init__(self, max_size=5000):
        self.max_size = max_size
        self.embeddings  = []   # (D,) vectors
        self.raw_features = []  # (10,) [xyz, intensity, nir, returns...]
        self.timestamps  = []
    
    def add(self, embedding, raw_feat, timestamp):
        if len(self.embeddings) >= self.max_size:
            # Ring: αντικατέστησε παλαιότερο
            self.embeddings.pop(0)
            self.raw_features.pop(0)
            self.timestamps.pop(0)
        self.embeddings.append(embedding)
        self.raw_features.append(raw_feat)
        self.timestamps.append(timestamp)
    
    def cluster(self, min_cluster_size=50):
        """
        DBSCAN clustering στο embedding space.
        Αν βρεθεί cluster > min_size → trigger learning mode.
        """
        from sklearn.cluster import DBSCAN
        if len(self.embeddings) < min_cluster_size:
            return None
        
        embs = np.array(self.embeddings)
        db = DBSCAN(eps=0.3, min_samples=20).fit(embs)
        labels = db.labels_
        
        # Βρες μεγαλύτερο cluster (εκτός noise=-1)
        from collections import Counter
        counts = Counter(l for l in labels if l != -1)
        if not counts:
            return None
        
        best_label, count = counts.most_common(1)[0]
        if count < min_cluster_size:
            return None
        
        mask = labels == best_label
        return {
            'embeddings': embs[mask],
            'raw_features': np.array(self.raw_features)[mask],
            'n_samples': count
        }
```

---

## 6. Fourier Prototypes — Η Πρωτότυπη Συνεισφορά

### 6.1 Το Πρόβλημα με Classic iCaRL

```
Classic iCaRL prototype: μ_k = (1/N) Σᵢ φ(xᵢ)

Αδυναμία: Ένα μοναδικό διάνυσμα δεν αποτυπώνει
          την πολυμορφία μιας κλάσης.

Παράδειγμα — Vegetation:
  - Χαμηλή βλάστηση (Z: 0-0.5m, NIR: medium)
  - Θάμνοι       (Z: 0.5-2m,   NIR: high)
  - Δέντρα       (Z: 2-20m,    NIR: very high, multiple returns)
  
  Mean embedding → "μέσο" vegetation που δεν μοιάζει με κανένα
  
  FFT των embeddings → αποτυπώνει ΟΛΑ τα modes της distribution
```

### 6.2 Fourier Prototype — Ορισμός

```
Έστω E_k = {φ(x₁), ..., φ(xₙ)} ⊂ ℝᴰ  (N embeddings της κλάσης k)

Βήμα 1: Για κάθε dimension d:
  signal_d = [φ(x₁)[d], φ(x₂)[d], ..., φ(xₙ)[d]]  ← χρονοσειρά

Βήμα 2: FFT
  F_d = FFT(signal_d)  →  complex coefficients

Βήμα 3: Κράτα top-K συχνότητες (μεγαλύτερο magnitude)
  Π_k[d] = TopK(F_d, K=16)  →  16 complex numbers per dimension

Αποτέλεσμα: Π_k ∈ ℂ^(D×K)  ← compact representation

Memory comparison (D=64, K=16, N=1000):
  Classic iCaRL exemplars:  1000 × 64 floats = 256KB per class
  Classic iCaRL prototype:  64 floats = 0.25KB per class
  Fourier prototype:        64 × 16 × 2 floats = 8KB per class  ← διατηρεί structure!
```

### 6.3 Reconstruction για Replay

```python
class FourierPrototype:
    def __init__(self, embedding_dim=64, top_k=16, n_reconstruct=100):
        self.D = embedding_dim
        self.K = top_k
        self.N = n_reconstruct
        self.coeffs = None   # (D, K) complex
        self.indices = None  # (D, K) int — ποιες συχνότητες κρατάμε
    
    def fit(self, embeddings: np.ndarray):
        """embeddings: (N, D)"""
        N, D = embeddings.shape
        self.coeffs  = np.zeros((D, self.K), dtype=complex)
        self.indices = np.zeros((D, self.K), dtype=int)
        
        for d in range(D):
            signal = embeddings[:, d]
            fft_vals = np.fft.rfft(signal)
            magnitudes = np.abs(fft_vals)
            top_idx = np.argsort(magnitudes)[-self.K:]
            self.indices[d] = top_idx
            self.coeffs[d]  = fft_vals[top_idx]
    
    def reconstruct(self, n=None) -> np.ndarray:
        """Δημιούργησε συνθετικά exemplars για replay."""
        n = n or self.N
        N_fft = n // 2 + 1
        synthetic = np.zeros((n, self.D))
        
        for d in range(self.D):
            full_fft = np.zeros(N_fft, dtype=complex)
            full_fft[self.indices[d]] = self.coeffs[d]
            signal = np.fft.irfft(full_fft, n=n)
            synthetic[:, d] = signal
        
        return synthetic  # (n, D) — synthetic embeddings για replay
    
    def similarity(self, query_embedding: np.ndarray) -> float:
        """Cosine similarity μεταξύ query και prototype mean."""
        proto_mean = self.reconstruct(n=50).mean(axis=0)
        cos_sim = np.dot(query_embedding, proto_mean) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(proto_mean) + 1e-8
        )
        return float(cos_sim)
```

### 6.4 Γιατί αυτό βοηθά στο edge

```
Memory budget Pi4 για CL memory:
  Classic replay buffer (200 exemplars × 64-dim): 200 × 64 × 4 bytes = 51KB per class
  Fourier prototype (K=16):                       64 × 16 × 8 bytes =  8KB per class
  
  Για 7 κλάσεις:
  Classic:  7 × 51KB = 357KB
  Fourier:  7 × 8KB  = 56KB  ← 6.4× μικρότερο
  
  Και αποτυπώνει καλύτερα τη multimodal distribution κάθε κλάσης.
```

---

## 7. Edge Deployment στο Raspberry Pi

### 7.1 Hardware Target

```
Raspberry Pi 4 Model B (4GB RAM)
  CPU:    ARM Cortex-A72 quad-core 1.8GHz
  RAM:    4GB LPDDR4
  Power:  5V/3A = 15W max (5-8W typical)
  
Drone power budget:
  Typical flight battery: 22.2V, 10Ah = ~222Wh
  Max flight time: 30 min
  Pi power consumption: 5-8W = 2.5-4Wh total flight ← αμελητέο (<2%)
  
Συμπέρασμα: η κατανάλωση του Pi δεν είναι πρόβλημα αν:
  - Δεν χρησιμοποιούμε GPU (δεν υπάρχει)
  - Δεν κάνουμε learning κατά πτήση (inference only)
  - Αποφεύγουμε memory thrashing
```

### 7.2 Inference Latency Budget

```
Στόχος: <100ms per patch (10 patches/sec = real-time για 10Hz LiDAR)

Breakdown TinyDualNet στο Pi4:
  LAZ loading & parsing:      ~5ms   (laspy)
  Subsampling 4096 pts:       ~2ms   (numpy random)
  BEV projection:             ~8ms   (numpy vectorized)
  PointNet-lite forward:      ~15ms  (ONNX runtime)
  SpectralBEV forward:        ~20ms  (numpy FFT + ONNX)
  OOD detection:              ~3ms   (matrix ops)
  Physics pre-filter:         ~2ms   (numpy thresholds)
  Post-processing:            ~2ms
  ─────────────────────────────────
  TOTAL:                     ~57ms  ✅ κάτω από 100ms

Σε σύγκριση:
  KPConv-small:              ~800ms ❌
  PointNet++ (large):        ~400ms ❌
  MiniPointNet (baseline):   ~200ms ⚠️ οριακό
```

### 7.3 Export Pipeline

```python
# 1. PyTorch → ONNX
import torch
import torch.onnx

model = TinyDualNet(num_classes=3)
model.load_state_dict(torch.load('checkpoint.pt'))
model.eval()

# Dummy inputs
pts_dummy = torch.randn(1, 4096, 9)   # (B, N, features)
bev_dummy = torch.randn(1, 6, 200, 200)

torch.onnx.export(
    model, (pts_dummy, bev_dummy),
    'tinydualnet.onnx',
    input_names=['points', 'bev'],
    output_names=['logits', 'embeddings'],
    opset_version=14,
    dynamic_axes={'points': {0: 'batch'}, 'bev': {0: 'batch'}}
)

# 2. ONNX → quantized (INT8) για Pi
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    'tinydualnet.onnx',
    'tinydualnet_int8.onnx',
    weight_type=QuantType.QInt8
)

# Model size comparison:
# Original FP32: ~200KB
# INT8 quantized: ~55KB  ← fits in L2 cache of Pi4 (1MB)
```

### 7.4 Runtime Memory στο Pi

```
Inference memory footprint:
  Model weights (INT8):       55KB
  Input buffer (LAZ patch):   ~6MB  (150K × 40bytes)
  BEV grid:                   ~1MB  (200×200×6×4bytes)
  Activations:                ~2MB  (peak, reusable)
  Unknown ring buffer:        ~20MB (5000 × 64dim × 4bytes + raw)
  Fourier prototypes (7 cls): <1MB
  ─────────────────────────────────
  TOTAL:                     ~30MB  (σε 4GB Pi = 0.75% RAM) ✅

Learning mode memory (offline):
  Replay exemplars:           ~50MB (reconstructed from Fourier prototypes)
  EWC Fisher matrices:        ~400KB (backbone params only)
  Optimizer state:            ~800KB
  ─────────────────────────────────
  TOTAL learning:            ~51MB  ✅ τρέχει άνετα στα 4GB
```

---

## 8. Πλήρης Εκπαίδευση Step-by-Step

### 8.1 Φάση 0 — Preprocessing

```python
# Για κάθε LAZ αρχείο:
1. Load με laspy
2. Merge classes: {3→5, 4→5}  (vegetation variants)
3. Filter: κράτα μόνο {1,2,5,6,9,17,64}
4. Voxelize @ 0.5m (μείωση ~40%)
5. Normalize Z: z_norm = (z - z_min) / (z_max - z_min + ε)
6. Normalize intensity/NIR: x = (x - mean) / std  (global stats)
7. Compute BEV grid (200×200, 6 channels)
8. Save ως .npz για γρήγορο loading

Output format per patch:
  pts:   (N, 9)  [x,y,z_norm, intensity_norm, nir_norm,
                   return_num, n_returns, scan_angle, last_return_flag]
  bev:   (6, 200, 200)
  labels: (N,)
```

### 8.2 Φάση 1 — Base Training

```
Dataset: train-00, train-01, train-02  (3000 patches)
Classes: Ground(2), Vegetation(5), Building(6)
         → Αφαίρεσε Water, Bridge, PermStruct patches

Loss: L = CE(logits, labels) + λ₁ · SupConLoss(embeddings, labels)
     λ₁ = 0.1

SupConLoss: αναγκάζει embeddings ίδιας κλάσης κοντά,
            διαφορετικών κλάσεων μακριά → καλύτερη OOD ανίχνευση

Optimizer: Adam, lr=1e-3, weight_decay=1e-4
Scheduler: CosineAnnealingLR (T_max=50)
Epochs: 50
Batch size: 16 patches

Metrics:
  Val mIoU (3 classes)
  Val OA
  Embedding separation (silhouette score)

Αναμενόμενο αποτέλεσμα: mIoU ~0.75-0.82 για 3 κλάσεις
```

### 8.3 Φάση 2 — Prototype Computation

```python
# Μετά το base training, ΠΡΙΝ deployment:
for class_id in [2, 5, 6]:
    embeddings = []
    for batch in val_loader:
        pts, bev, labels = batch
        mask = labels == class_id
        if mask.sum() == 0: continue
        emb = model.encode(pts, bev)[mask]  # (M, 64)
        embeddings.append(emb.numpy())
    
    embeddings = np.concatenate(embeddings)  # (N_total, 64)
    
    # Classic prototype (για baseline σύγκριση)
    classic_proto[class_id] = embeddings.mean(axis=0)
    
    # Fourier prototype (η πρότασή μας)
    fp = FourierPrototype(top_k=16)
    fp.fit(embeddings)
    fourier_proto[class_id] = fp

# Compute Fisher Information Matrix (για EWC)
fisher = compute_fisher(model, train_loader_base)
# fisher[param_name] = E[∂²L/∂θ²]  ← πόσο "σημαντική" είναι κάθε παράμετρος
```

### 8.4 Φάση 3 — OOD Threshold Calibration

```python
# Calibrate thresholds σε validation set με KNOWN + UNKNOWN patches
# Known: {Ground, Veg, Building}
# Unknown: {Water, Bridge} (κρατημένα εκτός training)

# Για Energy score:
energies_known   = compute_energy(model, known_val_patches)
energies_unknown = compute_energy(model, unknown_val_patches)

# Επιλογή τ_E: FPR@95TPR (standard metric)
# FPR = False Positive Rate (known classified as unknown)
# TPR = True Positive Rate (unknown correctly flagged)
τ_E = calibrate_threshold(energies_known, energies_unknown, target_tpr=0.95)

# Ομοίως για Mahalanobis
# Ομοίως για Physics score
```

### 8.5 Φάση 4 — Few-Shot Incremental Learning (κατά τη χρήση)

```
Trigger condition: unknown buffer cluster size > 50

Algorithm: FourierPrototype-iCaRL

Input:
  - novel_embeddings: (M, 64)  ← από buffer clustering
  - current_model: TinyDualNet
  - fisher: Fisher Information per parameter
  - fourier_protos: {2: FP, 5: FP, 6: FP}

Steps:
  1. Fit new Fourier Prototype για novel class:
     fp_new = FourierPrototype(); fp_new.fit(novel_embeddings)

  2. Extend cosine classifier:
     W_new = compute_prototype_mean(fp_new)  # (64,)
     W = torch.cat([W, W_new.unsqueeze(0)], dim=0)  # add row

  3. Generate replay exemplars από παλιές κλάσεις:
     for cls_id, fp in fourier_protos.items():
         replay_embs = fp.reconstruct(n=100)  # synthetic exemplars

  4. Fine-tune backbone με EWC:
     L_total = CE(novel_data) + CE(replay_data) + λ_EWC · Σ Fᵢ(θᵢ - θ*ᵢ)²
     
     EWC term: τιμωρεί αλλαγές σε παραμέτρους που είναι κρίσιμες
               για παλιές κλάσεις (high Fisher = κρίσιμη παράμετρος)

  5. Update Fisher Information για νέα κλάση

  6. Validate:
     mIoU(Ground), mIoU(Veg), mIoU(Building) vs baseline  ← forgetting check
     mIoU(Novel) per n_shots  ← few-shot performance
```

---

## 9. Evaluation Protocol

### 9.1 Πίνακας Αποτελεσμάτων (target)

```
                    mIoU  mIoU  mIoU  mIoU   BWT    FWT    Mem   Pi4
Method              Gnd   Veg   Bld   Water  (↑0)   (↑0)   (MB)  (ms)
────────────────────────────────────────────────────────────────────────
Heuristic rules     0.71  0.68  0.65  0.82   N/A    N/A    0     <1
Fine-tuning (LB)    0.41  0.38  0.29  0.61   -0.42  +0.03  0     57
EWC only            0.73  0.70  0.65  0.58   -0.08  +0.01  0.4   58
iCaRL (classic)     0.76  0.74  0.70  0.71   -0.04  +0.05  2.0   59
Ours (full)         0.78  0.76  0.72  0.74   -0.02  +0.07  0.35  62
────────────────────────────────────────────────────────────────────────
BWT: Backward Transfer (αρνητικό = forgetting · 0 = τέλειο)
FWT: Forward Transfer (πόσο βοηθά η παλιά γνώση τη νέα κλάση)
Mem: Replay buffer size
```

### 9.2 Few-Shot Curves

```
Για κάθε novel class (Water, Bridge):
  x-axis: n_shots (1, 5, 10, 20, 50, 100, 200)
  y-axis: mIoU(novel class)
  
  Συγκρίνεις:
    - Classic iCaRL με mean prototype
    - Ours με Fourier prototype
    - Heuristic baseline (flat line, δεν εξαρτάται από shots)
```

### 9.3 OOD Detection Metrics

```
AUROC:     Area Under ROC Curve  (↑1.0 = τέλειο)
FPR@95TPR: False Positive Rate at 95% True Positive Rate (↓0 = τέλειο)

Αναφορά benchmarks:
  Energy score μόνο:    AUROC ~0.82
  Mahalanobis μόνο:     AUROC ~0.85
  Physics μόνο:         AUROC ~0.78  (αλλά 0ms overhead)
  Ours (combination):   AUROC ~0.91
```

### 9.4 Degraded Conditions Testing

```
Σενάριο "Ομίχλη/Σκόνη":
  Προσθέτεις Gaussian noise στα intensity/NIR values
  σ = 10%, 20%, 30% του range

  Αναμένεις: Heuristics αποτυγχάνουν πρώτα (brittle thresholds)
             Μοντέλο πιο robust (μαθαίνει patterns, όχι thresholds)

Σενάριο "Νέος Sensor":
  Ο σενάριο αυτός δικαιολογεί το CL:
  Αν αλλάξει το sensor calibration (άλλος drone, άλλη τοποθεσία)
  το μοντέλο μαθαίνει → heuristic πρέπει manual re-tuning
```

---

## 10. Αντικριτική — Τι θα πουν οι κριτές

### Κριτική 1: "Απλά heuristics θα έκαναν το ίδιο"

**Απάντηση:**
> Συμπεριλαμβάνουμε heuristic baseline. Σε standard conditions είναι ανταγωνιστικό (Table X). Σε 3 σενάρια αποτυγχάνει:
> (1) Degraded conditions (noise), όπου το μοντέλο υπερτερεί κατά +X% mIoU.
> (2) Novel terrain (π.χ. frozen water), που χρειάζεται manual threshold re-tuning.
> (3) Autonomous adaptation: το μοντέλο μαθαίνει νέες κλάσεις χωρίς ανθρώπινη επέμβαση — αυτό δεν υποστηρίζεται από κανένα heuristic.

### Κριτική 2: "Δεν είναι πραγματικά UAV data, είναι ALS"

**Απάντηση:**
> Αναγνωρίζουμε αυτό το limitation (Section 4.5). Το FRACTAL προσομοιώνει το closest publicly available dataset με εξαιρετικά high density (40 pts/m²) που είναι συγκρίσιμη με UAV LiDAR (YellowScan: 200-800 pts/m² σε χαμηλό ύψος). Η μεθοδολογία (CL pipeline, OOD detection, Fourier prototypes) είναι dataset-agnostic. Μελλοντική εργασία περιλαμβάνει testing σε πραγματικά UAV datasets.

### Κριτική 3: "Το few-shot με 50 samples δεν είναι αρκετό"

**Απάντηση:**
> Παρουσιάζουμε full n-shot curves (1→200). Τα Fourier prototypes υπερτερούν των classic mean prototypes ιδιαίτερα σε 5-20 shots (Figure X), ακριβώς όπου ο περιορισμός πραγματικών συνθηκών βρίσκεται.

### Κριτική 4: "Το EWC έχει O(P²) complexity για Fisher matrix"

**Απάντηση:**
> Χρησιμοποιούμε diagonal approximation (standard στη βιβλιογραφία). Εφαρμόζουμε EWC μόνο στο backbone (PointNet ~15K params), όχι στο spectral branch. Fisher storage: 15K × 4 bytes = 60KB.

### Κριτική 5: "Γιατί FNO και όχι απλό CNN στο BEV;"

**Απάντηση:**
> Το Spectral BEV Encoder δεν είναι strict FNO — είναι learnable frequency mixing. Ablation study (Table Y) δείχνει +X% mIoU vs plain CNN με παρόμοια complexity, ιδιαίτερα για Water (spectrally distinct) και Vegetation (high-frequency pattern). Η φυσική ερμηνεία (material spectral signatures) υποστηρίζει την επιλογή.

---

## 11. Roadmap Κώδικα

### Δομή Αρχείων

```
Drone_cont_Learing/
├── data/
│   └── train/00/          ← 1000 LAZ patches
├── notebooks/
│   ├── eda_fractal.ipynb
│   └── run_eda.py
├── src/
│   ├── models/
│   │   ├── pointnet_lite.py      ← Branch A
│   │   ├── spectral_bev.py       ← Branch B (FNO-inspired)
│   │   ├── tinydualnet.py        ← Full model
│   │   └── cosine_classifier.py
│   ├── continual/
│   │   ├── ewc.py                ← EWC regularization
│   │   ├── fourier_prototype.py  ← Η πρωτότυπη συνεισφορά
│   │   ├── icarl.py              ← iCaRL adaptation
│   │   └── unknown_buffer.py    ← OOD buffer + clustering
│   ├── ood/
│   │   ├── energy_score.py
│   │   ├── mahalanobis.py
│   │   └── physics_ood.py       ← LiDAR-specific heuristic OOD
│   ├── data/
│   │   ├── fractal_dataset.py    ← PyTorch Dataset
│   │   ├── bev_projection.py
│   │   └── preprocessing.py
│   └── utils/
│       ├── metrics.py            ← mIoU, BWT, FWT, AUROC
│       └── export.py             ← ONNX export + quantization
├── configs/
│   ├── base_training.yaml
│   ├── cl_experiment.yaml
│   └── edge_inference.yaml
├── experiments/
│   ├── 01_base_training.py
│   ├── 02_ood_calibration.py
│   ├── 03_cl_experiments.py
│   └── 04_ablation_study.py
├── results/                      ← αποτελέσματα πειραμάτων
├── models/                       ← checkpoints
├── requirements.txt
├── TECHNICAL_REPORT.md           ← αυτό το αρχείο
└── project_plan.html
```

### Σειρά Υλοποίησης

```
Εβδομάδα 1-2:
  □ src/data/fractal_dataset.py    ← PyTorch Dataset + DataLoader
  □ src/data/bev_projection.py     ← BEV projection function
  □ src/data/preprocessing.py      ← normalization, class merging
  □ Verify EDA results

Εβδομάδα 3-4:
  □ src/models/pointnet_lite.py    ← Branch A
  □ src/models/spectral_bev.py     ← Branch B
  □ src/models/tinydualnet.py      ← Combined
  □ experiments/01_base_training.py

Εβδομάδα 5-6:
  □ src/ood/energy_score.py
  □ src/ood/mahalanobis.py
  □ src/ood/physics_ood.py
  □ experiments/02_ood_calibration.py

Εβδομάδα 7-9:
  □ src/continual/ewc.py
  □ src/continual/fourier_prototype.py  ← κύρια novel contribution
  □ src/continual/icarl.py
  □ src/continual/unknown_buffer.py
  □ experiments/03_cl_experiments.py

Εβδομάδα 10:
  □ experiments/04_ablation_study.py
  □ src/utils/export.py  ← ONNX + quantization
  □ Pi4 testing

Εβδομάδα 11-12:
  □ Συγγραφή αναφοράς
  □ Figures + Tables
  □ Παρουσίαση
```

---

## Σύνοψη Contributions

| # | Contribution | Novelty | Δυσκολία |
|---|--------------|---------|----------|
| 1 | Spectral BEV Encoder για LiDAR terrain | Υψηλή | Μέτρια |
| 2 | Fourier Prototype Memory για CL | Υψηλή | Χαμηλή-Μέτρια |
| 3 | Physics-guided OOD για LiDAR | Μέτρια | Χαμηλή |
| 4 | Open-World CL pipeline για UAV | Μέτρια | Υψηλή |
| 5 | Edge deployment analysis (Pi4) | Χαμηλή | Χαμηλή |

**Κλειδί:** Καμία μεμονωμένη contribution δεν είναι "τέλεια". Η δύναμη της εργασίας είναι ο **συνδυασμός** και η **πρακτική επαλήθευση σε πραγματικά δεδομένα** με **edge constraints**.

---

*Τελευταία ενημέρωση: 2026-06-01*
*Dataset: FRACTAL (IGNF) — 1000 patches, train-00*
