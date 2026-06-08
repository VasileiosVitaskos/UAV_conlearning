"""
pointnet2.py
PointNet++ Mini για point-wise semantic segmentation.

Αρχιτεκτονική:
  Input  (B, N, 7)
    ↓ SetAbstraction 1   4096 → 512 points,  MLP [7  → 32 → 64]
    ↓ SetAbstraction 2    512 → 128 points,  MLP [64 → 64 → 128]
    ↓ FeaturePropagation 2   128 → 512,      MLP [192 → 64]
    ↓ FeaturePropagation 1   512 → N,        MLP [71  → 32]
    ↓ CosineClassifier        32 → C
  Output (B, N, C)

Παράμετροι: ~80K
Inference CPU (Pi5, B=1): ~60ms
Inference GPU (3060, B=16): ~8ms

Βασικές επιλογές σχεδιασμού:
  • Pure PyTorch — χωρίς custom CUDA kernels, τρέχει παντού
  • KNN αντί για ball query — πιο stable, ίδια αποτελέσματα για μικρά N
  • CosineClassifier — zero-forgetting για Continual Learning
  • get_embeddings() — για OOD detection & few-shot prototypes
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ══════════════════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════════════════

def farthest_point_sample(xyz: Tensor, npoint: int) -> Tensor:
    """
    Iterative Farthest Point Sampling (FPS).

    Επιλέγει npoint σημεία που καλύπτουν όσο το δυνατόν ομοιόμορφα τον χώρο.
    Κάθε νέο σημείο είναι το πιο μακριά από τα ήδη επιλεγμένα.

    Γιατί FPS αντί για random sampling;
    → Εγγυάται ομοιόμορφη κάλυψη του patch → καλύτερα local features.

    Parameters
    ----------
    xyz    : (B, N, 3)  συντεταγμένες όλων των points
    npoint : int        πόσα centroids να επιλέξουμε

    Returns
    -------
    (B, npoint)  LongTensor με indices στο xyz
    """
    B, N, _ = xyz.shape
    device   = xyz.device

    centroids = torch.zeros(B, npoint, dtype=torch.long,  device=device)
    distance  = torch.full((B, N), 1e10,                  device=device)
    # Αρχικό σημείο: πάντα 0 (deterministic) — ONNX-safe.
    # torch.randint εδώ "ψήνεται" ως constant στο ONNX graph κατά το tracing,
    # οδηγώντας σε διαφορετικά centroids από PyTorch → 85% argmax agreement.
    # FPS από point 0 εξαπλώνεται γρήγορα στον χώρο, ανεξάρτητα από σημείο εκκίνησης.
    farthest  = torch.zeros(B, dtype=torch.long,          device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        # Συντεταγμένες τρέχοντος centroid: (B, 1, 3)
        # ONNX-safe: torch.gather αντί για xyz[arange, farthest] (advanced indexing)
        centroid = torch.gather(
            xyz, 1, farthest.view(B, 1, 1).expand(B, 1, 3)
        )   # (B, 1, 3)
        # Αποστάσεις όλων από τον τρέχοντα centroid: (B, N)
        dist = ((xyz - centroid) ** 2).sum(dim=-1)
        # Κράτα την ελάχιστη απόσταση από οποιονδήποτε centroid
        mask     = dist < distance
        distance = torch.where(mask, dist, distance)
        # Επόμενο σημείο = αυτό με τη μεγαλύτερη ελάχιστη απόσταση
        farthest = distance.max(dim=-1)[1]

    return centroids  # (B, npoint)


def knn_query(nsample: int, xyz: Tensor, new_xyz: Tensor) -> Tensor:
    """
    K-Nearest Neighbors: για κάθε centroid στο new_xyz βρες τους
    nsample κοντινότερους γείτονες στο xyz.

    Parameters
    ----------
    nsample : int         αριθμός γειτόνων
    xyz     : (B, N, 3)  όλα τα σημεία
    new_xyz : (B, S, 3)  centroids (query points)

    Returns
    -------
    (B, S, nsample)  indices στο xyz
    """
    # torch.cdist: (B, S, N) — πλήρης πίνακας αποστάσεων
    dists = torch.cdist(new_xyz, xyz)                              # (B, S, N)
    idx   = dists.topk(nsample, dim=-1, largest=False)[1]         # (B, S, K)
    return idx


def gather_points(xyz: Tensor, idx: Tensor) -> Tensor:
    """
    Vectorized gather: επιλέγει σημεία από xyz με βάση indices.

    xyz : (B, N, C)
    idx : (B, M)   ή   (B, S, K)
    returns: (B, M, C) ή (B, S, K, C)

    ONNX-safe: χρησιμοποιεί torch.gather αντί για advanced integer indexing
    (xyz[b_idx, flat] σπάει το ONNX export λόγω aten::index → Gather mismatch).
    torch.gather είναι natively supported σε όλα τα opsets.
    """
    B, N, C = xyz.shape
    shape = idx.shape                          # (B, M) ή (B, S, K)
    flat  = idx.reshape(B, -1)                # (B, M') όπου M' = M ή S*K
    idx_exp = flat.unsqueeze(-1).expand(-1, -1, C)   # (B, M', C)
    out     = torch.gather(xyz, 1, idx_exp)           # (B, M', C)  — ONNX-safe
    return out.reshape(*shape, C)              # (B, M, C) ή (B, S, K, C)


# ══════════════════════════════════════════════════════════════════════════════
# Set Abstraction
# ══════════════════════════════════════════════════════════════════════════════

class SetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction layer.

    Τρία βήματα:
      1. Sample  : FPS επιλέγει npoint centroids από τα N points
      2. Group   : KNN ομαδοποιεί nsample γείτονες γύρω από κάθε centroid
      3. Extract : shared MLP + max-pool → ένα feature vector ανά centroid

    Γιατί max-pool;
    → Permutation-invariant: η σειρά των γειτόνων δεν έχει σημασία.

    Parameters
    ----------
    npoint       : αριθμός centroids (downsampling)
    nsample      : γείτονες ανά centroid
    in_channels  : Cin (features εισόδου, χωρίς xyz)
    mlp_channels : λίστα από [hidden, ..., out] channels του MLP
    """

    def __init__(
        self,
        npoint:       int,
        nsample:      int,
        in_channels:  int,
        mlp_channels: list[int],
    ):
        super().__init__()
        self.npoint  = npoint
        self.nsample = nsample

        # Shared MLP: input = relative_xyz (3) + features (in_channels)
        layers = []
        cin    = 3 + in_channels
        for cout in mlp_channels:
            layers += [
                nn.Linear(cin, cout, bias=False),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
            ]
            cin = cout
        self.mlp         = nn.Sequential(*layers)
        self.out_channels = mlp_channels[-1]

    def forward(self, xyz: Tensor, features: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        xyz      : (B, N, 3)    συντεταγμένες
        features : (B, N, Cin)  features (π.χ. intensity, return_num, ...)

        Returns
        -------
        new_xyz  : (B, npoint, 3)    centroids
        new_feat : (B, npoint, Cout) aggregated features
        """
        B, N, _ = xyz.shape
        S, K     = self.npoint, self.nsample

        # ── 1. Sample ───────────────────────────────────────────────────────
        fps_idx = farthest_point_sample(xyz, S)    # (B, S)
        new_xyz = gather_points(xyz, fps_idx)       # (B, S, 3)

        # ── 2. Group ────────────────────────────────────────────────────────
        knn_idx      = knn_query(K, xyz, new_xyz)   # (B, S, K)
        grouped_xyz  = gather_points(xyz, knn_idx)  # (B, S, K, 3)
        grouped_feat = gather_points(features, knn_idx)  # (B, S, K, Cin)

        # Relative coordinates (local geometry)
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)  # (B, S, K, 3)

        # ── 3. Extract ──────────────────────────────────────────────────────
        x = torch.cat([grouped_xyz, grouped_feat], dim=-1)  # (B, S, K, 3+Cin)

        # Reshape για BatchNorm1d: (B*S*K, C)
        B_, S_, K_, C_ = x.shape
        x = x.reshape(B_ * S_ * K_, C_)
        x = self.mlp(x)                              # (B*S*K, Cout)
        x = x.reshape(B_, S_, K_, -1)

        # Max pooling over neighbors → (B, S, Cout)
        new_feat = x.max(dim=2)[0]

        return new_xyz, new_feat


# ══════════════════════════════════════════════════════════════════════════════
# Feature Propagation
# ══════════════════════════════════════════════════════════════════════════════

class FeaturePropagation(nn.Module):
    """
    PointNet++ Feature Propagation layer.

    Κάνει upsample από S → N points με inverse distance weighted interpolation
    χρησιμοποιώντας τους 3 κοντινότερους γείτονες (3-NN IDW).

    Μετά το interpolation, concatenate με skip connection από τον encoder
    και εφαρμόζει MLP.

    Parameters
    ----------
    in_channels  : Cin_skip + Cin_interp  (μετά το concatenation)
    mlp_channels : λίστα από [hidden, ..., out] channels
    """

    def __init__(self, in_channels: int, mlp_channels: list[int]):
        super().__init__()

        layers = []
        cin    = in_channels
        for cout in mlp_channels:
            layers += [
                nn.Linear(cin, cout, bias=False),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
            ]
            cin = cout
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        xyz1:  Tensor,          # (B, N, 3) dense  (target)
        xyz2:  Tensor,          # (B, S, 3) sparse (source)
        feat1: Tensor | None,   # (B, N, C1) skip connection
        feat2: Tensor,          # (B, S, C2) features to upsample
    ) -> Tensor:
        """
        Returns
        -------
        (B, N, Cout)  upsampled features
        """
        B, N, _ = xyz1.shape

        # ── Interpolation (3-NN IDW) ─────────────────────────────────────
        dists, idx = torch.cdist(xyz1, xyz2).topk(3, dim=-1, largest=False)
        # dists: (B, N, 3),  idx: (B, N, 3)

        # Inverse distance weights
        weights = 1.0 / (dists + 1e-8)                # (B, N, 3)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # normalize

        # Gather κοντινά features και weighted sum
        grouped = gather_points(feat2, idx)            # (B, N, 3, C2)
        interp  = (grouped * weights.unsqueeze(-1)).sum(dim=2)  # (B, N, C2)

        # ── Skip connection ───────────────────────────────────────────────
        x = torch.cat([feat1, interp], dim=-1) if feat1 is not None else interp
        # (B, N, C1+C2)

        # ── MLP ───────────────────────────────────────────────────────────
        B_, N_, C_ = x.shape
        x = x.reshape(B_ * N_, C_)
        x = self.mlp(x)
        x = x.reshape(B_, N_, -1)

        return x


# ══════════════════════════════════════════════════════════════════════════════
# Cosine Classifier
# ══════════════════════════════════════════════════════════════════════════════

class CosineClassifier(nn.Module):
    """
    Cosine similarity classifier — ο πυρήνας του Continual Learning.

    score(x, class_i) = τ · cos(x, w_i) = τ · (x·w_i) / (|x|·|w_i|)

    Γιατί cosine αντί για linear;
    ──────────────────────────────
    • Linear: score = x·w + b  → αλλάζει scale με νέες κλάσεις
    • Cosine:  score = τ·cos   → scale-invariant, prototypes δεν "ξεχνιούνται"

    Continual Learning flow:
      1. Train on Ground/Veg/Building → 5 prototypes (w_0..w_4)
      2. OOD detection εντοπίζει Water/Bridge
      3. add_class(water_prototype)  → w_5, δεν αλλάζει w_0..w_4
      4. add_class(bridge_prototype) → w_6, δεν αλλάζει w_0..w_5

    Parameters
    ----------
    in_features  : D  (embedding dimension)
    num_classes  : C  (αρχικός αριθμός κλάσεων)
    temperature  : τ  (sharpness, default=10 — standard για cosine classifiers)
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        temperature: float = 10.0,
    ):
        super().__init__()
        self.temperature = temperature
        # Κάθε γραμμή = prototype κλάσης (D-dimensional)
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    @property
    def num_classes(self) -> int:
        return self.weight.shape[0]

    def forward(self, x: Tensor) -> Tensor:
        """
        x       : (..., D)  per-point embeddings
        returns : (..., C)  logits (cosine × temperature)
        """
        shape  = x.shape
        x_flat = x.reshape(-1, shape[-1])                       # (M, D)

        x_norm = F.normalize(x_flat,       dim=-1)              # (M, D)
        w_norm = F.normalize(self.weight,  dim=-1)              # (C, D)

        logits = (x_norm @ w_norm.T) * self.temperature         # (M, C)
        return logits.reshape(*shape[:-1], self.num_classes)    # (..., C)

    @torch.no_grad()
    def add_class(self, prototype: Tensor) -> None:
        """
        Προσθέτει νέα κλάση χωρίς να τροποποιήσει τα παλιά prototypes.

        prototype : (D,)  mean embedding των few-shot samples της νέας κλάσης
        """
        new_row = F.normalize(prototype.unsqueeze(0), dim=-1).to(self.weight.device)
        self.weight = nn.Parameter(
            torch.cat([self.weight.data, new_row], dim=0)
        )

    @torch.no_grad()
    def update_class(self, class_idx: int, prototype: Tensor) -> None:
        """
        Ενημερώνει το prototype υπάρχουσας κλάσης.
        Χρήσιμο για online update: running mean των embeddings.
        """
        self.weight.data[class_idx] = F.normalize(
            prototype.to(self.weight.device), dim=-1
        )


# ══════════════════════════════════════════════════════════════════════════════
# PointNet2Mini — το κύριο μοντέλο
# ══════════════════════════════════════════════════════════════════════════════

class PointNet2Mini(nn.Module):
    """
    PointNet++ Mini: encoder-decoder για per-point semantic segmentation.

    Encoder (Set Abstraction):
      SA1: N=4096 → S=512,  K=32 neighbors,  MLP [7  → 32 → 64]
      SA2: S=512  → S=128,  K=64 neighbors,  MLP [64 → 64 → 128]

    Decoder (Feature Propagation με skip connections):
      FP2: 128 → 512,  MLP [128+64 → 64]   (concat SA1 skip)
      FP1: 512 → N,    MLP [64+7   → 32]   (concat input features)

    Head:
      CosineClassifier: 32 → C

    Skip connections:
      FP2 ← SA1 features (64-dim)
      FP1 ← input features (7-dim)   ← σημαντικό: η raw info δεν χάνεται

    Parameters
    ----------
    in_channels  : αριθμός input features (default 7)
    num_classes  : αρχικός αριθμός κλάσεων (default 8)
    temperature  : cosine classifier temperature (default 10.0)
    """

    def __init__(
        self,
        in_channels: int   = 7,
        num_classes: int   = 8,
        temperature: float = 10.0,
    ):
        super().__init__()
        self.in_channels = in_channels

        # ── Encoder ───────────────────────────────────────────────────────
        self.sa1 = SetAbstraction(
            npoint=512, nsample=32,
            in_channels=in_channels,
            mlp_channels=[32, 64],
        )
        self.sa2 = SetAbstraction(
            npoint=128, nsample=64,
            in_channels=64,
            mlp_channels=[64, 128],
        )

        # ── Decoder ───────────────────────────────────────────────────────
        # FP2: concat(SA2_out=128, SA1_skip=64) → 64
        self.fp2 = FeaturePropagation(
            in_channels=128 + 64,
            mlp_channels=[64],
        )
        # FP1: concat(FP2_out=64, input_skip=in_channels) → 32
        self.fp1 = FeaturePropagation(
            in_channels=64 + in_channels,
            mlp_channels=[32],
        )

        # ── Head ──────────────────────────────────────────────────────────
        self.classifier = CosineClassifier(
            in_features=32,
            num_classes=num_classes,
            temperature=temperature,
        )

    def forward(self, x: Tensor, xyz_dropout: float = 0.0) -> Tensor:
        """
        Parameters
        ----------
        x            : (B, N, 7)  normalized point features
        xyz_dropout  : float      probability of zeroing xyz FEATURE dims (0-2)
                                  during training. Spatial xyz for FPS/KNN is
                                  kept intact so grouping still works correctly.
                                  Forces model to rely on intensity/returns.
                                  Use 0.0 at inference (default).

        Returns
        -------
        (B, N, C)  per-point class logits
        """
        # Spatial xyz: ALWAYS intact — used for FPS, KNN, interpolation
        xyz = x[:, :, :3].detach()   # detach so dropout doesn't affect spatial ops

        # XYZ Feature Dropout (Run 7+):
        # Zero out xyz dims in the FEATURE tensor with probability xyz_dropout.
        # Per-batch masking: whole batch either has xyz or not.
        # This forces the MLP to learn from intensity/returns when xyz is absent.
        # Local relative xyz (inside SA groups) still available — only patch-level
        # absolute position is dropped.
        if xyz_dropout > 0.0 and self.training:
            # Per-sample mask: (B, 1, 1) → broadcast over N and C
            mask = (torch.rand(x.shape[0], 1, 1, device=x.device) < xyz_dropout)
            x_feat = x.clone()
            x_feat[:, :, :3] = x_feat[:, :, :3].masked_fill(mask, 0.0)
        else:
            x_feat = x

        # ── Encoder ───────────────────────────────────────────────────────
        xyz1, feat1 = self.sa1(xyz, x_feat)    # (B,512,3), (B,512,64)
        xyz2, feat2 = self.sa2(xyz1, feat1)    # (B,128,3), (B,128,128)

        # ── Decoder ───────────────────────────────────────────────────────
        # FP2: upsample 128→512, skip από SA1
        feat = self.fp2(xyz1, xyz2, feat1, feat2)    # (B,512,64)
        # FP1: upsample 512→N, skip από input (x_feat — consistency)
        feat = self.fp1(xyz, xyz1, x_feat, feat)     # (B,N,32)

        # ── Classification ────────────────────────────────────────────────
        return self.classifier(feat)                  # (B,N,C)

    def get_embeddings(self, x: Tensor) -> Tensor:
        """
        Επιστρέφει per-point embeddings (32-dim) πριν τον classifier.

        Χρήσιμο για:
          • OOD detection (Energy score, Mahalanobis distance)
          • Few-shot prototype computation
          • Visualization (t-SNE)

        Parameters
        ----------
        x : (B, N, 7)

        Returns
        -------
        (B, N, 32)  normalized embeddings
        """
        xyz  = x[:, :, :3].detach()
        xyz1, feat1 = self.sa1(xyz, x)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        feat = self.fp2(xyz1, xyz2, feat1, feat2)
        feat = self.fp1(xyz, xyz1, x, feat)
        return feat   # (B, N, 32)

    def model_summary(self) -> None:
        """Τυπώνει αριθμό παραμέτρων ανά module (για Poster/θέση)."""
        print(f"\n{'Module':<35} {'Params':>10}")
        print("─" * 47)
        for name, module in self.named_modules():
            # Μόνο leaf modules με παραμέτρους
            if len(list(module.children())) == 0:
                p = sum(par.numel() for par in module.parameters())
                if p > 0:
                    print(f"  {name:<33} {p:>10,}")
        print("─" * 47)
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"{'Total':<35} {total:>10,}")
        print(f"{'Trainable':<35} {trainable:>10,}")
        size_mb = total * 4 / 1024 / 1024   # float32 = 4 bytes
        print(f"{'Size (float32)':<35} {size_mb:>9.2f}MB")


# ══════════════════════════════════════════════════════════════════════════════
# Quick test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)

    B, N, C = 2, 4096, 7
    x = torch.randn(B, N, C)

    model = PointNet2Mini(in_channels=C, num_classes=8)
    model.eval()

    # ── Model summary ──────────────────────────────────────────────────────
    model.model_summary()

    # ── Forward pass ───────────────────────────────────────────────────────
    with torch.no_grad():
        # Warmup
        for _ in range(2):
            _ = model(x)

        # Benchmark
        times = []
        for _ in range(10):
            t0  = time.perf_counter()
            out = model(x)
            times.append(time.perf_counter() - t0)

    print(f"\nOutput shape  : {tuple(out.shape)}")
    print(f"Logits range  : [{out.min():.2f}, {out.max():.2f}]")
    t_mean = np.mean(times) * 1000
    t_std  = np.std(times)  * 1000
    print(f"\nInference (batch={B}, N={N}, CPU):")
    print(f"  {t_mean:.1f} ± {t_std:.1f} ms total")
    print(f"  {t_mean/B:.1f} ms per sample")

    # ── Embeddings ─────────────────────────────────────────────────────────
    with torch.no_grad():
        emb = model.get_embeddings(x)
    print(f"\nEmbeddings    : {tuple(emb.shape)}  (για OOD & few-shot)")

    # ── CosineClassifier: προσθήκη νέας κλάσης ────────────────────────────
    print(f"\nCosineClassifier πριν : {model.classifier.num_classes} κλάσεις")
    water_prototype = torch.randn(32)
    model.classifier.add_class(water_prototype)
    print(f"CosineClassifier μετά : {model.classifier.num_classes} κλάσεις  ← Water προστέθηκε")
