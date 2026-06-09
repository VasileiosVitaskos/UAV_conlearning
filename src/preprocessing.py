"""
preprocessing.py
Normalization pipeline για FRACTAL και YellowScan datasets.
Stats fit από train set με partial_fit (IncrementalScaler).
"""

import json
import numpy as np
from pathlib import Path


STATS_PATH = Path(__file__).parent.parent / "outputs" / "normalizer_stats.json"


def load_stats() -> dict:
    """Φόρτωση normalizer stats από JSON."""
    with open(STATS_PATH) as f:
        return json.load(f)


def normalize(
    x_rel:      np.ndarray,
    y_rel:      np.ndarray,
    z_rel:      np.ndarray,
    intensity:  np.ndarray,
    return_num: np.ndarray,
    n_returns:  np.ndarray,
    scan_angle: np.ndarray,
    is_fractal: bool = True,
    stats:      dict = None,
) -> np.ndarray:
    """
    Εφαρμόζει normalization στα 7 features.

    Parameters
    ----------
    x_rel, y_rel, z_rel : σχετικές συντεταγμένες (centered στο patch)
    intensity            : raw intensity values
    return_num           : return number (1-6)
    n_returns            : number of returns (1-6)
    scan_angle           : scan angle (LAS 1.4 units αν is_fractal=True, μοίρες αν False)
    is_fractal           : True → scan_angle × 0.006 / 60, False → scan_angle / 60
    stats                : normalizer stats dict (αν None, φορτώνει από JSON)

    Returns
    -------
    np.ndarray shape (N, 7), dtype float32
    """
    if stats is None:
        stats = load_stats()

    N = len(x_rel)
    out = np.zeros((N, 7), dtype=np.float32)

    # StandardScaler features
    out[:, 0] = (x_rel     - stats["x_rel"]["mean"])     / stats["x_rel"]["std"]
    out[:, 1] = (y_rel     - stats["y_rel"]["mean"])     / stats["y_rel"]["std"]
    out[:, 2] = (z_rel     - stats["z_rel"]["mean"])     / stats["z_rel"]["std"]
    # Run 6+: log1p transform before z-score (intensity is right-skewed, std/mean≈2).
    # Amplifies the Water vs Ground gap from 0.12 → ~1.4 normalized units.
    # Stats in normalizer_stats.json are pre-computed on log1p(raw_intensity).
    out[:, 3] = (np.log1p(intensity) - stats["intensity"]["mean"]) / stats["intensity"]["std"]

    # Fixed range features
    out[:, 4] = return_num / 6.0
    out[:, 5] = n_returns  / 6.0

    # Scan angle — διαφορετικές μονάδες ανά dataset
    if is_fractal:
        out[:, 6] = scan_angle * 0.006 / 60.0   # LAS 1.4: units 0.006° → μοίρες → /60
    else:
        out[:, 6] = scan_angle / 60.0            # YellowScan: ήδη μοίρες → /60

    return out


def extract_features(las, is_fractal: bool = True, valid_classes: set = None, stats: dict = None) -> tuple:
    """
    Φορτώνει ένα laspy object και επιστρέφει (X, y) normalized.

    Parameters
    ----------
    las           : laspy.LasData object
    is_fractal    : True για FRACTAL, False για YellowScan
    valid_classes : set κλάσεων που κρατάμε (default: {2,3,4,5,6,9,17,64})

    Returns
    -------
    X : np.ndarray (N, 7) normalized features
    y : np.ndarray (N,)   class labels (ή None αν unclassified)
    """
    if valid_classes is None:
        valid_classes = {2, 3, 4, 5, 6, 9, 17, 64}

    labels = np.array(las.classification, dtype=np.int32)
    mask   = np.isin(labels, list(valid_classes))

    x  = np.array(las.x)[mask]
    y  = np.array(las.y)[mask]
    z  = np.array(las.z)[mask]

    X = normalize(
        x_rel      = x - x.mean(),
        y_rel      = y - y.mean(),
        z_rel      = z - z.mean(),
        intensity  = np.array(las.intensity)[mask].astype(np.float32),
        return_num = np.array(las.return_number)[mask].astype(np.float32),
        n_returns  = np.array(las.number_of_returns)[mask].astype(np.float32),
        scan_angle = np.array(las.scan_angle if is_fractal else las.scan_angle_rank)[mask].astype(np.float32),
        is_fractal = is_fractal,
        stats      = stats,    # None → φορτώνει αυτόματα από JSON
    )

    return X, labels[mask]


if __name__ == "__main__":
    # Quick test
    import laspy
    from pathlib import Path

    f = sorted(Path("data/train/00").glob("*.laz"))[0]
    las = laspy.read(str(f))
    X, y = extract_features(las, is_fractal=True)

    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    print(f"y shape: {y.shape}, classes: {np.unique(y)}")
    print(f"\n{'Feature':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 44)
    names = ["x_rel", "y_rel", "z_rel", "intensity", "return_num", "n_returns", "scan_angle"]
    for i, name in enumerate(names):
        print(f"{name:<12} {X[:,i].mean():>8.3f} {X[:,i].std():>8.3f} {X[:,i].min():>8.3f} {X[:,i].max():>8.3f}")
