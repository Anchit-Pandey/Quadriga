"""
dataset.py — Dataset and DataLoader for CS3T-UNet
Paper: INFOCOM 2024 — Kang et al.

Supports loading from:
  (a) train_L5.mat / test_L5.mat  (pre-built sliding-window sequences,
      output of generate_data_paper_exact.m)
  (b) train_adp_norm.mat           (full normalised ADP; sliding window
      applied here in Python if .mat sequences not available)

Data format (MATLAB output → PyTorch):
  MATLAB: (N, T, 2, Nc, Nt)  float32
  Paper Fig.3 input: (Nf × Nt × 2T)  — real/imag of T frames combined
  This loader returns:
    X: (B, 2T, Nf, Nt)   input to model
    Y: (B, 2L, Nf, Nt)   target
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import scipy.io as sio
import h5py


# ─────────────────────────────────────────────
# LOAD .mat (supports both v7.3/HDF5 and older)
# ─────────────────────────────────────────────
def load_mat(path: str, key: str) -> np.ndarray:
    """Load a variable from a .mat file (v5 or v7.3/HDF5)."""
    try:
        data = sio.loadmat(path, variable_names=[key])
        arr  = data[key]
        return np.array(arr, dtype=np.float32)
    except Exception:
        # v7.3 HDF5 format
        with h5py.File(path, 'r') as f:
            arr = f[key][()]
        # HDF5 stores in column-major (MATLAB) order; transpose to row-major
        arr = arr.T if arr.ndim == 2 else np.transpose(arr)
        return np.array(arr, dtype=np.float32)


# ─────────────────────────────────────────────
# SLIDING WINDOW (used if loading full ADP)
# [PAPER §IV-A]: stride-1 window of length T+L
# ─────────────────────────────────────────────
def build_sequences(H: np.ndarray, T: int, L: int) -> tuple:
    """
    H: (N, num_frames, 2, Nc, Nt)
    Returns:
      X: (N_aug, T, 2, Nc, Nt)
      Y: (N_aug, L, 2, Nc, Nt)
    """
    N, F, C, Nf, Nt = H.shape
    win   = T + L
    nw    = F - win + 1
    total = N * nw
    X = np.zeros((total, T, C, Nf, Nt), dtype=np.float32)
    Y = np.zeros((total, L, C, Nf, Nt), dtype=np.float32)
    idx = 0
    for n in range(N):
        for w in range(nw):
            X[idx] = H[n, w      : w+T]
            Y[idx] = H[n, w+T    : w+T+L]
            idx += 1
    return X, Y


# ─────────────────────────────────────────────
# FORMAT CONVERSION: (N,T,2,Nf,Nt) → (N,2T,Nf,Nt)
# [PAPER §III-B]: "combine temporal and complex dimensions → Nf×Nt×2T"
# We do: dim0=N, dim1=2T (interleave real/imag per time step), dim2=Nf, dim3=Nt
# ─────────────────────────────────────────────
def to_model_input(arr: np.ndarray) -> np.ndarray:
    """
    arr: (N, T, 2, Nf, Nt)  — (sample, time, ri, freq, ant)
    Returns: (N, 2T, Nf, Nt)
    Interleaving: [real_t0, imag_t0, real_t1, imag_t1, ...]
    """
    N, T, C, Nf, Nt = arr.shape
    assert C == 2, f"Expected 2 (real/imag), got {C}"
    # Reshape: (N, T, 2, Nf, Nt) → (N, T*2, Nf, Nt)
    out = arr.reshape(N, T * 2, Nf, Nt)
    return out


# ─────────────────────────────────────────────
# PYTORCH DATASET
# ─────────────────────────────────────────────
class CSIDataset(Dataset):
    """
    Loads pre-built sliding-window sequences from .mat files.

    Args:
        x_path: path to X .mat or .npy file
        y_path: path to Y .mat or .npy file
        x_key : variable name in .mat for inputs  (default 'X_train_L5')
        y_key : variable name in .mat for targets (default 'Y_train_L5')
        use_npy: if True, load .npy instead of .mat
    """
    def __init__(self, x_path: str, y_path: str,
                 x_key: str = 'X_train_L5', y_key: str = 'Y_train_L5',
                 use_npy: bool = False):

        if use_npy:
            X_raw = np.load(x_path).astype(np.float32)
            Y_raw = np.load(y_path).astype(np.float32)
        else:
            X_raw = load_mat(x_path, x_key)
            Y_raw = load_mat(y_path, y_key)

        # X_raw: (N, T, 2, Nf, Nt) → model input (N, 2T, Nf, Nt)
        # Y_raw: (N, L, 2, Nf, Nt) → model target (N, 2L, Nf, Nt)
        N, T, C, Nf, Nt = X_raw.shape
        _, L, _, _, _   = Y_raw.shape

        self.X = torch.from_numpy(to_model_input(X_raw))  # (N, 2T, Nf, Nt)
        self.Y = torch.from_numpy(to_model_input(Y_raw))  # (N, 2L, Nf, Nt)

        self.N  = N
        self.T  = T
        self.L  = L
        self.Nf = Nf
        self.Nt = Nt

        # Sanity checks
        assert not torch.isnan(self.X).any(), "NaN in X"
        assert not torch.isnan(self.Y).any(), "NaN in Y"
        assert self.X.abs().max() <= 1.0 + 1e-4, \
            f"X values exceed ±1: max={self.X.abs().max():.4f}"
        assert self.Y.abs().max() <= 1.0 + 1e-4, \
            f"Y values exceed ±1: max={self.Y.abs().max():.4f}"

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __repr__(self):
        return (f"CSIDataset(N={self.N}, T={self.T}, L={self.L}, "
                f"Nf={self.Nf}, Nt={self.Nt})")


class CSIDatasetFromADP(Dataset):
    """
    Alternative loader: takes the full normalised ADP .mat and
    applies the sliding window here (useful if you only have
    train_adp_norm.mat rather than the pre-sliced train_L5.mat).
    """
    def __init__(self, adp_path: str, key: str = 'train_adp',
                 T: int = 10, L: int = 5):
        H = load_mat(adp_path, key)         # (N, 20, 2, 64, 64)
        X_raw, Y_raw = build_sequences(H, T, L)
        self.X = torch.from_numpy(to_model_input(X_raw))
        self.Y = torch.from_numpy(to_model_input(Y_raw))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ─────────────────────────────────────────────
# DATALOADER FACTORY
# ─────────────────────────────────────────────
def get_dataloaders(
    train_x_path: str,
    train_y_path: str,
    test_x_path:  str,
    test_y_path:  str,
    batch_size:   int   = 32,
    val_split:    float = 0.1,
    num_workers:  int   = 4,
    pin_memory:   bool  = True,
    use_npy:      bool  = False,
    x_train_key:  str   = 'X_train_L5',
    y_train_key:  str   = 'Y_train_L5',
    x_test_key:   str   = 'X_test_L5',
    y_test_key:   str   = 'Y_test_L5',
) -> dict:
    """
    Returns dict with 'train', 'val', 'test' DataLoaders.
    [PAPER §IV-A]: batch_size=32, lr=2e-3 for QuaDRiGa dataset.
    val_split carved from training set (paper does not specify val set;
    we reserve 10% of train for validation).   [ASSUMPTION: val_split=0.1]
    """
    train_full = CSIDataset(train_x_path, train_y_path,
                            x_train_key, y_train_key, use_npy)
    test_ds    = CSIDataset(test_x_path,  test_y_path,
                            x_test_key,  y_test_key,  use_npy)

    # Validation split
    n_val   = int(len(train_full) * val_split)
    n_train = len(train_full) - n_val
    train_ds, val_ds = random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    loader_kw = dict(batch_size=batch_size, num_workers=num_workers,
                     pin_memory=pin_memory)

    return {
        'train': DataLoader(train_ds, shuffle=True,  **loader_kw),
        'val':   DataLoader(val_ds,   shuffle=False, **loader_kw),
        'test':  DataLoader(test_ds,  shuffle=False, **loader_kw),
        'info':  {
            'n_train':   n_train,
            'n_val':     n_val,
            'n_test':    len(test_ds),
            'input_shape':  tuple(train_full.X.shape[1:]),
            'target_shape': tuple(train_full.Y.shape[1:]),
        }
    }


if __name__ == "__main__":
    # Synthetic test without real .mat files
    import tempfile, os
    T, L, Nf, Nt = 10, 5, 64, 64
    N_win = 6   # (20 - T - L + 1) = 6 windows per sample
    N_s   = 100

    X_fake = np.random.uniform(-1, 1, (N_s * N_win, T, 2, Nf, Nt)).astype(np.float32)
    Y_fake = np.random.uniform(-1, 1, (N_s * N_win, L, 2, Nf, Nt)).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        xp = os.path.join(td, 'X.npy')
        yp = os.path.join(td, 'Y.npy')
        np.save(xp, X_fake)
        np.save(yp, Y_fake)

        ds = CSIDataset(xp, yp, use_npy=True)
        print(ds)
        x, y = ds[0]
        print(f"X[0]: {x.shape}  Y[0]: {y.shape}")
        assert x.shape == (2*T, Nf, Nt)
        assert y.shape == (2*L, Nf, Nt)
        print("Dataset test PASSED")
