# CS3T-UNet — Full Research Reproduction
**Paper:** "Cross-shaped Separated Spatial-Temporal UNet Transformer For Accurate Channel Prediction"
IEEE INFOCOM 2024 — Kang, Hu, Chen, Huang, Zhang, Cheng

---

## Files

| File | Purpose |
|------|---------|
| `model.py` | Complete CS3T-UNet architecture |
| `dataset.py` | DataLoader from `.mat` / `.npy` files |
| `losses.py` | MSE loss + NMSE metric (paper Eq. 10) |
| `train.py` | Full training loop (AdamW, warmup+cosine, AMP) |
| `evaluate.py` | Standalone evaluation + all plots |
| `visualize.py` | Loss curves, NMSE, CSI heatmaps, error maps, temporal waterfall |
| `requirements.txt` | Dependencies |

---

## Quick Start

```bash
pip install -r requirements.txt

# Train using pre-sliced .mat sequences from MATLAB script
python train.py \
  --train_x train_L5.mat  --train_y train_L5.mat \
  --test_x  test_L5.mat   --test_y  test_L5.mat  \
  --x_train_key X_train_L5 --y_train_key Y_train_L5 \
  --x_test_key  X_test_L5  --y_test_key  Y_test_L5  \
  --epochs 400 --batch_size 32 --lr 2e-3 \
  --embed_dim 64 --num_blocks 2 2 6 2 \
  --out_dir outputs_L5

# Or train directly from full normalised ADP
python train.py \
  --use_adp_direct \
  --train_adp_path train_adp_norm.mat \
  --test_adp_path  test_adp_norm.mat  \
  --L 5 --out_dir outputs_L5

# Evaluate a saved checkpoint
python evaluate.py \
  --ckpt outputs_L5/best_model.pt \
  --test_x test_L5.mat --test_y test_L5.mat \
  --L 5 --out_dir eval_results
```

---

## Architecture — Paper Section Mapping

| Component | Code | Paper |
|-----------|------|-------|
| Temporal PE | `TemporalPositionalEncoding` | §III-C1, Eq.(7) |
| Cross-shaped spatial attention | `CrossShapedSpatialAttention` | §III-C1, Eq.(5)(6) |
| Group-wise temporal attention | `GroupWiseTemporalAttention` | §III-C1, Eq.(8) |
| CS3T Block | `CS3TBlock` | §III-C2, Fig.8 |
| Merge block | `MergeBlock` | §III-C, Fig.7a |
| Expand block | `ExpandBlock` | §III-C, Fig.7b |
| Patch embedding | `PatchEmbedding` | §III-B |
| 4-level UNet | `CS3TUNet` | §III-D, Fig.3 |
| Skip connections | `DecoderLayer.fuse` | §III-C |
| tanh output | `CS3TUNet.output_act` | §IV-A |
| NMSE metric | `nmse_db()` in losses.py | Eq.(10) |

---

## Hyperparameters (Paper §IV-A — QuaDRiGa Dataset)

| Parameter | Value |
|-----------|-------|
| Embedding dim C | 64 |
| Blocks per level | (2, 2, 6, 2) |
| Patch size | 2×2 |
| Optimizer | AdamW |
| Learning rate | 2e-3 |
| Batch size | 32 |
| Epochs | 400 |
| Warmup | 10 epochs |
| LR decay | Cosine to 1e-5 |
| Loss | MSE |
| T (history) | 10 |
| L (prediction) | 1 or 5 |
| Sliding window stride | 1 |

---

## Data Format

```
Input  X: (B, 2T, Nf, Nt)   — 2T channels: T frames × {real, imag}
Output Y: (B, 2L, Nf, Nt)   — 2L channels: L frames × {real, imag}

Channel layout: [real_t0, imag_t0, real_t1, imag_t1, ..., real_t{T-1}, imag_t{T-1}]
Spatial dims: Nf=64 subcarriers (delay axis), Nt=64 antennas (angle axis)
```

---

## Expected Results (Paper Table I — QuaDRiGa)

| L | Paper NMSE |
|---|-----------|
| 1 | −27.47 dB |
| 5 | −20.58 dB |
| Avg | −24.03 dB |

With mask-guided sparse attention (Table III): L=1 → −32.73 dB

---

## Explicit Assumptions

| Item | Assumption | Reason |
|------|-----------|--------|
| `stripe_width` | 7 | CSwin Transformer default (paper cites [13]) |
| `group_size` | 4 | Not specified; balances temporal granularity |
| Skip connection fusion | Linear(concat) | Paper unclear; add is alternative |
| Validation split | 10% of train | Paper doesn't specify val set |
| Random seed | 42 | Not stated in paper |
| Weight decay | 1e-4 | AdamW standard default |

---

## Training Outputs

```
outputs/
├── best_model.pt          ← best checkpoint by validation NMSE
├── ckpt_epochXXXX.pt      ← periodic checkpoints (every 50 epochs)
├── history.json           ← per-epoch loss, NMSE, learning rate
├── config.json            ← full training configuration
└── plots/
    ├── loss_curves.png         loss + LR schedule
    ├── nmse_curve.png          NMSE vs epoch with paper reference
    ├── csi_comparison_stepN.png  GT vs Predicted heatmaps
    ├── error_map.png           signed error per prediction step
    ├── temporal_sequence.png   waterfall across L steps
    ├── error_histogram.png     error distribution
    └── nmse_per_step.png       per-step NMSE vs paper Fig.10
```
