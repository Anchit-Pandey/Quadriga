"""
train.py — Training pipeline for CS3T-UNet
Paper: INFOCOM 2024 — Kang et al.

[PAPER §IV-A Training Scheme]:
  Optimizer : AdamW
  Epochs    : 400
  Warmup    : 10 epochs (LR linearly increases to initial LR)
  Batch size: 32   (QuaDRiGa dataset)
  LR        : 2e-3 (QuaDRiGa dataset)
  Loss      : MSE
"""

import os
import time
import json
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from model   import CS3TUNet, count_parameters
from dataset import get_dataloaders, CSIDatasetFromADP
from losses  import CompositeLoss, nmse_db, mae_metric
from visualize import (plot_loss_curves, plot_nmse_curve,
                        plot_csi_comparison, plot_error_map,
                        plot_temporal_sequence, plot_error_histogram)


# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────
# WARMUP + COSINE LR SCHEDULE  [PAPER §IV-A]
# "10 epochs warm-up where LR gradually increases"
# After warmup: cosine decay to 0
# ─────────────────────────────────────────────
def get_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(
                    max(1, total_epochs - warmup_epochs))
        return max(1e-5, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────
# SINGLE EPOCH TRAIN
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, scaler,
                use_amp: bool) -> dict:
    model.train()
    total_loss = 0.0
    total_nmse = 0.0
    total_mae  = 0.0
    n_batches  = 0

    for X, Y in loader:
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            pred = model(X)
            loss = criterion(pred, Y)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            batch_nmse = nmse_db(pred, Y)
            batch_mae  = mae_metric(pred, Y)

        total_loss += loss.item()
        total_nmse += batch_nmse
        total_mae  += batch_mae
        n_batches  += 1

    return {
        'loss': total_loss / n_batches,
        'nmse_db': total_nmse / n_batches,
        'mae':  total_mae  / n_batches,
    }


# ─────────────────────────────────────────────
# VALIDATION / TEST EPOCH
# ─────────────────────────────────────────────
@torch.no_grad()
def eval_epoch(model, loader, criterion, device,
               return_samples: bool = False) -> dict:
    model.eval()
    total_loss = 0.0
    total_nmse = 0.0
    total_mae  = 0.0
    n_batches  = 0
    samples    = None

    for i, (X, Y) in enumerate(loader):
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)

        pred = model(X)
        loss = criterion(pred, Y)

        total_loss += loss.item()
        total_nmse += nmse_db(pred, Y)
        total_mae  += mae_metric(pred, Y)
        n_batches  += 1

        if return_samples and i == 0:
            samples = (X.cpu(), Y.cpu(), pred.cpu())

    result = {
        'loss':    total_loss / n_batches,
        'nmse_db': total_nmse / n_batches,
        'mae':     total_mae  / n_batches,
    }
    if return_samples:
        result['samples'] = samples
    return result


# ─────────────────────────────────────────────
# CHECKPOINT UTILS
# ─────────────────────────────────────────────
def save_checkpoint(state: dict, path: str):
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer  and 'optimizer'  in ckpt: optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler  and 'scheduler'  in ckpt: scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt.get('epoch', 0), ckpt.get('best_nmse', float('inf'))


# ─────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────
def train(cfg: dict):
    set_seed(cfg['seed'])
    os.makedirs(cfg['out_dir'], exist_ok=True)

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = cfg.get('use_amp', True) and device.type == 'cuda'
    print(f"\n{'='*60}")
    print(f"  CS3T-UNet Training")
    print(f"  Device : {device}  |  AMP: {use_amp}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────
    if cfg.get('use_adp_direct', False):
        # Load from full ADP .mat files
        from torch.utils.data import DataLoader, random_split
        from dataset import CSIDatasetFromADP
        train_full = CSIDatasetFromADP(cfg['train_adp_path'], 'train_adp',
                                       T=cfg['T'], L=cfg['L'])
        test_ds    = CSIDatasetFromADP(cfg['test_adp_path'],  'test_adp',
                                       T=cfg['T'], L=cfg['L'])
        n_val   = int(len(train_full) * 0.1)
        n_train = len(train_full) - n_val
        train_ds, val_ds = random_split(train_full, [n_train, n_val],
            generator=torch.Generator().manual_seed(42))
        kw = dict(batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                  pin_memory=(device.type=='cuda'))
        loaders = {
            'train': DataLoader(train_ds, shuffle=True,  **kw),
            'val':   DataLoader(val_ds,   shuffle=False, **kw),
            'test':  DataLoader(test_ds,  shuffle=False, **kw),
            'info':  {'n_train': n_train, 'n_val': n_val,
                      'n_test': len(test_ds)}
        }
    else:
        loaders = get_dataloaders(
            train_x_path  = cfg['train_x_path'],
            train_y_path  = cfg['train_y_path'],
            test_x_path   = cfg['test_x_path'],
            test_y_path   = cfg['test_y_path'],
            batch_size    = cfg['batch_size'],
            num_workers   = cfg['num_workers'],
            pin_memory    = (device.type == 'cuda'),
            use_npy       = cfg.get('use_npy', False),
            x_train_key   = cfg.get('x_train_key', 'X_train_L5'),
            y_train_key   = cfg.get('y_train_key', 'Y_train_L5'),
            x_test_key    = cfg.get('x_test_key',  'X_test_L5'),
            y_test_key    = cfg.get('y_test_key',  'Y_test_L5'),
        )

    info = loaders['info']
    print(f"Train: {info['n_train']}  Val: {info['n_val']}  Test: {info['n_test']}")
    print(f"Input shape:  {info.get('input_shape', '(2T,Nf,Nt)')}")
    print(f"Target shape: {info.get('target_shape', '(2L,Nf,Nt)')}\n")

    # ── Model ─────────────────────────────────
    model = CS3TUNet(
        in_channels  = 2 * cfg['T'],
        out_channels = 2 * cfg['L'],
        spatial_h    = cfg['Nf'],
        spatial_w    = cfg['Nt'],
        embed_dim    = cfg['embed_dim'],
        num_blocks   = tuple(cfg['num_blocks']),
        num_heads    = cfg['num_heads'],
        stripe_width = cfg['stripe_width'],
        group_size   = cfg['group_size'],
        attn_drop    = cfg.get('attn_drop', 0.0),
        drop         = cfg.get('drop', 0.0),
    ).to(device)

    p = count_parameters(model)
    print(f"Parameters: {p['total_M']:.2f}M total  "
          f"({p['trainable_M']:.2f}M trainable)")
    print(f"Paper reports: ~19.64M\n")

    # ── Optimizer & Scheduler ─────────────────
    # [PAPER §IV-A]: AdamW, LR=2e-3, batch=32 for QuaDRiGa
    optimizer = AdamW(model.parameters(), lr=cfg['lr'],
                      weight_decay=cfg.get('weight_decay', 1e-4))
    scheduler = get_scheduler(optimizer,
                              warmup_epochs=cfg['warmup_epochs'],
                              total_epochs=cfg['epochs'])
    criterion = CompositeLoss(nmse_weight=cfg.get('nmse_weight', 0.0))
    scaler    = GradScaler(enabled=use_amp)

    # ── Resume ────────────────────────────────
    start_epoch = 0
    best_nmse   = float('inf')
    if cfg.get('resume') and os.path.exists(cfg['resume']):
        start_epoch, best_nmse = load_checkpoint(
            cfg['resume'], model, optimizer, scheduler)
        print(f"Resumed from epoch {start_epoch}, best NMSE={best_nmse:.2f} dB\n")

    # ── History ───────────────────────────────
    history = {'train_loss': [], 'val_loss': [],
               'train_nmse': [], 'val_nmse': [], 'lr': []}

    print(f"{'Epoch':>6}  {'LR':>8}  "
          f"{'TrLoss':>10}  {'TrNMSE(dB)':>12}  "
          f"{'VlLoss':>10}  {'VlNMSE(dB)':>12}  "
          f"{'Time':>8}")
    print("─" * 80)

    for epoch in range(start_epoch, cfg['epochs']):
        t0 = time.time()

        # Train
        tr = train_epoch(model, loaders['train'], optimizer, criterion,
                         device, scaler, use_amp)
        # Validate
        vl = eval_epoch(model, loaders['val'], criterion, device)
        # Step LR
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log
        history['train_loss'].append(tr['loss'])
        history['val_loss'].append(vl['loss'])
        history['train_nmse'].append(tr['nmse_db'])
        history['val_nmse'].append(vl['nmse_db'])
        history['lr'].append(current_lr)

        elapsed = time.time() - t0
        print(f"{epoch+1:>6}  {current_lr:>8.5f}  "
              f"{tr['loss']:>10.6f}  {tr['nmse_db']:>12.2f}  "
              f"{vl['loss']:>10.6f}  {vl['nmse_db']:>12.2f}  "
              f"{elapsed:>6.1f}s")

        # Save best
        if vl['nmse_db'] < best_nmse:
            best_nmse = vl['nmse_db']
            save_checkpoint({
                'epoch':      epoch + 1,
                'model':      model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'scheduler':  scheduler.state_dict(),
                'best_nmse':  best_nmse,
                'config':     cfg,
            }, os.path.join(cfg['out_dir'], 'best_model.pt'))
            print(f"  *** New best: {best_nmse:.2f} dB  (saved)")

        # Periodic checkpoint
        if (epoch + 1) % cfg.get('save_every', 50) == 0:
            save_checkpoint({
                'epoch':     epoch + 1,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_nmse': best_nmse,
                'config':    cfg,
            }, os.path.join(cfg['out_dir'], f'ckpt_epoch{epoch+1:04d}.pt'))

        # NaN guard
        if math.isnan(tr['loss']) or math.isnan(vl['loss']):
            print("NaN loss detected — stopping training.")
            break

    # ── Save history ──────────────────────────
    with open(os.path.join(cfg['out_dir'], 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # ── Final test evaluation ─────────────────
    print(f"\n{'='*60}")
    print("  Final Test Evaluation (best model)")
    print(f"{'='*60}")
    best_ckpt = os.path.join(cfg['out_dir'], 'best_model.pt')
    load_checkpoint(best_ckpt, model)
    test_res = eval_epoch(model, loaders['test'], criterion, device,
                          return_samples=True)
    print(f"  Test NMSE  : {test_res['nmse_db']:.2f} dB")
    print(f"  Test Loss  : {test_res['loss']:.6f}")
    print(f"  Test MAE   : {test_res['mae']:.6f}")
    print(f"\n  Paper reports (L=5): -20.58 dB on QuaDRiGa")
    print(f"  Your result        : {test_res['nmse_db']:.2f} dB")

    # ── Visualisation ─────────────────────────
    print("\nGenerating plots...")
    plot_dir = os.path.join(cfg['out_dir'], 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    plot_loss_curves(history, plot_dir)
    plot_nmse_curve(history, plot_dir)

    if 'samples' in test_res:
        X_s, Y_s, P_s = test_res['samples']
        plot_csi_comparison(Y_s, P_s, plot_dir, cfg['L'])
        plot_error_map(Y_s, P_s, plot_dir, cfg['L'])
        plot_temporal_sequence(Y_s, P_s, plot_dir, cfg['L'])
        plot_error_histogram(Y_s, P_s, plot_dir)

    print(f"\nAll outputs saved to: {cfg['out_dir']}")
    return history, test_res


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="CS3T-UNet Training")

    # Data paths
    p.add_argument('--train_x',   default='train_L5.mat')
    p.add_argument('--train_y',   default='train_L5.mat')
    p.add_argument('--test_x',    default='test_L5.mat')
    p.add_argument('--test_y',    default='test_L5.mat')
    p.add_argument('--x_train_key', default='X_train_L5')
    p.add_argument('--y_train_key', default='Y_train_L5')
    p.add_argument('--x_test_key',  default='X_test_L5')
    p.add_argument('--y_test_key',  default='Y_test_L5')
    p.add_argument('--use_npy',   action='store_true')
    # Alternative: load from full ADP
    p.add_argument('--use_adp_direct', action='store_true')
    p.add_argument('--train_adp_path', default='train_adp_norm.mat')
    p.add_argument('--test_adp_path',  default='test_adp_norm.mat')

    # Architecture [PAPER §IV-A defaults]
    p.add_argument('--embed_dim',    type=int, default=64)
    p.add_argument('--num_blocks',   type=int, nargs=4, default=[2,2,6,2])
    p.add_argument('--num_heads',    type=int, default=8)
    p.add_argument('--stripe_width', type=int, default=7)
    p.add_argument('--group_size',   type=int, default=4)
    p.add_argument('--attn_drop',    type=float, default=0.0)
    p.add_argument('--drop',         type=float, default=0.0)

    # Data dims
    p.add_argument('--T',  type=int, default=10,  help='Historical frames')
    p.add_argument('--L',  type=int, default=5,   help='Prediction frames')
    p.add_argument('--Nf', type=int, default=64,  help='Subcarriers')
    p.add_argument('--Nt', type=int, default=64,  help='Antennas')

    # Training [PAPER §IV-A]
    p.add_argument('--epochs',         type=int,   default=400)
    p.add_argument('--warmup_epochs',  type=int,   default=10)
    p.add_argument('--batch_size',     type=int,   default=32)
    p.add_argument('--lr',             type=float, default=2e-3)
    p.add_argument('--weight_decay',   type=float, default=1e-4)
    p.add_argument('--nmse_weight',    type=float, default=0.0)
    p.add_argument('--num_workers',    type=int,   default=4)
    p.add_argument('--use_amp',        action='store_true', default=True)
    p.add_argument('--save_every',     type=int,   default=50)
    p.add_argument('--resume',         type=str,   default='')

    # Output
    p.add_argument('--out_dir', default='./outputs')
    p.add_argument('--seed',    type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = vars(args)
    cfg['train_x_path'] = cfg.pop('train_x')
    cfg['train_y_path'] = cfg.pop('train_y')
    cfg['test_x_path']  = cfg.pop('test_x')
    cfg['test_y_path']  = cfg.pop('test_y')

    # Save config
    os.makedirs(cfg['out_dir'], exist_ok=True)
    with open(os.path.join(cfg['out_dir'], 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    train(cfg)
