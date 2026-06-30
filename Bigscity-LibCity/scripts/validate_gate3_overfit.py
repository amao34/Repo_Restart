"""
Gate 3: Single-batch overfit test.

Uses Flow-only smoke dataset (SUMO_BEIJING_FIXED_V2_FLOW_SMOKE).
Train on a selected high-signal single batch with dropout/drop_path disabled.
Expected:
- Loss decreases significantly
- R² approaches 1
- Predictions are not constant

Usage:
    cd Bigscity-LibCity
    python scripts/validate_gate3_overfit.py
"""

import sys
import os
import numpy as np
import torch
import scipy.sparse as sp

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_model, set_random_seed


def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    isolated_point_num = np.sum(np.where(d, 0, 1))
    d_flat = d.flatten()
    d_inv_sqrt = np.zeros_like(d_flat, dtype=np.float64)
    nonzero = d_flat > 0
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(d_flat[nonzero])
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian, isolated_point_num


def cal_lape(adj_mx, lape_dim=8):
    L, isolated_point_num = calculate_normalized_laplacian(adj_mx)
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    laplacian_pe = torch.from_numpy(
        EigVec[:, isolated_point_num + 1: lape_dim + isolated_point_num + 1]
    ).float()
    laplacian_pe.requires_grad = False
    return laplacian_pe


def select_overfit_batch(train_dataloader, data_feature, output_dim, device):
    """Pick a deterministic high-variance batch for the overfit probe."""
    scaler = data_feature['scaler']
    best_batch = None
    best_stats = None
    best_score = -1.0

    for batch_index, batch in enumerate(train_dataloader):
        batch.to_tensor(device)
        with torch.no_grad():
            y_true = batch['y'][..., :output_dim]
            y_true_inv = scaler.inverse_transform(y_true)
            y_np = y_true_inv.detach().cpu().numpy()

        target_std = float(y_np.std())
        positive_frac = float((np.abs(y_np) > 1e-8).mean())
        score = target_std * max(positive_frac, 1e-8)
        if score > best_score:
            best_score = score
            best_batch = batch
            best_stats = {
                'index': batch_index,
                'mean': float(y_np.mean()),
                'std': target_std,
                'min': float(y_np.min()),
                'max': float(y_np.max()),
                'positive_frac': positive_frac,
            }

    if best_batch is None:
        raise RuntimeError('No training batch is available for Gate 3.')
    return best_batch, best_stats


def main():
    # Use Flow-only smoke dataset
    dataset_name = 'SUMO_BEIJING_FIXED_V2_FLOW_SMOKE'
    config_file = 'sumo_pdformer_flow'
    num_steps = int(os.environ.get('GATE3_STEPS', '800'))

    other_args = {
        'dataset': dataset_name,
        'max_epoch': max(1, num_steps),
        'cache_dataset': False,
        # Disable all dropout for overfit test
        'drop': 0,
        'attn_drop': 0,
        'drop_path': 0,
        'use_curriculum_learning': False,
        'batch_size': 1,
        'learning_rate': 1e-3,
        # MSE aligns the optimization target with the R² overfit check. MAE
        # can prefer near-zero predictions on sparse flow batches.
        'set_loss': 'mse',
        'seed': 2026,
    }

    config = ConfigParser(
        task='traffic_state_pred',
        model='PDFormer',
        dataset=dataset_name,
        config_file=config_file,
        saved_model=False,
        train=True,
        other_args=other_args,
    )
    set_random_seed(config.get('seed', 2026))

    print('=' * 60)
    print('Gate 3: Single-Batch Overfit Test')
    print(f'Dataset: {dataset_name}')
    print('=' * 60)
    print()

    # Load dataset
    print('Loading dataset...')
    dataset = get_dataset(config)
    train_dataloader, _, _ = dataset.get_data()
    data_feature = dataset.get_data_feature()
    output_dim = data_feature.get('output_dim', config.get('output_dim', 1))

    print(f'  output_dim:   {output_dim}')
    print(f'  feature_dim:  {data_feature.get("feature_dim")}')

    # Compute Laplacian PE
    print('Computing Laplacian PE...')
    adj_mx = data_feature.get('adj_mx')
    lape_dim = config.get('lape_dim', 8)
    lap_mx = cal_lape(adj_mx, lape_dim).to(config.get('device', torch.device('cpu')))
    print(f'  lap_mx shape: {lap_mx.shape}')

    # Select one informative batch. Randomly taking the first batch is flaky
    # because SUMO flow is highly sparse and MAE can be minimized by a near-zero
    # predictor on many low-signal batches.
    device = config.get('device', torch.device('cpu'))
    single_batch, batch_stats = select_overfit_batch(train_dataloader, data_feature, output_dim, device)

    print(f'Batch X shape: {single_batch["X"].shape}')
    print(f'Batch y shape: {single_batch["y"].shape}')
    print(f'Selected batch index: {batch_stats["index"]}')
    print(
        'Target stats: '
        f'mean={batch_stats["mean"]:.6f}, '
        f'std={batch_stats["std"]:.6f}, '
        f'min={batch_stats["min"]:.6f}, '
        f'max={batch_stats["max"]:.6f}, '
        f'positive_frac={batch_stats["positive_frac"]:.4f}'
    )
    print()

    # Create model
    print('Creating model...')
    set_random_seed(config.get('seed', 2026))
    model = get_model(config, data_feature).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    print()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 1e-3))

    # Training loop
    print(f'Training for {num_steps} steps on single batch...')
    print()

    losses = []
    r2 = 0.0
    pred_std = 0.0
    for step in range(num_steps):
        model.train()
        optimizer.zero_grad()
        loss = model.calculate_loss(single_batch, batches_seen=step, lap_mx=lap_mx)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if step % 50 == 0 or step == num_steps - 1:
            with torch.no_grad():
                model.eval()
                pred = model.predict(single_batch, lap_mx=lap_mx)
                y_true = single_batch['y'][..., :output_dim]
                y_pred = pred[..., :output_dim]

                scaler = data_feature['scaler']
                y_true_inv = scaler.inverse_transform(y_true)
                y_pred_inv = scaler.inverse_transform(y_pred)

                y_true_flat = y_true_inv.cpu().flatten().numpy()
                y_pred_flat = y_pred_inv.cpu().flatten().numpy()
                ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
                ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                pred_std = y_pred_flat.std()

            print(f'  Step {step:4d}: loss={loss_val:.6f}, R²={r2:.4f}, pred_std={pred_std:.6f}')

    print()

    # Final evaluation
    passed = True

    loss_ratio = losses[-1] / losses[0] if losses[0] > 0 else 1.0
    print(f'Loss ratio (final/initial): {loss_ratio:.4f}')
    if loss_ratio > 0.5:
        print('[FAIL] Loss did not decrease significantly.')
        passed = False
    else:
        print('[PASS] Loss decreased significantly.')

    if r2 < 0.5:
        print(f'[FAIL] R²={r2:.4f} < 0.5, model is not fitting.')
        passed = False
    else:
        print(f'[PASS] R²={r2:.4f} >= 0.5, model is fitting.')

    if pred_std < 1e-6:
        print(f'[FAIL] Predictions are nearly constant (std={pred_std:.8f}).')
        passed = False
    else:
        print(f'[PASS] Predictions are not constant (std={pred_std:.6f}).')

    print()
    if passed:
        print('[PASS] Gate 3: Single-batch overfit test passed.')
    else:
        print('[FAIL] Gate 3: Single-batch overfit test failed.')

    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
