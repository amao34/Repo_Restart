"""
Gate 3: Single-batch overfit test.

Uses Flow-only smoke dataset (SUMO_BEIJING_FIXED_V2_FLOW_SMOKE).
Train on a single batch for 300 steps with dropout/drop_path disabled.
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
from libcity.model import get_model


def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    isolated_point_num = np.sum(np.where(d, 0, 1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
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


def main():
    # Use Flow-only smoke dataset
    dataset_name = 'SUMO_BEIJING_FIXED_V2_FLOW_SMOKE'
    config_file = 'sumo_pdformer_flow'

    other_args = {
        'dataset': dataset_name,
        'max_epoch': 1,
        'cache_dataset': False,
        # Disable all dropout for overfit test
        'drop': 0,
        'attn_drop': 0,
        'drop_path': 0,
        'use_curriculum_learning': False,
        'batch_size': 1,
        'learning_rate': 1e-3,
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

    print(f'  output_dim:   {data_feature.get("output_dim")}')
    print(f'  feature_dim:  {data_feature.get("feature_dim")}')

    # Compute Laplacian PE
    print('Computing Laplacian PE...')
    adj_mx = data_feature.get('adj_mx')
    lape_dim = config.get('lape_dim', 8)
    lap_mx = cal_lape(adj_mx, lape_dim).to(config.get('device', torch.device('cpu')))
    print(f'  lap_mx shape: {lap_mx.shape}')

    # Get one batch
    device = config.get('device', torch.device('cpu'))
    for batch in train_dataloader:
        batch.to_tensor(device)
        single_batch = batch
        break

    print(f'Batch X shape: {single_batch["X"].shape}')
    print(f'Batch y shape: {single_batch["y"].shape}')
    print()

    # Create model
    print('Creating model...')
    model = get_model(config, data_feature).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    print()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 1e-3))

    # Training loop
    num_steps = 300
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
                y_true = single_batch['y'][..., :config.get('output_dim', 1)]
                y_pred = pred[..., :config.get('output_dim', 1)]

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
