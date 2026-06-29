"""
Gate 1: Validate DataLoader dimensions.

Two-variable dataset (SUMO_BEIJING_FIXED_V2_SMOKE, output_dim=2):
    X: [B, 12, 1008, 10]
       - X[..., 0] = traffic_flow (scaled)
       - X[..., 1] = traffic_speed (scaled)
       - X[..., 2] = time-of-day (fraction of day)
       - X[..., 3:10] = day-of-week (one-hot)
    y: [B, 12, 1008, 10] (full feature_dim)

Flow-only dataset (SUMO_BEIJING_FIXED_V2_FLOW_SMOKE, output_dim=1):
    X: [B, 12, 1008, 9]
       - X[..., 0] = traffic_flow (scaled)
       - X[..., 1] = time-of-day (fraction of day)
       - X[..., 2:9] = day-of-week (one-hot)
    y: [B, 12, 1008, 9] (full feature_dim)

Usage:
    cd Bigscity-LibCity
    python scripts/validate_gate1_dataloader.py
"""

import sys
import os
import numpy as np
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

from libcity.config import ConfigParser
from libcity.data import get_dataset


def check_dataloader(dataset_name, config_file, expected_output_dim, expected_traffic_cols):
    """Validate dataloader dimensions for a given dataset."""
    print()
    print('=' * 60)
    print(f'Validating: {dataset_name}')
    print(f'Config: {config_file}')
    print('=' * 60)

    other_args = {
        'dataset': dataset_name,
        'max_epoch': 1,
        'cache_dataset': False,
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

    print(f'\nConfig:')
    print(f'  dataset:       {config["dataset"]}')
    print(f'  output_dim:    {config.get("output_dim", "N/A")}')
    print(f'  input_window:  {config.get("input_window", "N/A")}')
    print(f'  output_window: {config.get("output_window", "N/A")}')

    # Load dataset
    print('\nLoading dataset...')
    dataset = get_dataset(config)
    train_dataloader, _, _ = dataset.get_data()
    data_feature = dataset.get_data_feature()

    print(f'\nData features:')
    print(f'  num_nodes:    {data_feature.get("num_nodes", "N/A")}')
    print(f'  feature_dim:  {data_feature.get("feature_dim", "N/A")}')
    print(f'  output_dim:   {data_feature.get("output_dim", "N/A")}')
    print(f'  ext_dim:      {data_feature.get("ext_dim", "N/A")}')

    # Get first batch
    device = config.get('device', torch.device('cpu'))
    for batch in train_dataloader:
        batch.to_tensor(device)
        X = batch['X']
        y = batch['y']
        break

    print(f'\nBatch shapes:')
    print(f'  X shape: {X.shape}')
    print(f'  y shape: {y.shape}')

    # Validate
    passed = True
    output_dim = data_feature.get('output_dim', 1)
    feature_dim = data_feature.get('feature_dim', 10)

    if output_dim != expected_output_dim:
        print(f'[FAIL] output_dim={output_dim}, expected {expected_output_dim}')
        passed = False

    # Check traffic variable columns
    print(f'\nTraffic columns:')
    for i, col_name in enumerate(expected_traffic_cols):
        print(f'  X[..., {i}] ({col_name}): min={X[..., i].min().item():.4f}, max={X[..., i].max().item():.4f}')

    # Check time-of-day
    tod_idx = len(expected_traffic_cols)
    if feature_dim > tod_idx:
        print(f'  X[..., {tod_idx}] (time-of-day): min={X[..., tod_idx].min().item():.4f}, max={X[..., tod_idx].max().item():.4f}')

    # Check weekday one-hot
    dow_start = tod_idx + 1
    dow_end = dow_start + 7
    if feature_dim >= dow_end:
        dow_sum = X[..., dow_start:dow_end].sum(dim=-1)
        print(f'  X[..., {dow_start}:{dow_end}] (weekday): sum min={dow_sum.min().item():.4f}, max={dow_sum.max().item():.4f}')

    print()
    if passed:
        print(f'[PASS] Gate 1 ({dataset_name}): DataLoader dimensions correct.')
    else:
        print(f'[FAIL] Gate 1 ({dataset_name}): Dimension check failed.')

    return passed


def main():
    print('=' * 60)
    print('Gate 1: DataLoader Dimension Validation')
    print('=' * 60)

    all_passed = True

    # Check 1: Two-variable smoke dataset
    if not check_dataloader(
        dataset_name='SUMO_BEIJING_FIXED_V2_SMOKE',
        config_file='sumo_pdformer_smoke',
        expected_output_dim=2,
        expected_traffic_cols=['traffic_flow', 'traffic_speed']
    ):
        all_passed = False

    # Check 2: Flow-only smoke dataset (if exists)
    flow_smoke_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'raw_data', 'SUMO_BEIJING_FIXED_V2_FLOW_SMOKE'
    )
    if os.path.exists(flow_smoke_dir):
        if not check_dataloader(
            dataset_name='SUMO_BEIJING_FIXED_V2_FLOW_SMOKE',
            config_file='sumo_pdformer_flow',
            expected_output_dim=1,
            expected_traffic_cols=['traffic_flow']
        ):
            all_passed = False
    else:
        print('\n[SKIP] SUMO_BEIJING_FIXED_V2_FLOW_SMOKE not found.')

    print()
    if all_passed:
        print('[PASS] Gate 1: All DataLoader dimension checks passed.')
    else:
        print('[FAIL] Gate 1: Some checks failed.')

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
