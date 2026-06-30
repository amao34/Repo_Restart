"""
Gate 2: Validate time alignment.

Check that:
- X last step corresponds to time T
- y first step corresponds to time T+1 (5 min later)
- y last step corresponds to time T+12 (60 min later)

Usage:
    cd Bigscity-LibCity
    python scripts/validate_gate2_time_alignment.py
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


def check_time_alignment(dataset_name, config_file, expected_traffic_cols):
    print('=' * 60)
    print('Gate 2: Time Alignment Validation')
    print(f'Dataset: {dataset_name}')
    print(f'Config: {config_file}')
    print('=' * 60)
    print()

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

    # Load dataset
    print('Loading dataset...')
    dataset = get_dataset(config)

    # Access timestamps
    timesolts = dataset.timesolts
    print(f'Total timestamps: {len(timesolts)}')
    print(f'First timestamp: {timesolts[0]}')
    print(f'Last timestamp:  {timesolts[-1]}')
    print()

    input_window = config.get('input_window', 12)
    output_window = config.get('output_window', 12)

    # Validate time intervals
    passed = True
    for i in range(10):
        idx = i * 100
        if idx + input_window + output_window > len(timesolts):
            break
        x_last = timesolts[idx + input_window - 1]
        y_first = timesolts[idx + input_window]
        y_last = timesolts[idx + input_window + output_window - 1]

        diff_first = (y_first - x_last) / np.timedelta64(1, 'm')
        diff_last = (y_last - x_last) / np.timedelta64(1, 'm')

        if diff_first != 5:
            print(f'[FAIL] Sample {idx}: X_last to y_first gap = {diff_first} min, expected 5')
            passed = False
        if diff_last != 60:
            print(f'[FAIL] Sample {idx}: X_last to y_last gap = {diff_last} min, expected 60')
            passed = False

    if passed:
        print('[PASS] Gate 2: Time alignment is correct.')

    # Verify time-of-day feature alignment
    print()
    print('Time-of-day feature spot check:')
    train_dl, _, _ = dataset.get_data()
    device = config.get('device', torch.device('cpu'))
    for batch in train_dl:
        batch.to_tensor(device)
        X = batch['X']
        y = batch['y']
        break

    tod_idx = len(expected_traffic_cols)
    tod_x_last = X[0, -1, 0, tod_idx].item()  # last input step, node 0
    tod_y_first = y[0, 0, 0, tod_idx].item()   # first output step, node 0

    print(f'  X last step time-of-day:  {tod_x_last:.6f}')
    print(f'  y first step time-of-day: {tod_y_first:.6f}')
    print(f'  time-of-day feature index: {tod_idx}')

    step_increment = 5.0 / 1440.0
    expected_y_first = (tod_x_last + step_increment) % 1.0
    print(f'  Expected y first step:    {expected_y_first:.6f}')

    if abs(tod_y_first - expected_y_first) < 0.001:
        print('  [PASS] Time-of-day feature increment is correct.')
    else:
        print('  [FAIL] Time-of-day feature mismatch.')
        passed = False

    print()
    if passed:
        print(f'[PASS] Gate 2 ({dataset_name}): Time alignment validation passed.')
    else:
        print(f'[FAIL] Gate 2 ({dataset_name}): Time alignment check failed.')

    return passed


def main():
    all_passed = True

    if not check_time_alignment(
        dataset_name='SUMO_BEIJING_FIXED_V2_SMOKE',
        config_file='sumo_pdformer_smoke',
        expected_traffic_cols=['traffic_flow', 'traffic_speed']
    ):
        all_passed = False

    flow_smoke_dir = os.path.join(
        PROJECT_DIR, 'raw_data', 'SUMO_BEIJING_FIXED_V2_FLOW_SMOKE'
    )
    if os.path.exists(flow_smoke_dir):
        if not check_time_alignment(
            dataset_name='SUMO_BEIJING_FIXED_V2_FLOW_SMOKE',
            config_file='sumo_pdformer_flow',
            expected_traffic_cols=['traffic_flow']
        ):
            all_passed = False
    else:
        print('\n[SKIP] SUMO_BEIJING_FIXED_V2_FLOW_SMOKE not found.')

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
