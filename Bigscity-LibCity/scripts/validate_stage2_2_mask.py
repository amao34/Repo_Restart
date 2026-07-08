"""
Validate Stage 2.2 Flow+Speed joint-output data wiring without PDFormer DTW.

This uses TrafficStatePointDataset deliberately, so it checks dimensions and
Speed_Valid alignment without triggering PDFormer DTW/K-Shape cache generation.

Usage:
    cd Bigscity-LibCity
    python scripts/validate_stage2_2_mask.py
"""

import os
import sys

import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_evaluator


def main():
    dataset_name = 'SUMO_BEIJING_FIXED_V2'
    config = ConfigParser(
        task='traffic_state_pred',
        model='PDFormer',
        dataset=dataset_name,
        config_file='sumo_pdformer_flow_speed_joint',
        saved_model=False,
        train=True,
        other_args={
            'dataset_class': 'TrafficStatePointDataset',
            'max_epoch': 1,
            'cache_dataset': False,
        },
    )

    dataset = get_dataset(config)
    train_dataloader, _, _ = dataset.get_data()
    data_feature = dataset.get_data_feature()

    batch = next(iter(train_dataloader))
    batch.to_tensor(config.get('device', torch.device('cpu')))

    checks = [
        ('input_dim', data_feature.get('input_dim') == 2),
        ('output_dim', data_feature.get('output_dim') == 2),
        ('feature_dim', data_feature.get('feature_dim') == 10),
        ('X shape', tuple(batch['X'].shape[1:]) == (12, 1008, 10)),
        ('y shape', tuple(batch['y'].shape[1:]) == (12, 1008, 10)),
        ('speed_valid_y shape',
         tuple(batch['speed_valid_y'].shape[1:]) == (12, 1008, 1)),
    ]
    passed = True
    for name, ok in checks:
        print('[{}] {}'.format('PASS' if ok else 'FAIL', name))
        passed = passed and ok

    valid_rate = float(batch['speed_valid_y'].mean().item())
    print('speed_valid_y first-batch rate: {:.6f}'.format(valid_rate))
    if not (0 < valid_rate < 1):
        print('[FAIL] speed_valid_y rate should be between 0 and 1.')
        passed = False

    evaluator = get_evaluator(config)
    y_true = torch.ones(2, 3, 4, 2)
    y_pred = y_true + 0.5
    speed_valid = torch.ones(2, 3, 4, 1)
    evaluator.collect({
        'y_true': y_true,
        'y_pred': y_pred,
        'speed_valid': speed_valid,
    })
    result = evaluator.evaluate()
    evaluator_ok = result.get('Flow_MAE@1') == 0.5 and result.get('Speed_MAE@1') == 0.5
    print('[{}] FlowSpeedEvaluator smoke'.format('PASS' if evaluator_ok else 'FAIL'))
    passed = passed and evaluator_ok

    if passed:
        print('[PASS] Stage 2.2 mask/data wiring is ready.')
        return 0
    print('[FAIL] Stage 2.2 mask/data wiring has issues.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
