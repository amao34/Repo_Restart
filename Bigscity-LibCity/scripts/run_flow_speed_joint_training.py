"""
Stage 2.2: train PDFormer with Flow+Speed input and joint Flow+Speed output.

Speed loss and speed metrics use the Speed_Valid sidecar, so invalid speed
targets do not affect optimization or evaluation.

Usage:
    cd Bigscity-LibCity
    python scripts/run_flow_speed_joint_training.py
"""

import argparse
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

from libcity.pipeline import run_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='SUMO_BEIJING_FIXED_V2')
    parser.add_argument('--config_file', default='sumo_pdformer_flow_speed_joint')
    parser.add_argument('--saved_model', action='store_true', default=True)
    parser.add_argument('--no_train', action='store_true')
    args = parser.parse_args()

    dataset_dir = os.path.join(PROJECT_DIR, 'raw_data', args.dataset)
    speed_valid_path = os.path.join(dataset_dir, args.dataset + '_speed_valid.npz')
    if not os.path.exists(dataset_dir):
        raise SystemExit(
            'Dataset {} not found at {}. Restore the full 1008-node '
            'Flow+Speed dataset before running Stage 2.2.'.format(
                args.dataset, dataset_dir))
    if not os.path.exists(speed_valid_path):
        raise SystemExit(
            'Speed_Valid sidecar not found at {}. Stage 2.2 requires it for '
            'masked speed loss and metrics.'.format(speed_valid_path))

    print('=' * 60)
    print('Stage 2.2: Flow+Speed input -> Flow+Speed output')
    print('Dataset: {}'.format(args.dataset))
    print('Config: {}'.format(args.config_file))
    print('Input variables: traffic_flow, traffic_speed')
    print('Output variables: traffic_flow, traffic_speed')
    print('Speed mask: {}'.format(speed_valid_path))
    print('=' * 60)
    print()

    run_model(
        task='traffic_state_pred',
        model_name='PDFormer',
        dataset_name=args.dataset,
        config_file=args.config_file,
        saved_model=args.saved_model,
        train=not args.no_train,
    )

    print()
    print('[INFO] Stage 2.2 training/evaluation completed.')


if __name__ == '__main__':
    main()
