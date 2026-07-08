"""
Stage 2.1: train PDFormer with historical Flow+Speed input and Flow output.

This keeps the frozen Flow-only baseline untouched and tests whether historical
speed improves future flow prediction on the final 1008-node network.

Usage:
    cd Bigscity-LibCity
    python scripts/run_flow_speed_to_flow_training.py
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
    parser.add_argument('--config_file', default='sumo_pdformer_flow_speed_to_flow')
    parser.add_argument('--saved_model', action='store_true', default=True)
    parser.add_argument('--no_train', action='store_true')
    args = parser.parse_args()

    dataset_dir = os.path.join(PROJECT_DIR, 'raw_data', args.dataset)
    if not os.path.exists(dataset_dir):
        raise SystemExit(
            'Dataset {} not found at {}. Copy or generate the full two-variable '
            '1008-node dataset before running Stage 2.1.'.format(args.dataset, dataset_dir))

    print('=' * 60)
    print('Stage 2.1: Flow+Speed input -> Flow output')
    print('Dataset: {}'.format(args.dataset))
    print('Config: {}'.format(args.config_file))
    print('Input variables: traffic_flow, traffic_speed')
    print('Output variable: traffic_flow')
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
    print('[INFO] Stage 2.1 training/evaluation completed.')


if __name__ == '__main__':
    main()
