"""
Validate the collected 1008-node Flow+Speed dataset and Speed_Valid sidecar.

Usage:
  python scripts/validate_flow_speed_dataset.py --dataset SUMO_BEIJING_FIXED_V2
"""

import argparse
import csv
import json
import os

import numpy as np


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_DIR, 'raw_data')


def count_csv_rows(path):
    with open(path, 'r', newline='') as fin:
        return max(0, sum(1 for _ in fin) - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='SUMO_BEIJING_FIXED_V2')
    parser.add_argument('--expected-nodes', type=int, default=1008)
    parser.add_argument('--expected-interval', type=int, default=300)
    args = parser.parse_args()

    dataset_dir = os.path.join(RAW_DATA_DIR, args.dataset)
    config_path = os.path.join(dataset_dir, 'config.json')
    geo_path = os.path.join(dataset_dir, args.dataset + '.geo')
    rel_path = os.path.join(dataset_dir, args.dataset + '.rel')
    dyna_path = os.path.join(dataset_dir, args.dataset + '.dyna')
    speed_valid_path = os.path.join(dataset_dir, args.dataset + '_speed_valid.npz')

    missing = [
        path for path in [config_path, geo_path, rel_path, dyna_path, speed_valid_path]
        if not os.path.exists(path)
    ]
    if missing:
        raise SystemExit('Missing required files:\n' + '\n'.join(missing))

    with open(config_path, 'r') as fin:
        config = json.load(fin)
    info = config['info']
    data_col = info.get('data_col')
    if data_col != ['traffic_flow', 'traffic_speed']:
        raise SystemExit('Unexpected data_col: {}'.format(data_col))
    if int(info.get('output_dim')) != 2:
        raise SystemExit('Unexpected output_dim: {}'.format(info.get('output_dim')))
    if int(info.get('time_intervals')) != args.expected_interval:
        raise SystemExit('Unexpected time_intervals: {}'.format(info.get('time_intervals')))

    geo_rows = count_csv_rows(geo_path)
    rel_rows = count_csv_rows(rel_path)
    dyna_rows = count_csv_rows(dyna_path)
    if geo_rows != args.expected_nodes:
        raise SystemExit('Expected {} geo rows, got {}'.format(
            args.expected_nodes, geo_rows))
    if dyna_rows % args.expected_nodes != 0:
        raise SystemExit('Dyna rows {} are not divisible by {}'.format(
            dyna_rows, args.expected_nodes))
    num_timestamps = dyna_rows // args.expected_nodes

    with open(dyna_path, 'r', newline='') as fin:
        reader = csv.DictReader(fin)
        expected_header = [
            'dyna_id', 'type', 'time', 'entity_id',
            'traffic_flow', 'traffic_speed'
        ]
        if reader.fieldnames != expected_header:
            raise SystemExit('Unexpected dyna header: {}'.format(reader.fieldnames))
        first_rows = [next(reader) for _ in range(min(args.expected_nodes, dyna_rows))]
    first_entities = [int(row['entity_id']) for row in first_rows]
    if first_entities != list(range(len(first_entities))):
        raise SystemExit('First timestamp entity ids are not ordered 0..N-1.')

    sidecar = np.load(speed_valid_path)
    speed_valid = sidecar['speed_valid']
    if speed_valid.shape != (num_timestamps, args.expected_nodes):
        raise SystemExit('Speed_Valid shape {} does not match ({}, {}).'.format(
            speed_valid.shape, num_timestamps, args.expected_nodes))

    print('Dataset validation passed.')
    print('  dataset: {}'.format(args.dataset))
    print('  geo rows: {}'.format(geo_rows))
    print('  rel rows: {}'.format(rel_rows))
    print('  timestamps: {}'.format(num_timestamps))
    print('  dyna rows: {}'.format(dyna_rows))
    print('  speed_valid shape: {}'.format(speed_valid.shape))
    print('  speed_valid rate: {:.6f}'.format(float(speed_valid.mean())))


if __name__ == '__main__':
    os.chdir(PROJECT_DIR)
    main()
