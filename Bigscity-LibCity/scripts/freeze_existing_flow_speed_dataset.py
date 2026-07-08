"""
Freeze an existing 1008-node Flow+Speed dataset into raw_data.

This is for the recovered SUMO_BEIJING_FIXED_V2 artifacts that already contain
LibCity .dyna/.geo/.rel files and a CSV with Speed_Valid.

Example:
  python scripts/freeze_existing_flow_speed_dataset.py ^
    --source-dir D:\\Sumo\\tools\\2026-03-11-08-59-53\\SUMO_BEIJING_FIXED_V2 ^
    --speed-valid-csv D:\\Sumo\\tools\\2026-03-11-08-59-53\\beijing_base_v2_30days.csv ^
    --edge-ids-json D:\\Sumo\\tools\\2026-03-11-08-59-53\\beijing_base_v2_adj_edge_ids.json ^
    --source-manifest D:\\Sumo\\tools\\2026-03-11-08-59-53\\beijing_fixed_v2_manifest.json ^
    --dataset SUMO_BEIJING_FIXED_V2 ^
    --overwrite
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shutil

import numpy as np


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_DIR, 'raw_data')


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as fin:
        while True:
            chunk = fin.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def count_rows(path):
    with open(path, 'r', newline='') as fin:
        return max(0, sum(1 for _ in fin) - 1)


def read_geo_ids(path):
    with open(path, 'r', newline='') as fin:
        reader = csv.DictReader(fin)
        return [row['geo_id'] for row in reader]


def load_edge_ids(path):
    with open(path, 'r', encoding='utf-8') as fin:
        edge_ids = json.load(fin)
    if not isinstance(edge_ids, list):
        raise ValueError('edge ids json must be a list.')
    return [str(item) for item in edge_ids]


def normalize_config(source_config_path, dataset):
    with open(source_config_path, 'r', encoding='utf-8') as fin:
        config = json.load(fin)
    info = config.setdefault('info', {})
    info['data_col'] = ['traffic_flow', 'traffic_speed']
    info['data_files'] = [dataset]
    info['geo_file'] = dataset
    info['rel_file'] = dataset
    info['output_dim'] = 2
    info['time_intervals'] = int(info.get('time_intervals', 300))
    info['weight_col'] = 'link_weight'
    info['init_weight_inf_or_zero'] = info.get('init_weight_inf_or_zero', 'zero')
    info['set_weight_link_or_dist'] = info.get('set_weight_link_or_dist', 'link')
    info['calculate_weight_adj'] = bool(info.get('calculate_weight_adj', False))
    info['weight_adj_epsilon'] = float(info.get('weight_adj_epsilon', 0.1))
    return config


def parse_speed_valid(csv_path, edge_ids, expected_steps):
    num_nodes = len(edge_ids)
    speed_valid = np.zeros((expected_steps, num_nodes), dtype=np.bool_)
    edge_to_idx = {edge_id: idx for idx, edge_id in enumerate(edge_ids)}
    seen_counts = np.zeros(expected_steps, dtype=np.int32)
    first_time_edges = []
    valid_sum = 0
    rows = 0

    with open(csv_path, 'r', newline='') as fin:
        reader = csv.DictReader(fin)
        required = {'Time_Step', 'Edge_ID', 'Speed_Valid'}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError('Speed-valid CSV must contain {}'.format(sorted(required)))
        for row in reader:
            step = int(row['Time_Step'])
            if step < 0 or step >= expected_steps:
                raise ValueError('Time_Step {} is outside [0, {}).'.format(
                    step, expected_steps))
            edge_id = row['Edge_ID']
            if edge_id not in edge_to_idx:
                raise ValueError('Unknown Edge_ID in speed-valid CSV: {}'.format(edge_id))
            idx = edge_to_idx[edge_id]
            valid = float(row['Speed_Valid']) > 0
            speed_valid[step, idx] = valid
            seen_counts[step] += 1
            valid_sum += int(valid)
            rows += 1
            if step == 0:
                first_time_edges.append(edge_id)

    if rows != expected_steps * num_nodes:
        raise ValueError('Speed-valid rows {}, expected {}.'.format(
            rows, expected_steps * num_nodes))
    bad_steps = np.where(seen_counts != num_nodes)[0]
    if bad_steps.size:
        raise ValueError('Some time steps do not have {} edges, first bad: {}'.format(
            num_nodes, bad_steps[:10].tolist()))
    if first_time_edges != edge_ids:
        raise ValueError('Time step 0 edge order does not match edge ids json.')
    return speed_valid, valid_sum / rows


def copy_dataset_files(source_dir, dest_dir, dataset, config):
    os.makedirs(dest_dir, exist_ok=True)
    for ext in ['dyna', 'geo', 'rel']:
        src = os.path.join(source_dir, dataset + '.' + ext)
        dst = os.path.join(dest_dir, dataset + '.' + ext)
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
    masks_src = os.path.join(source_dir, dataset + '_masks.npz')
    if os.path.exists(masks_src):
        shutil.copy2(masks_src, os.path.join(dest_dir, dataset + '_masks.npz'))
    with open(os.path.join(dest_dir, 'config.json'), 'w', encoding='utf-8') as fout:
        json.dump(config, fout, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--speed-valid-csv', required=True)
    parser.add_argument('--edge-ids-json', required=True)
    parser.add_argument('--source-manifest')
    parser.add_argument('--dataset', default='SUMO_BEIJING_FIXED_V2')
    parser.add_argument('--expected-nodes', type=int, default=1008)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    source_config = os.path.join(args.source_dir, 'config.json')
    source_dyna = os.path.join(args.source_dir, args.dataset + '.dyna')
    source_geo = os.path.join(args.source_dir, args.dataset + '.geo')
    source_rel = os.path.join(args.source_dir, args.dataset + '.rel')
    for path in [
        source_config, source_dyna, source_geo, source_rel,
        args.speed_valid_csv, args.edge_ids_json
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    geo_ids = read_geo_ids(source_geo)
    if len(geo_ids) != args.expected_nodes:
        raise ValueError('Expected {} nodes, got {}'.format(
            args.expected_nodes, len(geo_ids)))
    edge_ids = load_edge_ids(args.edge_ids_json)
    if len(edge_ids) != len(geo_ids):
        raise ValueError('Edge id count {} != geo count {}'.format(
            len(edge_ids), len(geo_ids)))

    dyna_rows = count_rows(source_dyna)
    if dyna_rows % len(geo_ids) != 0:
        raise ValueError('Dyna rows {} are not divisible by {}'.format(
            dyna_rows, len(geo_ids)))
    num_steps = dyna_rows // len(geo_ids)
    speed_valid, speed_valid_rate = parse_speed_valid(
        args.speed_valid_csv, edge_ids, num_steps)

    dest_dir = os.path.join(RAW_DATA_DIR, args.dataset)
    if os.path.exists(dest_dir):
        if not args.overwrite:
            raise SystemExit(
                'Destination exists: {}. Use --overwrite to replace it.'.format(dest_dir))
        shutil.rmtree(dest_dir)

    config = normalize_config(source_config, args.dataset)
    copy_dataset_files(args.source_dir, dest_dir, args.dataset, config)

    edge_map_path = os.path.join(dest_dir, args.dataset + '_edge_map.csv')
    with open(edge_map_path, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(['entity_id', 'sumo_edge_id'])
        for geo_id, edge_id in zip(geo_ids, edge_ids):
            writer.writerow([geo_id, edge_id])

    timestamps = []
    with open(source_dyna, 'r', newline='') as fin:
        reader = csv.DictReader(fin)
        for idx, row in enumerate(reader):
            if idx % len(geo_ids) == 0:
                timestamps.append(row['time'])
    speed_valid_path = os.path.join(dest_dir, args.dataset + '_speed_valid.npz')
    np.savez_compressed(
        speed_valid_path,
        speed_valid=speed_valid,
        timestamps=np.array(timestamps, dtype='U20'),
        geo_ids=np.array(geo_ids),
        sumo_edge_ids=np.array(edge_ids),
        valid_rule='Speed_Valid > 0 from {}'.format(
            os.path.basename(args.speed_valid_csv))
    )

    manifest = {
        'dataset': args.dataset,
        'frozen_at_utc': dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'source_dir': os.path.abspath(args.source_dir),
        'source_dyna_sha256': sha256_file(source_dyna),
        'source_speed_valid_csv': os.path.abspath(args.speed_valid_csv),
        'source_speed_valid_csv_sha256': sha256_file(args.speed_valid_csv),
        'source_edge_ids_json': os.path.abspath(args.edge_ids_json),
        'source_edge_ids_json_sha256': sha256_file(args.edge_ids_json),
        'source_manifest': os.path.abspath(args.source_manifest) if args.source_manifest else None,
        'source_manifest_sha256': (
            sha256_file(args.source_manifest)
            if args.source_manifest and os.path.exists(args.source_manifest)
            else None
        ),
        'num_nodes': len(geo_ids),
        'num_timestamps': num_steps,
        'dyna_rows': dyna_rows,
        'time_intervals': config['info']['time_intervals'],
        'data_col': ['traffic_flow', 'traffic_speed'],
        'speed_valid_file': os.path.basename(speed_valid_path),
        'speed_valid_shape': list(speed_valid.shape),
        'speed_valid_rate': speed_valid_rate,
        'edge_map_file': os.path.basename(edge_map_path),
        'notes': [
            'Speed_Valid is a sidecar mask, not a third model target.',
            'traffic_speed in .dyna remains the model input/output variable.'
        ]
    }
    if args.source_manifest and os.path.exists(args.source_manifest):
        with open(args.source_manifest, 'r', encoding='utf-8') as fin:
            manifest['source_manifest_summary'] = json.load(fin)
    with open(os.path.join(dest_dir, 'manifest.json'), 'w', encoding='utf-8') as fout:
        json.dump(manifest, fout, indent=4)

    print('Frozen dataset at {}'.format(dest_dir))
    print('  nodes: {}'.format(len(geo_ids)))
    print('  timestamps: {}'.format(num_steps))
    print('  dyna rows: {}'.format(dyna_rows))
    print('  speed_valid: {}'.format(speed_valid_path))
    print('  speed_valid_rate: {:.6f}'.format(speed_valid_rate))


if __name__ == '__main__':
    os.chdir(PROJECT_DIR)
    main()
