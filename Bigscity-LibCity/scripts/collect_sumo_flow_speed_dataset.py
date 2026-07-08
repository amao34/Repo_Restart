"""
Collect a full Flow+Speed atomic dataset from a SUMO simulation.

The script writes:
  raw_data/<dataset>/<dataset>.dyna
  raw_data/<dataset>/<dataset>.geo
  raw_data/<dataset>/<dataset>.rel
  raw_data/<dataset>/<dataset>_speed_valid.npz
  raw_data/<dataset>/config.json
  raw_data/<dataset>/manifest.json

Important:
  The existing 1008-node data uses internal entity ids 0..1007. A stable mapping
  from those ids to SUMO edge ids is required; do not guess it.

Example:
  python scripts/collect_sumo_flow_speed_dataset.py ^
      --sumocfg path\\to\\scenario.sumocfg ^
      --edge-map path\\to\\entity_edge_map.csv ^
      --num-steps 864 ^
      --dataset SUMO_BEIJING_FIXED_V2

To create a mapping template:
  python scripts/collect_sumo_flow_speed_dataset.py ^
      --write-edge-map-template entity_edge_map_template.csv
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import sys

import numpy as np


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_DIR, 'raw_data')
DEFAULT_REFERENCE_DATASET = 'SUMO_BEIJING_FIXED_V2_FLOW'


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as fin:
        while True:
            chunk = fin.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_geo_ids(reference_dataset):
    geo_path = os.path.join(RAW_DATA_DIR, reference_dataset, reference_dataset + '.geo')
    if not os.path.exists(geo_path):
        raise FileNotFoundError('Reference geo file not found: {}'.format(geo_path))
    with open(geo_path, 'r', newline='') as fin:
        reader = csv.DictReader(fin)
        if 'geo_id' not in reader.fieldnames:
            raise ValueError('geo_id column not found in {}'.format(geo_path))
        return [row['geo_id'] for row in reader]


def write_edge_map_template(path, geo_ids):
    with open(path, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(['entity_id', 'sumo_edge_id'])
        for geo_id in geo_ids:
            writer.writerow([geo_id, ''])
    print('Wrote edge-map template to {}'.format(path))


def load_edge_map(path, geo_ids, identity_edge_ids=False):
    if identity_edge_ids:
        return {geo_id: geo_id for geo_id in geo_ids}
    if path is None:
        raise ValueError(
            'Missing --edge-map. The 1008-node dataset uses internal ids, so a '
            'stable entity_id -> sumo_edge_id mapping is required. Use '
            '--write-edge-map-template to create a template.')
    with open(path, 'r', newline='') as fin:
        reader = csv.DictReader(fin)
        required = {'entity_id', 'sumo_edge_id'}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                'Edge map must contain columns: entity_id,sumo_edge_id')
        mapping = {}
        for row in reader:
            entity_id = str(row['entity_id'])
            sumo_edge_id = str(row['sumo_edge_id']).strip()
            if not sumo_edge_id:
                raise ValueError('Empty sumo_edge_id for entity_id {}'.format(entity_id))
            mapping[entity_id] = sumo_edge_id

    missing = [geo_id for geo_id in geo_ids if geo_id not in mapping]
    extra = [entity_id for entity_id in mapping if entity_id not in set(geo_ids)]
    if missing:
        raise ValueError('Edge map is missing {} entity ids, first: {}'.format(
            len(missing), missing[:10]))
    if extra:
        raise ValueError('Edge map has {} unknown entity ids, first: {}'.format(
            len(extra), extra[:10]))
    return mapping


def copy_static_files(reference_dataset, dataset, output_dir):
    ref_dir = os.path.join(RAW_DATA_DIR, reference_dataset)
    for ext in ['geo', 'rel']:
        src = os.path.join(ref_dir, reference_dataset + '.' + ext)
        dst = os.path.join(output_dir, dataset + '.' + ext)
        if not os.path.exists(src):
            raise FileNotFoundError('Reference .{} file not found: {}'.format(ext, src))
        shutil.copy2(src, dst)


def write_config(dataset, output_dir):
    config = {
        "geo": {
            "including_types": ["Point"],
            "Point": {
                "geo_id": "num",
                "type": "enum",
                "coordinates": "GeoShape"
            }
        },
        "usr": {},
        "rel": {
            "including_types": ["geo"],
            "geo": {
                "rel_id": "num",
                "type": "enum",
                "origin_id": "num",
                "destination_id": "num",
                "link_weight": "num"
            }
        },
        "dyna": {
            "including_types": ["state"],
            "state": {
                "dyna_id": "num",
                "type": "enum",
                "time": "time",
                "entity_id": "num",
                "traffic_flow": "num",
                "traffic_speed": "num"
            }
        },
        "info": {
            "data_col": ["traffic_flow", "traffic_speed"],
            "data_files": [dataset],
            "geo_file": dataset,
            "rel_file": dataset,
            "output_dim": 2,
            "time_intervals": 300,
            "weight_col": "link_weight",
            "init_weight_inf_or_zero": "zero",
            "set_weight_link_or_dist": "link",
            "calculate_weight_adj": False,
            "weight_adj_epsilon": 0.1
        }
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as fout:
        json.dump(config, fout, indent=4)


def iso_timestamp(start_datetime, offset_seconds):
    return (start_datetime + dt.timedelta(seconds=offset_seconds)).strftime(
        '%Y-%m-%dT%H:%M:%SZ')


def collect(args):
    try:
        import traci
    except ImportError as exc:
        raise SystemExit(
            'Python package traci is not available in this environment. '
            'Install SUMO tools or run in the SUMO collection environment.') from exc

    geo_ids = load_geo_ids(args.reference_dataset)
    if len(geo_ids) != args.expected_nodes:
        raise ValueError('Expected {} nodes, got {} from reference dataset {}'.format(
            args.expected_nodes, len(geo_ids), args.reference_dataset))
    edge_map = load_edge_map(args.edge_map, geo_ids, args.identity_edge_ids)
    sumo_edge_ids = [edge_map[geo_id] for geo_id in geo_ids]

    output_dir = os.path.join(RAW_DATA_DIR, args.dataset)
    if os.path.exists(output_dir):
        if not args.overwrite:
            raise SystemExit(
                'Output dataset already exists: {}. Use --overwrite to replace it.'.format(
                    output_dir))
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    copy_static_files(args.reference_dataset, args.dataset, output_dir)
    write_config(args.dataset, output_dir)

    start_datetime = dt.datetime.strptime(args.start_datetime, '%Y-%m-%dT%H:%M:%SZ')
    if args.num_steps is None and args.end is None:
        raise ValueError('Set --num-steps for fixed-length collection, or set --end.')
    if args.num_steps is not None:
        collect_times = [args.begin + i * args.interval for i in range(args.num_steps)]
    else:
        collect_times = list(range(args.begin, args.end, args.interval))
    if not collect_times:
        raise ValueError('No collection timestamps were generated.')

    speed_valid = np.zeros((len(collect_times), len(geo_ids)), dtype=np.bool_)
    timestamps = np.array(
        [iso_timestamp(start_datetime, sec - args.begin) for sec in collect_times],
        dtype='U20')

    dyna_path = os.path.join(output_dir, args.dataset + '.dyna')
    cmd = [args.sumo_binary, '-c', args.sumocfg, '--start', '--quit-on-end',
           '--no-step-log', 'true']
    for item in args.sumo_arg:
        cmd.extend(item.split())

    print('Starting SUMO collection:')
    print('  sumocfg: {}'.format(args.sumocfg))
    print('  dataset: {}'.format(args.dataset))
    print('  nodes: {}'.format(len(geo_ids)))
    print('  steps: {}'.format(len(collect_times)))
    print('  interval: {} seconds'.format(args.interval))

    traci.start(cmd)
    try:
        existing_edges = set(traci.edge.getIDList())
        missing_edges = [edge_id for edge_id in sumo_edge_ids if edge_id not in existing_edges]
        if missing_edges:
            raise ValueError(
                'Mapped SUMO edges are not in the loaded network. Missing {}, first: {}'.format(
                    len(missing_edges), missing_edges[:20]))

        with open(dyna_path, 'w', newline='') as fout:
            writer = csv.writer(fout)
            writer.writerow([
                'dyna_id', 'type', 'time', 'entity_id',
                'traffic_flow', 'traffic_speed'
            ])
            dyna_id = 0
            for step_idx, sim_time in enumerate(collect_times):
                traci.simulationStep(sim_time)
                timestamp = timestamps[step_idx]
                for edge_idx, (geo_id, edge_id) in enumerate(zip(geo_ids, sumo_edge_ids)):
                    flow = float(traci.edge.getLastStepVehicleNumber(edge_id))
                    speed = float(traci.edge.getLastStepMeanSpeed(edge_id))
                    free_speed = float(traci.edge.getMaxSpeed(edge_id))
                    valid = flow > args.valid_min_flow and speed > args.valid_min_speed
                    speed_valid[step_idx, edge_idx] = valid
                    if not valid and args.invalid_speed_fill == 'freeflow':
                        speed = free_speed
                    elif not valid and args.invalid_speed_fill == 'nan':
                        speed = float('nan')
                    writer.writerow([
                        dyna_id, 'state', timestamp, geo_id,
                        args.float_format.format(flow),
                        args.float_format.format(speed)
                    ])
                    dyna_id += 1
                if (step_idx + 1) % args.log_every == 0 or step_idx == 0:
                    print('  collected {}/{} timestamps'.format(
                        step_idx + 1, len(collect_times)))
    finally:
        traci.close()

    speed_valid_path = os.path.join(output_dir, args.dataset + '_speed_valid.npz')
    np.savez_compressed(
        speed_valid_path,
        speed_valid=speed_valid,
        timestamps=timestamps,
        geo_ids=np.array(geo_ids),
        sumo_edge_ids=np.array(sumo_edge_ids),
        valid_min_flow=args.valid_min_flow,
        valid_min_speed=args.valid_min_speed,
        invalid_speed_fill=args.invalid_speed_fill
    )

    manifest = {
        'dataset': args.dataset,
        'created_at_utc': dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'sumocfg': os.path.abspath(args.sumocfg),
        'sumocfg_sha256': file_sha256(args.sumocfg),
        'edge_map': os.path.abspath(args.edge_map) if args.edge_map else None,
        'edge_map_sha256': file_sha256(args.edge_map) if args.edge_map else None,
        'reference_dataset': args.reference_dataset,
        'num_nodes': len(geo_ids),
        'num_timestamps': len(collect_times),
        'time_intervals': args.interval,
        'begin': args.begin,
        'end': collect_times[-1] + args.interval,
        'start_datetime': args.start_datetime,
        'data_col': ['traffic_flow', 'traffic_speed'],
        'speed_valid_file': os.path.basename(speed_valid_path),
        'speed_valid_rate': float(speed_valid.mean()),
        'valid_rule': {
            'traffic_flow_gt': args.valid_min_flow,
            'traffic_speed_gt': args.valid_min_speed
        },
        'invalid_speed_fill': args.invalid_speed_fill,
        'traffic_flow_semantics': 'SUMO edge.getLastStepVehicleNumber at each collection timestamp',
        'traffic_speed_semantics': 'SUMO edge.getLastStepMeanSpeed at each collection timestamp'
    }
    with open(os.path.join(output_dir, 'manifest.json'), 'w') as fout:
        json.dump(manifest, fout, indent=4)

    print()
    print('Collection complete.')
    print('  dyna: {}'.format(dyna_path))
    print('  speed_valid: {}'.format(speed_valid_path))
    print('  speed_valid_rate: {:.6f}'.format(float(speed_valid.mean())))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sumocfg', help='Path to SUMO .sumocfg file.')
    parser.add_argument('--sumo-binary', default='sumo')
    parser.add_argument('--sumo-arg', action='append', default=[],
                        help='Additional SUMO CLI args, for example "--seed 2026".')
    parser.add_argument('--dataset', default='SUMO_BEIJING_FIXED_V2')
    parser.add_argument('--reference-dataset', default=DEFAULT_REFERENCE_DATASET)
    parser.add_argument('--edge-map',
                        help='CSV with columns entity_id,sumo_edge_id.')
    parser.add_argument('--identity-edge-ids', action='store_true',
                        help='Use geo_id as SUMO edge id. Only use if ids really match.')
    parser.add_argument('--write-edge-map-template',
                        help='Write a mapping template and exit.')
    parser.add_argument('--expected-nodes', type=int, default=1008)
    parser.add_argument('--begin', type=int, default=0)
    parser.add_argument('--end', type=int)
    parser.add_argument('--num-steps', type=int,
                        help='Fixed number of 5-minute slots to collect.')
    parser.add_argument('--interval', type=int, default=300)
    parser.add_argument('--start-datetime', default='2026-01-05T00:00:00Z',
                        help='Timestamp label for --begin, UTC ISO format.')
    parser.add_argument('--valid-min-flow', type=float, default=0.0)
    parser.add_argument('--valid-min-speed', type=float, default=0.0)
    parser.add_argument('--invalid-speed-fill',
                        choices=['measured', 'freeflow', 'nan'],
                        default='freeflow')
    parser.add_argument('--float-format', default='{:.6g}')
    parser.add_argument('--log-every', type=int, default=12)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    geo_ids = load_geo_ids(args.reference_dataset)
    if args.write_edge_map_template:
        write_edge_map_template(args.write_edge_map_template, geo_ids)
        sys.exit(0)
    if not args.sumocfg:
        parser.error('--sumocfg is required unless --write-edge-map-template is used.')
    if not os.path.exists(args.sumocfg):
        parser.error('--sumocfg does not exist: {}'.format(args.sumocfg))
    return args


def main():
    os.chdir(PROJECT_DIR)
    collect(parse_args())


if __name__ == '__main__':
    main()
