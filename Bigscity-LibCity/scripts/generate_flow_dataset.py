"""
Generate Flow-only dataset (SUMO_BEIJING_FIXED_V2_FLOW) from the 2-variable dataset.

Creates an independent atomic dataset with ONLY traffic_flow (no traffic_speed).
This ensures DataEmbedding feature indexing is correct:
    0: traffic_flow
    1: time-of-day (added by load_external)
    2-8: weekday one-hot (added by add_day_in_week)

output_dim = 1 (natural, since only 1 traffic variable)

Usage:
    cd Bigscity-LibCity
    python scripts/generate_flow_dataset.py
"""

import os
import json
import shutil
import pandas as pd
from tqdm import tqdm

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, 'raw_data', 'SUMO_BEIJING_FIXED_V2')
DST_DIR = os.path.join(PROJECT_DIR, 'raw_data', 'SUMO_BEIJING_FIXED_V2_FLOW')

SRC_PREFIX = 'SUMO_BEIJING_FIXED_V2'
DST_PREFIX = 'SUMO_BEIJING_FIXED_V2_FLOW'


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    # 1. Generate Flow-only .dyna file
    src_dyna = os.path.join(SRC_DIR, f'{SRC_PREFIX}.dyna')
    dst_dyna = os.path.join(DST_DIR, f'{DST_PREFIX}.dyna')

    print(f'Reading {src_dyna}...')
    print(f'Writing {dst_dyna} (Flow-only)...')

    with open(src_dyna, 'r') as fin:
        header = fin.readline().strip()
        # Original: dyna_id,type,time,entity_id,traffic_flow,traffic_speed
        # Target:   dyna_id,type,time,entity_id,traffic_flow
        cols = header.split(',')
        flow_col_idx = cols.index('traffic_flow')
        # Keep: dyna_id, type, time, entity_id, traffic_flow
        keep_indices = list(range(4)) + [flow_col_idx]  # first 4 + traffic_flow

        with open(dst_dyna, 'w') as fout:
            # Write new header
            new_cols = [cols[i] for i in keep_indices]
            fout.write(','.join(new_cols) + '\n')

            # Write data lines
            line_count = 0
            for line in tqdm(fin, desc='Processing dyna'):
                parts = line.strip().split(',')
                new_parts = [parts[i] for i in keep_indices]
                fout.write(','.join(new_parts) + '\n')
                line_count += 1

    print(f'  Written {line_count} data lines')

    # 2. Copy .geo file (unchanged, rename prefix)
    print('Copying .geo file...')
    shutil.copy2(
        os.path.join(SRC_DIR, f'{SRC_PREFIX}.geo'),
        os.path.join(DST_DIR, f'{DST_PREFIX}.geo')
    )

    # 3. Copy .rel file (unchanged, rename prefix)
    print('Copying .rel file...')
    shutil.copy2(
        os.path.join(SRC_DIR, f'{SRC_PREFIX}.rel'),
        os.path.join(DST_DIR, f'{DST_PREFIX}.rel')
    )

    # 4. Copy masks.npz if exists
    src_masks = os.path.join(SRC_DIR, f'{SRC_PREFIX}_masks.npz')
    if os.path.exists(src_masks):
        print('Copying masks.npz...')
        shutil.copy2(src_masks, os.path.join(DST_DIR, f'{DST_PREFIX}_masks.npz'))

    # 5. Generate config.json for Flow-only dataset
    print('Generating config.json...')
    flow_config = {
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
                "traffic_flow": "num"
            }
        },
        "info": {
            "data_col": ["traffic_flow"],
            "data_files": [DST_PREFIX],
            "geo_file": DST_PREFIX,
            "rel_file": DST_PREFIX,
            "output_dim": 1,
            "time_intervals": 300,
            "weight_col": "link_weight",
            "init_weight_inf_or_zero": "zero",
            "set_weight_link_or_dist": "link",
            "calculate_weight_adj": False,
            "weight_adj_epsilon": 0.1
        }
    }
    with open(os.path.join(DST_DIR, 'config.json'), 'w') as f:
        json.dump(flow_config, f, indent=4)

    # Verify
    print('\nVerification:')
    print(f'  Source: {SRC_DIR}')
    print(f'  Output: {DST_DIR}')
    print(f'  Data lines: {line_count}')
    print(f'  data_col: ["traffic_flow"]')
    print(f'  output_dim: 1')
    print()
    print('Data feature layout after load_external + add_day_in_week:')
    print('  0: traffic_flow')
    print('  1: time-of-day (fraction of day)')
    print('  2-8: weekday one-hot (7 columns)')
    print()
    print('Flow-only dataset generated successfully!')


if __name__ == '__main__':
    main()
