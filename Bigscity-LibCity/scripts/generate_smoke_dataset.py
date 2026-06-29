"""
Generate 3-day smoke datasets from the full SUMO datasets.

Creates two smoke datasets:
1. SUMO_BEIJING_FIXED_V2_SMOKE (2-variable: flow + speed)
2. SUMO_BEIJING_FIXED_V2_FLOW_SMOKE (Flow-only)

Each has 864 time steps (3 days * 288 steps/day), 1008 nodes.

Usage:
    cd Bigscity-LibCity
    python scripts/generate_smoke_dataset.py
"""

import os
import json
import shutil

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)

# 3 days: 864 time steps * 1008 nodes = 870,912 data rows
NUM_DAYS = 3
STEPS_PER_DAY = 288  # 24h * 60min / 5min
NUM_NODES = 1008
NUM_STEPS = NUM_DAYS * STEPS_PER_DAY  # 864
NUM_DATA_ROWS = NUM_STEPS * NUM_NODES  # 870,912


def generate_smoke(src_name, dst_name, dyna_cols):
    """Generate a smoke dataset from a source dataset.

    Args:
        src_name: Source dataset directory name
        dst_name: Destination smoke dataset directory name
        dyna_cols: List of column names to keep in .dyna (besides dyna_id, type, time, entity_id)
    """
    src_dir = os.path.join(PROJECT_DIR, 'raw_data', src_name)
    dst_dir = os.path.join(PROJECT_DIR, 'raw_data', dst_name)
    os.makedirs(dst_dir, exist_ok=True)

    src_prefix = src_name
    dst_prefix = dst_name

    # 1. Generate smoke .dyna file (first 3 days)
    print(f'\n{"="*60}')
    print(f'Generating {dst_name}')
    print(f'{"="*60}')

    src_dyna = os.path.join(src_dir, f'{src_prefix}.dyna')
    dst_dyna = os.path.join(dst_dir, f'{dst_prefix}.dyna')

    print(f'Generating smoke .dyna file ({NUM_DATA_ROWS} data rows)...')
    with open(src_dyna, 'r') as fin, open(dst_dyna, 'w') as fout:
        header = fin.readline()
        fout.write(header)
        for i, line in enumerate(fin):
            if i >= NUM_DATA_ROWS:
                break
            fout.write(line)
    print(f'  Written to {dst_dyna}')

    # 2. Copy .geo file (unchanged)
    print('Copying .geo file...')
    shutil.copy2(
        os.path.join(src_dir, f'{src_prefix}.geo'),
        os.path.join(dst_dir, f'{dst_prefix}.geo')
    )

    # 3. Copy .rel file (unchanged)
    print('Copying .rel file...')
    shutil.copy2(
        os.path.join(src_dir, f'{src_prefix}.rel'),
        os.path.join(dst_dir, f'{dst_prefix}.rel')
    )

    # 4. Copy masks.npz if exists
    src_masks = os.path.join(src_dir, f'{src_prefix}_masks.npz')
    if os.path.exists(src_masks):
        print('Copying masks.npz...')
        shutil.copy2(src_masks, os.path.join(dst_dir, f'{dst_prefix}_masks.npz'))

    # 5. Generate config.json
    print('Generating config.json...')
    # Build dyna state fields
    state_fields = {
        "dyna_id": "num",
        "type": "enum",
        "time": "time",
        "entity_id": "num"
    }
    for col in dyna_cols:
        state_fields[col] = "num"

    smoke_config = {
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
            "state": state_fields
        },
        "info": {
            "data_col": dyna_cols,
            "data_files": [dst_prefix],
            "geo_file": dst_prefix,
            "rel_file": dst_prefix,
            "output_dim": len(dyna_cols),
            "time_intervals": 300,
            "weight_col": "link_weight",
            "init_weight_inf_or_zero": "zero",
            "set_weight_link_or_dist": "link",
            "calculate_weight_adj": False,
            "weight_adj_epsilon": 0.1
        }
    }
    with open(os.path.join(dst_dir, 'config.json'), 'w') as f:
        json.dump(smoke_config, f, indent=4)

    print(f'  output_dim: {len(dyna_cols)}')
    print(f'  data_col: {dyna_cols}')
    print(f'  Time steps: {NUM_STEPS} ({NUM_DAYS} days)')
    print(f'  Nodes: {NUM_NODES}')
    print(f'  Output dir: {dst_dir}')
    print(f'  Smoke dataset generated successfully!')


def main():
    # 1. Two-variable smoke dataset (flow + speed)
    generate_smoke(
        src_name='SUMO_BEIJING_FIXED_V2',
        dst_name='SUMO_BEIJING_FIXED_V2_SMOKE',
        dyna_cols=['traffic_flow', 'traffic_speed']
    )

    # 2. Flow-only smoke dataset
    flow_src = os.path.join(PROJECT_DIR, 'raw_data', 'SUMO_BEIJING_FIXED_V2_FLOW')
    if os.path.exists(flow_src):
        generate_smoke(
            src_name='SUMO_BEIJING_FIXED_V2_FLOW',
            dst_name='SUMO_BEIJING_FIXED_V2_FLOW_SMOKE',
            dyna_cols=['traffic_flow']
        )
    else:
        print('\n[SKIP] SUMO_BEIJING_FIXED_V2_FLOW not found.')
        print('Run scripts/generate_flow_dataset.py first to create it.')


if __name__ == '__main__':
    main()
