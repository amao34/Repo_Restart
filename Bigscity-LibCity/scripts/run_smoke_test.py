"""
Smoke test: Run 1 epoch on 2-variable smoke dataset.
Verifies: data loads, DTW generates, K-Shape generates, forward works.

Usage:
    cd Bigscity-LibCity
    python scripts/run_smoke_test.py
"""

import sys
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

from libcity.pipeline import run_model


def main():
    print('=' * 60)
    print('Smoke Test: 1 epoch on SUMO_BEIJING_FIXED_V2_SMOKE')
    print('Config: sumo_pdformer_smoke (output_dim=2)')
    print('=' * 60)
    print()

    run_model(
        task='traffic_state_pred',
        model_name='PDFormer',
        dataset_name='SUMO_BEIJING_FIXED_V2_SMOKE',
        config_file='sumo_pdformer_smoke',
        saved_model=False,
        train=True,
    )

    print()
    print('[PASS] Smoke test completed without errors.')


if __name__ == '__main__':
    main()
