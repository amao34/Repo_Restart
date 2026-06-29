"""
Gate 4: Baseline training (5 epochs) on Flow-only dataset.

Dataset: SUMO_BEIJING_FIXED_V2_FLOW
Config: sumo_pdformer_flow (output_dim=1, set_loss=mae)

Verifies:
- Training loss decreases
- Validation loss is normal
- No NaN in output
- R² grows from negative toward positive
- Inverse transform is correct

Usage:
    cd Bigscity-LibCity
    python scripts/run_baseline_training.py
"""

import sys
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

from libcity.pipeline import run_model


def main():
    print('=' * 60)
    print('Gate 4: Baseline Training (5 epochs)')
    print('Dataset: SUMO_BEIJING_FIXED_V2_FLOW')
    print('Config: sumo_pdformer_flow')
    print('Output: Flow-only (output_dim=1)')
    print('=' * 60)
    print()

    run_model(
        task='traffic_state_pred',
        model_name='PDFormer',
        dataset_name='SUMO_BEIJING_FIXED_V2_FLOW',
        config_file='sumo_pdformer_flow',
        saved_model=True,
        train=True,
    )

    print()
    print('[INFO] Baseline training completed.')
    print('Check:')
    print('  - Training loss decreasing?')
    print('  - Validation loss normal (not NaN)?')
    print('  - R² growing from negative toward positive?')
    print('  - Inverse transform producing reasonable values?')


if __name__ == '__main__':
    main()
