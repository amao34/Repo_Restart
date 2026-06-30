"""
Run all validation gates in sequence.
Follows the corrected execution order:
  1. Generate Flow-only dataset
  2. Generate smoke datasets
  3. Validate loss
  4. Gate 1: DataLoader dimensions
  5. Gate 2: Time alignment
  6. Smoke test (2-variable, 1 epoch)
  7. Gate 3: Single-batch overfit capability probe (Flow-only)
  8. Gate 4: Baseline training (Flow-only, 30 epochs) [optional]

Usage:
    cd Bigscity-LibCity
    python scripts/run_all_validations.py
"""

import sys
import os
import subprocess

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(script_name, description):
    """Run a validation script and return success/failure."""
    print()
    print('=' * 70)
    print(f'Running: {description}')
    print(f'Script:  {script_name}')
    print('=' * 70)
    print()

    script_path = os.path.join(SCRIPTS_DIR, script_name)
    result = subprocess.run([sys.executable, script_path], cwd=os.path.dirname(SCRIPTS_DIR))

    if result.returncode != 0:
        print(f'\n[FAIL] {description} failed.')
        return False
    print(f'\n[PASS] {description} passed.')
    return True


def main():
    print('=' * 70)
    print('SUMO BEIJING FIXED V2 - Full Validation Pipeline')
    print('=' * 70)

    # Step 6: Generate Flow-only dataset
    if not run_script('generate_flow_dataset.py', 'Generate Flow-Only Dataset'):
        return 1

    # Generate smoke datasets
    if not run_script('generate_smoke_dataset.py', 'Generate Smoke Datasets'):
        return 1

    # Validate loss includes zero targets
    if not run_script('validate_loss_zero.py', 'Validate Loss Zero-Target'):
        return 1

    # Gate 1: DataLoader dimensions
    if not run_script('validate_gate1_dataloader.py', 'Gate 1: DataLoader Dimensions'):
        return 1

    # Gate 2: Time alignment
    if not run_script('validate_gate2_time_alignment.py', 'Gate 2: Time Alignment'):
        return 1

    # Smoke test: 1 epoch (2-variable)
    if not run_script('run_smoke_test.py', 'Smoke Test: 1 Epoch (2-variable)'):
        return 1

    # Gate 3: Single-batch overfit capability probe (Flow-only)
    if not run_script('validate_gate3_overfit.py', 'Gate 3: Single-Batch Overfit Probe (Flow-only)'):
        return 1

    print()
    print('=' * 70)
    print('[ALL PASS] All validation gates passed!')
    print()
    print('Next step: Run Gate 4 baseline training:')
    print('  python scripts/run_baseline_training.py')
    print()
    print('Then compute the same-split HA baseline:')
    print('  python scripts/run_ha_baseline.py')
    print('=' * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
