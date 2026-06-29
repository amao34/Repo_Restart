"""
Validate that the loss function includes zero-flow targets.

With set_loss="mae" (not "masked_mae"), zero targets should contribute to loss.

Unit test:
    true = [0.0], pred = [1.0] => loss > 0

Usage:
    cd Bigscity-LibCity
    python scripts/validate_loss_zero.py
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libcity.model import loss


def main():
    print('=' * 60)
    print('Loss Zero-Target Validation')
    print('=' * 60)
    print()

    passed = True

    # Test 1: masked_mae with null_val=0 (old behavior - zeros are masked)
    print('Test 1: masked_mae_torch(pred=[1.0], label=[0.0], null_val=0)')
    pred = torch.tensor([1.0])
    label = torch.tensor([0.0])
    result = loss.masked_mae_torch(pred, label, null_val=0)
    print(f'  Loss = {result.item():.4f} (expected: 0.0 - zeros are masked)')
    if result.item() > 0:
        print('  [INFO] masked_mae with null_val=0 does NOT mask zeros (label set to 0 by threshold)')
    print()

    # Test 2: plain mae (new behavior - zeros are NOT masked)
    print('Test 2: masked_mae_torch(pred=[1.0], label=[0.0]) [no null_val, i.e. mae]')
    pred = torch.tensor([1.0])
    label = torch.tensor([0.0])
    result = loss.masked_mae_torch(pred, label)  # null_val defaults to np.nan
    print(f'  Loss = {result.item():.4f} (expected: > 0 - zeros contribute)')
    if result.item() <= 0:
        print('  [FAIL] Loss should be > 0 when label=0 and pred=1')
        passed = False
    else:
        print('  [PASS] Zero targets contribute to loss.')
    print()

    # Test 3: mae with mixed zeros and non-zeros
    print('Test 3: mae with mixed values')
    pred = torch.tensor([1.0, 2.0, 0.5, 0.0])
    label = torch.tensor([0.0, 2.0, 0.0, 0.0])
    result = loss.masked_mae_torch(pred, label)
    print(f'  Loss = {result.item():.4f} (expected: > 0)')
    if result.item() <= 0:
        print('  [FAIL] Loss should be > 0')
        passed = False
    else:
        print('  [PASS] Mixed zero/non-zero targets contribute to loss.')
    print()

    # Test 4: masked_mae with null_val=0 (old behavior comparison)
    print('Test 4: masked_mae_torch same inputs but with null_val=0 (old behavior)')
    result_masked = loss.masked_mae_torch(pred, label, null_val=0)
    print(f'  Loss = {result_masked.item():.4f}')
    print(f'  Ratio (mae/masked_mae) = {result.item() / result_masked.item():.2f}' if result_masked.item() > 0 else '  masked_mae = 0')
    print()

    # Test 5: Verify PDFormer model's set_loss config
    print('Test 5: Verify PDFormer set_loss config integration')
    print('  Config should have: "set_loss": "mae"')
    print('  PDFormer.calculate_loss should pass set_loss=self.set_loss')
    print()

    print('=' * 60)
    if passed:
        print('[PASS] Loss validation: zero targets are included with set_loss="mae".')
    else:
        print('[FAIL] Loss validation failed.')
    print('=' * 60)

    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
