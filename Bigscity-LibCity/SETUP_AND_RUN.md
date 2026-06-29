# PDFormer Baseline Setup Guide

## Three Types of Config Files

1. **Atomic dataset config** (`raw_data/SUMO_BEIJING_FIXED_V2/config.json`) — describes data schema
2. **Model default config** (`libcity/config/model/traffic_state_pred/PDFormer.json`) — model hyperparams
3. **Experiment override config** (root dir, e.g. `sumo_pdformer_smoke.json`) — overrides both above

## Modified Files
- `libcity/model/traffic_flow_prediction/PDFormer.py` — added `self.set_loss` config support

## New Files
| File | Purpose |
|------|---------|
| `sumo_pdformer_smoke.json` | 2-variable smoke config (output_dim=2) |
| `sumo_pdformer_flow.json` | Flow-only config (output_dim=1, set_loss=mae) |
| `scripts/generate_flow_dataset.py` | Generate Flow-only dataset (SUMO_BEIJING_FIXED_V2_FLOW) |
| `scripts/generate_smoke_dataset.py` | Generate 3-day smoke datasets |
| `scripts/validate_loss_zero.py` | Validate loss includes zero targets |
| `scripts/validate_gate1_dataloader.py` | Validate DataLoader dimensions |
| `scripts/validate_gate2_time_alignment.py` | Validate time alignment |
| `scripts/validate_gate3_overfit.py` | Single-batch overfit test |
| `scripts/run_smoke_test.py` | 1 epoch smoke test (2-variable) |
| `scripts/run_baseline_training.py` | 5 epoch baseline (Flow-only) |
| `scripts/run_all_validations.py` | Run all gates in sequence |

---

## Setup on New Machine

```bash
conda create -n pdformer_clean python=3.9.7
conda activate pdformer_clean
cd E:\PythonProject\Restart\Bigscity-LibCity
pip install -r requirements.txt
python --version > environment.txt
pip freeze >> environment.txt
```

---

## Execution Order

### Quick Run (All Gates)
```bash
python scripts/run_all_validations.py
```

### Step-by-Step

```bash
# Step 6: Generate Flow-only dataset (only traffic_flow column)
python scripts/generate_flow_dataset.py

# Generate smoke datasets (3 days each)
python scripts/generate_smoke_dataset.py

# Validate loss includes zero targets
python scripts/validate_loss_zero.py

# Gate 1: DataLoader dimensions
python scripts/validate_gate1_dataloader.py

# Gate 2: Time alignment
python scripts/validate_gate2_time_alignment.py

# Smoke test: 1 epoch (2-variable, output_dim=2)
python scripts/run_smoke_test.py

# Gate 3: Single-batch overfit (Flow-only)
python scripts/validate_gate3_overfit.py

# Gate 4: 5 epoch baseline training (Flow-only)
python scripts/run_baseline_training.py
```

---

## Key Design Decisions

### Why Two Datasets?
- **SUMO_BEIJING_FIXED_V2**: 2 variables (flow + speed), `output_dim=2`
- **SUMO_BEIJING_FIXED_V2_FLOW**: 1 variable (flow only), `output_dim=1`

Cannot set `output_dim=1` on the 2-variable dataset because PDFormer uses `output_dim` to determine feature indexing. Setting it to 1 would cause time features to be misaligned.

### Flow-only Dataset Layout
```
0: traffic_flow
1: time-of-day (added by load_external)
2-8: weekday one-hot (added by add_day_in_week)
```

### Config Details

**sumo_pdformer_smoke.json** (2-variable smoke):
```json
{
    "batch_size": 2,
    "max_epoch": 1,
    "set_loss": "masked_mae",
    "cache_dataset": false
}
```

**sumo_pdformer_flow.json** (Flow-only):
```json
{
    "batch_size": 4,
    "max_epoch": 5,
    "set_loss": "mae",
    "cache_dataset": false
}
```
- `set_loss: "mae"` — standard MAE, zero-flow targets contribute to loss
- `set_loss: "masked_mae"` — masked MAE, zeros are excluded (81.2% of Flow data)

---

## First Run Notes

1. **DTW computation is slow**: First run computes DTW for 1008 nodes (~10-30 min)
2. **K-Shape clustering**: Also slow on first run, cached after
3. **After success**: Back up cache to `cache_backup/`

### Expected Shapes
- DTW matrix: `(1008, 1008)`
- Short-path matrix: `(1008, 1008)`
- Adjacency matrix: `(1008, 1008)`
- Pattern keys: `(16, 3, output_dim)`

---

## Git Commit Structure

```
commit 1: chore: initialize official PDFormer baseline
commit 2: data: add SUMO_BEIJING_FIXED_V2 configuration
commit 3: test: validate custom dataset shapes and alignment
commit 4: fix: include zero-flow targets in training loss
commit 5: test: add single-batch overfit check
commit 6: baseline: train original PDFormer on SUMO dataset
```

## After Baseline Passes

Only after Flow R² > 0.612 and RMSE < 0.857:
```bash
git checkout -b feature/intent-fusion
```
