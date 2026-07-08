# PDFormer Baseline Setup Guide

## Current Route After Baseline Freeze

- Formal network: `SUMO_BEIJING_FIXED_V2` with 1008 nodes.
- Frozen Flow-only dataset: `SUMO_BEIJING_FIXED_V2_FLOW`.
- Frozen Flow+Speed dataset: `raw_data/SUMO_BEIJING_FIXED_V2`, with
  `SUMO_BEIJING_FIXED_V2_speed_valid.npz` as the speed-valid sidecar.
- Frozen baseline tag: `baseline-flow-pdformer-fixed-v2`.
- Current engineering step: run Stage 2.2, Flow+Speed input with joint
  Flow+Speed output and Speed_Valid-masked speed loss/metrics.
- Formal naming: use information-induced congestion or route-guidance-induced
  congestion for the research problem. Braess-related terms are background only,
  not method/module/metric names.

## Three Types of Config Files

1. **Atomic dataset config** (`raw_data/SUMO_BEIJING_FIXED_V2/config.json`) - describes data schema
2. **Model default config** (`libcity/config/model/traffic_state_pred/PDFormer.json`) - model hyperparams
3. **Experiment override config** (root dir, e.g. `sumo_pdformer_smoke.json`) - overrides both above

## Modified Files
- `libcity/model/traffic_flow_prediction/PDFormer.py` - added `self.set_loss` config support

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
| `scripts/run_baseline_training.py` | 30 epoch baseline (Flow-only) |
| `scripts/collect_sumo_flow_speed_dataset.py` | Re-collect full Flow+Speed dataset and Speed_Valid if the frozen artifact is unavailable |
| `scripts/freeze_existing_flow_speed_dataset.py` | Restore/freeze an existing Flow+Speed artifact into `raw_data` |
| `scripts/validate_flow_speed_dataset.py` | Validate full Flow+Speed dataset before training |
| `scripts/run_flow_speed_to_flow_training.py` | Stage 2.1 Flow+Speed input, Flow output |
| `sumo_pdformer_flow_speed_joint.json` | Stage 2.2 Flow+Speed input, joint Flow+Speed output |
| `scripts/validate_stage2_2_mask.py` | Validate Stage 2.2 Speed_Valid data wiring without DTW/K-Shape |
| `scripts/run_flow_speed_joint_training.py` | Stage 2.2 Flow+Speed input, joint Flow+Speed output |
| `libcity/evaluator/flow_speed_evaluator.py` | Flow/Speed separate metrics with Speed_Valid masking |
| `scripts/run_ha_baseline.py` | Same-split Historical Average baseline |
| `scripts/run_all_validations.py` | Run all gates in sequence |
| `libcity/data/dataset/dtw_utils.py` | Shared parallel DTW distance helper |

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

# Gate 1 uses the lightweight point dataset on smoke data.
# It checks tensor layout only and intentionally skips PDFormer DTW/K-Shape.

# Gate 2: Time alignment
python scripts/validate_gate2_time_alignment.py

# Smoke test: 1 epoch (2-variable, output_dim=2)
python scripts/run_smoke_test.py

# Gate 3: Single-batch overfit (Flow-only)
python scripts/validate_gate3_overfit.py

# Gate 4: 30 epoch baseline training (Flow-only)
python scripts/run_baseline_training.py

# Same-split Historical Average baseline
python scripts/run_ha_baseline.py

# Stage 2.1: Flow+Speed input, Flow-only output
python scripts/run_flow_speed_to_flow_training.py

# Stage 2.2: validate Speed_Valid wiring without DTW/K-Shape
python scripts/validate_stage2_2_mask.py

# Stage 2.2: Flow+Speed input, joint Flow+Speed output
python scripts/run_flow_speed_joint_training.py
```

---

## Key Design Decisions

### Dataset Layout
- **SUMO_BEIJING_FIXED_V2**: full Flow+Speed dataset. Stage 2.1 uses
  `input_dim=2` and `output_dim=1`; Stage 2.2 uses `input_dim=2`,
  `output_dim=2`, and Speed_Valid-masked speed loss/metrics.
- **SUMO_BEIJING_FIXED_V2_FLOW**: frozen Flow-only baseline dataset,
  `input_dim=1` and `output_dim=1`.

PDFormer now separates:

- `input_dim`: number of traffic variables read by PDFormer.
- `output_dim`: number of traffic variables predicted and evaluated.

Stage 2.1 uses `input_dim=2` and `output_dim=1`.
Stage 2.2 uses `input_dim=2` and `output_dim=2`.

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
    "max_epoch": 30,
    "set_loss": "mae",
    "lr_warmup_epoch": 5,
    "patience": 8,
    "cache_dataset": false
}
```
- `set_loss: "mae"` - standard MAE, zero-flow targets contribute to loss
- `set_loss: "masked_mae"` - masked MAE, zeros are excluded (81.2% of Flow data)

---

## First Run Notes

1. **DTW computation is train-only and parallelized**: First run computes DTW for 1008 nodes using only the train split. Cache names include train rate, radius, and output_dim, for example `dtw_SUMO_BEIJING_FIXED_V2_FLOW_train0.6_r6_out1.npy`.
2. **K-Shape clustering**: Also slow on first run. Cache names include train rate, output_dim, scaler, candidate days, cluster count, and max iterations.
3. **Stage 2.2 has separate `out2` caches**: the first joint Flow+Speed run will compute `dtw_SUMO_BEIJING_FIXED_V2_train0.6_r6_out2.npy` and `pattern_keys_kshape_SUMO_BEIJING_FIXED_V2_train0.6_out2_standard_days14_s3_k16_iter5.npy`.
4. **After success**: Back up cache to `cache_backup/`

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

## After Stage 2.2 Is Stable

Only after joint Flow+Speed output is validated with Speed_Valid-masked speed
metrics:
```bash
git checkout -b feature/intent-fusion
```
