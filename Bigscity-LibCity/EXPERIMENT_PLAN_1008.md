# 1008-node experiment plan

This project now treats `SUMO_BEIJING_FIXED_V2` as the final 1008-node network
for the formal open-loop and closed-loop experiments.

## Positioning

- The frozen baseline is `SUMO_BEIJING_FIXED_V2_FLOW`.
- The frozen tag is `baseline-flow-pdformer-fixed-v2`.
- The current formal Flow-only baseline must not be tuned further just to chase
  small metric changes.
- The research problem should be named information-induced congestion or
  route-guidance-induced congestion. Braess-related terms may appear only as
  theoretical background or related work, not as method, module, or metric names.

## Immediate order

1. Keep the frozen Flow-only PDFormer baseline as the reference.
2. Use the recovered full two-variable dataset
   `raw_data/SUMO_BEIJING_FIXED_V2`.
3. Keep the frozen Flow+Speed dataset and `Speed_Valid` sidecar as the
   official state-model data source.
4. Complete open-loop evaluation metrics: MAE, RMSE, R2, WAPE, horizon metrics,
   and masked MAPE as an auxiliary metric.
5. Stage 2.1 is frozen: Flow+Speed input with Flow-only output reached
   Flow MAE about 0.2472 and is stable against the frozen Flow-only baseline.
6. Run Stage 2.2: Flow+Speed input with joint Flow+Speed output.
7. In Stage 2.2, train Speed only on `Speed_Valid=1` targets and report Flow
   and Speed metrics separately.
8. After Stage 2.2 is stable, add travel-time metrics.
9. Only after the state model is stable, define route intent data:
   `A_plan`, `A_exec`, and `A_realized`.

## Current runnable entries

```bash
python scripts/validate_gate1_dataloader.py
python scripts/validate_stage2_2_mask.py
python scripts/run_ha_baseline.py
python scripts/run_flow_speed_to_flow_training.py
python scripts/run_flow_speed_joint_training.py
```

## Frozen Flow+Speed Data

The full two-variable dataset has been restored and frozen at:

```text
raw_data/SUMO_BEIJING_FIXED_V2
```

It contains:

```text
SUMO_BEIJING_FIXED_V2.dyna
SUMO_BEIJING_FIXED_V2.geo
SUMO_BEIJING_FIXED_V2.rel
SUMO_BEIJING_FIXED_V2_masks.npz
SUMO_BEIJING_FIXED_V2_speed_valid.npz
SUMO_BEIJING_FIXED_V2_edge_map.csv
config.json
manifest.json
```

Validated shape:

```text
nodes: 1008
timestamps: 8640
dyna rows: 8709120
speed_valid shape: (8640, 1008)
speed_valid rate: 0.206682
```

If the dataset is lost again, restore it from the SUMO artifact folder first.
Only re-collect from SUMO when the artifact folder is unavailable or the
simulation setup itself changes. To re-collect, first create and fill the edge
mapping:

```bash
python scripts/collect_sumo_flow_speed_dataset.py \
  --write-edge-map-template edge_map_SUMO_BEIJING_FIXED_V2_template.csv
```

Fill `sumo_edge_id` for every internal `entity_id` 0..1007. Then collect:

```bash
python scripts/collect_sumo_flow_speed_dataset.py \
  --sumocfg path/to/scenario.sumocfg \
  --edge-map edge_map_SUMO_BEIJING_FIXED_V2.csv \
  --dataset SUMO_BEIJING_FIXED_V2 \
  --num-steps <number_of_5min_slots> \
  --overwrite
```

Validate the frozen dataset:

```bash
python scripts/validate_flow_speed_dataset.py --dataset SUMO_BEIJING_FIXED_V2
```

`Speed_Valid` is saved as a sidecar file:

```text
raw_data/SUMO_BEIJING_FIXED_V2/SUMO_BEIJING_FIXED_V2_speed_valid.npz
```

It should be used for speed loss and speed metrics, but should not be treated as
a third traffic variable in the first Flow+Speed state models.

## Stage 2.2

Stage 2.2 uses:

```text
config: sumo_pdformer_flow_speed_joint
runner: scripts/run_flow_speed_joint_training.py
dataset: SUMO_BEIJING_FIXED_V2
input_dim: 2
output_dim: 2
loss: flow_speed_masked_mae
evaluator: FlowSpeedEvaluator
```

Run the lightweight wiring check before formal training:

```bash
python scripts/validate_stage2_2_mask.py
```

Formal training:

```bash
python scripts/run_flow_speed_joint_training.py
```

The first formal Stage 2.2 run will build `out2` DTW and K-Shape caches. This
is expected and can take several hours on the 1008-node full dataset.

For smoke validation of Stage 2.1, `validate_gate1_dataloader.py` uses
`SUMO_BEIJING_FIXED_V2_SMOKE` with `input_dim=2` and `output_dim=1`.

For formal Stage 2.1 training, `run_flow_speed_to_flow_training.py` expects the
full two-variable dataset at:

```text
raw_data/SUMO_BEIJING_FIXED_V2
```

The current workspace contains the full Flow-only dataset, the full Flow+Speed
dataset, and the two-variable smoke dataset.

## Do not start yet

- Do not implement FiLM, late fusion, residual gates, or cross-attention before
  Flow+Speed state prediction is checked.
- Do not use Braess as a formal safety-layer or metric name.
- Do not switch to reinforcement learning for the first closed-loop controller.
