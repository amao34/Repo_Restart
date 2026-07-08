"""
Historical Average baseline with the same sample split as PDFormer.

The baseline uses only training targets to build per-time-slot averages, then
evaluates each forecast horizon on the held-out test samples.

Usage:
    cd Bigscity-LibCity
    python scripts/run_ha_baseline.py
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def metric_row(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    err = y_pred - y_true
    abs_err = np.abs(err)
    sq_err = err ** 2

    mask = np.abs(y_true) > 1e-8
    masked_abs = abs_err[mask]
    masked_sq = sq_err[mask]
    masked_true = y_true[mask]

    mae = float(abs_err.mean())
    mse = float(sq_err.mean())
    rmse = float(np.sqrt(mse))
    wape = float(abs_err.sum() / max(np.abs(y_true).sum(), 1e-8))
    masked_mae = float(masked_abs.mean()) if mask.any() else np.nan
    masked_mse = float(masked_sq.mean()) if mask.any() else np.nan
    masked_rmse = float(np.sqrt(masked_mse)) if mask.any() else np.nan
    masked_mape = (
        float((masked_abs / np.abs(masked_true)).mean()) if mask.any() else np.nan
    )

    ss_res = float(sq_err.sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    true_var = float(np.var(y_true))
    evar = 1.0 - float(np.var(err)) / true_var if true_var > 0 else np.nan

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'WAPE': wape,
        'masked_MAE': masked_mae,
        'masked_MSE': masked_mse,
        'masked_RMSE': masked_rmse,
        'masked_MAPE': masked_mape,
        'R2': r2,
        'EVAR': evar,
    }


def load_dyna_array(dataset, output_dim):
    raw_dir = os.path.join(PROJECT_DIR, 'raw_data', dataset)
    raw_config = load_json(os.path.join(raw_dir, 'config.json'))
    info = raw_config['info']
    data_file = info.get('data_files', [dataset])[0]
    data_cols = info.get('data_col', [])[:output_dim]
    if len(data_cols) != output_dim:
        raise ValueError(
            'Expected {} data columns, got {}'.format(output_dim, data_cols))

    geo_path = os.path.join(raw_dir, '{}.geo'.format(info.get('geo_file', data_file)))
    dyna_path = os.path.join(raw_dir, '{}.dyna'.format(data_file))
    num_nodes = pd.read_csv(geo_path).shape[0]

    print('Loading {}'.format(dyna_path))
    dyna = pd.read_csv(dyna_path, usecols=['time', 'entity_id'] + data_cols)
    if dyna.shape[0] % num_nodes != 0:
        raise ValueError(
            'Dyna rows {} are not divisible by num_nodes {}'.format(
                dyna.shape[0], num_nodes))

    first_entities = dyna['entity_id'].iloc[:num_nodes].to_numpy()
    expected_entities = np.sort(first_entities)
    if not np.array_equal(first_entities, expected_entities):
        print('Dyna rows are not sorted by entity_id within the first time slot; sorting.')
        dyna = dyna.sort_values(['time', 'entity_id'])

    num_timestamps = dyna.shape[0] // num_nodes
    values = dyna[data_cols].to_numpy(dtype=np.float32)
    values = values.reshape(num_timestamps, num_nodes, output_dim)
    return values, raw_config


def compute_ha(dataset, config_file):
    config_path = os.path.join(PROJECT_DIR, '{}.json'.format(config_file))
    config = load_json(config_path)
    raw_config = load_json(os.path.join(PROJECT_DIR, 'raw_data', dataset, 'config.json'))
    info = raw_config['info']

    train_rate = float(config.get('train_rate', 0.7))
    eval_rate = float(config.get('eval_rate', 0.1))
    input_window = int(config.get('input_window', 12))
    output_window = int(config.get('output_window', 12))
    output_dim = int(config.get('output_dim', info.get('output_dim', 1)))
    time_intervals = int(info.get('time_intervals', 300))
    points_per_day = 24 * 3600 // time_intervals

    data, _ = load_dyna_array(dataset, output_dim)
    num_timestamps, num_nodes, _ = data.shape
    sample_t = np.arange(input_window - 1, num_timestamps - output_window)
    num_samples = sample_t.shape[0]
    num_test = round(num_samples * (1 - train_rate - eval_rate))
    num_train = round(num_samples * train_rate)
    num_val = num_samples - num_test - num_train
    train_t = sample_t[:num_train]
    test_t = sample_t[-num_test:]

    print('Dataset: {}'.format(dataset))
    print('Raw data shape: {}'.format(data.shape))
    print(
        'Samples: train={}, eval={}, test={}, input_window={}, output_window={}'.
        format(num_train, num_val, num_test, input_window, output_window))

    records = []
    global_train_mean = data[train_t].mean(axis=0)
    for horizon in range(1, output_window + 1):
        train_indices = train_t + horizon
        train_slots = train_indices % points_per_day

        sums = np.zeros((points_per_day, num_nodes, output_dim), dtype=np.float64)
        np.add.at(sums, train_slots, data[train_indices])
        counts = np.bincount(train_slots, minlength=points_per_day).astype(np.float64)

        slot_mean = np.empty_like(sums)
        for slot in range(points_per_day):
            if counts[slot] > 0:
                slot_mean[slot] = sums[slot] / counts[slot]
            else:
                slot_mean[slot] = global_train_mean

        test_indices = test_t + horizon
        test_slots = test_indices % points_per_day
        y_true = data[test_indices]
        y_pred = slot_mean[test_slots]

        row = {'horizon': horizon}
        row.update(metric_row(y_true, y_pred))
        records.append(row)

    return pd.DataFrame(records).set_index('horizon')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='SUMO_BEIJING_FIXED_V2_FLOW')
    parser.add_argument('--config_file', default='sumo_pdformer_flow')
    parser.add_argument('--output_dir', default='./libcity/cache/ha_baseline')
    args = parser.parse_args()

    result = compute_ha(args.dataset, args.config_file)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        'HA_{}_{}.csv'.format(args.dataset, args.config_file))
    result.to_csv(output_path)
    print()
    print(result)
    print()
    print('Saved HA baseline result at {}'.format(output_path))


if __name__ == '__main__':
    main()
