import os
import json
import datetime
from logging import getLogger

import pandas as pd
import torch

from libcity.evaluator.abstract_evaluator import AbstractEvaluator
from libcity.model import loss
from libcity.utils import ensure_dir


class FlowSpeedEvaluator(AbstractEvaluator):
    """Evaluate Flow on all targets and Speed on Speed_Valid targets only."""

    def __init__(self, config):
        self.metrics = config.get('metrics', [
            'Flow_MAE', 'Flow_RMSE', 'Flow_WAPE', 'Flow_R2',
            'Speed_MAE', 'Speed_RMSE', 'Speed_WAPE', 'Speed_R2',
            'Speed_Valid_Rate',
        ])
        self.allowed_metrics = [
            'Flow_MAE', 'Flow_RMSE', 'Flow_WAPE', 'Flow_R2',
            'Flow_masked_MAPE',
            'Speed_MAE', 'Speed_RMSE', 'Speed_WAPE', 'Speed_R2',
            'Speed_MAPE', 'Speed_Valid_Rate', 'Joint_MAE',
        ]
        self.save_modes = config.get('save_mode', ['csv'])
        self.mode = config.get('evaluator_mode', 'single')
        self.config = config
        self.len_timeslots = 0
        self.result = {}
        self.intermediate_result = {}
        self._check_config()
        self._logger = getLogger()

    def _check_config(self):
        if not isinstance(self.metrics, list):
            raise TypeError('Evaluator metrics must be a list')
        for metric in self.metrics:
            if metric not in self.allowed_metrics:
                raise ValueError(
                    'the metric {} is not allowed in FlowSpeedEvaluator'.format(
                        str(metric)))

    @staticmethod
    def _safe_r2(preds, labels):
        try:
            return float(loss.r2_score_torch(preds, labels))
        except Exception:
            return float('nan')

    @staticmethod
    def _masked_values(preds, labels, mask):
        mask = mask.bool()
        if mask.sum().item() == 0:
            return None, None
        return preds[mask], labels[mask]

    @staticmethod
    def _mape(preds, labels, eps=1e-8):
        denom = torch.clamp(torch.abs(labels), min=eps)
        return torch.mean(torch.abs(preds - labels) / denom)

    def _compute_metric(self, metric, y_pred, y_true, speed_valid):
        flow_pred = y_pred[..., 0]
        flow_true = y_true[..., 0]
        speed_pred = y_pred[..., 1]
        speed_true = y_true[..., 1]
        speed_mask = speed_valid.squeeze(-1).bool()

        if metric == 'Flow_MAE':
            return loss.masked_mae_torch(flow_pred, flow_true).item()
        if metric == 'Flow_RMSE':
            return loss.masked_rmse_torch(flow_pred, flow_true).item()
        if metric == 'Flow_WAPE':
            return loss.wape_torch(flow_pred, flow_true).item()
        if metric == 'Flow_R2':
            return self._safe_r2(flow_pred, flow_true)
        if metric == 'Flow_masked_MAPE':
            return loss.masked_mape_torch(flow_pred, flow_true, null_val=0).item()
        if metric == 'Speed_Valid_Rate':
            return speed_mask.float().mean().item()

        masked_pred, masked_true = self._masked_values(
            speed_pred, speed_true, speed_mask)
        if masked_pred is None:
            return float('nan')

        if metric == 'Speed_MAE':
            return torch.mean(torch.abs(masked_pred - masked_true)).item()
        if metric == 'Speed_RMSE':
            return torch.sqrt(torch.mean((masked_pred - masked_true) ** 2)).item()
        if metric == 'Speed_WAPE':
            return loss.wape_torch(masked_pred, masked_true).item()
        if metric == 'Speed_R2':
            return self._safe_r2(masked_pred, masked_true)
        if metric == 'Speed_MAPE':
            return self._mape(masked_pred, masked_true).item()
        if metric == 'Joint_MAE':
            flow_mae = loss.masked_mae_torch(flow_pred, flow_true)
            speed_mae = torch.mean(torch.abs(masked_pred - masked_true))
            return (flow_mae + speed_mae).item()
        raise ValueError('Unsupported metric {}'.format(metric))

    def collect(self, batch):
        if not isinstance(batch, dict):
            raise TypeError('evaluator.collect input is not a dict')
        y_true = batch['y_true']
        y_pred = batch['y_pred']
        if y_true.shape != y_pred.shape:
            raise ValueError("batch['y_true'].shape is not equal to batch['y_pred'].shape")
        if y_true.shape[-1] < 2:
            raise ValueError('FlowSpeedEvaluator requires y_true/y_pred last dim >= 2')
        if 'speed_valid' not in batch:
            raise ValueError('FlowSpeedEvaluator requires `speed_valid` in evaluate input')
        speed_valid = batch['speed_valid']
        if speed_valid.shape[:3] != y_true.shape[:3]:
            raise ValueError(
                'speed_valid shape {} is not aligned with y_true {}'.format(
                    tuple(speed_valid.shape), tuple(y_true.shape)))
        if speed_valid.ndim == 3:
            speed_valid = speed_valid.unsqueeze(-1)

        self.len_timeslots = y_true.shape[1]
        for i in range(1, self.len_timeslots + 1):
            if self.mode.lower() == 'average':
                pred_i = y_pred[:, :i]
                true_i = y_true[:, :i]
                mask_i = speed_valid[:, :i]
            elif self.mode.lower() == 'single':
                pred_i = y_pred[:, i - 1]
                true_i = y_true[:, i - 1]
                mask_i = speed_valid[:, i - 1]
            else:
                raise ValueError(
                    'Error parameter evaluator_mode={}, please set `single` or `average`.'.format(
                        self.mode))
            for metric in self.metrics:
                self.intermediate_result[metric + '@' + str(i)] = [
                    self._compute_metric(metric, pred_i, true_i, mask_i)]

    def evaluate(self):
        for i in range(1, self.len_timeslots + 1):
            for metric in self.metrics:
                values = self.intermediate_result[metric + '@' + str(i)]
                self.result[metric + '@' + str(i)] = sum(values) / len(values)
        return self.result

    def save_result(self, save_path, filename=None):
        self._logger.info('Note that you select the {} mode to evaluate!'.format(self.mode))
        self.evaluate()
        ensure_dir(save_path)
        if filename is None:
            filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + \
                       self.config['model'] + '_' + self.config['dataset']

        if 'json' in self.save_modes:
            self._logger.info('Evaluate result is ' + json.dumps(self.result))
            with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') as f:
                json.dump(self.result, f)
            self._logger.info('Evaluate result is saved at ' +
                              os.path.join(save_path, '{}.json'.format(filename)))

        dataframe = {}
        if 'csv' in self.save_modes:
            for metric in self.metrics:
                dataframe[metric] = []
            for i in range(1, self.len_timeslots + 1):
                for metric in self.metrics:
                    dataframe[metric].append(self.result[metric + '@' + str(i)])
            dataframe = pd.DataFrame(dataframe, index=range(1, self.len_timeslots + 1))
            dataframe.to_csv(os.path.join(save_path, '{}.csv'.format(filename)), index=False)
            self._logger.info('Evaluate result is saved at ' +
                              os.path.join(save_path, '{}.csv'.format(filename)))
            self._logger.info('\n' + str(dataframe))
            return dataframe
        return self.result

    def clear(self):
        self.result = {}
        self.intermediate_result = {}
