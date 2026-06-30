import os
import warnings
import numpy as np
from libcity.data.dataset import TrafficStatePointDataset
from libcity.data.dataset.dtw_utils import compute_dtw_distance_matrix
from libcity.data.utils import generate_dataloader

warnings.filterwarnings('ignore', message='h5py not installed.*', category=UserWarning)
from tslearn.clustering import TimeSeriesKMeans, KShape


class PDFormerDataset(TrafficStatePointDataset):

    def __init__(self, config):
        self.type_short_path = config.get('type_short_path', 'hop')
        super().__init__(config)
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'pdformer_point_based_{}.npz'.format(self.parameters_str))
        self.points_per_hour = 3600 // self.time_intervals
        self.dtw_matrix = self._get_dtw()
        self.points_per_day = 24 * 3600 // self.time_intervals
        self.cand_key_days = config.get("cand_key_days", 14)
        self.s_attn_size = config.get("s_attn_size", 3)
        self.n_cluster = config.get("n_cluster", 16)
        self.cluster_max_iter = config.get("cluster_max_iter", 5)
        self.cluster_method = config.get("cluster_method", "kshape")

    def _get_dtw(self):
        dtw_radius = self.config.get('dtw_radius', 6)
        cache_name = 'dtw_{}_train{}_r{}_out{}.npy'.format(
            self.dataset, self.train_rate, dtw_radius, self.output_dim)
        cache_path = os.path.join('./libcity/cache/dataset_cache', cache_name)
        if os.path.exists(cache_path):
            dtw_matrix = np.load(cache_path)
            self._logger.info('Load DTW matrix from {}'.format(cache_path))
            return dtw_matrix

        for ind, filename in enumerate(self.data_files):
            current_df = self._load_dyna(filename)
            if ind == 0:
                df = current_df
            else:
                df = np.concatenate((df, current_df), axis=0)

        points_per_day = 24 * 3600 // self.time_intervals
        total_days = df.shape[0] // points_per_day
        train_days = max(1, int(total_days * self.train_rate))
        train_steps = train_days * points_per_day
        train_df = df[:train_steps, :, :self.output_dim]
        daily_data = train_df.reshape(
            train_days, points_per_day, self.num_nodes, self.output_dim)
        data_mean = daily_data.mean(axis=0)

        self._logger.info(
            'Computing DTW from train split only: total_days={}, '
            'train_days={}, train_steps={}'.format(
                total_days, train_days, train_steps))
        dtw_workers = self.config.get('dtw_workers', os.cpu_count() or 1)
        chunk_size = max(1, int(self.config.get('dtw_pair_chunk_size', 2048)))
        dtw_distance = compute_dtw_distance_matrix(
            np.swapaxes(data_mean, 0, 1), radius=dtw_radius,
            workers=dtw_workers, chunk_size=chunk_size, logger=self._logger)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, dtw_distance)
        self._logger.info('Saved DTW matrix at {}'.format(cache_path))
        return dtw_distance

    def _load_rel(self):
        self.sd_mx = None
        super()._load_rel()
        self._logger.info('Max adj_mx value = {}'.format(self.adj_mx.max()))
        self.sh_mx = self.adj_mx.copy()
        if self.type_short_path == 'hop':
            self.sh_mx[self.sh_mx > 0] = 1
            self.sh_mx[self.sh_mx == 0] = 511
            for i in range(self.num_nodes):
                self.sh_mx[i, i] = 0
            for k in range(self.num_nodes):
                for i in range(self.num_nodes):
                    for j in range(self.num_nodes):
                        self.sh_mx[i, j] = min(self.sh_mx[i, j], self.sh_mx[i, k] + self.sh_mx[k, j], 511)
            np.save('{}.npy'.format(self.dataset), self.sh_mx)

    def _calculate_adjacency_matrix(self):
        self._logger.info("Start Calculate the weight by Gauss kernel!")
        self.sd_mx = self.adj_mx.copy()
        distances = self.adj_mx[~np.isinf(self.adj_mx)].flatten()
        std = distances.std()
        self.adj_mx = np.exp(-np.square(self.adj_mx / std))
        self.adj_mx[self.adj_mx < self.weight_adj_epsilon] = 0
        if self.type_short_path == 'dist':
            self.sd_mx[self.adj_mx == 0] = np.inf
            for k in range(self.num_nodes):
                for i in range(self.num_nodes):
                    for j in range(self.num_nodes):
                        self.sd_mx[i, j] = min(self.sd_mx[i, j], self.sd_mx[i, k] + self.sd_mx[k, j])

    def get_data(self):
        x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
        if self.data is None:
            self.data = {}
            if self.cache_dataset and os.path.exists(self.cache_file_name):
                x_train, y_train, x_val, y_val, x_test, y_test = self._load_cache_train_val_test()
            else:
                x_train, y_train, x_val, y_val, x_test, y_test = self._generate_train_val_test()
        self.feature_dim = x_train.shape[-1]
        self.ext_dim = self.feature_dim - self.output_dim
        self.scaler = self._get_scalar(self.scaler_type,
                                       x_train[..., :self.output_dim], y_train[..., :self.output_dim])
        self.ext_scaler = self._get_scalar(self.ext_scaler_type,
                                           x_train[..., self.output_dim:], y_train[..., self.output_dim:])
        x_train[..., :self.output_dim] = self.scaler.transform(x_train[..., :self.output_dim])
        y_train[..., :self.output_dim] = self.scaler.transform(y_train[..., :self.output_dim])
        x_val[..., :self.output_dim] = self.scaler.transform(x_val[..., :self.output_dim])
        y_val[..., :self.output_dim] = self.scaler.transform(y_val[..., :self.output_dim])
        x_test[..., :self.output_dim] = self.scaler.transform(x_test[..., :self.output_dim])
        y_test[..., :self.output_dim] = self.scaler.transform(y_test[..., :self.output_dim])
        if self.normal_external:
            x_train[..., self.output_dim:] = self.ext_scaler.transform(x_train[..., self.output_dim:])
            y_train[..., self.output_dim:] = self.ext_scaler.transform(y_train[..., self.output_dim:])
            x_val[..., self.output_dim:] = self.ext_scaler.transform(x_val[..., self.output_dim:])
            y_val[..., self.output_dim:] = self.ext_scaler.transform(y_val[..., self.output_dim:])
            x_test[..., self.output_dim:] = self.ext_scaler.transform(x_test[..., self.output_dim:])
            y_test[..., self.output_dim:] = self.ext_scaler.transform(y_test[..., self.output_dim:])
        train_data = list(zip(x_train, y_train))
        eval_data = list(zip(x_val, y_val))
        test_data = list(zip(x_test, y_test))
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data, eval_data, test_data, self.feature_name,
                                self.batch_size, self.num_workers, pad_with_last_sample=self.pad_with_last_sample)
        self.num_batches = len(self.train_dataloader)
        self.pattern_key_file = os.path.join(
            './libcity/cache/dataset_cache/',
            'pattern_keys_{}_{}_train{}_out{}_{}_days{}_s{}_k{}_iter{}'.format(
                self.cluster_method, self.dataset, self.train_rate,
                self.output_dim, self.scaler_type, self.cand_key_days,
                self.s_attn_size, self.n_cluster, self.cluster_max_iter))
        if not os.path.exists(self.pattern_key_file + '.npy'):
            cand_key_time_steps = self.cand_key_days * self.points_per_day
            pattern_cand_keys = x_train[:cand_key_time_steps, :self.s_attn_size, :, :self.output_dim].swapaxes(1, 2).reshape(-1, self.s_attn_size, self.output_dim)
            self._logger.info("Clustering...")
            if self.cluster_method == "kshape":
                km = KShape(n_clusters=self.n_cluster, max_iter=self.cluster_max_iter).fit(pattern_cand_keys)
            else:
                km = TimeSeriesKMeans(n_clusters=self.n_cluster, metric="softdtw", max_iter=self.cluster_max_iter).fit(pattern_cand_keys)
            self.pattern_keys = km.cluster_centers_
            np.save(self.pattern_key_file, self.pattern_keys)
            self._logger.info("Saved at file " + self.pattern_key_file + ".npy")
        else:
            self.pattern_keys = np.load(self.pattern_key_file + ".npy")
            self._logger.info("Loaded file " + self.pattern_key_file + ".npy")
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, "sd_mx": self.sd_mx, "sh_mx": self.sh_mx,
                "ext_dim": self.ext_dim, "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim, "num_batches": self.num_batches,
                "dtw_matrix": self.dtw_matrix, "pattern_keys": self.pattern_keys}
