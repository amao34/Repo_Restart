import os
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
from fastdtw import fastdtw
from tqdm import tqdm


_DTW_SERIES = None
_DTW_RADIUS = None
_DTW_DIST = None


def _init_dtw_worker(series, radius, dist):
    global _DTW_SERIES, _DTW_RADIUS, _DTW_DIST
    warnings.filterwarnings('ignore')
    _DTW_SERIES = series
    _DTW_RADIUS = radius
    _DTW_DIST = dist


def _iter_pair_chunks(num_nodes, chunk_size):
    chunk = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            chunk.append((i, j))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def _fastdtw_distance(x, y, radius, dist):
    if dist is None:
        return fastdtw(x, y, radius=radius)[0]
    return fastdtw(x, y, radius=radius, dist=dist)[0]


def _compute_dtw_pair_chunk(pair_chunk):
    return _compute_dtw_pair_chunk_for_series(
        _DTW_SERIES, _DTW_RADIUS, _DTW_DIST, pair_chunk)


def _compute_dtw_pair_chunk_for_series(series, radius, dist, pair_chunk):
    rows = np.empty(len(pair_chunk), dtype=np.int32)
    cols = np.empty(len(pair_chunk), dtype=np.int32)
    distances = np.empty(len(pair_chunk), dtype=np.float64)
    for index, (i, j) in enumerate(pair_chunk):
        rows[index] = i
        cols[index] = j
        distances[index] = _fastdtw_distance(series[i], series[j], radius, dist)
    return rows, cols, distances


def _set_distances(matrix, result):
    rows, cols, distances = result
    matrix[rows, cols] = distances
    matrix[cols, rows] = distances
    return len(distances)


def _compute_dtw_sequential(series, radius, dist, chunk_size, desc, total_pairs):
    dtw_distance = np.zeros((series.shape[0], series.shape[0]), dtype=np.float64)
    with tqdm(total=total_pairs, desc=desc, unit='pair') as pbar:
        for pair_chunk in _iter_pair_chunks(series.shape[0], chunk_size):
            result = _compute_dtw_pair_chunk_for_series(
                series, radius, dist, pair_chunk)
            pbar.update(_set_distances(dtw_distance, result))
    return dtw_distance


def _compute_dtw_parallel(series, radius, dist, workers, chunk_size, desc, total_pairs):
    dtw_distance = np.zeros((series.shape[0], series.shape[0]), dtype=np.float64)
    chunks = _iter_pair_chunks(series.shape[0], chunk_size)
    pending = set()
    max_pending = max(1, workers * 2)

    with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_dtw_worker,
            initargs=(series, radius, dist)) as executor:
        with tqdm(total=total_pairs, desc=desc, unit='pair') as pbar:
            for _ in range(max_pending):
                try:
                    pending.add(executor.submit(
                        _compute_dtw_pair_chunk, next(chunks)))
                except StopIteration:
                    break

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pbar.update(_set_distances(dtw_distance, future.result()))
                    try:
                        pending.add(executor.submit(
                            _compute_dtw_pair_chunk, next(chunks)))
                    except StopIteration:
                        pass
    return dtw_distance


def compute_dtw_distance_matrix(series, radius=6, dist=None, workers=None,
                                chunk_size=2048, logger=None, desc='DTW'):
    """Compute a symmetric DTW distance matrix for node-major time series."""
    series = np.asarray(series)
    num_nodes = series.shape[0]
    dtw_distance = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    total_pairs = num_nodes * (num_nodes - 1) // 2
    if total_pairs == 0:
        return dtw_distance

    if workers is None:
        workers = os.cpu_count() or 1
    workers = min(total_pairs, max(1, int(workers)))
    if chunk_size is None:
        chunk_size = 2048
    chunk_size = max(1, int(chunk_size))

    if logger is not None:
        logger.info(
            'Computing DTW matrix ({} nodes, {} workers, {} pairs, '
            'chunk_size={}, radius={})...'.format(
                num_nodes, workers, total_pairs, chunk_size, radius))

    if workers == 1:
        return _compute_dtw_sequential(
            series, radius, dist, chunk_size, desc, total_pairs)

    try:
        return _compute_dtw_parallel(
            series, radius, dist, workers, chunk_size, desc, total_pairs)
    except Exception as err:
        if logger is not None:
            logger.warning(
                'Parallel DTW failed ({}); falling back to sequential DTW.'.format(
                    err))
        return _compute_dtw_sequential(
            series, radius, dist, chunk_size, desc, total_pairs)
