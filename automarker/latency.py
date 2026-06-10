#!/usr/bin/env python3
"""
Latency and scalability benchmark for AutoMarker adaptive neighborhood expansion.
"""

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '8')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '8')
os.environ.setdefault('BLIS_NUM_THREADS', '8')

import json
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

from automarker.core import (
    load_sample_dirs_from_txt,
    estimate_adaptive_radius_params,
    moving_average_smooth,
)

EPS = 1e-8


@dataclass
class LatencyMetrics:
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    std_latency_ms: float
    throughput_qps: float
    total_time_ms: float


class AutoMarkerLatencyBenchmark:
    def __init__(self, output_dir: str, seed: int = 42):
        self.output_dir = output_dir
        self.seed = seed
        np.random.seed(seed)
        os.makedirs(output_dir, exist_ok=True)

    def load_real_data(self, root_dir: str, group1_txt: str, group2_txt: str, n_per_group: int, feature_dim: int | None = None):
        group1_dirs = load_sample_dirs_from_txt(group1_txt)
        group2_dirs = load_sample_dirs_from_txt(group2_txt)
        n1 = n_per_group // max(len(group1_dirs), 1) if n_per_group else None
        n2 = n_per_group // max(len(group2_dirs), 1) if n_per_group else None

        feats1, labels1 = self._load_single_group(root_dir, group1_dirs, 1, n1, self.seed, feature_dim)
        feats2, labels2 = self._load_single_group(root_dir, group2_dirs, 2, n2, self.seed + 1, feature_dim)

        features = np.concatenate([feats1, feats2], axis=0)
        labels = np.concatenate([labels1, labels2], axis=0)
        return features, labels

    def _load_single_group(self, root_dir: str, sample_dirs: List[str], group_label: int, n_per_sample: int | None, seed: int, feature_dim: int | None):
        import pickle

        rng = np.random.default_rng(seed)
        level_dir = os.path.join(root_dir, 'cell')
        all_features = []
        all_labels = []

        for slide_name in sample_dirs:
            slide_dir = os.path.join(level_dir, slide_name)
            if not os.path.isdir(slide_dir):
                continue
            slide_feats = []
            pkl_files = sorted([f for f in os.listdir(slide_dir) if f.endswith('.pkl')])
            for fname in pkl_files:
                fpath = os.path.join(slide_dir, fname)
                try:
                    with open(fpath, 'rb') as f:
                        data = pickle.load(f)
                    feats = np.asarray(data['features'], dtype=np.float32)
                    if feats.ndim != 2:
                        continue
                    if feature_dim is not None:
                        feats = feats[:, :feature_dim]
                    slide_feats.append(feats)
                except Exception:
                    continue
            if not slide_feats:
                continue
            slide_feats = np.concatenate(slide_feats, axis=0)
            if n_per_sample is not None and len(slide_feats) > n_per_sample:
                idx = rng.choice(len(slide_feats), size=n_per_sample, replace=False)
                slide_feats = slide_feats[idx]
            all_features.append(slide_feats)
            all_labels.append(np.full(len(slide_feats), group_label, dtype=np.int32))

        if not all_features:
            raise RuntimeError(f'No cells loaded for group {group_label}')
        return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)

    def measure_query_latency(self, features: np.ndarray, labels: np.ndarray, n_queries: int, n_repetitions: int = 3, radius_steps: int = 50):
        kdtree = KDTree(features)
        n1 = np.sum(labels == 1)
        n2 = np.sum(labels == 2)
        norm_factor = n2 / n1 if n1 > 0 else 1.0
        radius_params = estimate_adaptive_radius_params(features)
        radius_sequence = np.linspace(radius_params['r_min'], radius_params['r_max'], radius_steps)

        latencies = []
        for _ in range(n_repetitions):
            query_indices = np.random.choice(len(features), min(n_queries, len(features)), replace=False)
            for qp in features[query_indices]:
                start = time.perf_counter()
                lfc_trajectory = []
                dominant_direction = None
                for r_idx, r in enumerate(radius_sequence):
                    indices = kdtree.query_ball_point(qp, r=r)
                    if len(indices) < 3:
                        lfc_trajectory.append(0.0)
                        continue
                    win_labels = labels[indices]
                    n1_win = np.sum(win_labels == 1)
                    n2_win = np.sum(win_labels == 2)
                    dynamic_pseudo = 5.0 * (radius_params['r_min'] / (r + EPS))
                    current_lfc = np.log2((n1_win * norm_factor + dynamic_pseudo) / (n2_win + dynamic_pseudo))
                    if dominant_direction is None and (n1_win > 0 or n2_win > 0):
                        dominant_direction = np.sign(current_lfc)
                    if dominant_direction is not None and np.sign(current_lfc) != dominant_direction:
                        current_lfc = lfc_trajectory[-1] if lfc_trajectory else 0.0
                    lfc_trajectory.append(current_lfc)
                    if len(lfc_trajectory) >= 7:
                        smoothed = moving_average_smooth(lfc_trajectory, 5)
                        if len(smoothed) >= 3:
                            d2_lfc = smoothed[-1] - 2 * smoothed[-2] + smoothed[-3]
                            if d2_lfc <= 0 and r_idx >= 5:
                                break
                latencies.append((time.perf_counter() - start) * 1000.0)

        lat = np.asarray(latencies, dtype=float)
        return LatencyMetrics(
            avg_latency_ms=float(np.mean(lat)),
            p50_latency_ms=float(np.percentile(lat, 50)),
            p95_latency_ms=float(np.percentile(lat, 95)),
            p99_latency_ms=float(np.percentile(lat, 99)),
            min_latency_ms=float(np.min(lat)),
            max_latency_ms=float(np.max(lat)),
            std_latency_ms=float(np.std(lat)),
            throughput_qps=float(len(lat) / (np.sum(lat) / 1000.0)),
            total_time_ms=float(np.sum(lat)),
        )

    def experiment_query_scalability(self, features: np.ndarray, labels: np.ndarray, query_sizes: List[int]):
        rows = []
        for q in query_sizes:
            metrics = self.measure_query_latency(features, labels, n_queries=q, n_repetitions=3, radius_steps=50)
            rows.append({'dimension': 'query_scalability', 'query_size': q, **asdict(metrics)})
        return pd.DataFrame(rows)

    def experiment_data_scalability(self, features: np.ndarray, labels: np.ndarray, n_points_list: List[int], query_size: int):
        rows = []
        for n_points in n_points_list:
            if n_points > len(features):
                continue
            idx = np.random.choice(len(features), n_points, replace=False)
            metrics = self.measure_query_latency(features[idx], labels[idx], n_queries=query_size, n_repetitions=3, radius_steps=50)
            rows.append({'dimension': 'data_scalability', 'n_points': int(n_points), 'query_size': int(query_size), **asdict(metrics)})
        return pd.DataFrame(rows)

    def experiment_dimension_scalability(self, features: np.ndarray, labels: np.ndarray, dims: List[int], query_size: int):
        rows = []
        for d in dims:
            if d > features.shape[1]:
                continue
            reduced = PCA(n_components=d).fit_transform(features)
            metrics = self.measure_query_latency(reduced, labels, n_queries=query_size, n_repetitions=3, radius_steps=50)
            rows.append({'dimension': 'dimension_scalability', 'n_features': int(d), 'query_size': int(query_size), **asdict(metrics)})
        return pd.DataFrame(rows)

    def experiment_radius_steps(self, features: np.ndarray, labels: np.ndarray, radius_steps_list: List[int], query_size: int):
        rows = []
        for steps in radius_steps_list:
            metrics = self.measure_query_latency(features, labels, n_queries=query_size, n_repetitions=3, radius_steps=steps)
            rows.append({'dimension': 'radius_steps', 'radius_steps': int(steps), 'query_size': int(query_size), **asdict(metrics)})
        return pd.DataFrame(rows)

    def save_results(self, df: pd.DataFrame):
        csv_path = os.path.join(self.output_dir, 'latency_benchmark_results.csv')
        json_path = os.path.join(self.output_dir, 'latency_benchmark_results.json')
        df.to_csv(csv_path, index=False)
        payload = {k: v.to_dict(orient='records') for k, v in df.groupby('dimension')}
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        return csv_path, json_path


def parse_int_list(arg: str) -> List[int]:
    return [int(x.strip()) for x in str(arg).split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser(description='Latency and scalability benchmark for AutoMarker adaptive neighborhood expansion.')
    parser.add_argument('--root_dir', type=str, default='/path/to/features')
    parser.add_argument('--group1_txt', type=str, default='/path/to/group1.txt')
    parser.add_argument('--group2_txt', type=str, default='/path/to/group2.txt')
    parser.add_argument('--output_dir', type=str, default='/path/to/output_latency')
    parser.add_argument('--n_per_group', type=int, default=50000)
    parser.add_argument('--feature_dim', type=int, default=7)
    parser.add_argument('--query_sizes', type=str, default='100,500,1000,5000')
    parser.add_argument('--n_points_list', type=str, default='5000,10000,25000,50000')
    parser.add_argument('--dims', type=str, default='2,3,5,7')
    parser.add_argument('--radius_steps_list', type=str, default='10,20,30,50')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    benchmark = AutoMarkerLatencyBenchmark(args.output_dir, seed=args.seed)
    features, labels = benchmark.load_real_data(args.root_dir, args.group1_txt, args.group2_txt, args.n_per_group, feature_dim=args.feature_dim)

    all_frames = [
        benchmark.experiment_query_scalability(features, labels, parse_int_list(args.query_sizes)),
        benchmark.experiment_data_scalability(features, labels, parse_int_list(args.n_points_list), query_size=min(parse_int_list(args.query_sizes))),
        benchmark.experiment_dimension_scalability(features, labels, parse_int_list(args.dims), query_size=min(parse_int_list(args.query_sizes))),
        benchmark.experiment_radius_steps(features, labels, parse_int_list(args.radius_steps_list), query_size=min(parse_int_list(args.query_sizes))),
    ]
    combined = pd.concat(all_frames, ignore_index=True)
    csv_path, json_path = benchmark.save_results(combined)
    print(f'Saved CSV: {csv_path}')
    print(f'Saved JSON: {json_path}')


if __name__ == '__main__':
    main()
