#!/usr/bin/env python3
"""
AutoMarker region-level parameter sweep.

This script performs resumable region-level AutoMarker parameter sweeps with cached preprocessing.
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
import math
import argparse
import hashlib
import pickle
from itertools import product

import numpy as np
import pandas as pd
import psutil
from scipy.spatial import KDTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from automarker.core import (
    load_sample_dirs_from_txt,
    estimate_adaptive_radius_params,
    compute_dynamic_lfc_with_expansion,
    get_pvalue_filtered_regions,
)

EPS = 1e-8


def set_seed(seed):
    np.random.seed(seed)


def get_process_memory_mb():
    return float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2))


def parse_list_arg(arg, cast_fn):
    return [cast_fn(x.strip()) for x in arg.split(',') if x.strip()]


def stable_token(config):
    payload = json.dumps(config, sort_keys=True)
    return hashlib.md5(payload.encode('utf-8')).hexdigest()[:12]


def maybe_int(x):
    return int(x) if float(x).is_integer() else float(x)


def extract_coords(data, n_cells):
    if 'centroids' in data:
        c = np.asarray(data['centroids'], dtype=np.float32)
        if c.ndim == 2 and c.shape[1] == 2 and c.shape[0] == n_cells:
            x0 = float(data.get('patch_x', data.get('tile_x', 0.0)))
            y0 = float(data.get('patch_y', data.get('tile_y', 0.0)))
            c = c.copy()
            c[:, 0] += x0
            c[:, 1] += y0
            return c

    x0 = float(data.get('patch_x', data.get('tile_x', 0.0)))
    y0 = float(data.get('patch_y', data.get('tile_y', 0.0)))
    # Coarse tile-level coordinates. This is less precise than centroids but stable for sweep ranking.
    return np.repeat(np.array([[x0, y0]], dtype=np.float32), n_cells, axis=0)


def load_single_group(root_dir, sample_dirs, group_label, n_per_sample, seed, feature_dim=None):
    rng = np.random.default_rng(seed)
    all_features, all_slides, all_groups, all_coords = [], [], [], []
    channel_names = None

    for slide_name in sample_dirs:
        slide_dir = os.path.join(root_dir, slide_name)
        if not os.path.isdir(slide_dir):
            print(f"[WARN] missing slide dir: {slide_dir}")
            continue

        slide_feats, slide_coords = [], []
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
                n_cells = feats.shape[0]
                coords = extract_coords(data, n_cells)

                if channel_names is None:
                    raw_names = list(data.get('channel_names', []))
                    if raw_names:
                        channel_names = raw_names[:feats.shape[1]]
                    else:
                        channel_names = [f'Ch_{i}' for i in range(feats.shape[1])]

                slide_feats.append(feats)
                slide_coords.append(coords)
            except Exception as exc:
                print(f"[WARN] failed loading {fpath}: {exc}")

        if not slide_feats:
            continue

        slide_feats = np.concatenate(slide_feats, axis=0)
        slide_coords = np.concatenate(slide_coords, axis=0)

        if n_per_sample is not None and len(slide_feats) > n_per_sample:
            idx = rng.choice(len(slide_feats), size=n_per_sample, replace=False)
            slide_feats = slide_feats[idx]
            slide_coords = slide_coords[idx]

        all_features.append(slide_feats)
        all_coords.append(slide_coords)
        all_slides.append(np.array([slide_name] * len(slide_feats), dtype='U64'))
        all_groups.append(np.array([group_label] * len(slide_feats), dtype=np.int32))
        print(f"Loaded {slide_name}: {len(slide_feats)} cells")

    if not all_features:
        raise RuntimeError(f'No cells loaded for group {group_label}')

    return {
        'features': np.concatenate(all_features, axis=0),
        'coords': np.concatenate(all_coords, axis=0),
        'slides': np.concatenate(all_slides, axis=0),
        'groups': np.concatenate(all_groups, axis=0),
        'channel_names': channel_names,
    }


def dataset_cache_paths(output_folder):
    cache_dir = os.path.join(output_folder, 'dataset_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return {
        'dir': cache_dir,
        'npz': os.path.join(cache_dir, 'dataset_arrays.npz'),
        'meta': os.path.join(cache_dir, 'dataset_meta.json'),
        'entropy': os.path.join(cache_dir, 'entropy_cache.npz'),
    }


def load_or_build_dataset(args):
    paths = dataset_cache_paths(args.output_folder)
    cache_key = {
        'root_dir': args.root_dir,
        'group1_txt': args.group1_txt,
        'group2_txt': args.group2_txt,
        'n_per_group': args.n_per_group,
        'feature_dim': args.feature_dim,
        'seed': args.seed,
    }

    if os.path.exists(paths['npz']) and os.path.exists(paths['meta']):
        with open(paths['meta'], 'r') as f:
            meta = json.load(f)
        if meta.get('cache_key') == cache_key:
            print('\nUsing cached dataset arrays')
            data = np.load(paths['npz'], allow_pickle=True)
            return {
                'features': data['features'],
                'coords': data['coords'],
                'slides': data['slides'],
                'groups': data['groups'],
                'channel_names': meta['channel_names'],
                'paths': paths,
            }

    print('\nBuilding dataset cache from raw features')
    group1_dirs = load_sample_dirs_from_txt(args.group1_txt)
    group2_dirs = load_sample_dirs_from_txt(args.group2_txt)
    n1 = args.n_per_group // max(len(group1_dirs), 1) if args.n_per_group else None
    n2 = args.n_per_group // max(len(group2_dirs), 1) if args.n_per_group else None

    d1 = load_single_group(args.root_dir, group1_dirs, 1, n1, args.seed, feature_dim=args.feature_dim)
    d2 = load_single_group(args.root_dir, group2_dirs, 2, n2, args.seed + 1, feature_dim=args.feature_dim)

    out = {
        'features': np.concatenate([d1['features'], d2['features']], axis=0),
        'coords': np.concatenate([d1['coords'], d2['coords']], axis=0),
        'slides': np.concatenate([d1['slides'], d2['slides']], axis=0),
        'groups': np.concatenate([d1['groups'], d2['groups']], axis=0),
        'channel_names': d1['channel_names'],
        'paths': paths,
    }

    np.savez_compressed(paths['npz'], features=out['features'], coords=out['coords'], slides=out['slides'], groups=out['groups'])
    with open(paths['meta'], 'w') as f:
        json.dump({'cache_key': cache_key, 'channel_names': out['channel_names']}, f, indent=2)
    return out


def build_slide_lookup(dataset):
    slide_order = []
    slide_to_group = {}
    for slide, group in zip(dataset['slides'], dataset['groups']):
        if slide not in slide_to_group:
            slide_order.append(slide)
            slide_to_group[slide] = int(group)
    return slide_order, slide_to_group


def compute_entropy_cache(dataset, args):
    paths = dataset['paths']
    entropy_key = {
        'entropy_cluster_k': args.entropy_cluster_k,
        'entropy_k': args.entropy_k,
        'n_cells': int(len(dataset['features'])),
        'feature_dim': int(dataset['features'].shape[1]),
    }
    if os.path.exists(paths['entropy']):
        data = np.load(paths['entropy'], allow_pickle=True)
        meta = json.loads(str(data['meta']))
        if meta == entropy_key:
            print('Using cached entropy values')
            return data['entropy_values']

    print('Computing entropy cache')
    feats = dataset['features']
    feats_z = (feats - feats.mean(axis=0, keepdims=True)) / (feats.std(axis=0, keepdims=True) + EPS)
    km = MiniBatchKMeans(n_clusters=args.entropy_cluster_k, random_state=args.seed, batch_size=4096, n_init='auto')
    cluster_labels = km.fit_predict(feats_z)

    entropy_values = np.zeros(len(feats), dtype=np.float32)
    slide_order, _ = build_slide_lookup(dataset)
    n_clusters = int(cluster_labels.max()) + 1
    for slide in slide_order:
        idx = np.where(dataset['slides'] == slide)[0]
        if len(idx) <= 1:
            continue
        k = min(args.entropy_k + 1, len(idx))
        nbrs = NearestNeighbors(n_neighbors=k)
        nbrs.fit(dataset['coords'][idx])
        neigh_idx = nbrs.kneighbors(return_distance=False)
        slide_clusters = cluster_labels[idx]
        for local_i, neighbors in enumerate(neigh_idx):
            neigh_clusters = slide_clusters[neighbors[1:]] if len(neighbors) > 1 else slide_clusters[neighbors]
            counts = np.bincount(neigh_clusters, minlength=n_clusters).astype(np.float64)
            probs = counts[counts > 0] / counts.sum()
            if len(probs) == 0:
                continue
            entropy_values[idx[local_i]] = float(-np.sum(probs * np.log(probs + EPS)) / np.log(n_clusters + EPS))

    np.savez_compressed(paths['entropy'], entropy_values=entropy_values, meta=json.dumps(entropy_key, sort_keys=True))
    return entropy_values


def per_slide_proportion(mask, slides, slide_order, slide_to_group):
    rows = []
    for slide in slide_order:
        slide_mask = slides == slide
        total = int(slide_mask.sum())
        selected = int(np.logical_and(mask, slide_mask).sum())
        value = selected / total if total > 0 else 0.0
        rows.append({'slide': slide, 'group': slide_to_group[slide], 'selected': selected, 'total': total, 'value': value})
    return pd.DataFrame(rows)


def safe_mwu(values, groups):
    g1 = np.asarray(values)[np.asarray(groups) == 1]
    g2 = np.asarray(values)[np.asarray(groups) == 2]
    if len(g1) == 0 or len(g2) == 0:
        return 1.0
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(g1, g2, alternative='two-sided').pvalue)
    except Exception:
        return 1.0


def cosine_to_prototype(features, prototype):
    feats = features / (np.linalg.norm(features, axis=1, keepdims=True) + EPS)
    proto = prototype.reshape(1, -1)
    proto = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + EPS)
    return (feats @ proto.T).ravel()


def evaluate_automarker(dataset, entropy_values, args, combo):
    features = dataset['features']
    labels = dataset['groups']
    slides = dataset['slides']
    slide_order, slide_to_group = build_slide_lookup(dataset)

    t0 = time.time()
    mem0 = get_process_memory_mb()

    radius_params = estimate_adaptive_radius_params(
        features,
        k_neighbors=combo['radius_knn_k'],
        percentile=combo['radius_percentile'],
        r_min_scale_factor=combo['r_min_scale_factor'],
        r_max_scale_factor=combo['r_max_scale_factor'],
    )
    kdtree = KDTree(features)
    n1 = max(int(np.sum(labels == 1)), 1)
    n2 = max(int(np.sum(labels == 2)), 1)
    norm_factor = n2 / n1

    lfc_results = compute_dynamic_lfc_with_expansion(
        features,
        labels,
        kdtree,
        norm_factor,
        density_col=None,
        radius_params=radius_params,
        radius_steps=combo['radius_steps'],
        min_samples=3,
        smooth_window=combo['smooth_window'],
        use_geometric_steps=False,
        base_pseudo=combo['base_pseudo'],
        min_total_support=combo['min_total_support'],
        fallback_min_support=combo['fallback_min_support'],
    )
    regions = get_pvalue_filtered_regions(
        features,
        labels,
        slides,
        lfc_results,
        n_neighbors=args.graph_neighbors,
        min_size_ratio=args.min_size_ratio,
        p_value_threshold=1.0,
    )

    marker_rows = []
    summary_rows = []
    for ridx, region in enumerate(regions, start=1):
        member_indices = np.asarray(region.get('member_indices', []), dtype=np.int64)
        if member_indices.size == 0:
            continue

        p_value = float(region.get('p_value', 1.0))
        mean_entropy = float(np.mean(entropy_values[member_indices])) if member_indices.size > 0 else np.nan
        marker_id = f'AutoMarker_R{ridx}'
        marker_rows.append({
            'Marker': marker_id,
            'p_value': p_value,
            '-log10(p)': -math.log10(max(p_value, 1e-300)),
            'selected_cells': int(member_indices.size),
            'mean_spot_entropy': mean_entropy,
            'region_score': float(region['score']),
            'region_lfc': float(region['lfc']),
            'high_response_channels': json.dumps(region.get('high_response_channels', [])),
        })
        summary_rows.append({'p_value': p_value, 'entropy': mean_entropy})

    runtime = time.time() - t0
    peak_memory = max(mem0, get_process_memory_mb())

    if summary_rows:
        pvals = np.array([r['p_value'] for r in summary_rows], dtype=float)
        ent = np.array([r['entropy'] for r in summary_rows], dtype=float)
        result = {
            'Method': 'AutoMarker',
            'Median -log10(p)': float(np.median(-np.log10(np.clip(pvals, 1e-300, 1.0)))),
            'Best p': float(np.min(pvals)),
            'Sig.markers': int(np.sum(pvals < 0.05)),
            'Mean Spot Entropy': float(np.mean(ent)),
            'Total runtime': float(runtime),
            'Peak memory': float(peak_memory),
            'n_regions': int(len(regions)),
            'n_markers_kept': int(len(marker_rows)),
        }
    else:
        result = {
            'Method': 'AutoMarker',
            'Median -log10(p)': np.nan,
            'Best p': np.nan,
            'Sig.markers': 0,
            'Mean Spot Entropy': np.nan,
            'Total runtime': float(runtime),
            'Peak memory': float(peak_memory),
            'n_regions': int(len(regions)),
            'n_markers_kept': 0,
        }

    return result, pd.DataFrame(marker_rows)


def combo_dir(output_folder, combo):
    token = stable_token(combo)
    return os.path.join(output_folder, 'sweep_checkpoints', token)


def run_or_resume_combo(dataset, entropy_values, args, combo):
    cdir = combo_dir(args.output_folder, combo)
    os.makedirs(cdir, exist_ok=True)
    summary_path = os.path.join(cdir, 'summary.json')
    markers_path = os.path.join(cdir, 'markers.csv')
    combo_path = os.path.join(cdir, 'combo.json')

    if (not args.force_rerun_combo) and os.path.exists(summary_path) and os.path.exists(markers_path):
        print(f"\nReusing combo {os.path.basename(cdir)}")
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        return summary

    print(f"\nRunning combo {os.path.basename(cdir)}: {combo}")
    summary, markers_df = evaluate_automarker(dataset, entropy_values, args, combo)
    full_summary = {**combo, **summary}
    with open(summary_path, 'w') as f:
        json.dump(full_summary, f, indent=2)
    with open(combo_path, 'w') as f:
        json.dump(combo, f, indent=2)
    markers_df.to_csv(markers_path, index=False)
    return full_summary


def collect_summaries(output_folder):
    base = os.path.join(output_folder, 'sweep_checkpoints')
    rows = []
    if not os.path.isdir(base):
        return pd.DataFrame()
    for name in sorted(os.listdir(base)):
        summary_path = os.path.join(base, name, 'summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                rows.append(json.load(f))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Resumable AutoMarker parameter sweep for Orion CRC')
    parser.add_argument('--root_dir', type=str, default='/c23227/hwx/MarkerFinder/temp/mifcsv/cell_raw')
    parser.add_argument('--group1_txt', type=str, default='/c23227/hwx/XCellAligner_old/datasets/orion_crc/good.txt')
    parser.add_argument('--group2_txt', type=str, default='/c23227/hwx/XCellAligner_old/datasets/orion_crc/bad.txt')
    parser.add_argument('--output_folder', type=str, default='/c23227/hwx/XCellAligner_old/output/orion_crc_automarker_sweep')
    parser.add_argument('--n_per_group', type=int, default=120000)
    parser.add_argument('--feature_dim', type=int, default=None)
    parser.add_argument('--entropy_cluster_k', type=int, default=12)
    parser.add_argument('--entropy_k', type=int, default=8)
    parser.add_argument('--radius_steps', type=int, default=50)
    parser.add_argument('--smooth_window', type=int, default=5)
    parser.add_argument('--sim_threshold', type=float, default=0.75)
    parser.add_argument('--graph_neighbors', type=int, default=10)
    parser.add_argument('--min_size_ratio', type=float, default=0.005)
    parser.add_argument('--min_marker_cells', type=int, default=200)
    parser.add_argument('--radius_knn_k_list', type=str, default='5,10,20')
    parser.add_argument('--radius_percentile_list', type=str, default='5,10,20')
    parser.add_argument('--r_min_scale_factor_list', type=str, default='0.01,0.03,0.05')
    parser.add_argument('--r_max_scale_factor_list', type=str, default='0.2,0.5,1.0')
    parser.add_argument('--base_pseudo_list', type=str, default='0.5,1,2,5')
    parser.add_argument('--min_total_support_list', type=str, default='20,50,100,200')
    parser.add_argument('--fallback_min_support_list', type=str, default='5,10,20,50')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force_rerun_combo', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_folder, exist_ok=True)

    print('\n' + '=' * 72)
    print('AutoMarker Orion CRC Parameter Sweep')
    print('=' * 72)
    print(f'Feature root: {args.root_dir}')
    print(f'Output dir:   {args.output_folder}')
    print(f'Fixed outer params: radius_steps={args.radius_steps}, smooth_window={args.smooth_window}, sim_threshold={args.sim_threshold}, graph_neighbors={args.graph_neighbors}, min_size_ratio={args.min_size_ratio}, min_marker_cells={args.min_marker_cells}')

    dataset = load_or_build_dataset(args)
    print(f"Total loaded cells: {len(dataset['features'])}")
    print(f"Feature dim: {dataset['features'].shape[1]}")

    entropy_values = compute_entropy_cache(dataset, args)

    radius_knn_k_list = parse_list_arg(args.radius_knn_k_list, int)
    radius_percentile_list = parse_list_arg(args.radius_percentile_list, float)
    r_min_scale_factor_list = parse_list_arg(args.r_min_scale_factor_list, float)
    r_max_scale_factor_list = parse_list_arg(args.r_max_scale_factor_list, float)
    base_pseudo_list = parse_list_arg(args.base_pseudo_list, float)
    min_total_support_list = parse_list_arg(args.min_total_support_list, int)
    fallback_min_support_list = parse_list_arg(args.fallback_min_support_list, int)

    combos = []
    for radius_knn_k, radius_percentile, r_min_scale_factor, r_max_scale_factor, base_pseudo, min_total_support, fallback_min_support in product(
        radius_knn_k_list,
        radius_percentile_list,
        r_min_scale_factor_list,
        r_max_scale_factor_list,
        base_pseudo_list,
        min_total_support_list,
        fallback_min_support_list,
    ):
        combos.append({
            'radius_knn_k': int(radius_knn_k),
            'radius_percentile': float(radius_percentile),
            'r_min_scale_factor': float(r_min_scale_factor),
            'r_max_scale_factor': float(r_max_scale_factor),
            'base_pseudo': float(base_pseudo),
            'min_total_support': int(min_total_support),
            'fallback_min_support': int(fallback_min_support),
            'radius_steps': int(args.radius_steps),
            'smooth_window': int(args.smooth_window),
        })

    print(f'Total parameter combinations: {len(combos)}')
    for combo in combos:
        run_or_resume_combo(dataset, entropy_values, args, combo)
        summary_df = collect_summaries(args.output_folder)
        if not summary_df.empty:
            sort_cols = ['Sig.markers', 'Median -log10(p)', 'Best p', 'Mean Spot Entropy']
            ascending = [False, False, True, True]
            sortable = summary_df.copy()
            sortable['Median -log10(p)'] = sortable['Median -log10(p)'].fillna(-1)
            sortable['Best p'] = sortable['Best p'].fillna(1.0)
            sortable['Mean Spot Entropy'] = sortable['Mean Spot Entropy'].fillna(1.0)
            sortable = sortable.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
            sortable.to_csv(os.path.join(args.output_folder, 'sweep_results.csv'), index=False)
            primary_cols = [
                'radius_knn_k', 'radius_percentile', 'r_min_scale_factor', 'r_max_scale_factor',
                'base_pseudo', 'min_total_support', 'fallback_min_support',
                'radius_steps', 'smooth_window',
                'Median -log10(p)', 'Best p', 'Sig.markers', 'Mean Spot Entropy',
                'Total runtime', 'Peak memory', 'n_regions', 'n_markers_kept'
            ]
            sortable[primary_cols].head(20).to_csv(os.path.join(args.output_folder, 'sweep_top20.csv'), index=False)

    final_df = pd.read_csv(os.path.join(args.output_folder, 'sweep_results.csv'))
    show_cols = [
        'radius_knn_k', 'radius_percentile', 'r_min_scale_factor', 'r_max_scale_factor',
        'base_pseudo', 'min_total_support', 'fallback_min_support',
        'radius_steps', 'smooth_window',
        'Median -log10(p)', 'Best p', 'Sig.markers', 'Mean Spot Entropy',
        'Total runtime', 'Peak memory'
    ]
    print('\nTop parameter settings:')
    print(final_df[show_cols].head(10).to_string(index=False))
    print(f"\nSaved sweep table to: {os.path.join(args.output_folder, 'sweep_results.csv')}")


if __name__ == '__main__':
    main()
