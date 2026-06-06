#!/usr/bin/env python3
"""
AutoMarker core implementation.
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.sparse import csgraph, csr_matrix
from sklearn.neighbors import NearestNeighbors, kneighbors_graph


EPS = 1e-6


def load_sample_dirs_from_txt(txt_file):
    """Load non-empty sample directory names from a text file."""
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f'Missing sample list: {txt_file}')

    with open(txt_file, 'r', encoding='utf-8') as handle:
        return [line.strip() for line in handle if line.strip()]



def estimate_adaptive_radius_params(
    features,
    k_neighbors=5,
    percentile=10,
    r_min_scale_factor=0.05,
    r_max_scale_factor=0.5,
):
    """Estimate a reasonable radius range from local feature-space density."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or len(features) == 0:
        raise ValueError('features must be a non-empty 2D array')

    base_scale = float(np.std(features))
    n_samples = min(len(features), 10000)
    sample_indices = np.random.choice(len(features), n_samples, replace=False)
    sampled_features = features[sample_indices]

    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, n_jobs=-1)
    nbrs.fit(sampled_features)
    distances, _ = nbrs.kneighbors(sampled_features)
    mean_knn_dist = distances[:, 1:].mean(axis=1)

    r_min_knn = float(np.percentile(mean_knn_dist, percentile))
    r_min_scale = float(base_scale * r_min_scale_factor)
    r_min = max(r_min_knn, r_min_scale)
    r_max = float(base_scale * r_max_scale_factor)

    if r_min >= r_max:
        r_max = r_min * 5.0

    return {
        'r_min': r_min,
        'r_max': r_max,
        'base_scale': base_scale,
        'mean_knn_dist': float(np.mean(mean_knn_dist)),
    }



def moving_average_smooth(trajectory, window_size=3):
    """Apply centered moving-average smoothing to a 1D trajectory."""
    if len(trajectory) < window_size:
        return trajectory
    if window_size % 2 == 0:
        window_size += 1

    smoothed = []
    half_window = window_size // 2
    for idx in range(len(trajectory)):
        start = max(0, idx - half_window)
        end = min(len(trajectory), idx + half_window + 1)
        smoothed.append(float(np.mean(trajectory[start:end])))
    return smoothed



def compute_dynamic_lfc_with_expansion(
    search_domain,
    labels,
    kdtree,
    norm_factor,
    density_col=None,
    radius_params=None,
    radius_steps=50,
    min_samples=5,
    smooth_window=3,
    use_geometric_steps=False,
    base_pseudo=5.0,
    min_total_support=200,
    fallback_min_support=20,
):
    """Compute smoothed dynamic log-fold-change trajectories over expanding neighborhoods."""
    search_domain = np.asarray(search_domain, dtype=np.float32)
    labels = np.asarray(labels)
    n_samples = len(search_domain)

    if radius_params is None:
        radius_params = estimate_adaptive_radius_params(search_domain)

    r_min = radius_params['r_min']
    r_max = radius_params['r_max']
    if use_geometric_steps:
        radius_sequence = np.geomspace(r_min, r_max, radius_steps)
    else:
        radius_sequence = np.linspace(r_min, r_max, radius_steps)

    lfc_num = np.zeros(n_samples, dtype=np.float32)
    lfc_rel = np.zeros(n_samples, dtype=np.float32) if density_col is not None else None
    adaptive_radius = np.zeros(n_samples, dtype=np.float32)
    support_scores = np.zeros(n_samples, dtype=np.float32)
    valid_mask = np.zeros(n_samples, dtype=bool)
    trajectories = []
    smoothed_trajectories = []

    for idx in range(n_samples):
        query_point = search_domain[idx]
        lfc_trajectory = []
        dominant_direction = None

        for step_idx, radius in enumerate(radius_sequence):
            member_indices = kdtree.query_ball_point(query_point, r=float(radius))
            if len(member_indices) < min_samples:
                lfc_trajectory.append(0.0)
                continue

            window_labels = labels[member_indices]
            n_group1 = int(np.sum(window_labels == 1))
            n_group2 = int(np.sum(window_labels == 2))
            dynamic_pseudo = base_pseudo * (r_min / (float(radius) + EPS))
            current_lfc = np.log2((n_group1 * norm_factor + dynamic_pseudo) / (n_group2 + dynamic_pseudo))

            if dominant_direction is None and (n_group1 > 0 or n_group2 > 0):
                dominant_direction = np.sign(current_lfc)
            if dominant_direction is not None and np.sign(current_lfc) != dominant_direction:
                current_lfc = lfc_trajectory[-1] if lfc_trajectory else 0.0

            lfc_trajectory.append(float(current_lfc))

            if len(lfc_trajectory) >= smooth_window + 2:
                smoothed = moving_average_smooth(lfc_trajectory, smooth_window)
                if len(smoothed) >= 3:
                    d2_lfc = smoothed[-1] - 2 * smoothed[-2] + smoothed[-3]
                    if d2_lfc <= 0 and step_idx >= smooth_window:
                        if len(member_indices) >= min_total_support:
                            adaptive_radius[idx] = float(radius)
                            lfc_num[idx] = float(np.clip(smoothed[-2], -8.0, 8.0))
                            support_scores[idx] = float(abs(lfc_num[idx]) * np.log1p(len(member_indices)))
                            valid_mask[idx] = True

                            if density_col is not None:
                                win_density = density_col[member_indices]
                                d1 = np.mean(win_density[window_labels == 1]) if n_group1 > 0 else 0.0
                                d2 = np.mean(win_density[window_labels == 2]) if n_group2 > 0 else 0.0
                                lfc_rel[idx] = float(np.log2((d1 + EPS) / (d2 + EPS)))

                            trajectories.append(lfc_trajectory)
                            smoothed_trajectories.append(smoothed)
                        break

        if not valid_mask[idx] and lfc_trajectory:
            smoothed = moving_average_smooth(lfc_trajectory, smooth_window)
            if abs(smoothed[-1]) > 0.1:
                adaptive_radius[idx] = float(r_max)
                lfc_num[idx] = float(np.clip(smoothed[-1], -8.0, 8.0))
                member_indices = kdtree.query_ball_point(query_point, r=float(r_max))
                if len(member_indices) >= fallback_min_support:
                    support_scores[idx] = float(abs(lfc_num[idx]) * np.log1p(len(member_indices)))
                    valid_mask[idx] = True

    result = {
        'lfc_num': lfc_num,
        'radius': adaptive_radius,
        'support_score': support_scores,
        'valid_mask': valid_mask,
        'radius_params': radius_params,
        'trajectories': trajectories,
        'smoothed_trajectories': smoothed_trajectories,
    }
    if lfc_rel is not None:
        result['lfc_rel'] = lfc_rel
    return result



def get_pvalue_filtered_regions(
    feats,
    labels,
    names,
    lfc_results,
    n_neighbors=10,
    min_size_ratio=0.01,
    p_value_threshold=0.05,
    exclusivity_strength=0.0,
):
    """Build same-sign connected components and keep regions with significant slide-level differences."""
    del exclusivity_strength  # kept only for backward-compatible calls
    from scipy.spatial.distance import cdist

    feats = np.asarray(feats, dtype=np.float32)
    labels = np.asarray(labels)
    names = np.asarray(names)

    valid_mask = np.asarray(lfc_results['valid_mask'], dtype=bool)
    valid_feats = feats[valid_mask]
    valid_lfc = np.asarray(lfc_results['lfc_num'])[valid_mask]
    valid_names = names[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    n_total = len(feats)

    if len(valid_feats) == 0:
        return []

    min_samples_limit = int(n_total * min_size_ratio)
    adjacency = kneighbors_graph(valid_feats, n_neighbors, mode='connectivity', include_self=False)
    rows, cols = adjacency.nonzero()
    lfc_signs = np.sign(valid_lfc)
    keep_edges = (lfc_signs[rows] == lfc_signs[cols]) & (lfc_signs[rows] != 0)

    refined_adj = csr_matrix((np.ones(np.sum(keep_edges)), (rows[keep_edges], cols[keep_edges])), shape=adjacency.shape)
    _, component_labels = csgraph.connected_components(refined_adj, directed=False)
    unique_components, component_sizes = np.unique(component_labels, return_counts=True)

    unique_slide_names, slide_totals = np.unique(names, return_counts=True)
    slide_total_dict = dict(zip(unique_slide_names, slide_totals))
    slide_to_group = {slide_name: int(group) for slide_name, group in zip(names, labels)}

    filtered_regions = []
    for comp_id, count in zip(unique_components, component_sizes):
        if count < min_samples_limit:
            continue

        component_mask = component_labels == comp_id
        component_feats = valid_feats[component_mask]
        component_lfc = valid_lfc[component_mask]
        component_names = valid_names[component_mask]
        component_indices = valid_indices[component_mask]

        local_best_idx = int(np.argmax(np.abs(component_lfc)))
        center_feat = component_feats[local_best_idx].reshape(1, -1)
        internal_dists = cdist(component_feats, center_feat, metric='cosine').ravel()
        dist_threshold = float(np.percentile(internal_dists, 95))

        mu = component_feats.mean(axis=0)
        std = component_feats.std(axis=0)
        feature_lower = mu - 2 * std
        feature_upper = mu + 2 * std

        unique_component_names, component_name_counts = np.unique(component_names, return_counts=True)
        hit_dict = dict(zip(unique_component_names, component_name_counts))

        sample_ratios = []
        group1_ratios = []
        group2_ratios = []
        for slide_name in unique_slide_names:
            cells_in_region = int(hit_dict.get(slide_name, 0))
            total_cells_in_slide = int(slide_total_dict[slide_name])
            ratio_percent = 100.0 * cells_in_region / total_cells_in_slide if total_cells_in_slide > 0 else 0.0
            sample_ratios.append({
                'sample_name': slide_name,
                'group': int(slide_to_group[slide_name]),
                'cells_in_region': cells_in_region,
                'total_cells_in_sample': total_cells_in_slide,
                'ratio_percent': float(ratio_percent),
            })
            if slide_to_group[slide_name] == 1:
                group1_ratios.append(ratio_percent)
            else:
                group2_ratios.append(ratio_percent)

        try:
            _, p_value = mannwhitneyu(group1_ratios, group2_ratios, alternative='two-sided')
            p_value = float(p_value)
        except Exception:
            p_value = 1.0

        if p_value <= p_value_threshold:
            filtered_regions.append({
                'center_idx': int(component_indices[local_best_idx]),
                'member_indices': component_indices.tolist(),
                'lfc': float(np.mean(component_lfc)),
                'score': float(abs(np.mean(component_lfc)) * np.sqrt(count)),
                'p_value': p_value,
                'size': int(count),
                'g1_ratios': group1_ratios,
                'g2_ratios': group2_ratios,
                'feature_lower': feature_lower,
                'feature_upper': feature_upper,
                'dist_threshold': dist_threshold,
                'center_feat': center_feat,
                'sample_ratios_df': pd.DataFrame(sample_ratios).sort_values(['group', 'sample_name']).reset_index(drop=True),
            })

    return sorted(filtered_regions, key=lambda region: region['score'], reverse=True)
