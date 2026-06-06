#!/usr/bin/env python3
"""
Region traceback script.

1. load region fingerprints from a JSON file;
2. load `.pkl` feature file;
3. assign each cell to at most one region with high-confidence matching;
4. save the traceback result as JSON.
"""

import argparse
import json
import os
import pickle
from collections import Counter, defaultdict

import numpy as np


DEFAULT_DOMAIN = 7



def load_target_profiles(json_path, domain=DEFAULT_DOMAIN):
    """Load region fingerprints from a JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f'Missing fingerprint JSON: {json_path}')

    with open(json_path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)

    profiles = []
    for index, region_key in enumerate(sorted(data.keys(), key=lambda x: int(x.split('_')[1])), 1):
        info = data[region_key]
        lower = np.asarray(info['feature_lower'], dtype=np.float32)
        upper = np.asarray(info['feature_upper'], dtype=np.float32)
        seed = np.asarray(info['seed_feature'], dtype=np.float32)
        profiles.append({
            'id': region_key,
            'index': index,
            'lower': lower[:domain],
            'upper': upper[:domain],
            'seed': seed[:domain],
            'center_feat': seed[:domain],
            'dist_threshold': float(info.get('dist_threshold', 0.3)),
            'lfc': float(info.get('lfc', 0.0)),
            'size': int(info.get('size_in_discovery', 0)),
        })
    return profiles



def compute_match_confidence(feature_vector, region_profile, domain=DEFAULT_DOMAIN):
    """Compute cosine/box/combined confidence for one cell-region pair."""
    feat = np.asarray(feature_vector[:domain], dtype=np.float32)
    center = np.asarray(region_profile['center_feat'][:domain], dtype=np.float32)

    feat_norm = feat / (np.linalg.norm(feat) + 1e-8)
    center_norm = center / (np.linalg.norm(center) + 1e-8)
    cosine_sim = float(np.dot(feat_norm, center_norm))
    cosine_dist = float(1.0 - cosine_sim)

    lower = np.asarray(region_profile['lower'][:domain], dtype=np.float32)
    upper = np.asarray(region_profile['upper'][:domain], dtype=np.float32)
    within_box = (feat >= lower) & (feat <= upper)
    within_box_ratio = float(np.sum(within_box) / len(within_box))

    combined_score = float(cosine_sim * 0.6 + within_box_ratio * 0.4)
    return {
        'cosine_sim': cosine_sim,
        'cosine_dist': cosine_dist,
        'within_box_ratio': within_box_ratio,
        'combined_score': combined_score,
    }



def find_region_with_confidence(
    sample_feat,
    target_profiles,
    min_cosine_sim=0.7,
    min_box_ratio=0.5,
    min_combined=0.65,
    require_box_majority=True,
    domain=DEFAULT_DOMAIN,
):
    """Assign one cell to the best-matching region if it passes all thresholds."""
    best_match = None
    best_confidence = -1.0
    best_details = None

    for profile in target_profiles:
        confidence = compute_match_confidence(sample_feat, profile, domain=domain)
        passes_cosine = confidence['cosine_sim'] >= min_cosine_sim
        passes_box = confidence['within_box_ratio'] >= min_box_ratio
        passes_combined = confidence['combined_score'] >= min_combined

        if require_box_majority:
            passes_box = passes_box and (confidence['within_box_ratio'] >= 0.5)

        if passes_cosine and passes_box and passes_combined:
            if confidence['combined_score'] > best_confidence:
                best_confidence = confidence['combined_score']
                best_match = profile
                best_details = confidence

    if best_match is None:
        return None

    return {
        'region_id': best_match['id'],
        'region_index': best_match['index'],
        'cosine_sim': best_details['cosine_sim'],
        'cosine_dist': best_details['cosine_dist'],
        'within_box_ratio': best_details['within_box_ratio'],
        'combined_score': best_details['combined_score'],
    }



def traceback_cells(
    features,
    target_profiles,
    min_cosine_sim=0.7,
    min_box_ratio=0.5,
    min_combined=0.65,
    domain=DEFAULT_DOMAIN,
):
    """Run high-confidence region traceback for all cells in one feature matrix."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)

    assignments = []
    region_to_cells = defaultdict(list)
    for cell_index, feature_vector in enumerate(features):
        match = find_region_with_confidence(
            feature_vector,
            target_profiles,
            min_cosine_sim=min_cosine_sim,
            min_box_ratio=min_box_ratio,
            min_combined=min_combined,
            domain=domain,
        )
        if match is None:
            assignments.append({
                'cell_index': cell_index,
                'matched': False,
            })
        else:
            assignments.append({
                'cell_index': cell_index,
                'matched': True,
                **match,
            })
            region_to_cells[match['region_id']].append(cell_index)

    region_counts = Counter({region_id: len(cell_ids) for region_id, cell_ids in region_to_cells.items()})
    return {
        'assignments': assignments,
        'region_to_cells': dict(region_to_cells),
        'region_counts': dict(region_counts),
        'matched_cells': int(sum(1 for row in assignments if row['matched'])),
        'unmatched_cells': int(sum(1 for row in assignments if not row['matched'])),
        'total_cells': int(len(assignments)),
    }



def load_feature_pkl(pkl_path):
    """Load one AutoMarker-compatible cell-level feature file."""
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f'Missing feature pkl: {pkl_path}')

    with open(pkl_path, 'rb') as handle:
        data = pickle.load(handle)
    if 'features' not in data:
        raise KeyError(f"'features' not found in {pkl_path}")
    return data



def run_traceback(pkl_path, json_path, output_json, min_cosine_sim, min_box_ratio, min_combined, domain):
    """Run the full traceback pipeline for one `.pkl` file."""
    data = load_feature_pkl(pkl_path)
    target_profiles = load_target_profiles(json_path, domain=domain)
    result = traceback_cells(
        features=data['features'],
        target_profiles=target_profiles,
        min_cosine_sim=min_cosine_sim,
        min_box_ratio=min_box_ratio,
        min_combined=min_combined,
        domain=domain,
    )

    output = {
        'input_pkl': pkl_path,
        'fingerprint_json': json_path,
        'thresholds': {
            'min_cosine_sim': min_cosine_sim,
            'min_box_ratio': min_box_ratio,
            'min_combined': min_combined,
            'domain': domain,
        },
        'summary': {
            'total_cells': result['total_cells'],
            'matched_cells': result['matched_cells'],
            'unmatched_cells': result['unmatched_cells'],
            'match_rate': float(result['matched_cells'] / result['total_cells']) if result['total_cells'] > 0 else 0.0,
        },
        'region_counts': result['region_counts'],
        'assignments': result['assignments'],
    }

    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as handle:
        json.dump(output, handle, indent=2)
    return output



def main():
    parser = argparse.ArgumentParser(description='AutoMarker region traceback for one cell-level feature file.')
    parser.add_argument('--json_path', type=str, required=True, help='Path to the region fingerprint JSON file.')
    parser.add_argument('--input_pkl', type=str, required=True, help='Path to one cell-level feature `.pkl` file.')
    parser.add_argument('--output_json', type=str, required=True, help='Path to save the traceback result JSON.')
    parser.add_argument('--domain', type=int, default=DEFAULT_DOMAIN, help='Number of feature channels used for matching.')
    parser.add_argument('--min_cosine_sim', type=float, default=0.7, help='Minimum cosine similarity threshold.')
    parser.add_argument('--min_box_ratio', type=float, default=0.5, help='Minimum in-box ratio threshold.')
    parser.add_argument('--min_combined', type=float, default=0.65, help='Minimum combined confidence threshold.')
    args = parser.parse_args()

    output = run_traceback(
        pkl_path=args.input_pkl,
        json_path=args.json_path,
        output_json=args.output_json,
        min_cosine_sim=args.min_cosine_sim,
        min_box_ratio=args.min_box_ratio,
        min_combined=args.min_combined,
        domain=args.domain,
    )
    print(
        f"Processed {output['summary']['total_cells']} cells; "
        f"matched {output['summary']['matched_cells']} cells; "
        f"saved traceback JSON to {args.output_json}"
    )


if __name__ == '__main__':
    main()
