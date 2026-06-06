#!/usr/bin/env python3
"""
HE-to-feature inference entry point.

This script generates an AutoMarker-compatible feature file by:

1. segment cells from an HE image;
2. extract per-cell features with CTransPath;
3. run the upstream Aligner inference model;
4. derive cell-level features as [signal channels, density channel];
5. save a single `.pkl` file for downstream AutoMarker analysis.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils import load_cellpose_model
from models import TransformerEncoder

try:
    from module.TransPath.ctran import ctranspath
except ImportError as exc:
    raise ImportError(
        'CTransPath is required for HE feature extraction. '
        'Please make sure the main project dependencies are available.'
    ) from exc

try:
    from skimage import io
except ImportError as exc:
    raise ImportError('scikit-image is required to read HE images.') from exc


DIM = 7
REMOVE_DIMS = [1, 3, 4]
DEFAULT_CHANNEL_NAMES = [f'AutoMarkerFeature_{idx}' for idx in range(DIM)] + ['Density']


def extract_env_features_from_cell_data(cell_data, k_channels_per_cell=3, k_cell_types_per_patch=5):
    """Build a lightweight ENV representation from cell-level features."""
    original_features = cell_data.get('features')
    if original_features is None:
        return None, None, None

    n_cells = original_features.shape[0]
    if n_cells == 0:
        env_features = {
            'sparse': True,
            'freq_dict': {},
            'num_cells': 0,
            'feature_dim': DIM,
            'encoding_info': {
                'max_code': 2 ** DIM - 1,
                'n_unique_codes': 0,
                'k_channels_per_cell': k_channels_per_cell,
                'k_cell_types_per_patch': k_cell_types_per_patch,
            },
        }
        env_cell_masks = np.zeros(cell_data.get('image_shape', (512, 512)), dtype=np.uint32)
        stats = {'cells_retained': 0, 'n_unique_codes': 0}
        return env_features, env_cell_masks, stats

    features = original_features[:, :DIM]
    binary_per_cell = np.zeros_like(features, dtype=np.uint8)
    for idx in range(n_cells):
        top_k_indices = np.argsort(features[idx])[-k_channels_per_cell:][::-1]
        binary_per_cell[idx, top_k_indices] = 1

    powers = np.array([2 ** idx for idx in range(DIM)], dtype=np.uint32)
    cell_codes = np.zeros(n_cells, dtype=np.uint32)
    for idx in range(n_cells):
        active_channels = np.where(binary_per_cell[idx] > 0)[0]
        if len(active_channels) > 0:
            cell_codes[idx] = np.sum(powers[active_channels], dtype=np.uint32)

    valid_codes = cell_codes[cell_codes > 0]
    if len(valid_codes) == 0:
        env_features = {
            'sparse': True,
            'freq_dict': {},
            'num_cells': 0,
            'feature_dim': DIM,
            'encoding_info': {
                'max_code': 2 ** DIM - 1,
                'n_unique_codes': 0,
                'k_channels_per_cell': k_channels_per_cell,
                'k_cell_types_per_patch': k_cell_types_per_patch,
            },
        }
        env_cell_masks = np.zeros(cell_data.get('image_shape', (512, 512)), dtype=np.uint32)
        stats = {'cells_retained': 0, 'n_unique_codes': 0}
        return env_features, env_cell_masks, stats

    unique_codes, counts = np.unique(valid_codes, return_counts=True)
    sorted_idx = np.argsort(counts)[::-1]
    n_to_keep = min(k_cell_types_per_patch, len(unique_codes))
    top_codes = unique_codes[sorted_idx[:n_to_keep]]
    keep_mask = np.isin(cell_codes, top_codes)
    filtered_codes = cell_codes.copy()
    filtered_codes[~keep_mask] = 0

    original_cell_masks = cell_data.get('cell_masks')
    if original_cell_masks is not None:
        env_cell_masks = np.zeros_like(original_cell_masks, dtype=np.uint32)
        for cell_idx in range(n_cells):
            if filtered_codes[cell_idx] > 0:
                mask_value = cell_idx + 1
                env_cell_masks[original_cell_masks == mask_value] = filtered_codes[cell_idx]
    else:
        env_cell_masks = np.zeros(cell_data.get('image_shape', (512, 512)), dtype=np.uint32)

    final_codes = filtered_codes[filtered_codes > 0]
    final_unique_codes, final_counts = np.unique(final_codes, return_counts=True) if len(final_codes) > 0 else (np.array([], dtype=np.uint32), np.array([], dtype=np.int64))
    total_retained_cells = int(np.sum(filtered_codes > 0))

    freq_dict = {}
    for code, count in zip(final_unique_codes, final_counts):
        if code > 0:
            freq_dict[int(code)] = float(count / total_retained_cells) if total_retained_cells > 0 else 0.0

    env_features = {
        'sparse': True,
        'freq_dict': freq_dict,
        'num_cells': total_retained_cells,
        'feature_dim': DIM,
        'encoding_info': {
            'max_code': 2 ** DIM - 1,
            'n_unique_codes': int(len(final_unique_codes)),
            'k_channels_per_cell': k_channels_per_cell,
            'k_cell_types_per_patch': k_cell_types_per_patch,
        },
    }
    stats = {'cells_retained': total_retained_cells, 'n_unique_codes': int(len(final_unique_codes))}
    return env_features, env_cell_masks, stats


def masks_to_centroids(masks):
    """Compute one centroid per segmented cell instance."""
    centroids = []
    instance_ids = []
    for instance_id in np.unique(masks):
        if instance_id == 0:
            continue
        coords = np.column_stack(np.where(masks == instance_id))
        if len(coords) == 0:
            continue
        centroids.append([float(coords[:, 1].mean()), float(coords[:, 0].mean())])
        instance_ids.append(int(instance_id))
    return np.asarray(centroids, dtype=np.float32), instance_ids



def extract_cell_features_for_inference_gpu(image_path, cellpose_model, ctranspath_model, device, max_cells=1024):
    """Extract per-cell CTransPath features following extract_feature_main.py."""
    image = io.imread(image_path)
    image_tensor = torch.from_numpy(image).float().to(device)

    with torch.no_grad():
        masks, _, _ = cellpose_model.eval(image, diameter=None, channels=[0, 0])

    masks_tensor = torch.from_numpy(masks).to(device=device, dtype=torch.int32)
    ctranspath_model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    unique_labels = torch.unique(masks_tensor)
    unique_labels = unique_labels[unique_labels != 0]
    max_area = (masks_tensor > 0).sum().float()

    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    cell_features = []
    for label in unique_labels:
        cell_mask = (masks_tensor == label).float()
        cell_area = cell_mask.sum()
        cell_mask_4d = cell_mask.unsqueeze(0).unsqueeze(0)
        grad_x = F.conv2d(cell_mask_4d, sobel_x, padding=1)
        grad_y = F.conv2d(cell_mask_4d, sobel_y, padding=1)
        cell_perimeter = torch.sqrt(grad_x ** 2 + grad_y ** 2).sum()
        roundness = 4 * np.pi * (cell_area / (cell_perimeter ** 2 + 1e-6))

        normalized_area = cell_area * 1e4 / (max_area + 1e-6)
        normalized_perimeter = cell_perimeter * 1e4 / (max_area + 1e-6)
        normalized_roundness = roundness * 1e4

        cell_image_tensor = image_tensor * cell_mask.unsqueeze(-1)
        cell_image_tensor = cell_image_tensor.permute(2, 0, 1)
        cell_image_pil = transforms.ToPILImage()(cell_image_tensor.cpu())
        cell_image_tensor = preprocess(cell_image_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            features = ctranspath_model(cell_image_tensor).squeeze(0)

        morphology_features = torch.tensor(
            [normalized_area.item(), normalized_perimeter.item(), normalized_roundness.item()],
            dtype=torch.float32,
            device=device,
        )
        expanded_morphology = torch.tile(morphology_features, (features.shape[0] // 3 + 1,))[:features.shape[0]]
        features = features * expanded_morphology
        cell_features.append(features)

    if cell_features:
        features_tensor = torch.stack(cell_features)
        num_cells = features_tensor.shape[0]
        if num_cells < max_cells:
            padding = torch.zeros((max_cells - num_cells, features_tensor.shape[1]), device=device)
            features_tensor = torch.cat([features_tensor, padding], dim=0)
        elif num_cells > max_cells:
            features_tensor = features_tensor[:max_cells]
        features_tensor = features_tensor.unsqueeze(0)
    else:
        features_tensor = torch.zeros((1, max_cells, 1000), device=device)

    return features_tensor, masks, image



def load_aligner_model(model_path, device, input_dim=1000, hidden_dim=512, n_heads=4, num_layers=6, output_dim=12, max_cells=1024):
    """Load the pretrained Aligner inference model."""
    model = TransformerEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        num_layers=num_layers,
        output_dim=output_dim,
        max_cells=max_cells,
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model



def run_inference(image_path, model_path, output_pkl, max_cells=1024):
    """Generate an AutoMarker-compatible cell-level feature file from one HE image."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cellpose_model = load_cellpose_model(model_type='cyto', device=device)
    ctranspath_model = ctranspath().to(device)
    ctranspath_model.eval()
    model = load_aligner_model(model_path, device=device, max_cells=max_cells)

    features_tensor, masks, original_image = extract_cell_features_for_inference_gpu(
        image_path=image_path,
        cellpose_model=cellpose_model,
        ctranspath_model=ctranspath_model,
        device=device,
        max_cells=max_cells,
    )

    centroids, instance_ids = masks_to_centroids(masks)
    num_cells = min(len(instance_ids), max_cells)

    if num_cells == 0:
        payload = {
            'features': np.zeros((0, DIM + 1), dtype=np.float32),
            'centroids': np.zeros((0, 2), dtype=np.float32),
            'cell_ids': [],
            'cell_masks': masks,
            'channel_names': DEFAULT_CHANNEL_NAMES,
            'image_path': image_path,
        }
        os.makedirs(os.path.dirname(output_pkl) or '.', exist_ok=True)
        with open(output_pkl, 'wb') as handle:
            pickle.dump(payload, handle)
        return payload

    mask_array = np.zeros(max_cells, dtype=np.float32)
    mask_array[:num_cells] = 1.0
    mask_tensor = torch.tensor(mask_array, dtype=torch.float32, device=device).unsqueeze(0)

    interception = {}

    def hook_fn(module, module_input, module_output):
        del module_output
        interception['x'] = module_input[0].detach()

    target_layer = model.transformer.layers[-1].self_attn
    handle = target_layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        model_output = model(features_tensor, mask=mask_tensor)
    handle.remove()

    if isinstance(model_output, tuple):
        logits_tensor = model_output[2]
    else:
        logits_tensor = model_output

    if logits_tensor.ndim == 2:
        logits_tensor = logits_tensor.unsqueeze(0)

    actual_logits = logits_tensor[0, :num_cells].detach().cpu().numpy()
    keep_dims = [idx for idx in range(actual_logits.shape[1]) if idx not in REMOVE_DIMS]
    actual_logits = actual_logits[:, keep_dims]

    x_intercepted = interception['x']
    embed_dim = target_layer.embed_dim
    qkv = F.linear(x_intercepted, target_layer.in_proj_weight, target_layer.in_proj_bias)
    q_all = qkv[:, :, :embed_dim]
    k_all = qkv[:, :, embed_dim:2 * embed_dim]
    seq_len, batch_size, _ = q_all.shape
    num_heads = target_layer.num_heads
    head_dim = embed_dim // num_heads
    q = q_all.view(seq_len, batch_size, num_heads, head_dim).permute(1, 2, 0, 3)
    k = k_all.view(seq_len, batch_size, num_heads, head_dim).permute(1, 2, 0, 3)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
    attn_probs = F.softmax(attn_scores, dim=-1).mean(dim=1)

    similarity_matrix = attn_probs[0, 1:num_cells + 1, 1:num_cells + 1].detach().cpu().numpy()
    density = np.sum(similarity_matrix, axis=1).reshape(-1, 1)

    signal = actual_logits[:, :DIM]
    z_cell = np.concatenate([signal, density], axis=1).astype(np.float32)

    weights = density / (np.sum(density) + 1e-6)
    z_tissue = np.sum(signal * weights, axis=0).astype(np.float32)

    temp_cell_data = {
        'features': z_cell,
        'cell_masks': masks,
        'image_shape': masks.shape,
    }
    env_features, env_cell_masks, env_stats = extract_env_features_from_cell_data(
        temp_cell_data,
        k_channels_per_cell=3,
        k_cell_types_per_patch=5,
    )

    payload = {
        'features': z_cell,
        'centroids': centroids[:num_cells].astype(np.float32),
        'cell_ids': instance_ids[:num_cells],
        'cell_masks': masks,
        'channel_names': DEFAULT_CHANNEL_NAMES,
        'image_path': image_path,
        'tissue_features': z_tissue,
        'tissue_channel_names': DEFAULT_CHANNEL_NAMES[:DIM],
        'env_features': env_features,
        'env_cell_masks': env_cell_masks,
        'env_stats': env_stats,
    }

    os.makedirs(os.path.dirname(output_pkl) or '.', exist_ok=True)
    with open(output_pkl, 'wb') as handle:
        pickle.dump(payload, handle)
    return payload



def main():
    parser = argparse.ArgumentParser(description='Generate AutoMarker-compatible cell-level features from an HE image.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to an HE image file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained Aligner weights.')
    parser.add_argument('--output_pkl', type=str, required=True, help='Output path for the generated feature .pkl file.')
    parser.add_argument('--max_cells', type=int, default=1024, help='Maximum number of cells processed per image.')
    args = parser.parse_args()

    payload = run_inference(
        image_path=args.image_path,
        model_path=args.model_path,
        output_pkl=args.output_pkl,
        max_cells=args.max_cells,
    )
    print(f"Saved {len(payload['features'])} cell-level features to {args.output_pkl}")


if __name__ == '__main__':
    main()
