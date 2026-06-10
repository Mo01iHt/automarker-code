# AutoMarker Code Package

This directory contains a structured code package for feature generation, region discovery, parameter sweeps, and region traceback.

## Package layout

```
package_root/
|- automarker/
|  |- __init__.py
|  |- core.py
|  |- upstream.py
|  |- sweep.py
|  |- latency.py
|  `- traceback.py
|- scripts/
|  |- run_he_to_multimodal_features.py
|  |- run_region_sweep.py
|  |- run_latency_benchmark.py
|  `- run_traceback_regions.py
|- requirements.txt
`- README.md
```

## Components

- `automarker.core`: region discovery and statistical filtering
- `automarker.upstream`: HE-to-feature inference
- `automarker.sweep`: region-level benchmarking and hyperparameter sweeps
- `automarker.latency`: latency and scalability benchmarking for adaptive neighborhood expansion
- `automarker.traceback`: high-confidence region traceback for cell-level features

## Entry points

### 1. Generate features from HE

```bash
python scripts/run_he_to_multimodal_features.py --image_path /path/to/image.png --model_path /path/to/aligner_weights.pt --output_pkl /path/to/output.pkl
```

### 2. Run region-level parameter sweep

```bash
python scripts/run_region_sweep.py --root_dir /path/to/features --group1_txt /path/to/group1.txt --group2_txt /path/to/group2.txt --output_folder /path/to/output_sweep
```

### 3. Run latency and scalability benchmark

```bash
python scripts/run_latency_benchmark.py --root_dir /path/to/features --group1_txt /path/to/group1.txt --group2_txt /path/to/group2.txt --output_dir /path/to/output_latency
```

### 4. Run region traceback on one cell-level feature file

```bash
python scripts/run_traceback_regions.py --json_path /path/to/region_feature_fingerprints.json --input_pkl /path/to/cell_feature.pkl --output_json /path/to/traceback.json
```

## Expected input format

### Cell-level feature `.pkl` files
Expected fields include:
- `features`: shape `(n_cells, n_features)`
- `centroids`: shape `(n_cells, 2)`
- `cell_ids`: instance identifiers
- `channel_names`: feature names

### Sweep input
The sweep pipeline expects:
- per-slide directories containing `.pkl` files
- two text files listing slide directory names for the two clinical groups

## Notes on evaluation

The sweep pipeline uses region-level evaluation. Each discovered region is evaluated using its original member cells and original region-level p-value, rather than by expanding the region into a global marker mask.

## Reproducibility

The sweep module supports cached preprocessing and resumable checkpoints to reduce recomputation after interruption.
