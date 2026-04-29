#!/usr/bin/env python
"""
Training script for encoding models across layers with GPU acceleration.
Trains GPU-accelerated linear encoders for each layer of a model on neural data.

Uses PyTorch with CUDA support for fast training on large feature sets.

Usage:
    python train_encoding_models.py --model Qwen3-VL-2B-Instruct --neural_dataset TVSD --dataset things_stimuli --roi IT --verbose
"""

import argparse
import numpy as np
import h5py
from pathlib import Path
import json
import sys
from sklearn.metrics import r2_score, mean_squared_error
from utils.predictive_alignement import ModelBrainDataset, SGDEncoder
from utils.inspection_utils import load_tsvd_dataset, load_eeg2_dataset, load_nsd_dataset


def load_neural_data(neural_dataset_name, roi, subject=None):
    """
    Load neural response data for the given dataset, subject, and ROI.
    
    Uses specialized loader functions based on dataset type:
    - TVSD (macaque): load_tsvd_dataset
    - EEG2 (human EEG): load_eeg2_dataset
    - NSD (human fMRI): load_nsd_dataset
    
    Parameters:
    neural_dataset_name (str): Dataset name ("TVSD", "EEG2"/"things_eeg2", or "NSD")
    roi (str): Region of interest (ROI) name
    subject (str): Subject identifier. Defaults per dataset if None:
                   - TVSD: "monkeyF"
                   - EEG2: "sub-01"
                   - NSD: "subj01"
    
    Returns:
    tuple: (y_train, y_test, stimuli_train, stimuli_test)
    """
    # Normalize dataset name
    dataset_lower = neural_dataset_name.lower()
    
    if dataset_lower in ["tvsd", "things_tvsd"]:
        if subject is None:
            subject = "monkeyF"
        y_train, stimuli_train = load_tsvd_dataset(split="train", subject=subject, roi=roi)
        y_test, stimuli_test = load_tsvd_dataset(split="test", subject=subject, roi=roi)
        
    elif dataset_lower in ["eeg2", "things_eeg2"]:
        if subject is None:
            subject = "sub-01"
        y_train, stimuli_train = load_eeg2_dataset(split="train", subject=subject, roi=roi)
        y_test, stimuli_test = load_eeg2_dataset(split="test", subject=subject, roi=roi)
        
    elif dataset_lower in ["nsd"]:
        if subject is None:
            subject = "subj01"
        y_train, stimuli_train = load_nsd_dataset(split="train", subject=subject, roi=roi)
        y_test, stimuli_test = load_nsd_dataset(split="test", subject=subject, roi=roi)
        
    else:
        raise ValueError(f"Unknown dataset: {neural_dataset_name}. Must be one of: TVSD, EEG2, NSD")
    
    return y_train, y_test, stimuli_train, stimuli_test


def get_layer_activations(activations_path, layer_name, stimuli_ids):
    """
    Load activations for a specific layer.
    
    Parameters:
    activations_path (str): Path to the h5 file with model activations
    layer_name (str): Name of the layer
    stimuli_ids (array-like): Stimulus IDs to load
    
    Returns:
    array-like: Activations (n_samples, n_features)
    """
    with h5py.File(activations_path, 'r') as f:
        layer_data = f['features'][layer_name][:]
        activation_indexes = list(f['ids'][:])
        
        # Map stimulus IDs to indices
        data_indexes = [activation_indexes.index(stimulus_id) for stimulus_id in stimuli_ids]
        X = layer_data[data_indexes, :]
    
    return X


def get_model_layers(model_name, dataset_name):
    """
    Get list of layers from model activations file.
    
    Parameters:
    model_name (str): Name of the model
    dataset_name (str): Name of the dataset
    
    Returns:
    list: List of layer names
    """
    activations_path = f"/shared/NX-414/extracted_features/{model_name}/{dataset_name}.h5"
    
    if not Path(activations_path).exists():
        raise FileNotFoundError(f"Model activations not found at {activations_path}")
    
    with h5py.File(activations_path, 'r') as f:
        layers = list(f['features'].keys())
    
    return layers


def train_layer_encoder(model_name, dataset_name, neural_dataset_name, roi, layer_name, subject=None, verbose=True):
    """
    Train and evaluate an encoding model for a single layer.
    
    Parameters:
    model_name (str): Name of the model
    dataset_name (str): Name of the dataset  
    neural_dataset_name (str): Name of the neural dataset
    roi (str): Region of interest
    layer_name (str): Name of the layer
    subject (str): Subject identifier (optional, defaults per dataset)
    verbose (bool): Print progress and results
    
    Returns:
    dict: Results dictionary with metrics
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Processing Layer: {layer_name}")
        print(f"{'='*70}")
    
    # Load neural data
    if verbose:
        print(f"Loading neural data for ROI: {roi} (subject: {subject or 'default'})...")
    y_train, y_test, stimuli_train, stimuli_test = load_neural_data(neural_dataset_name, roi, subject=subject)
    if verbose:
        print(f"  Neural data shapes - y_train: {y_train.shape}, y_test: {y_test.shape}")
    
    # Load model activations for this layer
    if verbose:
        print(f"Loading model activations...")
    activations_path = f"/shared/NX-414/extracted_features/{model_name}/{dataset_name}.h5"
    with h5py.File(activations_path, 'r') as f:
        layer_data = f['features'][layer_name][:]
        activation_indexes = np.array(f['ids'][:])
    
    # Map stimulus IDs to activations
    data_indexes_train = np.array([np.where(activation_indexes == sid)[0][0] for sid in stimuli_train])
    data_indexes_test = np.array([np.where(activation_indexes == sid)[0][0] for sid in stimuli_test])
    
    X_train = layer_data[data_indexes_train, :]
    X_test = layer_data[data_indexes_test, :]
    
    if verbose:
        print(f"  Feature shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
        print(f"  Target shapes - y_train: {y_train.shape}, y_test: {y_test.shape}")
    
    # Train encoder with batch processing and GPU acceleration
    if verbose:
        print(f"Initializing GPU-accelerated encoder with data normalization...")
    encoder = SGDEncoder(alpha=0.0001, max_iter=100, batch_size=256, learning_rate=0.001, random_state=42)
    
    # Fit with verbose output to see training progress
    if verbose:
        print(f"Training encoder on GPU (batch_size=256, lr=0.001)...")
    encoder.fit(X_train, y_train, batch_size=256, verbose=verbose)
    
    # Evaluate on test set
    if verbose:
        print(f"Evaluating on test set...")
    y_pred = encoder.predict(X_test)
    
    # Calculate metrics per unit (voxel/neuron)
    r2_scores = []
    mse_scores = []
    
    for i in range(y_test.shape[1]):
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        r2_scores.append(r2)
        mse_scores.append(mse)
    
    results = {
        'layer': layer_name,
        'model': model_name,
        'dataset': dataset_name,
        'roi': roi,
        'X_train_shape': X_train.shape,
        'X_test_shape': X_test.shape,
        'y_train_shape': y_train.shape,
        'y_test_shape': y_test.shape,
        'r2_mean': np.mean(r2_scores),
        'r2_std': np.std(r2_scores),
        'r2_median': np.median(r2_scores),
        'r2_min': np.min(r2_scores),
        'r2_max': np.max(r2_scores),
        'mse_mean': np.mean(mse_scores),
        'mse_std': np.std(mse_scores),
        'n_units': len(r2_scores)
    }
    
    if verbose:
        print(f"\nResults for {layer_name}:")
        print(f"  R² Mean: {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
        print(f"  R² Range: [{results['r2_min']:.4f}, {results['r2_max']:.4f}]")
        print(f"  MSE Mean: {results['mse_mean']:.4f} ± {results['mse_std']:.4f}")
        print(f"  Units analyzed: {results['n_units']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train encoding models for neural data using model activations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset names:
  - "TVSD" or "things_tvsd": macaque electrophysiology (THINGS dataset)
  - "EEG2" or "things_eeg2": human EEG responses (THINGS dataset)  
  - "NSD": human fMRI data

Examples:
  # TVSD (macaque IT) - uses default monkeyF
  python train_encoding_models.py --model Qwen3-VL-2B-Instruct --dataset TVSD --roi IT --verbose
  
  # TVSD with monkeyN
  python train_encoding_models.py --model Qwen3-VL-2B-Instruct --dataset TVSD --roi V1 --subject monkeyN --verbose
  
  # EEG2 (human) - uses default sub-01
  python train_encoding_models.py --model ViT-L-14 --dataset EEG2 --roi occipital_parietal --verbose
  
  # EEG2 with different subject
  python train_encoding_models.py --model ViT-L-14 --dataset EEG2 --roi occipital --subject sub-05 --verbose
  
  # NSD (fMRI) - uses default subj01
  python train_encoding_models.py --model Qwen3-VL-2B-Instruct --dataset NSD --roi V1d --verbose
  
  # NSD with different subject
  python train_encoding_models.py --model Qwen3-VL-2B-Instruct --dataset NSD --roi V1v --subject subj03 --verbose
        """
    )
    
    parser.add_argument(
        '--model', type=str, required=True,
        help='Model name (e.g., "Qwen3-VL-2B-Instruct", "ViT-L-14")'
    )
    parser.add_argument(
        '--neural_dataset', type=str, required=True,
        help='Dataset name (TVSD, EEG2, or NSD)'
    )
    parser.add_argument(
        '--dataset', type=str, required=True,
        help='Stimuli dataset name (e.g., "things_stimuli")'
    )
    parser.add_argument(
        '--roi', type=str, required=True,
        help='Target ROI (e.g., "IT", "V1", "V4")'
    )
    parser.add_argument(
        '--subject', type=str, default=None,
        help='Subject identifier (optional, defaults per dataset: monkeyF for TVSD, sub-01 for EEG2, subj01 for NSD)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print detailed progress'
    )
    parser.add_argument(
        '--save-results', action='store_true', default=True,
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./results',
        help='Directory to save results (default: ./results)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.verbose:
        print("="*70)
        print("ENCODING MODEL TRAINING SCRIPT")
        print("="*70)
        print(f"Model: {args.model}")
        print(f"Dataset: {args.dataset}")
        print(f"ROI: {args.roi}")
        print(f"Subject: {args.subject or 'default'}")
        print(f"Verbose: {args.verbose}")
        print("="*70)
    
    try:
        # Get layers
        if args.verbose:
            print("\nRetrieving layer information...")
        layers = get_model_layers(args.model, args.dataset)
        if args.verbose:
            print(f"Found {len(layers)} layers: {layers}")
        
        # Train encoder for each layer
        all_results = []
        for i, layer in enumerate(layers, 1):
            if args.verbose:
                print(f"\n[{i}/{len(layers)}]", end=" ")
            
            results = train_layer_encoder(
                args.model, args.dataset, args.neural_dataset, args.roi, layer,
                subject=args.subject,
                verbose=args.verbose
            )
            all_results.append(results)
        
        # Summary
        if args.verbose:
            print(f"\n{'='*70}")
            print("SUMMARY")
            print(f"{'='*70}")
            print(f"Layers processed: {len(all_results)}")
            print(f"Mean R² across layers: {np.mean([r['r2_mean'] for r in all_results]):.4f}")
            print(f"Best R² (layer mean): {max([r['r2_mean'] for r in all_results]):.4f}")
            print(f"Worst R² (layer mean): {min([r['r2_mean'] for r in all_results]):.4f}")
        
        # Save results
        if args.save_results:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            subject_str = f"_{args.subject}" if args.subject else ""
            output_file = output_dir / f"{args.model}_{args.dataset}_{args.roi}{subject_str}_results.json"
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            if args.verbose:
                print(f"\nResults saved to: {output_file}")
        
        print("\n✓ Training completed successfully!")
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
