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


def train_layer_encoder(model_name, dataset_name, neural_dataset_name, roi, layer_name, subject=None, 
                       max_epochs=500, min_epochs=20, patience=10, tolerance=1e-5, 
                       batch_size=256, learning_rate=0.001, verbose=True):
    """
    Train and evaluate an encoding model for a single layer.
    
    Parameters:
    model_name (str): Name of the model
    dataset_name (str): Name of the dataset  
    neural_dataset_name (str): Name of the neural dataset
    roi (str): Region of interest
    layer_name (str): Name of the layer
    subject (str): Subject identifier (optional, defaults per dataset)
    max_epochs (int): Maximum number of training epochs
    min_epochs (int): Minimum epochs before early stopping allowed
    patience (int): Number of epochs with no improvement before early stopping
    tolerance (float): Tolerance for convergence detection
    batch_size (int): Batch size for training
    learning_rate (float): Learning rate for optimizer
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
    
    # Create ModelBrainDataset for proper train/test/val management
    if verbose:
        print(f"Creating dataset...")
    dataset = ModelBrainDataset(
        y_train=y_train, y_test=y_test,
        stimuli_train=stimuli_train, stimuli_test=stimuli_test,
        model_name=model_name, dataset_name=dataset_name, layer_name=layer_name
    )
    
    if verbose:
        print(f"  Feature shapes - X_train: {dataset.X_train.shape}, X_test: {dataset.X_test.shape}")
        print(f"  Target shapes - y_train: {dataset.y_train.shape}, y_test: {dataset.y_test.shape}")
    
    # Initialize encoder with GPU acceleration
    if verbose:
        print(f"Initializing GPU-accelerated encoder with hyperparameter tuning...")
        print(f"  Training params: max_epochs={max_epochs}, min_epochs={min_epochs}, patience={patience}, tol={tolerance:.1e}")
    encoder = SGDEncoder(
        alpha=0.0001, 
        max_iter=max_epochs,
        min_iter=min_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate, 
        early_stopping_patience=patience,
        early_stopping_tol=tolerance,
        random_state=42
    )
    
    # Fit and evaluate with hyperparameter selection via cross-validation
    if verbose:
        print(f"Training with hyperparameter selection (3-fold CV)...")
        print(f"  Testing regularization: [0.01, 0.1, 0.2]")
    validation_folds = 3
    results = encoder.fit_and_evaluate(
        dataset,
        alphas=[0.01, 0.1, 0.2],
        cv=validation_folds,
        val_size=1/validation_folds,
        scoring='r2',
        verbose=verbose
    )
    
    y_pred = results['y_pred_test']
    y_test = results['y_test']
    
    # Calculate metrics per unit (voxel/neuron)
    r2_scores = []
    mse_scores = []
    
    for i in range(y_test.shape[1]):
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        r2_scores.append(r2)
        mse_scores.append(mse)
    
    layer_results = {
        'layer': layer_name,
        'model': model_name,
        'dataset': dataset_name,
        'roi': roi,
        'subject': subject,
        'X_train_shape': dataset.X_train.shape,
        'X_test_shape': dataset.X_test.shape,
        'y_train_shape': dataset.y_train.shape,
        'y_test_shape': dataset.y_test.shape,
        'best_alpha': results['best_alpha'],
        'cv_score_mean': results['cv_score'],
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
        print(f"  Best Alpha: {layer_results['best_alpha']:.1e}")
        print(f"  CV R² Score: {layer_results['cv_score_mean']:.4f}")
        print(f"  Test R² Mean: {layer_results['r2_mean']:.4f} ± {layer_results['r2_std']:.4f}")
        print(f"  Test R² Range: [{layer_results['r2_min']:.4f}, {layer_results['r2_max']:.4f}]")
        print(f"  Test MSE Mean: {layer_results['mse_mean']:.4f} ± {layer_results['mse_std']:.4f}")
        print(f"  Units analyzed: {layer_results['n_units']}")
        
        # Diagnostic: Count negative R²
        n_negative = np.sum(np.array(r2_scores) < 0)
        if n_negative > 0:
            pct_negative = 100 * n_negative / len(r2_scores)
            print(f"  ⚠️  WARNING: {n_negative}/{len(r2_scores)} units ({pct_negative:.1f}%) have NEGATIVE R²")
            print(f"     (Model worse than predicting mean)")
            print(f"     REC: Try higher alpha, more epochs, or check data quality")
        else:
            print(f"  ✓ All units have positive R²")
    
    return layer_results


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
    parser.add_argument(
        '--max-epochs', type=int, default=1000,
        help='Maximum number of training epochs (default: 1000 - allow longer training)'
    )
    parser.add_argument(
        '--min-epochs', type=int, default=20,
        help='Minimum number of epochs before early stopping allowed (default: 100 - let model train more)'
    )
    parser.add_argument(
        '--patience', type=int, default=10,
        help='Number of epochs with no improvement before early stopping (default: 15)'
    )
    parser.add_argument(
        '--tolerance', type=float, default=1e-4,
        help='Tolerance for convergence detection (default: 1e-4 - less strict)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=2048,
        help='Batch size for training (default: 2048)'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.0001,
        help='Learning rate for optimizer (default: 0.0001)'
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
        
        # Create output files
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        subject_str = f"_{args.subject}" if args.subject else ""
        txt_file = output_dir / f"{args.model[:10]}_{args.dataset}_{args.neural_dataset}_{args.roi}_{subject_str}_results.txt"
        json_file = output_dir / f"{args.model[:10]}_{args.dataset}_{args.neural_dataset}_{args.roi}_{subject_str}_results.json"
        
        # Write txt header
        with open(txt_file, 'w') as f:
            f.write(f"Encoding Model Training Results\n")
            f.write(f"{'='*70}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Neural Dataset: {args.neural_dataset}\n")
            f.write(f"ROI: {args.roi}\n")
            f.write(f"Subject: {args.subject or 'default'}\n")
            f.write(f"{'='*70}\n\n")
        
        # Train encoder for each layer
        all_results = []
        for i, layer in enumerate(layers, 1):
            if args.verbose:
                print(f"\n[{i}/{len(layers)}]", end=" ")
            
            results = train_layer_encoder(
                args.model, args.dataset, args.neural_dataset, args.roi, layer,
                subject=args.subject,
                max_epochs=args.max_epochs,
                min_epochs=args.min_epochs,
                patience=args.patience,
                tolerance=args.tolerance,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                verbose=args.verbose
            )
            all_results.append(results)
            
            # Progressively append results to txt file
            with open(txt_file, 'a') as f:
                f.write(f"Layer: {results['layer']}\n")
                f.write(f"  Best Alpha: {results['best_alpha']:.1e}\n")
                f.write(f"  CV R² Score: {results['cv_score_mean']:.4f}\n")
                f.write(f"  Test R² Mean: {results['r2_mean']:.4f} ± {results['r2_std']:.4f}\n")
                f.write(f"  Test R² Range: [{results['r2_min']:.4f}, {results['r2_max']:.4f}]\n")
                f.write(f"  Test R² Median: {results['r2_median']:.4f}\n")
                f.write(f"  Test MSE Mean: {results['mse_mean']:.4f} ± {results['mse_std']:.4f}\n")
                f.write(f"  Units analyzed: {results['n_units']}\n")
                f.write(f"  X shape: {results['X_train_shape']} → {results['X_test_shape']}\n")
                f.write(f"  y shape: {results['y_train_shape']} → {results['y_test_shape']}\n")
                f.write(f"\n")
        
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
            # Save JSON
            with open(json_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # Append summary to txt file
            with open(txt_file, 'a') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"SUMMARY\n")
                f.write(f"{'='*70}\n")
                f.write(f"Layers processed: {len(all_results)}\n")
                f.write(f"Mean R² across layers: {np.mean([r['r2_mean'] for r in all_results]):.4f}\n")
                f.write(f"Best R² (layer mean): {max([r['r2_mean'] for r in all_results]):.4f}\n")
                f.write(f"Worst R² (layer mean): {min([r['r2_mean'] for r in all_results]):.4f}\n")
            
            if args.verbose:
                print(f"\nResults saved to:")
                print(f"  TXT: {txt_file}")
                print(f"  JSON: {json_file}")
        
        print("\n✓ Training completed successfully!")
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
