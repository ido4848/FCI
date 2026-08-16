"""
Calculate Functional Complexity Index (FCI) from a trained TCN model.

The FCI is calculated from the AUC (Area Under the ROC Curve) of the trained model:
    FCI = log10(1000 * (1 - AUC)) / log10(1000 * (1 - 0.9))

Where:
- AUC = 0.9 corresponds to FCI = 1.0 (baseline complexity)
- AUC = 0.999 corresponds to FCI ≈ 0.0 (near-perfect prediction, low complexity)

Usage:
    python calculate_fci.py --neuron_tcn_folder <path_to_trained_tcn>
"""

import os
import sys
import glob
import pickle
import argparse
import numpy as np


def auc_to_fci(auc, minimal_auc=0.9, maximal_auc=0.999, mult=1000):
    """
    Convert AUC to Functional Complexity Index (FCI).
    
    Parameters
    ----------
    auc : float
        Area Under the ROC Curve (between 0.5 and 1.0)
    minimal_auc : float
        AUC value corresponding to FCI = 1.0 (default: 0.9)
    maximal_auc : float
        AUC value corresponding to FCI ≈ 0.0 (default: 0.999)
    mult : float
        Multiplier for the logarithmic transformation (default: 1000)
    
    Returns
    -------
    float
        Functional Complexity Index (higher = more complex)
    """
    # Clip AUC to valid range
    auc = np.clip(auc, minimal_auc, maximal_auc)
    
    fci = np.log10(mult * (1 - auc)) / np.log10(mult * (1 - minimal_auc))
    return fci


def load_checkpoint(checkpoint_path):
    """
    Load a checkpoint written by train_neuron_tcn.py.

    The light checkpoints (`model_<epoch>_<step>_light.pkl`) are plain pickles, while the full
    ones (`model_<epoch>_<step>`) are written with torch.save.
    """
    try:
        return pickle.load(open(checkpoint_path, 'rb'))
    except pickle.UnpicklingError:
        import torch
        return torch.load(checkpoint_path, map_location='cpu')


def load_best_auc_from_tcn_folder(neuron_tcn_folder):
    """
    Load the best AUC from a trained TCN folder.

    Parameters
    ----------
    neuron_tcn_folder : str
        Path to the trained TCN folder

    Returns
    -------
    dict
        Dictionary containing AUC values and other metrics
    """
    models_folder = os.path.join(neuron_tcn_folder, 'models')

    # the light checkpoints hold the same best_save as the full ones, at a fraction of the size
    # and without needing torch, so prefer them
    model_files = glob.glob(os.path.join(models_folder, 'model_*_light.pkl'))
    if not model_files:
        model_files = [f for f in glob.glob(os.path.join(models_folder, 'model_*_*'))
                       if not f.endswith('.pkl')]

    if not model_files:
        raise FileNotFoundError(f"No model checkpoints found in {models_folder}")

    # Sort by step number (second number in filename)
    def get_step(f):
        try:
            return int(os.path.basename(f).split('_')[2])
        except:
            return 0

    model_files.sort(key=get_step)
    latest_model = model_files[-1]

    print(f"Loading checkpoint from: {latest_model}")
    checkpoint = load_checkpoint(latest_model)

    best_save = checkpoint.get('best_save', {})

    return best_save


def load_results_from_pkl(results_pkl_path):
    """
    Load results directly from a results pickle file.
    
    Parameters
    ----------
    results_pkl_path : str
        Path to a results pickle file (e.g., model_X_Y_test_results.pkl)
    
    Returns
    -------
    dict
        Dictionary containing AUC and other metrics
    """
    return pickle.load(open(results_pkl_path, 'rb'))


def main():
    parser = argparse.ArgumentParser(
        description='Calculate Functional Complexity Index (FCI) from a trained TCN model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Calculate FCI from a trained TCN folder
    python calculate_fci.py --neuron_tcn_folder /path/to/neuron_tcn
    
    # Calculate FCI from a specific results file
    python calculate_fci.py --results_pkl /path/to/model_X_Y_test_results.pkl
    
    # Calculate FCI from a specific AUC value
    python calculate_fci.py --auc 0.95
        """
    )
    
    parser.add_argument('--neuron_tcn_folder', type=str, default=None,
                        help='Path to the trained TCN folder')
    parser.add_argument('--results_pkl', type=str, default=None,
                        help='Path to a results pickle file')
    parser.add_argument('--auc', type=float, default=None,
                        help='Directly specify an AUC value to convert to FCI')
    parser.add_argument('--use_normalized', action='store_true',
                        help='Use normalized AUC instead of regular AUC')
    
    args = parser.parse_args()
    
    if args.auc is not None:
        # Direct AUC to FCI conversion
        auc = args.auc
        fci = auc_to_fci(auc)
        print(f"\nAUC: {auc:.6f}")
        print(f"FCI: {fci:.6f}")
        return {'auc': auc, 'fci': fci}
    
    elif args.results_pkl is not None:
        # Load from results pickle file
        results = load_results_from_pkl(args.results_pkl)
        auc = results.get('auc', None)
        if auc is None:
            raise ValueError("No 'auc' key found in results file")
        
    elif args.neuron_tcn_folder is not None:
        # Load from TCN folder
        best_save = load_best_auc_from_tcn_folder(args.neuron_tcn_folder)
        
        if args.use_normalized:
            auc = best_save.get('normalized_auc', None)
            auc_type = 'normalized_auc'
        else:
            auc = best_save.get('auc', None)
            auc_type = 'auc'
        
        if auc is None or auc < 0:
            raise ValueError(f"No valid '{auc_type}' found in model checkpoint")
        
        # Print additional metrics
        print("\n" + "="*50)
        print("Best Model Metrics")
        print("="*50)
        print(f"Epoch: {best_save.get('ep', 'N/A')}")
        print(f"Steps: {best_save.get('steps', 'N/A')}")
        print(f"Training time: {best_save.get('total_training_time_in_seconds', 0)/3600:.2f} hours")
        print("-"*50)
        print(f"AUC: {best_save.get('auc', 'N/A'):.6f}" if best_save.get('auc') else "AUC: N/A")
        print(f"Normalized AUC: {best_save.get('normalized_auc', 'N/A'):.6f}" if best_save.get('normalized_auc') else "Normalized AUC: N/A")
        print(f"PRC-AUC: {best_save.get('prc_auc', 'N/A'):.6f}" if best_save.get('prc_auc') else "PRC-AUC: N/A")
        print(f"Somatic Voltage Explained Variance: {best_save.get('somatic_voltage_explained_variance', 'N/A'):.2f}%" if best_save.get('somatic_voltage_explained_variance') else "Somatic Voltage Explained Variance: N/A")
        print(f"Soma RMSE: {best_save.get('soma_RMSE', 'N/A'):.4f}" if best_save.get('soma_RMSE') else "Soma RMSE: N/A")
    else:
        parser.print_help()
        sys.exit(1)
    
    # Calculate FCI
    fci = auc_to_fci(auc)
    
    print("-"*50)
    print(f"Using {'Normalized ' if args.use_normalized else ''}AUC: {auc:.6f}")
    print(f"\n>>> Functional Complexity Index (FCI): {fci:.6f} <<<")
    print("="*50)
    
    # Interpretation
    print("\nInterpretation:")
    if fci >= 0.8:
        print("  High functional complexity - the neuron's input-output")
        print("  relationship is difficult to predict.")
    elif fci >= 0.5:
        print("  Moderate functional complexity.")
    else:
        print("  Low functional complexity - the neuron's input-output")
        print("  relationship is relatively easy to predict.")
    
    return {'auc': auc, 'fci': fci}


if __name__ == "__main__":
    main()
