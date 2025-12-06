import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pandas as pd
import math

import model.baseline_net as baseline_net
import model.stereo_net as stereo_net
import model.data_loader as data_loader
import utils.base_utils as utils
import utils.experiment_utils as experiment_utils

from evaluate import evaluate 

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='data/processed/davis.csv', 
                    help="Path to dataset CSV file")
parser.add_argument('--model_dir', default='experiments/experiment_model', 
                    help="Directory containing the best_model.pt and test_indices.npy")
parser.add_argument('--model_type', choices=['baseline', 'experiment'], default='experiment',
                    help="Which model to test")

args = parser.parse_args()


if __name__ == "__main__":
    if args.model_type == 'baseline':
        args.model_dir = 'experiments/base_model'
    else:
        args.model_dir = 'experiments/experiment_model'
    json_path = os.path.join(args.model_dir, 'params.json')
    
    assert os.path.isfile(json_path), f"No json configuration file found at {json_path}. Point model_dir to the experiment folder."
    
    params = utils.Params(json_path)
    params.cuda = torch.cuda.is_available()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.info(f"Testing model: {args.model_type}")

    df = pd.read_csv(args.data_path)
    df = df.dropna(subset=['affinity']).reset_index(drop=True)
    df['affinity'] = -np.log10(df['affinity'] / 1e9)

    descriptors = None
    if args.model_type == 'experiment':
        descriptors = np.array([experiment_utils.featurize_molecule(smi) for smi in df['ligands'].values])

    test_idx_path = os.path.join(args.model_dir, 'test_indices.npy')
    try:
        test_idx = np.load(test_idx_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Test indices not found at {test_idx_path}. Did you run train.py first?")
        
    logging.info(f"Test Set Size (80/10/10 Split): {len(test_idx)} samples")

    dataset = data_loader.DTIDataset(df, descriptors=descriptors)
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=params.batch_size,
        shuffle=False, 
        collate_fn=lambda b: data_loader.collate_fn(b, model_type=args.model_type)
    )

    if args.model_type == 'baseline':
        model = baseline_net.DeepDTA(
            len(dataset.protein_vocab)+1, 
            len(dataset.ligand_vocab)+1,
            channel=params.channel,
            protein_kernel_size=params.protein_kernel_size,
            ligand_kernel_size=params.ligand_kernel_size
        )
    else:
        model = stereo_net.StereoDTA(
            pro_vocab_size = len(dataset.protein_vocab) + 1,
            lig_vocab_size = len(dataset.ligand_vocab) + 1,
            stereo_vocab_size = len(dataset.stereo_vocab) + 1,
            descriptor_size = 200 
        )

    if params.cuda:
        model = model.cuda()

    weights_path = os.path.join(args.model_dir, "best_model.pt")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Could not find best_model.pt at {weights_path}. Check if model_dir points to the experiment directory.")
        
    logging.info(f"Loading checkpoint from: {weights_path}")

    try:
        checkpoint = torch.load(weights_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logging.info("Model weights loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model weights: {e}")
        exit(1) 

    loss_fn = nn.MSELoss()

    test_metrics, y_true, y_pred = evaluate(model, loss_fn, test_loader, params, model_type=args.model_type)
    
    test_loss = test_metrics['loss']
    
    print("\n" + "="*35)
    print(f"TEST RESULTS ({args.model_type.upper()})")
    print("="*35)
    print(f"Test Loss (MSE): {test_loss:.5f}")
    print(f"RMSE             : {test_metrics['rmse']:.5f}")
    print(f"CI               : {test_metrics['ci']:.5f}") 
    print("="*35 + "\n")
    
    np.save(os.path.join(args.model_dir, "y_true.npy"), y_true)
    np.save(os.path.join(args.model_dir, "y_pred.npy"), y_pred)

    metrics_to_save = {k: [v] for k, v in test_metrics.items() if k in ['mse', 'rmse', 'ci']} 
    metrics_df = pd.DataFrame(metrics_to_save)
    metrics_df.to_csv(os.path.join(args.model_dir, "metrics.csv"), index=False)
    
    logging.info(f"Test metrics, true, and pred saved to {args.model_dir}")
