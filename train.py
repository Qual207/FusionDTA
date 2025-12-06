import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import json

import model.baseline_net as baseline_net
import model.stereo_net as stereo_net
import model.data_loader as data_loader
import utils.base_utils as utils
import utils.experiment_utils as experiment_utils
from evaluate import evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='data/processed/davis.csv',
                    help="Path to dataset CSV file")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json and saved weights")
parser.add_argument('--model_type', choices=['baseline', 'experiment'], default='baseline',
                    help="Which model to train: baseline (DeepDTA) or experiment (Stereo+Descriptors)")

args = parser.parse_args()

if args.model_type == 'baseline':
    args.model_dir = 'experiments/base_model'
else:
    args.model_dir = 'experiments/experiment_model'

model_dir = args.model_dir

os.makedirs(model_dir, exist_ok=True)

#Single-epoch train function. Handles both baseline and experimental models. 
#Performs forward pass, computes loss, backpropagation, optimizer step, and keeps a running average of the loss. 
#Uses tqdm for progress display.
def train(model, optimizer, loss_fn, dataloader, params, model_type='baseline'):
    model.train()
    loss_avg = utils.RunningAverage()
    
    with tqdm(total=len(dataloader)) as t:
        for batch in dataloader:
            if model_type == 'baseline':
                pr, lig, y = batch
                if params.cuda:
                    pr, lig, y = pr.cuda(), lig.cuda(), y.cuda()
                pred = model(pr, lig)
            else:
                pr, lig, stereo, desc, y = batch
                if params.cuda:
                    pr, lig, stereo, desc, y = pr.cuda(), lig.cuda(), stereo.cuda(), desc.cuda(), y.cuda()
                pred = model(pr, lig, stereo, desc)

            optimizer.zero_grad()
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            loss_avg.update(loss.item())
            t.update(1)
    
    return loss_avg()

#Train across multiple epochs
def train_and_evaluate(model, train_loader, val_loader, optimizer, loss_fn, params, model_dir, model_type='baseline'):
    best_val_loss = float('inf')

    for epoch in range(params.num_epochs):
        logging.info(f"Epoch {epoch+1}/{params.num_epochs}")
        train_loss = train(model, optimizer, loss_fn, train_loader, params, model_type=args.model_type)
        

        val_metrics, _, _ = evaluate(model, loss_fn, val_loader, params, model_type=model_type)

        val_loss = val_metrics['loss']
        is_best = val_loss <= best_val_loss

        if is_best:
            logging.info(f"- Found new best model (val_loss={val_loss:.4f})")
            best_val_loss = val_loss
            
            #save BEST model
            utils.save_checkpoint(model, optimizer, os.path.join(model_dir, "best_model.pt"))
            utils.save_dict_to_json(val_metrics, os.path.join(model_dir, "metrics_val_best.json"))

        #save last epoch if needed
        # utils.save_dict_to_json(val_metrics, os.path.join(model_dir, "metrics_val_last.json"))

    utils.save_checkpoint(model, optimizer, os.path.join(model_dir, "last_model.pt"))

    logging.info("Training complete!")


if __name__ == "__main__":
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), f"No json configuration file found at {json_path}"
    params = utils.Params(json_path)
    params.cuda = torch.cuda.is_available()

    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    #Load and preprocess dataset
    logging.info("Loading datasets...")
    df = pd.read_csv(args.data_path)
    df = df.dropna(subset=['affinity']).reset_index(drop=True)
    df['affinity'] = -np.log10(df['affinity'] / 1e9) 

    descriptors = None
    if args.model_type == 'experiment':
        logging.info("Generating descriptors")
        descriptors = np.array([experiment_utils.featurize_molecule(smi) for smi in df['ligands'].values])

    dataset = data_loader.DTIDataset(df, descriptors=descriptors)

    logging.info("Data splitting")
    unique_ligands = df['ligands'].unique()

    train_val_ligands, test_ligands, _, _ = train_test_split(
        unique_ligands, unique_ligands, test_size=0.1, random_state=230, shuffle=True
    )

    train_ligands, val_ligands, _, _ = train_test_split(
        train_val_ligands, train_val_ligands, test_size=(1/9), random_state=230, shuffle=True
    )

    train_idx = df[df['ligands'].isin(train_ligands)].index.values
    val_idx = df[df['ligands'].isin(val_ligands)].index.values
    test_idx = df[df['ligands'].isin(test_ligands)].index.values

    np.save(os.path.join(args.model_dir, 'test_indices.npy'), test_idx)
    
    #MODEL INSTANTIATION
    if args.model_type == 'baseline':
        model = baseline_net.DeepDTA(
            len(dataset.protein_vocab)+1, len(dataset.ligand_vocab)+1,
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
    
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=params.batch_size,
        shuffle=True,
        collate_fn=lambda b: data_loader.collate_fn(b, model_type=args.model_type)
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=params.batch_size,
        collate_fn=lambda b: data_loader.collate_fn(b, model_type=args.model_type)
    )

    logging.info(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test_idx)}")
    

    train_and_evaluate(model, train_loader, val_loader, optimizer, loss_fn, params, args.model_dir, args.model_type)
    
    logging.info("\nFinished Training")
    logging.info(f"Test indices saved to {os.path.join(args.model_dir, 'test_indices.npy')}.")