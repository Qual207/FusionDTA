import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader
#from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import numpy as np
import pandas as pd
import logging
import json
from tqdm import tqdm
import utils.experiment_utils as experiment_utils

#DTIDataset class
class DTIDataset(Dataset):
    def __init__(self, df, seqlen=2000, smilen=200, descriptors=None):
        self.df = df
        self.proteins = df['proteins'].values
        self.ligands = df['ligands'].values
        self.affinity = df['affinity'].values
        self.descriptors = descriptors
        self.smilelen = smilen
        self.seqlen = seqlen

        #build vocab for protein and ligand characters
        self.protein_vocab = set()
        self.ligand_vocab = set()
        self.stereo_vocab = set()
        for prot in self.proteins:
            self.protein_vocab.update(list(prot))
        for lig in self.ligands:
            self.ligand_vocab.update(list(lig))
            self.stereo_vocab.update(experiment_utils.extract_stereo_tokens(lig))

        self.protein_dict = {tok: i+1 for i, tok in enumerate(self.protein_vocab)}
        self.ligand_dict = {tok: i+1 for i, tok in enumerate(self.ligand_vocab)}
        self.stereo_dict = {tok: i+1 for i, tok in enumerate(self.stereo_vocab)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pr = self.proteins[idx][:self.seqlen]
        lig = self.ligands[idx][:self.smilelen]
        target = self.affinity[idx]

        #token to id conversion
        protein_ids = [self.protein_dict[x] for x in pr] + [0]*(self.seqlen-len(pr))
        ligand_ids = [self.ligand_dict[x] for x in lig] + [0]*(self.smilelen-len(lig))

        stereo_tokens = experiment_utils.extract_stereo_tokens(lig)
        stereo_ids = [self.stereo_dict[x] for x in stereo_tokens] + [0]*(self.smilelen - len(stereo_tokens))

        if self.descriptors is not None:
            desc = self.descriptors[idx]
            return {
                'protein': torch.tensor(protein_ids, dtype=torch.long),
                'ligand': torch.tensor(ligand_ids, dtype=torch.long),
                'stereo': torch.tensor(stereo_ids, dtype=torch.long),
                'descriptor': torch.tensor(desc, dtype=torch.float),
                'affinity': torch.tensor(target, dtype=torch.float)
            }
        else:
            return {
                'protein': torch.tensor(protein_ids, dtype=torch.long),
                'ligand': torch.tensor(ligand_ids, dtype=torch.long),
                'stereo': torch.tensor(stereo_ids, dtype=torch.long),
                'affinity': torch.tensor(target, dtype=torch.float)
            }

#stacks into batch tensor (different for baseline and experiment models)
def collate_fn(batch, model_type='baseline'):
    if model_type == 'baseline':
        proteins = torch.stack([b['protein'] for b in batch])
        ligands  = torch.stack([b['ligand'] for b in batch])
        y        = torch.stack([b['affinity'] for b in batch])
        return proteins, ligands, y
    else:
        proteins = torch.stack([b['protein'] for b in batch])
        ligands  = torch.stack([b['ligand'] for b in batch])
        stereo   = torch.stack([b['stereo'] for b in batch])
        desc     = torch.stack([b['descriptor'] for b in batch])
        y        = torch.stack([b['affinity'] for b in batch])
        return proteins, ligands, stereo, desc, y