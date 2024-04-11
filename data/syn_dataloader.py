from typing import List, Union
import torch, os
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import sys
from torch.utils.data import DataLoader, TensorDataset, random_split
# fpath = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(os.path.join(fpath, '../../data/ml_mmrf'))
# sys.path.append(os.path.join(fpath, '../../data/'))
# print(sys.path)

from torch.nn.utils.rnn import pad_sequence


class SynDataModule2(pl.LightningDataModule):
    def __init__(self, hparams, dataset, seed=1):
        super().__init__()
        # self.hparams = hparams
        self.hparams.update(hparams)
        self.generator = torch.Generator().manual_seed(seed)
        self.dataset = dataset
        ddata = {}

        ddata['x'] = torch.from_numpy(self.dataset.x)
        ddata['trt'] = torch.from_numpy(self.dataset.u)

        ddata['gene'] = torch.from_numpy(self.dataset.g)

        ddata['z_gt'] = torch.from_numpy(np.argmax(self.dataset.z_cat, axis=-1)).to(torch.float32)

        ddata['v'] = torch.from_numpy(self.dataset.v)
        ddata['s'] = torch.from_numpy(self.dataset.s)

        mask = torch.ones_like(ddata['x'])
        ddata['m'] = mask

        # ddata['v'] = torch.from_numpy(self.dataset.v)

        self.hparams['dim_data'] = ddata['x'].shape[-1]
        self.hparams['dim_treat'] = ddata['trt'].shape[-1]
        self.hparams['n_ggroup'] = ddata['v'].shape[-1]

        self.ddata = ddata

        batch_size = self.hparams['bs']
        Z_gt = self.ddata['z_gt']
        if self.hparams["gpu"]:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        X = self.ddata['x'].to(torch.float32).to(device)
        Trt = self.ddata['trt'].to(torch.float32).to(device)
        Mask = self.ddata['m'].to(torch.float32).to(device)
        V_gene_cluster = self.ddata['v'].to(torch.float32).to(device)
        S_gene_info = self.ddata['s'].to(torch.float32).to(device)
        Genetic = self.ddata['gene'].to(torch.float32).to(device)
        Z_gt = Z_gt.to(device)

        data = TensorDataset(X, Trt, V_gene_cluster, S_gene_info, Genetic, Z_gt, Mask)
        proportions = [0.5, 0.1, 0.4]
        self.data_tr, self.data_test, self.data_val = random_split(data, [int(p * len(data)) for p in proportions], generator=self.generator)

    def pad_seq(self, X_observations):
        X_observations = [torch.from_numpy(i) for i in X_observations]
        return pad_sequence(X_observations, batch_first=True)

    def setup(self, stage):
        '''
        When adding a dataset (e.g. VA MM dataset), the data loading function should return
        the following structure:
            key: fold number, val: dictionary ==> {key: 'train', 'test', 'valid', \
                val: dictionary ==> {key: data type ('x','m'), val: np matrix}}
        See load_mmrf() function in data.py file in ml_mmrf folder.
        '''

        self.train_loader = DataLoader(self.data_tr, batch_size=self.hparams['bs'], shuffle=False)
        self.test_loader = DataLoader(self.data_test, batch_size=len(self.data_test), shuffle=False)
        self.valid_loader = DataLoader(self.data_val, batch_size=len(self.data_val), shuffle=False)


    def train_dataloader(self):

        return self.train_loader
    def test_dataloader(self):

        return self.test_loader
    def val_dataloader(self):
        
        
        return self.valid_loader
