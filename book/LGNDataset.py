import os.path as osp
import torch
from torch.utils.data import Dataset, TensorDataset
from torch.nn import ConstantPad2d
import itertools
import numpy as np
import uproot
import glob
import multiprocessing
from pathlib import Path
import yaml
from tqdm.notebook import tqdm
import awkward as ak
from utils import get_file_handler

import os
from itertools import islice
from math import inf

import logging

class LGNDataset():
    def __init__(self, features, labels, spectators,
                 n_events=-1, file_names=None, npad = 0, remove_unlabeled=True):
        """
        Initialize parameters of LGN dataset
        Args:
            root (str): path
            n_events (int): how many events to process (-1=all)
            n_events_merge (int): how many events to merge
            file_names (list of strings): file names
            remove_unlabeled (boolean): remove unlabeled data samples
        """
        self.features = features
        self.labels = labels
        self.spectators = spectators
        self.n_events = n_events
        self.file_names = file_names
        self.npad = npad
        self.remove_unlabeled = remove_unlabeled
        self.datas = {'x':[], # list of 4-momenta for each event
                      'y':[] # targets
                      }

    @property
    def raw_file_names(self):
        """
        Determines which file is being processed
        """
        if self.file_names is None:
            return ['root://eospublic.cern.ch//eos/opendata/cms/datascience/HiggsToBBNtupleProducerTool/HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC/train/ntuple_merged_10.root']
        else:
            return self.file_names


    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        """
        Handles conversion of dataset file at raw_path into Deep Sets dataset.

        Args:
            raw_path (str): The absolute path to the dataset file
            k (int): Number of process (0,...,max_events // n_proc) to determine where to read file
        """
        for raw_path in self.raw_file_names:
            with uproot.open(raw_path, **get_file_handler(raw_path)) as root_file:

                tree = root_file['deepntuplizer/tree']

                feature_array = tree.arrays(self.features,
                                            entry_stop=self.n_events,
                                            library='ak')

                label_array_all = tree.arrays(self.labels,
                                              entry_stop=self.n_events,
                                              library='np')

                spec_array = tree.arrays(self.spectators,
                                         entry_stop=self.n_events,
                                         library='np')

            n_samples = label_array_all[self.labels[0]].shape[0]
            y = np.zeros((n_samples, 2))
            y[:, 0] = label_array_all['sample_isQCD'] * (label_array_all['label_QCD_b'] +
                                                         label_array_all['label_QCD_bb'] +
                                                         label_array_all['label_QCD_c'] +
                                                         label_array_all['label_QCD_cc'] +
                                                         label_array_all['label_QCD_others'])
            y[:, 1] = label_array_all['label_H_bb']

            z = np.stack([spec_array[spec] for spec in self.spectators], axis=1)
            
            for i in tqdm(range(n_samples)):
                if self.remove_unlabeled:
                    if np.sum(y[i:i+1], axis=1) == 0:
                        continue
                pt = feature_array['track_pt'][i].to_numpy()
                mass = feature_array['track_mass'][i].to_numpy()
                eta = feature_array['track_etarel'][i].to_numpy()+spec_array['fj_eta'][i]
                phi = feature_array['track_phirel'][i].to_numpy()+spec_array['fj_phi'][i]
                energy = np.sqrt(pt*pt*np.cosh(eta)*np.cosh(eta) + mass*mass)
                px = pt*np.cos(phi)
                py = pt*np.sin(phi)
                pz = pt*np.sinh(eta)
                fourvec = [energy,px,py,pz]
                x = torch.tensor(fourvec, dtype=torch.float).T
                if x.size(dim=0)<self.npad:
                    x = ConstantPad2d((0,0,0,self.npad-x.size(dim=0)), 0.)(x)
                else:
                    x = x[:self.npad,:]
                x = x[None, :]
                self.datas['x'].append(x)
                self.datas['y'].append(torch.tensor(y[i:i+1], dtype=torch.long))


    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.datas[idx]

class PmuDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        
        return {'Pmu': x, 'output': y}
    
    def __len__(self):
        return len(self.x)
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset path", required=True)
    parser.add_argument("--n-events", type=int, default=-1, help="number of events (-1 means all)")
    args = parser.parse_args()

    with open('definitions.yml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        definitions = yaml.load(file, Loader=yaml.FullLoader)

    features = definitions['features']
    spectators = definitions['spectators']
    labels = definitions['labels']

    dsdata = DeepSetsDataset(args.dataset, features, labels, spectators,
                         n_events=args.n_events)
