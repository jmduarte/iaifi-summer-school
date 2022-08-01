import os.path as osp
import torch
from torch_geometric.data import Dataset, Data
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


class LorentzGraphDataset(Dataset):
    def __init__(
        self,
        root,
        features,
        labels,
        spectators,
        transform=None,
        pre_transform=None,
        start_event=0,
        stop_event=1000,
        n_events_merge=1000,
        file_names=None,
        remove_unlabeled=True,
    ):
        """
        Initialize parameters of graph dataset
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
        self.n_events = stop_event - start_event
        self.start_event = start_event
        self.stop_event = stop_event
        self.n_events_merge = n_events_merge
        self.file_names = file_names
        self.remove_unlabeled = remove_unlabeled
        super(LorentzGraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """
        Determines which file is being processed
        """
        if self.file_names is None:
            return [
                "root://eospublic.cern.ch//eos/opendata/cms/datascience/HiggsToBBNtupleProducerTool/HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC/train/ntuple_merged_10.root"
            ]
        else:
            return self.file_names

    @property
    def processed_file_names(self):
        """
        Returns a list of all the files in the processed files directory
        """
        proc_list = glob.glob(osp.join(self.processed_dir, "data*.pt"))
        return_list = list(map(osp.basename, proc_list))
        return return_list

    def len(self):
        return len(self.processed_file_names)

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        """
        Handles conversion of dataset file at raw_path into graph dataset.

        Args:
            raw_path (str): The absolute path to the dataset file
            k (int): Number of process (0,...,max_events // n_proc) to determine where to read file
        """
        for raw_path in self.raw_file_names:
            with uproot.open(raw_path, **get_file_handler(raw_path)) as root_file:

                tree = root_file["deepntuplizer/tree"]

                feature_array = tree.arrays(
                    self.features,
                    entry_start=self.start_event,
                    entry_stop=self.stop_event,
                    library="ak",
                )

                label_array_all = tree.arrays(
                    self.labels,
                    entry_start=self.start_event,
                    entry_stop=self.stop_event,
                    library="np",
                )

                spec_array = tree.arrays(
                    self.spectators,
                    entry_start=self.start_event,
                    entry_stop=self.stop_event,
                    library="np",
                )

            n_samples = label_array_all[self.labels[0]].shape[0]
            y = np.zeros((n_samples, 2))
            y[:, 0] = label_array_all["sample_isQCD"] * (
                label_array_all["label_QCD_b"]
                + label_array_all["label_QCD_bb"]
                + label_array_all["label_QCD_c"]
                + label_array_all["label_QCD_cc"]
                + label_array_all["label_QCD_others"]
            )
            y[:, 1] = label_array_all["label_H_bb"]

            z = np.stack([spec_array[spec] for spec in self.spectators], axis=1)

            for i in tqdm(range(n_samples)):
                if i % self.n_events_merge == 0:
                    datas = []
                if self.remove_unlabeled:
                    if np.sum(y[i : i + 1], axis=1) == 0:
                        continue
                n_particles = len(feature_array[self.features[0]][i])
                if n_particles < 2:
                    continue
                pt = feature_array["track_pt"][i].to_numpy()
                mass = feature_array["track_mass"][i].to_numpy()
                eta = (
                    feature_array["track_etarel"][i].to_numpy()
                    + spec_array["fj_eta"][i]
                )
                phi = (
                    feature_array["track_phirel"][i].to_numpy()
                    + spec_array["fj_phi"][i]
                )
                energy = np.sqrt(pt * pt * np.cosh(eta) * np.cosh(eta) + mass * mass)
                px = pt * np.cos(phi)
                py = pt * np.sin(phi)
                pz = pt * np.sinh(eta)
                pairs = np.stack(
                    [
                        [m, n]
                        for (m, n) in itertools.product(
                            range(n_particles), range(n_particles)
                        )
                        if m != n
                    ]
                )
                edge_index = torch.tensor(pairs, dtype=torch.long)
                edge_index = edge_index.t().contiguous()
                # x = torch.tensor([feature_array[feat][i].to_numpy() for feat in self.features], dtype=torch.float).T
                fourvec = [energy, px, py, pz]
                x = torch.tensor(fourvec, dtype=torch.float).T
                u = torch.tensor(z[i], dtype=torch.float)
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    y=torch.tensor(y[i : i + 1], dtype=torch.long),
                )
                data.u = torch.unsqueeze(u, 0)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                datas.append([data])

                if i % self.n_events_merge == self.n_events_merge - 1:
                    datas = sum(datas, [])
                    torch.save(
                        datas, osp.join(self.processed_dir, "data_{}.pt".format(i))
                    )

    def get(self, idx):
        p = osp.join(self.processed_dir, self.processed_file_names[idx])
        data = torch.load(p)
        return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset path", required=True)
    parser.add_argument(
        "--n-events", type=int, default=-1, help="number of events (-1 means all)"
    )
    parser.add_argument(
        "--n-events-merge", type=int, default=1000, help="number of events to merge"
    )
    args = parser.parse_args()

    with open("definitions.yml") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        definitions = yaml.load(file, Loader=yaml.FullLoader)

    features = definitions["features"]
    spectators = definitions["spectators"]
    labels = definitions["labels"]

    gdata = GraphDataset(
        args.dataset,
        features,
        labels,
        spectators,
        start_event=args.start_event,
        stop_event=args.stop_event,
        n_events_merge=args.n_events_merge,
    )
