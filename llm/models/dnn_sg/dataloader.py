import torch
from torch.utils.data import Dataset
import numpy as np

from utils import *


class DnnDataset(Dataset):

    def __init__(self, data_dir, seed=10, tp='train', suffix=''):
        sample_df = pd.read_csv(pathjoin(data_dir, 'samples_df_%s_%s.csv' % (tp, suffix)))
        sample_df = sample_df.groupby('sid').apply(lambda x: x[x.vacc_pred == x.vacc_pred.mode().iloc[0]].drop(['sid'], axis=1).iloc[0]).reset_index()
        self.enc_nm_inputs, self.vacc = [], []
        for i, row in sample_df.iterrows():
            self.enc_nm_inputs.append([row.vsp, row.vgap, row.vrelsp])
            self.vacc.append(row.vacc_pred)
        self.enc_nm_inputs = np.stack(self.enc_nm_inputs)
        self.vacc = np.array(self.vacc)

    def __len__(self):
        return len(self.enc_nm_inputs)

    def __getitem__(self, item):
        return (
                    item,
                    self.enc_nm_inputs[item],
                    self.vacc[item],
        )