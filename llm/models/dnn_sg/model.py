import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

sys.path.append('.')
from model_utils import *


class DnnModel(nn.Module):

    def __init__(self, feature_configs_dict, nm_cols, hidden_size=60, Ts=0.1, device='cpu'):
        super(DnnModel, self).__init__()

        self.feature_configs_dict = feature_configs_dict
        self.nm_cols = nm_cols
        self.Ts = Ts

        self.normalization_layer = NormalizationLayer(feature_configs_dict, nm_cols, device, nm_method='meanstd')

        feature_size = len(nm_cols)
        self.dnn_network = nn.Sequential(nn.Linear(feature_size, hidden_size),
                                         nn.Tanh(),
                                         nn.Linear(hidden_size, hidden_size),
                                         nn.Tanh(),
                                         nn.Linear(hidden_size, hidden_size),
                                         nn.Tanh(),
                                         nn.Linear(hidden_size, 1))

        self.traj_id, self.frame_id = None, None

    def forward(self, X_nm_enc):
        X_nm_enc = self.normalization_layer(X_nm_enc)
        X = X_nm_enc
        vacc = self.dnn_network(X)
        return vacc