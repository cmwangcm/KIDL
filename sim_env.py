import math

import numpy as np
import torch

from utils import *


class SimEnv(object):
    def __init__(self, traj_id, Ts, verbose=True):
        self.traj_id = traj_id
        self.Ts = Ts
        self.frame_list = []
        self.vgap_sim, self.vsp_sim, self.vrelsp_sim, self.vacc_sim, self.vlocy_sim = [], [], [], [], []
        self.collision_loc = []
        self.verbose = verbose

        self.model_params = {}

    def reset(self):
        self.vgap_sim, self.vsp_sim, self.vrelsp_sim, self.vacc_sim, self.vlocy_sim, self.collision_loc = [], [], [], [], [], []

    def simulate_DNN(self, model_func, style_index, cf_pairs_df_sub, X_nm, enc_len, dec_len, device, temperature=0):
        n = len(cf_pairs_df_sub)
        X_num_i = torch.tensor(X_nm[:1]).float().to(device)

        row = cf_pairs_df_sub.iloc[0]
        vsp, vgap, vrelsp, vacc, vlocy = row.vsp, row.vgap, row.vrelsp, row.vacc, row.vlocy

        model_func.eval()
        with torch.no_grad():
            for i in range(n):
                vacc = model_func(X_num_i)
                vacc = vacc.numpy()[0, 0]
                vacc = np.random.randn() * temperature + vacc
                self.frame_list.append(row.frame_id)
                self.vgap_sim.append(vgap)
                self.vsp_sim.append(vsp)
                self.vrelsp_sim.append(vrelsp)
                self.vacc_sim.append(vacc)
                self.vlocy_sim.append(vlocy)
                if i >= n - 1:
                    break
                row = cf_pairs_df_sub.iloc[i+1]
                vsp_ = max(0, vsp + vacc * self.Ts)
                vrelsp_ = vsp_ - row.lvsp
                vlocy_ = vlocy + self.Ts * (vsp_ + vsp) / 2
                vgap_ = row.lvlocy - vlocy_ - row.lvlen
                vgap, vsp, vrelsp, vlocy = vgap_, vsp_, vrelsp_, vlocy_
                collision_loc = 0
                if vgap <= 0:
                    collision_loc = 1
                self.collision_loc.append(collision_loc)

                X_num_i_pred = np.array([vsp, vgap, vrelsp]).reshape((1, -1))
                X_num_i = torch.tensor(X_num_i_pred, dtype=torch.float32)

            if self.verbose:
                if max(self.collision_loc) == 1:
                    print('traj_id: %s   Collision' % self.traj_id)
                else:
                    print('traj_id: %s   Finish' % self.traj_id)