import torch.nn as nn

from sim_env import SimEnv
from utils import *

def IDM(params, vgap, vsp, vrelsp):
    v0, T0, s0, amax, b = params
    vss = s0 + max(0, vsp * T0 + vsp * vrelsp / (2 * np.sqrt(amax * b)))
    vacc = amax * (1 - (vsp / v0) ** 4 - (vss / vgap) ** 2)
    return vacc

def calc_error(sim_list, real_list, weight_list=None, collision_loc=None, error_type='rmse', agg_type='mean'):
    assert len(sim_list) == len(real_list), 'len(sim_list) == len(real_list)'
    if weight_list is None:
        weight_arr = np.ones(len(sim_list))
    else:
        weight_arr = np.array(weight_list)
    if error_type == 'mse':
        sim_error = np.sum(((np.array(sim_list) - np.array(real_list)) ** 2) * weight_arr)
        if agg_type == 'mean':
            sim_error /= sum(weight_arr)
    elif error_type == 'mae':
        sim_error = np.sum((np.abs(np.array(sim_list) - np.array(real_list))) * weight_arr)
        if agg_type == 'mean':
            sim_error /= sum(weight_arr)
    elif error_type == 'rmse':
        sim_error = np.sqrt(np.sum(((np.array(sim_list) - np.array(real_list)) ** 2) * weight_arr) / sum(weight_arr))
        if agg_type != 'mean':
            raise Exception('RMSE only support mean agg_type')
    else:
        raise Exception('Unknown error')
    return sim_error + sum(collision_loc) * 1000 if collision_loc is not None else sim_error

def get_X_nm_ct(model_data_dir, tp, traj_list):
    df_base = pd.read_csv(pathjoin(model_data_dir, 'df_base_%s.csv' % tp))
    df_nm = pd.read_csv(pathjoin(model_data_dir, 'df_nm_%s.csv' % tp))
    df_ct = pd.read_csv(pathjoin(model_data_dir, 'df_ct_%s.csv' % tp))
    X_nm_dict, X_ct_dict = {}, {}
    for traj_id in traj_list:
        traj_index = np.where(df_base.traj_id == traj_id)[0]
        X_nm_dict[traj_id] = df_nm.iloc[traj_index].values
        X_ct_dict[traj_id] = df_ct.iloc[traj_index].values
    return X_nm_dict, X_ct_dict

def traj_simulate(traj_id, model, simulate_func, X_nm_dict, cf_pairs_df, enc_len, dec_len, device, style_index=None, Ts=0.1, seed=1, **kwargs):
    traj_id = int(traj_id)
    np.random.seed(traj_id+seed)

    cf_pairs_df_sub = cf_pairs_df[cf_pairs_df.traj_id == traj_id]
    cf_pairs_df_dec = cf_pairs_df_sub[enc_len-1:]
    vgap = cf_pairs_df_dec.vgap.values

    env = SimEnv(traj_id=traj_id, Ts=Ts, verbose=False)
    eval('env.%s' % simulate_func)(model, style_index, cf_pairs_df_sub, X_nm_dict[traj_id], enc_len, dec_len, device, **kwargs)
    vgap_error = calc_error(env.vgap_sim, vgap, collision_loc=None, error_type='rmse')

    traj_sim_dict_sub = {
        'frame_list': cf_pairs_df_dec.frame_id.tolist(),
        'vgap_list': cf_pairs_df_dec.vgap.tolist(),
        'vsp_list': cf_pairs_df_dec.vsp.tolist(),
        'vlocy_list': cf_pairs_df_dec.vlocy.tolist(),
        'vacc_list': cf_pairs_df_dec.vacc.tolist(),
        'lvsp_list': cf_pairs_df_dec.lvsp.tolist(),
        'lvlocy_list': cf_pairs_df_dec.lvlocy.tolist(),
        'vgap_sim': env.vgap_sim,
        'vsp_sim': env.vsp_sim,
        'vlocy_sim': env.vlocy_sim,
        'vacc_sim': env.vacc_sim,
        'collision_loc': env.collision_loc,
        'vgap_error': vgap_error
    }

    return [traj_id, traj_sim_dict_sub]

class NormalizationLayer(nn.Module):
    def __init__(self, feature_configs_dict, nm_cols, device, nm_method='meanstd'):
        super(NormalizationLayer, self).__init__()
        self.mean_, self.std_ = torch.zeros(len(nm_cols), device=device), torch.ones(len(nm_cols), device=device)
        self.min_val, self.max_val = torch.zeros(len(nm_cols), device=device), torch.ones(len(nm_cols), device=device)
        self.nm_method = nm_method
        for i, col in enumerate(nm_cols):
            if col not in feature_configs_dict:
                continue
            if nm_method == 'meanstd':
                self.mean_[i] = feature_configs_dict[col]['mean_']
                self.std_[i] = feature_configs_dict[col]['std_']
            elif nm_method == 'maxmin':
                self.max_val[i] = feature_configs_dict[col]['max_val']
                self.min_val[i] = feature_configs_dict[col]['min_val']
            elif nm_method == 'raw':
                pass
            else:
                raise Exception('No such nm_method %s' % nm_method)

    def forward(self, inputs):
        if self.nm_method == 'meanstd':
            return (inputs - self.mean_) / self.std_
        elif self.nm_method == 'maxmin':
            return (inputs - self.min_val) / (self.max_val - self.min_val)
        else:
            return inputs


class CategorizationLayer(nn.Module):
    def __init__(self, feature_configs_dict, ct_cols, device):
        super(CategorizationLayer, self).__init__()
        self.feature_configs_dict = feature_configs_dict
        self.device = device
        self.bins = []
        for col in ct_cols:
            self.bins.append(feature_configs_dict[col].get('bins', []))

    def forward(self, inputs):
        inputs_bin = []
        for i in range(inputs.shape[-1]):
            if len(self.bins[i]) > 0:
                bins = torch.tensor(self.bins[i], device=self.device)
                inputs_i_bin = torch.bucketize(inputs[..., i:i+1].contiguous(), bins)
            else:
                inputs_i_bin = inputs[..., i:i+1]
            inputs_bin.append(inputs_i_bin)
        inputs_bin = torch.concat(inputs_bin, dim=-1)
        return inputs_bin

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, activation=nn.Tanh, bias=True):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size, bias)
        self.activation = activation()
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias)

    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        out += identity
        out = self.activation(out)
        return out