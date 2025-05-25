import argparse
import sys
import os
import pickle
import numpy as np
import pandas as pd
from time import time
from collections import OrderedDict
from sklearn.metrics import median_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from multiprocessing import Pool
from functools import partial
import warnings
warnings.filterwarnings("ignore")

from model import *
from utils import *
from data_utils import DATASET_DICT
from model_utils import *
from dataloader import *
from sim_env import *
from data_utils import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def locate_eq_states(model):
    inputs_e, vgap_e_dict = [], {}
    for i, vsp_e in enumerate(range(1, 41)):
        vgap_e_list, flag = [], False
        for vgap in np.linspace(1, 100, 1000):
            vacc = model(torch.tensor([[vsp_e, vgap, 0.]], dtype=torch.float32, requires_grad=False)).detach().numpy()[0, 0]
            if abs(vacc) < 1e-2:
                flag = True
                vgap_e_list.append(vgap)
                break
        if not flag:
            continue
        vgap_e_dict[vsp_e] = vgap_e_list
        vgap_e = vgap_e_list[-1]
        inputs_e.append([vsp_e, vgap_e, 0.])
    return inputs_e, vgap_e_dict

def calc_string_stability(model, device='cpu'):
    if STB_FACTOR == 0.:
        return None
    inputs_e, vgap_e_dict = locate_eq_states(model)
    if len(inputs_e) < 1:
        return None
    input_tensor = torch.tensor(inputs_e, dtype=torch.float32, requires_grad=True).to(device)
    output = model(input_tensor)
    input_gradients = torch.autograd.grad(outputs=output, inputs=input_tensor,
                                          grad_outputs=torch.ones_like(output),
                                          create_graph=True, retain_graph=True)[0]
    string_stb_list = input_gradients[:, 0] ** 2 - 2 * input_gradients[:, 1] + 2 * input_gradients[:, 0] * input_gradients[:, 2]
    return string_stb_list

def train_epoch(model, data_loader, optimizer, loss_func, device):
    model.train()

    epoch_loss, total_num = 0, 0
    for i, (_, X_nm_enc, y_enc) in enumerate(data_loader):
        optimizer.zero_grad()

        X_nm_enc = X_nm_enc.float().to(device)
        y_enc = y_enc.float().to(device)

        X_nm_enc.requires_grad_(True)
        ypred = model(X_nm_enc)
        string_stb_list = calc_string_stability(model, device)
        loss = loss_func(ypred, y_enc.unsqueeze(-1), X_nm_enc, string_stb_list)
        cnt = ypred.size(0)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * cnt
        total_num += cnt

    return epoch_loss / total_num

def val_epoch(model, data_loader, loss_func, device):
    model.eval()

    epoch_loss, total_num = 0, 0
    for i, (_, X_nm_enc, y_enc) in enumerate(data_loader):
        X_nm_enc = X_nm_enc.float().to(device)
        y_enc = y_enc.float().to(device)

        X_nm_enc.requires_grad_(True)
        ypred = model(X_nm_enc)
        string_stb_list = calc_string_stability(model, device)
        loss = loss_func(ypred, y_enc.unsqueeze(-1), X_nm_enc, string_stb_list, verbose=True, stage='val')
        X_nm_enc.requires_grad_(False)

        cnt = ypred.size(0)
        epoch_loss += loss.item() * cnt
        total_num += cnt

    return epoch_loss / total_num

def loss(ypred, ytrue, X, string_stb_list=None, verbose=False, stage='train'):
    global train_val_monitor, epoch_cnt
    loss = nn.MSELoss(reduction='mean')(ypred, ytrue)
    gradients = torch.autograd.grad(outputs=ypred, inputs=X, grad_outputs=torch.ones_like(ypred), create_graph=True, retain_graph=True)[0]
    mono_loss = torch.relu(-gradients[:, 1]).mean() + torch.relu(gradients[:, 2]).mean()
    string_stb_val = None
    if string_stb_list is not None:
        string_stb_val = torch.relu(-torch.min(string_stb_list))
        if verbose:
            print('mono_loss: %.4f, string_stb: %.3f' % (mono_loss.detach().numpy(), string_stb_val.detach().numpy()))
        loss = loss + string_stb_val * STB_FACTOR
    loss = loss + mono_loss * MONO_FACTOR
    train_val_monitor.append([epoch_cnt, stage, loss.detach().numpy(), mono_loss.detach().numpy(), string_stb_val.detach().numpy() if string_stb_val is not None else None])
    if stage == 'val':
        epoch_cnt += 1
    return loss

def train_val_test_pipeline(model):
    train_set = DnnDataset(SAMPLE_DIR, tp='train', suffix=SUFFIX)
    val_set = DnnDataset(SAMPLE_DIR, tp='val', suffix=SUFFIX)
    train_sampler = RandomSampler(train_set)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE if BATCH_SIZE > 0 else len(train_set), sampler=train_sampler, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=len(val_set), sampler=None, num_workers=0, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    min_val_loss = float('inf')
    loss_list, not_descending_cnt = [], 0
    for epoch in range(EPOCH_NUM):
        start_time = time()
        train_loss = train_epoch(model, train_loader, optimizer, loss, device)
        val_loss = val_epoch(model, val_loader, loss, device)
        train_loss, val_loss = round(train_loss, 3), round(val_loss, 3)
        end_time = time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print('Epoch: %s | Time: %sm %ss' % (str(epoch + 1).zfill(2), epoch_mins, epoch_secs))
        print('\tTrain Loss: %.3f | Val Loss: %.3f' % (train_loss, val_loss))
        loss_list.append(val_loss)

        if val_loss >= min_val_loss:
            not_descending_cnt += 1
            if not_descending_cnt >= STOP_NUM and epoch != EPOCH_NUM - 1:
                print('\nEarly Stopped ...')
                break
        else:
            not_descending_cnt = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), pathjoin(RESULT_DIR, '%s_%s.pt' % (MODEL, SUFFIX)))
            print()
            print('model saved with validation loss', val_loss)
            print()

def simulate_test_pipeline(model, model_data_dir, cf_pairs_df, Ts, device, seed=1):
    state_dict = torch.load(pathjoin(RESULT_DIR, '%s_%s.pt' % (MODEL, SUFFIX)), map_location=device)
    model.load_state_dict(state_dict)
    traj_list = sorted(cf_pairs_df.traj_id.unique())
    X_nm_dict, _ = get_X_nm_ct(model_data_dir, 'test', traj_list)
    traj_params_df, traj_sim_dict = [], {}
    for i, traj_id in enumerate(traj_list):
        traj_id, traj_sim_dict_sub = traj_simulate(traj_id, model=model, simulate_func='simulate_DNN',
                                                   X_nm_dict=X_nm_dict, cf_pairs_df=cf_pairs_df, enc_len=1,
                                                   dec_len=None, Ts=Ts, device=device, seed=seed)
        traj_sim_dict[traj_id] = traj_sim_dict_sub
    with open(pathjoin(RESULT_DIR, '%s/sim_dict_%s_%s.pkl' % (DATASET_T, MODEL, SUFFIX)), 'wb') as f:
        f.write(pickle.dumps(traj_sim_dict))
    return traj_sim_dict

def print_result(traj_sim_dict, style_index=None):
    if style_index is not None:
        traj_sim_dict = traj_sim_dict[style_index]
    error_list = []
    for traj_id, traj_dict in traj_sim_dict.items():
        error = calc_error(traj_dict['vgap_sim'], traj_dict['vgap_list'])
        error_list.append(error)
    print('\nvgap simulation error:')
    print(pd.Series(error_list).describe())

    collision_cnt = 0
    for traj_id, traj_dict in traj_sim_dict.items():
        collision_cnt_i = int(sum(traj_dict['collision_loc']) > 0)
        collision_cnt += collision_cnt_i
        if collision_cnt_i > 0:
            print(traj_id)
    print('\nCollision count: %s' % collision_cnt)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_t', type=str, default='ngsim_i80')
    parser.add_argument('--model', type=str, default='dnn_sg_mode_mono_cons')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--llm', type=str, default='deepseek-coder')
    parser.add_argument('--suffix', type=str, default='zero-shot-dist')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--mono_factor', type=float, default=5000.)
    parser.add_argument('--stb_factor', type=float, default=0.9)
    parser.add_argument('--stop_num', type=int, default=10)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--need_train', type=str, default='True')
    parser.add_argument('--need_predict', type=str, default='True')
    args = parser.parse_args()
    print(args)

    LLM = args.llm
    SUFFIX = args.suffix
    MODEL = args.model
    NEED_TRAIN = eval(args.need_train)
    NEED_PREDICT = eval(args.need_predict)
    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    ORIGINAL_DATA_DIR = pathjoin(ABS_PATH, '../../../data/')
    SAMPLE_DIR = pathjoin(ABS_PATH, 'data/samples/%s/' % LLM)
    RESULT_DIR = pathjoin(ABS_PATH, 'result/%s' % LLM)
    BATCH_SIZE = args.batch_size
    EPOCH_NUM = 10000
    LEARNING_RATE = args.learning_rate
    MONO_FACTOR = args.mono_factor
    STB_FACTOR = args.stb_factor
    SEED = int(args.seed)
    STOP_NUM = args.stop_num

    set_seed(SEED)
    pathmake(RESULT_DIR)

    device = torch.device('cpu')

    feature_configs_dict = pickle.load(open(pathjoin(SAMPLE_DIR, 'feature_configs_%s.pkl' % SUFFIX), 'rb'))
    nm_cols = feature_configs_dict['nm_cols']
    del feature_configs_dict['nm_cols']

    model = DnnModel(feature_configs_dict, nm_cols, hidden_size=16, device=device).to(device)

    if NEED_TRAIN:
        train_val_monitor, epoch_cnt = [], 0
        train_val_test_pipeline(model)
        train_val_monitor_df = pd.DataFrame(train_val_monitor, columns=['epoch', 'stage', 'loss', 'mono_loss', 'string_stb'])
        train_val_monitor_df.to_csv(pathjoin(RESULT_DIR, '%s_%s_monitor.csv' % (MODEL, SUFFIX)), index=False)

    if NEED_PREDICT:

        if args.dataset_t == 'all':
            dataset_list = ['highd', 'ngsim_i80', 'ngsim_us101']
        else:
            dataset_list = [args.dataset_t]

        for dataset in dataset_list:
            print('\nDataset: %s\n' % dataset)
            DATASET_T = dataset
            DATA_DIR = pathjoin(ABS_PATH, 'data/%s/' % DATASET_T)

            pathmake(pathjoin(RESULT_DIR, DATASET_T))
            Ts = DATASET_DICT[DATASET_T]

            cf_pairs_df = pd.read_csv(pathjoin(ORIGINAL_DATA_DIR, 'cf_pairs_df_%s.csv' % DATASET_T)).sort_values(['traj_id', 'frame_id']).reset_index(drop=True)
            df_base_test = pd.read_csv(pathjoin(DATA_DIR, 'df_base_test.csv'))
            cf_pairs_df = cf_pairs_df.merge(df_base_test[['traj_id']].drop_duplicates(), on='traj_id')
            traj_sim_dict = simulate_test_pipeline(model, DATA_DIR, cf_pairs_df, Ts=Ts, device=device, seed=SEED)

            print_result(traj_sim_dict)