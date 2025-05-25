import pandas as pd
import numpy as np
import os
import pickle
import shutil
import argparse
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import sys
sys.path.append('.')

from data_utils import normalize_df, categorize_df
from utils import *

def get_seq_data(df, cols, enc_len=1, dec_len=1, skip=10):
    X_enc_list, X_dec_list, start_points_df = [], [], []
    for traj_id in sorted(df.traj_id.unique()):
        df_sub = df[df.traj_id == traj_id]
        n = len(df_sub)
        seq_len = enc_len + dec_len
        sample_cnt = n - seq_len + 1
        if sample_cnt <= 0:
            raise Exception('sample_cnt must be greater than 0')
        start_points = list(range(0, sample_cnt, skip))
        enc_points = [list(range(start_point, start_point + enc_len)) for start_point in start_points]
        dec_points = [list(range(start_point + enc_len, start_point + enc_len + dec_len)) for start_point in start_points]

        X = df_sub[cols].values
        n, m = len(start_points), X.shape[1]
        X_enc = np.take(X, enc_points, axis=0)
        X_dec = np.take(X, dec_points, axis=0)
        X_enc_list.append(X_enc)
        X_dec_list.append(X_dec)
        start_points_df.extend(list(zip([traj_id] * n, start_points, [enc_len] * n, [dec_len] * n)))
    X_enc_seq, X_dec_seq = np.concatenate(X_enc_list), np.concatenate(X_dec_list)
    start_points_df = pd.DataFrame(start_points_df, columns=['traj_id', 'start_point', 'enc_len', 'dec_len'])
    return start_points_df, X_enc_seq, X_dec_seq

def get_features(df, nm_cols, ct_cols, test_size):
    traj_list = sorted(df.traj_id.unique())
    traj_list_train, traj_list_test = train_test_split(traj_list, test_size=test_size, random_state=SEED)
    traj_list_train, traj_list_val = train_test_split(traj_list_train, test_size=test_size, random_state=SEED)
    df_train, df_val, df_test = df[df.traj_id.isin(traj_list_train)], df[df.traj_id.isin(traj_list_val)], df[df.traj_id.isin(traj_list_test)]

    feature_configs_dict = {'nm_cols': nm_cols, 'ct_cols': ct_cols}
    if len(nm_cols) > 0:
        feature_configs_dict = normalize_df(df_train.copy(), feature_configs_dict, nm_cols)
    if len(ct_cols) > 0:
        feature_configs_dict = categorize_df(df_train.copy(), feature_configs_dict, ct_cols, n_bins=10)
    with open(pathjoin(DATA_DIR, 'feature_configs.pkl'), 'wb') as f:
        pickle.dump(feature_configs_dict, f)

    base_cols = ['traj_id', 'frame_id', 'vid', 'lvid', 'lane_id', 'vlen', 'lvlen', 'vsp', 'vgap', 'vrelsp', 'vlocy', 'vacc', 'lvsp', 'lvlocy']
    for tp in ['train', 'val', 'test']:
        df_tp = eval('df_%s' % tp)
        df_tp[base_cols].to_csv(pathjoin(DATA_DIR, 'df_base_%s.csv' % tp), index=False)
        df_tp[nm_cols].to_csv(pathjoin(DATA_DIR, 'df_nm_%s.csv' % tp), index=False)
        df_tp[ct_cols].to_csv(pathjoin(DATA_DIR, 'df_ct_%s.csv' % tp), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ngsim_i80')
    parser.add_argument('--data_dir', type=str, default='models/dnn_sg/data/')
    args = parser.parse_args()

    DATASET = args.dataset
    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    ORIGINAL_DATA_DIR = pathjoin(ABS_PATH, '../../data/')
    DATA_DIR = pathjoin(ABS_PATH, args.data_dir, args.dataset)
    TEST_SIZE = 0.2
    SEED = 10
    np.random.seed(SEED)
    pathmake(DATA_DIR)

    df = pd.read_csv(pathjoin(ORIGINAL_DATA_DIR, 'cf_pairs_df_%s.csv' % DATASET)).sort_values(['traj_id', 'frame_id']).reset_index(drop=True)
    get_features(df, nm_cols=['vsp', 'vgap', 'vrelsp'], ct_cols=['traj_id', 'lane_id'], test_size=TEST_SIZE)