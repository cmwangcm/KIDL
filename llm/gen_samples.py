import numpy as np
import pandas as pd
import re
import argparse
from time import time, sleep
from functools import partial
from sklearn.model_selection import train_test_split
from scipy.stats import truncnorm
import sys

sys.path.append('.')
from utils import *
from data_utils import *
from llm.llm_utils import *
from system_msg import SYSTEM_MSG_DICT

def gen_normal_states(Ts=0.1):
    def get_truncated_normal(mean=0., sd=1., low=0., upp=10.):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    sid = np.array(list(range(TOTAL_NUM))).reshape((-1, 1))
    vsp = get_truncated_normal(15, 15, RANGE_DICT['vsp'][0], RANGE_DICT['vsp'][1]).rvs((TOTAL_NUM, 1))
    vgap = get_truncated_normal(15, 15, RANGE_DICT['vgap'][0], RANGE_DICT['vgap'][1]).rvs((TOTAL_NUM, 1))
    vrelsp = np.random.randn(TOTAL_NUM, 1) * 2
    lvsp = vsp - vrelsp
    no_neg_lvsp = lvsp >= 0
    no_collision = vgap + Ts * ((lvsp - vsp) + (lvsp - (vsp - 5 * Ts))) / 2 >= 0.1
    index_list = np.where(np.logical_and(no_collision, no_neg_lvsp))[0]
    return np.concatenate([sid, vsp, vgap, vrelsp], axis=1)[index_list]

def construct_user_msg(inputs):
    user_msg_list = ["Here is the current scenario: vsp:{vsp}, vgap:{vgap}, lvsp:{lvsp}. Think step by step to predict vacc in the next time step.".format(vsp=round(inputs[i][1], 2), vgap=round(inputs[i][2], 2), lvsp=round(inputs[i][1] - inputs[i][3], 2))
                     for i in range(len(inputs))]
    states_msg_df = pd.DataFrame(inputs, columns=['sid', 'vsp', 'vgap', 'vrelsp'])
    states_msg_df['sid'] = states_msg_df.sid.astype(int)
    states_msg_df['user_msg'] = user_msg_list
    return states_msg_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default='deepseek-coder')
    parser.add_argument('--system_msg_name', type=str, default='zero-shot-dist')
    parser.add_argument('--save_dir', type=str, default='models/dnn_sg/')
    parser.add_argument('--sample_num', type=int, default=10000)
    parser.add_argument('--multiplier', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.9)
    args = parser.parse_args()
    print(args)

    LLM = args.llm
    TEMPERATUR = args.temperature
    GEN_METHOD = args.gen_method
    SYSTEM_MSG_NAME = args.system_msg_name
    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = pathjoin(ABS_PATH, args.save_dir, 'data/samples/%s' % LLM)
    pathmake(DATA_DIR)

    SEED = 10
    MAX_WORKERS = 50
    TOTAL_NUM = 1000000
    SAMPLE_NUM = args.sample_num
    MULTIPLIER = args.multiplier
    SAMPLE_NUM_EACH = min(200, SAMPLE_NUM)
    TS = 0.1
    np.random.seed(SEED)

    RANGE_DICT = {'vsp': [0, 40],
                  'vgap': [0.1, 100]}

    system_msg = SYSTEM_MSG_DICT[SYSTEM_MSG_NAME]

    inputs = gen_normal_states(TS)
    print("Total number:", len(inputs))
    states_df = construct_user_msg(inputs)

    nm_cols = ['vsp', 'vgap', 'vrelsp']
    feature_configs_dict = {'nm_cols': nm_cols}
    feature_configs_dict = normalize_df(states_df.copy(), feature_configs_dict, nm_cols)
    with open(pathjoin(DATA_DIR, 'feature_configs_%s.pkl' % SYSTEM_MSG_NAME), 'wb') as f:
        pickle.dump(feature_configs_dict, f)

    sid_list = sorted(states_df.sid.unique())
    sid_list_sub = np.random.choice(sid_list, size=SAMPLE_NUM, replace=False)
    states_df_sub = states_df[states_df.sid.isin(sid_list_sub)]
    user_msg_list = [[row.sid, row.user_msg] for i, row in states_df_sub.iterrows()] * MULTIPLIER

    res, total_cnt = [], 0
    for i in range((SAMPLE_NUM * MULTIPLIER) // SAMPLE_NUM_EACH):
        start_i = time()
        res_i = multi_thread_request(partial(llm_infer, model=LLM, system_msg=system_msg, extract_result_func=extract_float_result, temperature=TEMPERATUR),
                                     user_msg_list[SAMPLE_NUM_EACH*i:SAMPLE_NUM_EACH*(i+1)], max_workers=MAX_WORKERS)
        res.extend(res_i)
        total_cnt += len(res_i)
        printf('Total cnt: %s' % total_cnt)

    res_df = pd.DataFrame(res, columns=['sid', 'vacc_pred', 'content', 'logprobs'])
    res_df = res_df.merge(states_df, on='sid')
    res_df.to_csv(pathjoin(DATA_DIR, 'samples_df_raw_%s.csv' % SYSTEM_MSG_NAME), index=False)
    res_df = res_df.dropna().reset_index(drop=True)
    res_df['vacc_pred'] = res_df.vacc_pred.apply(lambda x: np.clip(x, -5., 5.))
    res_df.to_csv(pathjoin(DATA_DIR, 'samples_df_%s.csv' % SYSTEM_MSG_NAME), index=False)

    sid_list = sorted(res_df.sid.unique())
    sid_list_train, sid_list_test = train_test_split(sid_list, test_size=0.1, random_state=SEED)
    sid_list_train, sid_list_val = train_test_split(sid_list_train, test_size=0.1, random_state=SEED)
    res_df_train = res_df[res_df.sid.isin(sid_list_train)]
    res_df_val = res_df[res_df.sid.isin(sid_list_val)]
    res_df_test = res_df[res_df.sid.isin(sid_list_test)]

    res_df_train.to_csv(pathjoin(DATA_DIR, 'samples_df_train_%s.csv' % SYSTEM_MSG_NAME), index=False)
    res_df_val.to_csv(pathjoin(DATA_DIR, 'samples_df_val_%s.csv' % SYSTEM_MSG_NAME), index=False)
    res_df_test.to_csv(pathjoin(DATA_DIR, 'samples_df_test_%s.csv' % SYSTEM_MSG_NAME), index=False)