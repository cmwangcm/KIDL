import sys
sys.path.append('.')

from utils import *
from data_utils import *


if __name__ == '__main__':
    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = pathjoin(ABS_PATH, '../data/')
    MIN_TIME = 300

    raw_v_df = pd.read_table(pathjoin(DATA_DIR, 'ngsim/DATA (NO MOTORCYCLES).txt'), header=None,
                             names=['vid', 'frame_id', 'lane_id', 'vlocy', 'vsp', 'vacc', 'vlen', 'vcid', 'fvid', 'lvid'])

    raw_lv_df = raw_v_df.drop(['lvid', 'fvid'], axis=1) \
                        .rename(columns={'vid': 'lvid', 'vlocy': 'lvlocy', 'vsp': 'lvsp', 'vacc': 'lvacc',
                                         'vlen': 'lvlen', 'vcid': 'lvcid'}).copy()
    raw_df = raw_v_df.merge(raw_lv_df, on=['lvid', 'lane_id', 'frame_id'], how='inner') \
                     .sort_values(['vid', 'frame_id']).reset_index(drop=True)
    raw_df = raw_df[(raw_df.vcid == 2) & (raw_df.lvcid == 2)]

    traj_seq = []
    traj_id = 0
    for i, row in raw_df[['vid', 'lane_id', 'lvid']].drop_duplicates().iterrows():
        vid, lane_id, lvid = row.vid, row.lane_id, row.lvid
        raw_df_sub = raw_df[(raw_df.vid == vid) & (raw_df.lane_id == lane_id) & (raw_df.lvid == lvid)]
        frame_list = sorted(raw_df_sub.frame_id.tolist()) + [-1]
        traj_seq_sub, num = [], 0
        n = len(frame_list)
        for i in range(1, n):
            frame_id_prev, frame_id_cur = frame_list[i-1], frame_list[i]
            if frame_id_cur - frame_id_prev != 1:
                if len(traj_seq_sub) >= MIN_TIME:
                    traj_seq.append([traj_id, vid, lane_id, lvid, num, traj_seq_sub])
                    traj_id += 1
                    num += 1
                traj_seq_sub = []
            else:
                traj_seq_sub += [frame_id_prev]

    traj_seq_df = pd.DataFrame(traj_seq, columns=['traj_id', 'vid', 'lane_id', 'lvid', 'num', 'seq'])
    traj_frame_df = traj_seq_df.explode('seq').rename(columns={'seq': 'frame_id'})[['traj_id', 'frame_id', 'vid']]
    df = raw_df.merge(traj_frame_df, on=['vid', 'frame_id'], how='inner').sort_values(['traj_id', 'frame_id']).reset_index(drop=True)
    df['vgap'] = df.lvlocy - df.vlocy - df.lvlen
    df['vrelsp'] = df.vsp - df.lvsp
    df['vacc'] = df.groupby('traj_id').vacc.shift(-1)
    df['lvacc'] = df.groupby('traj_id').lvacc.shift(-1)
    df.dropna(inplace=True, subset=['vacc', 'lvacc'])

    traj_exclude = df[df.vgap <= 0].traj_id.unique()
    df = df[~df.traj_id.isin(traj_exclude)]
    df = df.sort_values(['traj_id', 'frame_id']).reset_index(drop=True)

    df = df.sort_values(['traj_id', 'frame_id']).reset_index(drop=True)
    print("Number of trajectories: %s" % len(df.traj_id.unique()))

    df.to_csv(pathjoin(DATA_DIR, 'cf_pairs_df_ngsim_i80.csv'), index=False)
