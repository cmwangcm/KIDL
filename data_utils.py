import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

DATASET_DICT = {
    'ngsim_i80': 0.1,
}

def normalize_df(df_train, feature_configs_dict, nm_cols):
    for col in nm_cols:
        feature_configs_dict[col] = feature_configs_dict.get(col, {})

        mean_ = df_train.loc[:, col].mean()
        std_ = df_train.loc[:, col].std()
        std_ = std_ if std_ > 0.01 else 1
        feature_configs_dict[col]['mean_'] = mean_
        feature_configs_dict[col]['std_'] = std_

        min_val = df_train.loc[:, col].min()
        max_val = df_train.loc[:, col].max()
        feature_configs_dict[col]['min_val'] = min_val
        feature_configs_dict[col]['max_val'] = max_val

    return feature_configs_dict

def categorize_df(df_train, feature_configs_dict, ct_cols, n_bins=5):
    for col in ct_cols:
        if df_train.loc[:, col].dtype in ('int64', 'int32', 'string'):
            feature_configs_dict[col] = feature_configs_dict.get(col, {})
        else:
            kbins = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
            kbins.fit(df_train.loc[:, col].values.reshape((-1, 1)))
            feature_configs_dict[col] = feature_configs_dict.get(col, {})
            feature_configs_dict[col]['bins'] = list(kbins.bin_edges_[0])
    return feature_configs_dict