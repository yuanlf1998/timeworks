# -*- coding: utf-8 -*-
# Author: Linfeng
# Date: 2025-12-09
# 整合了模型侧（DLinear, PatchTST）和 TFB 的代码

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from . import dataset_config
from timeworks.utils.cprint import cprint

dict_712 = {'train': {'start': 0, 'end': 0.7},
            'val': {'start': 0.7, 'end': 0.8},
            'test': {'start': 0.8, 'end': 1},
            'all': {'start': 0, 'end': 1}
            }
dict_622 = {'train': {'start': 0, 'end': 0.6},
            'val': {'start': 0.6, 'end': 0.8},
            'test': {'start': 0.8, 'end': 1},
            'all': {'start': 0, 'end': 1}
            }


def read_raw(dataset_name: str, flag='train', norm=True, style='traditional', outlier_threshold=0.99) -> pd.DataFrame:
    """
        读csv, train test split, norm
        2025/12/16 适配 tfb 的长度
        2025/12/17 增加 clean_outliers逻辑
    """
    if dataset_name in dataset_config.tfb_datasets:
        folder = dataset_config.tfb_root_path
        style = 'tfb'
    elif dataset_name in dataset_config.traditional_datasets:
        folder = dataset_config.root_path
    else:
        raise NotImplementedError

    dataset_info = dataset_config.config_dict[dataset_name]

    filepath = folder + dataset_name + '.csv'
    df = pd.read_csv(filepath)
    if dataset_name in dataset_config.tfb_datasets:
        # 处理 tfb 格式（列存储）的数据
        df = df.pivot(index='date', columns='cols', values='data')
    else:
        # 需要 drop 掉date 列
        df = df.iloc[:, 1:]

    # train test split
    if style == 'traditional':
        n = len(df)
    elif style == 'tfb':
        # 从 meta 文件获取长度
        filepath = dataset_config.tfb_root_path + 'FORECAST_META.csv'
        meta_data = pd.read_csv(filepath) 
        n = meta_data[meta_data['file_name']==dataset_name + '.csv']['length'].item()
        
    split_method = dataset_info.split
    index_dict = dict_712 if split_method == '712' else dict_622
    start_idx = int(index_dict[flag]['start'] * n)
    end_idx = int(index_dict[flag]['end'] * n)

    # slice
    target = df.iloc[start_idx: end_idx]
    if norm:
        # norm
        train_len = int(len(df) * index_dict['train']['end'])
        df_train = df.iloc[:train_len]
        scaler = StandardScaler().fit(df_train)
        target = scaler.transform(target)

    return target


def norm(data, dataset_name):
    def get_normalizer(dataset_name):
        df_train = read_raw(dataset_name, 'train', norm=False)
        scaler = StandardScaler().fit(df_train)
        return scaler

    def apply_scalar(pred_data, scaler):
        """
        apply 2d scalar to 3d data 
        pred_data: np.ndarray, shape (n, w, c)
        scaler: fitted sklearn scaler (fit on train data of shape (n_train, c))
        """
        n, w, c = pred_data.shape
        
        # 1. reshape to (-1, c)
        pred_2d = pred_data.reshape(-1, c)
        
        # 2. apply scaler
        pred_scaled_2d = scaler.transform(pred_2d)
        
        # 3. reshape back to (n, w, c)
        pred_scaled = pred_scaled_2d.reshape(n, w, c)
        
        return pred_scaled

    scalar = get_normalizer(dataset_name)
    if len(data.shape) == 2:
        data = scalar.transform(data)
    else: 
        data = apply_scalar(data, scalar)
    return data 


def load_data(dataset_name: str, flag='train', window_size=96, style='traditional', norm=True, 
              clean=True, verbose=True) -> np.ndarray:
    """
        return windowed data after preprocessing 
        TODO: 目前只支持 lookback-pred 等长的情况，后面用到再改
    """
    # read raw
    cprint('Loading data', verbose)
    data = read_raw(dataset_name, flag, norm=norm, style=style)

    def clean_outliers(data: np.ndarray) -> np.ndarray:
        """
            去除极端值, 按 99 分位数切割
        """
        cleaned = data.copy()
        N, C = cleaned.shape

        for c in range(C):
            col = cleaned[:, c]

            # 不插值的分位数（取最近样本）
            q01 = np.quantile(col, 0.01, method="nearest")
            q99 = np.quantile(col, 0.99, method="nearest")

            mask = (col < q01) | (col > q99)
            cleaned[mask, c] = 0

        return cleaned

    if clean:
        cprint('Cleaning', verbose)
        data = clean_outliers(data)

    # window
    enc_in = data.shape[1]

    cprint('Reshaping', verbose)
    data_windowed = np.lib.stride_tricks.sliding_window_view(
        data, (window_size, enc_in))
    data_windowed = np.squeeze(data_windowed)

    print(f'{dataset_name} {flag}:')
    print(data_windowed.shape)
    return data_windowed


# 测试
if __name__ == '__main__':
    pass 

    # for dataset_name in dataset_config.traditional_datasets + dataset_config.tfb_datasets:
    #     for flag in ['train', 'val', 'test']:
    #         data_windowed = load_data(dataset_name, flag)
    
    # print('Test passed')
    
    data = load_data('ETTm1', 'test')
    data_norm = norm(data, 'ETTm1')
    
