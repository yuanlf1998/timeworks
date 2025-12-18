# -*- coding: utf-8 -*-
# Author: Linfeng
# Date: 2025-12-09
# Error funcs 

import numpy as np
from timeworks.data import norm, load_data


def _mse(y_true: np.ndarray, y_predict: np.ndarray):
    """
        Calculates the mse of a series or an array 
    """
    return np.mean((y_true - y_predict) ** 2, axis=1).mean(axis=0)


def mse(y_predict: np.ndarray, dataset_name: str, y_true=None, 
        style='tfb',         
        return_by_dims=False):
    """
        包装后的 mse 函数, 支持自动加载真实值和自动标准化
    """
    # 默认再做一次标准化
    y_predict = norm(y_predict, dataset_name)

    if y_true is None:
        y_true = load_data(dataset_name, flag='test', style=style, verbose=False)
    
    error = _mse(y_true, y_predict)
    if return_by_dims:
        return error
    else:
        return np.mean(error)
    

    
