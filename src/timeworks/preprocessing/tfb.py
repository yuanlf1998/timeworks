# -*- coding: utf-8 -*-
# Author: Linfeng
# Date: 2025-12-09
# 处理 tfb 结果格式的各种问题

import base64
import os
import pickle
import time
import tarfile 

import numpy as np
import pandas as pd

from tqdm import tqdm


def decode_data(src_dir: str, dst_folder: str = '/home/yuanlinfeng/data/prediction'):
    """
    tfb 的结果以 base64 格式存在 csv 的列中，在官方代码基础上实现
    官方的代码会把每个 timestep 存到一个单独的路径下，修改了这一点
    Load the result CSV file and decode the base64-encoded 'inference_data' and 'actual_data' columns.

    :param src_dir: tar.gz fileath
    :return: None. The decoded data will be saved as CSV files in corresponding folders.
    """
    
    # example filepath: /home/yuanlinfeng/code/TFB-master/result/ETTm1/Crossformer/Crossformer.1765106206.e0562d6a3a084a03.1349716.csv.tar.gz
    
    # 自动识别dataset name 和 model name
    folder = os.path.dirname(src_dir)
    # 先 model name
    model_name = os.path.basename(folder)
    print(f'Model: {model_name}')
    # 再 dataset_name
    folder = os.path.dirname(folder)
    dataset_name = os.path.basename(folder)
    print(f'Dataset: {dataset_name}')
    
    dst_dir = os.path.join(dst_folder, dataset_name, model_name)
    os.makedirs(dst_dir, exist_ok=True)
    
    # 先解压成 csv
    print('解压tar.gz')
    temp_dir = '/home/yuanlinfeng/code/TFB-master/result-decoded/'
    with tarfile.open(src_dir, "r:gz") as tar:
        tar.extractall(path=temp_dir)   # 解压到指定目录
        
    # 解压后的 csv 路径
    filename = os.path.basename(src_dir).replace('.tar.gz', '')
    filename = os.path.join(temp_dir, filename)
    
    print(f"解压完成：{filename}")

    data = pd.read_csv(filename)  # Read the CSV file with encoded columns
    
    y_true_list = []
    y_pred_list = []
    
    for index, row in tqdm(data.iterrows()):
        # Decode base64 strings and deserialize them back to original DataFrames
        decoded_inference_data = base64.b64decode(row["inference_data"])
        decoded_actual_data = base64.b64decode(row["actual_data"])
        inference_data = pickle.loads(decoded_inference_data)
        actual_data = pickle.loads(decoded_actual_data)
        # TODO: 感觉可能会报错，看看要不要增加容错
        # save 
        y_true_filename = os.path.join(dst_dir, 'true.npy')
        if os.path.exists(y_true_filename):
            print(y_true_filename)
            print('文件已存在')
            return 
        y_pred_filename = os.path.join(dst_dir, 'pred.npy')
        
        np.save(y_true_filename, actual_data)
        np.save(y_pred_filename, inference_data)


# test 
if __name__ == '__main__':
    filepath_list = ['/home/yuanlinfeng/code/TFB-master/result/ETTm2/Crossformer/Crossformer.1766398221.e0562d6a3a084a03.1241839.csv.tar.gz',
                     '/home/yuanlinfeng/code/TFB-master/result/ETTm2/DUET/DUET.1766392842.e0562d6a3a084a03.1045571.csv.tar.gz', 
                     '/home/yuanlinfeng/code/TFB-master/result/ETTm2/FiLM/FiLM.1766397432.e0562d6a3a084a03.1052239.csv.tar.gz', 
                     '/home/yuanlinfeng/code/TFB-master/result/ETTm2/Informer/Informer.1766397581.e0562d6a3a084a03.1223438.csv.tar.gz', 
                     '/home/yuanlinfeng/code/TFB-master/result/ETTm2/MICN/MICN.1766397717.e0562d6a3a084a03.1228790.csv.tar.gz', 
                     '/home/yuanlinfeng/code/TFB-master/result/ETTm2/PatchTST/PatchTST.1766398336.e0562d6a3a084a03.1251981.csv.tar.gz', 
                     '/home/yuanlinfeng/code/TFB-master/result/ETTm2/TimesNet/TimesNet.1766397938.e0562d6a3a084a03.1233870.csv.tar.gz']
    for filepath in filepath_list:
    # filepath = '/home/yuanlinfeng/code/TFB-master/result/ETTm1/Crossformer/Crossformer.1765166214.e0562d6a3a084a03.2947050.csv.tar.gz'
        decode_data(filepath)
    print('Testcase passed')