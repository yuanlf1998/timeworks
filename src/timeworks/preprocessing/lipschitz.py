import torch
import numpy as np
import matplotlib.pyplot as plt

from timeworks.utils.cprint import cprint
from timeworks.data import load_data
from timeworks.preprocessing.drop_plat import remove_plat

device = torch.device('cuda')


def calc_pair_k(x, y):
    '''
        评估 k 的核心逻辑，其中x和 y 是窗口后的(N, W)
    '''
    X = torch.from_numpy(x).to(device)
    Y = torch.from_numpy(y).to(device)

    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(X.shape[0], -1)

    n_samples = len(x)
    n_dim = X.shape[1] if len(X.shape) == 2 else X.shape[1] * X.shape[2]
    # print(n_dim)

    Dx = torch.cdist(X, X, p=2)   # (n_samples, n_samples)
    Dy = torch.cdist(Y, Y, p=2)   # (n_samples, n_samples)

    Dx = Dx.detach().cpu() / np.sqrt(n_dim)
    Dy = Dy.detach().cpu() / np.sqrt(n_dim)

    Dx = torch.where(Dx < 1e-3, 1, Dx)

    ks = Dy / Dx
    return ks


def calc_k(data_windowed: np.ndarray, pred: np.ndarray, interval=20,
           q=100, channel_independence=True,
           preprocess=True, verbose=False, return_matrix=False):
    '''
        计算所有 x 和 y样本的 k，主要是一些预处理的逻辑
        x, y: (N, W, C) 或 (N, W)
    '''
    cprint(f'q: {q}, interval: {interval}', verbose)
    
    sample_idx = np.arange(len(pred))[::interval]
    
    # sample 
    x = data_windowed[:len(pred)][sample_idx]
    y = pred[sample_idx]
    
    n_dim = x.shape[-1]
    
    if channel_independence:
        # 逐 channel 计算，最后求平均
        k_by_dim = []
        k_matrix_list = []
        
        # 遍历各个channel
        for i in range(n_dim):
            cprint(f'Channel {i}', verbose)
            if preprocess:
                # 先筛选，再重新采样 
                x_dim, y_dim = remove_plat(data_windowed[:len(pred), :, i], pred[:, :, i])
                p = len(x_dim) / len(data_windowed)
                cprint(f'{p*100:.2f}%', verbose)
                if len(x_dim) < len(x)*0.1:
                    print('Abnormal!')
                    k_by_dim.append(np.nan)
                    continue 
                sample_idx = np.arange(len(x_dim))[::interval]
                x_dim = x_dim[sample_idx]
                y_dim = y_dim[sample_idx]
            else:
                x_dim = x[:, :, i] 
                y_dim = y[:, :, i]
            # start calculate inside channel 
            k_matrix = calc_pair_k(x_dim, y_dim)
            k_matrix = k_matrix.numpy()
            k = np.percentile(k_matrix, q)
            cprint(f'k: {k}', verbose)

            k_by_dim.append(k)
            k_matrix_list.append(k_matrix)

        if verbose: 
            plt.bar(np.arange(n_dim), k_by_dim)
        k_by_dim = [k for k in k_by_dim if k >= 0]
        print(round(np.mean(k_by_dim), 4))

        if return_matrix:
            return k_by_dim, k_matrix_list
        else: 
            return k_by_dim
    
    else: 
        # treat as a whole 
        k_matrix = calc_pair_k(x, y)
        k_matrix = k_matrix.numpy()
        result = np.percentile(k_matrix, q)
        print(round(result, 4))
        if return_matrix:
            return result, k_matrix
        else:
            return result


def calc_k_pair(data_windowed: np.ndarray, y_true: np.ndarray, pred: np.ndarray,
                interval=20, q=100, channel_independence=True,
                preprocess=True, verbose=False, return_matrix=True):
    """
    Compute k for (data, y_true) and (data, pred) with optional matrices.

    Returns (k_true, k_pred, k_matrix_true, k_matrix_pred) when return_matrix is True,
    otherwise returns (k_true, k_pred).
    """
    k_true, k_matrix_true = calc_k(
        data_windowed,
        y_true,
        interval=interval,
        q=q,
        channel_independence=channel_independence,
        preprocess=preprocess,
        verbose=verbose,
        return_matrix=True,
    )
    k_pred, k_matrix_pred = calc_k(
        data_windowed,
        pred,
        interval=interval,
        q=q,
        channel_independence=channel_independence,
        preprocess=preprocess,
        verbose=verbose,
        return_matrix=True,
    )
    if return_matrix:
        return k_true, k_pred, k_matrix_true, k_matrix_pred
    return k_true, k_pred


def dataset_k(dataset_name: str, flag='train', look_ahead=96, **calc_kwargs):
    """
    Convenience helper that computes the dataset's intrinsic Lipschitz k by
    pairing consecutive windows from the same dataset (shifted by `look_ahead`).

    Parameters
    ----------
    dataset_name :
        Name present in `dataset_config`.
    flag :
        Split flag passed down to `load_data`.
    look_ahead :
        Gap between the input/output windows, defaulting to 96 steps.
    calc_kwargs :
        Additional keyword arguments forwarded to `calc_k`.
    """
    dataset = load_data(dataset_name, flag)
    if len(dataset) <= look_ahead:
        raise ValueError("Dataset is shorter than the requested look_ahead shift.")

    x = dataset[:-look_ahead]
    y = dataset[look_ahead:]
    return calc_k(x, y, **calc_kwargs)
    

if __name__ == '__main__':
    folder = '/home/yuanlinfeng/data/prediction/ETTm1/PatchTST'
    y_true = np.load(folder + '/true.npy')
    y_pred = np.load(folder + '/pred.npy')
    
    ks = calc_k(y_true, y_pred, verbose=True, q=100)
    
