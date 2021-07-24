import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

tune_round = 3
selected_nn = 3
noise_percent = 0.20

def get_random_noise(dataShape, comb_maxmin):
    print(f"get random noise args: {dataShape}, {comb_maxmin.shape}")
    # 0 is row
    # 1 is column
    total_noise = int(np.ceil(dataShape[0]*noise_percent))
    # random for each
    _rd = None
    for i in range(comb_maxmin.shape[0]):
        _p = np.random.randint(comb_maxmin[i][0], comb_maxmin[i][1], (total_noise, 1))
        if _rd is None:
            _rd = _p
        else:
            _rd = np.hstack((_rd, _p))
    return _rd


def vizz(data_point, name=""):
  fig, ax = plt.subplots()
  ax.scatter(data_point[:, 0], data_point[:, 1])
  fig.savefig(name, dpi=100)
  plt.close()

def algorithm1(dps):
    _dps = dps.copy()
    last_columns_index = _dps.shape[1]-1
    for i in range(tune_round):
        for dpidx in range(_dps.shape[0]):
            filter_dps = _dps[np.isin(dps_default_idx, [dpidx], invert=True)]
            cdp = _dps[dpidx, :last_columns_index].copy().reshape((1, 2))
            # 
            dist = pairwise_distances(filter_dps[::, :last_columns_index], cdp, metric='euclidean')
            dist = dist.ravel()
            dist_idx = np.argsort(dist)
            # get min dist by selected_nn
            min_dist_selected_idx = dist_idx[:selected_nn]
            new_dp = np.array([np.mean(filter_dps[min_dist_selected_idx, :last_columns_index], axis=0)])
            _dps[dpidx, :last_columns_index] = new_dp
        
        vizz(_dps, f'./plot/{i}.png')
        
    return _dps

def get_comb_maxmin_each_column(X):
    _get_min = X.min(axis=0)
    _get_max = X.max(axis=0)
    return np.vstack((_get_min, _get_max)).T

if __name__ == '__main__':
    fileName = 'a1'
    data_path = f'./data/{fileName}.csv'
    X = pd.read_csv(data_path).to_numpy()
    #vizz(X, f'./plot/{fileName}-raw_data.png')
    comb_maxmin = get_comb_maxmin_each_column(X)

    _rd = get_random_noise(X.shape, comb_maxmin) 

    X = np.vstack((X, _rd))
    # origin_training_data = origin_data.loc[:, ['DFA', 'PPE']]
    # origin_label_data = origin_data[['status']]
    # origin_training_data = origin_data.drop(columns=['name', 'status'])
    # y = origin_label_data.to_numpy().ravel()
    dps = X.copy()
    dps_default_idx = np.arange(dps.shape[0])
    dps = np.append(dps, dps_default_idx.reshape(dps_default_idx.shape[0], 1), 1)
    algorithm1(dps)

