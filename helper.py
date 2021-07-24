import numpy as np

def get_comb_maxmin_each_column(X):
    _get_min = X.min(axis=0)
    _get_max = X.max(axis=0)
    return np.vstack((_get_min, _get_max)).T

def get_random_noise(dataShape, comb_maxmin, noise_percent):
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