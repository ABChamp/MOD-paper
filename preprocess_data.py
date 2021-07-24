
import pandas as pd
# http://cs.uef.fi/sipu/datasets/
def preprocess_data():
    d_list = {
        'a1': {'url': './raw_data/a1.txt', 'ex_type': 'txt'},
        'a2': {'url': './raw_data/a2.txt', 'ex_type': 'txt'},
        'a3': {'url': './raw_data/a3.txt', 'ex_type': 'txt'},
        'uci-parkinsons': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data', 
            'ex_type': 'pd_csv',
            'ex_detail': {'sep': ','}},
    }

    for k, v in d_list.items():
        if v['ex_type'] == 'txt':
            with open(v['url']) as f:
                lines = [line.rstrip().lstrip().split(' ') for line in f]
                df = pd.DataFrame(lines)
                df.to_csv(f'./data/{k}.csv', index=False)
        elif v['ex_type'] == 'pd_csv':
            df = pd.read_csv(v['url'], **v['ex_detail'])
            df.to_csv(f'./data/{k}.csv', index=False)

if __name__ == '__main__':
    preprocess_data() 