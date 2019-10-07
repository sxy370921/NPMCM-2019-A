import pandas as pd
import numpy as np

class MyDataset:

    def __init__(self, file, batch_size):
        data = pd.read_csv(file)
        print('data shape:', data.shape)
        self._num_samples = data.shape[0]
        feat_cols = ['dy', 'd', 'RS Power', 'dx',  'dh', 'l', 'th_d',
                        'log3', 'log2', 'pl', 'A_mean', 'dth_h', 'd_hv', 'Index']
        label_name = 'RSRP'
        self.batch_size = batch_size
        self.num = self._num_samples // batch_size
        self._index = np.arange(self._num_samples)
        self._idx_bat = [self._index[i:i+batch_size] for i in range(0, self._num_samples, batch_size)]
        self.bit_len = 20

        self.feat = data[feat_cols].values
        self.label = data[label_name].values.reshape(-1,1)



    def __len__(self):
        return self.num

    def shufle_index(self):
        np.random.shuffle(self._index)
        self._idx_bat = [self._index[i:i+self.batch_size] for i in range(0, self._num_samples, self.batch_size)]

    def __getitem__(self, idx):
        idx_bat = self._idx_bat[idx]
        return self.feat[idx_bat], self.label[idx_bat]


if __name__ == '__main__':
    # root_path = 'E:\\AAA19研赛\\2019年中国研究生数学建模竞赛赛题\\2019年中国研究生数学建模竞赛A题\\'
    dataset = MyDataset('train_data_new.csv', 1000)
    print(dataset[1][0].shape, dataset[1][1].shape)