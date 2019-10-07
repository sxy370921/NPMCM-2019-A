import pandas as pd
import numpy as np

data = pd.read_csv('train_data_new.csv')
test_num = 100000
index = np.arange(data.shape[0])
test_index = np.random.choice(index, test_num, replace=False)
test_data = data.iloc[test_index]
train_index = list(set(index)-set(test_index))
train_data = data.iloc[train_index]

print(test_data.shape, train_data.shape)

test_data.to_csv('test_data_new_split.csv')
train_data.to_csv('train_data_new_split.csv')