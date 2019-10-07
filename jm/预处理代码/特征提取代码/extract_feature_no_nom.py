import pandas as pd
import numpy as np
from glob2 import glob
from multiprocessing import Pool


def extract_feature(df):
    feat_cols_lx = ['dy', 'd', 'RS Power', 'dx', 'dh', 'l', 'th_d',
                    'log3', 'log2', 'pl', 'A_mean', 'dth_h', 'd_hv']

    feat_cols_lb = ['Cell Clutter Index', 'Clutter Index']

    def fun1(t):
        return 33.9 * np.log10(t)

    def fun2(t):
        return -13.82 * np.log10(t+.2)

    def fun3(t):
        return (44.9 - 6.55 * np.log10(t['Height']+.2)) * np.log10(t['d']+.2)

    df['dx'] = df['X'] - df['Cell X']  # 01
    df['dy'] = df['Y'] - df['Cell Y']  # 02
    df['d'] = df.apply(lambda t: (t['dx'] ** 2 + t['dy'] ** 2) ** .5, axis=1)  # 03
    df['dh'] = (df['Cell Altitude'] + df['Cell Building Height'] + df['Height'] \
                - df['Altitude'] - df['Building Height']).abs()  # 04
    df['th_d'] = df['Mechanical Downtilt'] + df['Electrical Downtilt']  # 05

    df['Cell Z'] = df['Cell Altitude'] + df['Cell Building Height'] + df['Height']  # temp cell z
    df['Z'] = df['Altitude'] + df['Building Height']  # temp z
    df['l'] = df.apply(
        lambda t: ((t['Cell X'] - t['X']) ** 2 + (t['Cell Y'] - t['Y']) ** 2 + (t['Cell Z'] - t['Z']) ** 2) ** .5
        , axis=1)  # 06
    df['log1'] = df['Frequency Band'].map(fun1)
    df['log2'] = df['Height'].map(fun2)  # 07
    df['log3'] = df.apply(fun3, axis=1)  # 08
    df['pl'] = df.apply(lambda t: 46.3 + t['log1'] + t['log2'] + t['log3'] - .001, axis=1)  # 09
    df['th_d'] = df['Mechanical Downtilt'] + df['Electrical Downtilt']  # 10
    df['A_mean'] = (df['Cell Altitude'] + df['Altitude']) / 2  # 11
    df['d_hv'] = df.apply(lambda t: t['Height'] - t['d'] * np.tan(t['th_d'] / 180 * np.pi), axis=1)  # 12
    df['temp_d'] = df['d']
    df.loc[df['temp_d'] == 0, 'temp_d'] = 1.0
    df['dth_h'] = df.apply(lambda t: np.abs(np.arccos((t['dx'] * np.sin(t['Azimuth'] / 180 * np.pi) + t['dy'] * np.cos(
        t['Azimuth'] / 180 * np.pi)) / t['temp_d']) / np.pi * 180), axis=1)  # 13

    df[df.isnull()] = .001

    feat1 = df[feat_cols_lx].values
    temp =  df[feat_cols_lb].values
    feat2 = (temp[:,0]==temp[:,1]).astype('float32').reshape(-1,1)  # 14

    return np.hstack((feat1, feat2))


# def get_data_label(path):
#     files = glob(path)
#     data, label = [], []
#     for i, f in enumerate(files):
#         df = pd.read_csv(f)
#         label.append(df.iloc[:,-1].values.reshape(-1,1))
#         data.append(extract_feature(df))
#         print(i)
#     return np.vstack(data), np.vstack(label)


if __name__ == '__main__':

    feat_cols = ['dy', 'd', 'RS Power', 'dx', 'dh', 'l', 'th_d',
                    'log3', 'log2', 'pl', 'A_mean', 'dth_h', 'd_hv', 'Index']

    path = './raw_dataset/train_set/*.csv'
    data = []
    files = glob(path)
    for i, f in enumerate(files):
        data.append(pd.read_csv(f))
    df_mer = pd.concat(data, axis=0, ignore_index=False)

    print('merge finished!')

    label = df_mer['RSRP'].values
    
    tasks = np.array_split(df_mer, 8)

    p = Pool(8)
    data = p.map(extract_feature, tasks)

    feat = np.vstack(data)

    # feat = extract_feature(data)

    train_df = pd.DataFrame(feat, columns = feat_cols)
    train_df['RSRP'] = label
    train_df.to_csv('train_data_new.csv')

