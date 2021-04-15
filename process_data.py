import numpy as np
import pandas as pd

data_dir = './stock_data/2020_all_stock_return.csv'
data = pd.read_csv(data_dir, encoding='gbk')

original_codes = data.columns[1:]
np.save('columns.npy', original_codes)

original_codes = np.load('columns.npy')
original_codes = list(original_codes)