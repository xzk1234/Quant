import os
import time
import pandas as pd

root = './models/'
model_roots = os.listdir(root)
results_dict = {}
for item in model_roots:
    log_dir = os.path.join(root, item, 'log.txt')
    with open(log_dir, 'r') as f:
        data = f.readlines()
        data = data[::-1]
        for line in data:
            if 'Best Test Acc' in line:
                acc = float(line[-5:])
                results_dict['Code' + item[:6]] = acc * 100
                break
df = pd.DataFrame.from_dict(results_dict, orient='index', columns=['Acc'])
df = df.reset_index().rename(columns={'index': 'Stock_code'})
print(df)
df.to_csv('all_test_results.csv', index=False, encoding='gbk')