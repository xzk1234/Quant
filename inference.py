from TendencyPredict import *
from load_data_from_JQuant import create_table

parser = argparse.ArgumentParser(description='Quant')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--display_step', type=int, default=100)
parser.add_argument('--n_steps_encoder', type=int, default=10)
parser.add_argument('--n_hidden_encoder', type=int, default=128)
parser.add_argument('--n_input_decoder', type=int, default=1)
parser.add_argument('--n_steps_decoder', type=int, default=10)
parser.add_argument('--n_hidden_decoder', type=int, default=128)
parser.add_argument('--n_classes', type=int, default=2)
args = parser.parse_args()
'''
data_dir = './stock_data/2020_all_stock_return.csv'
data = pd.read_csv(data_dir, encoding='gbk')
total_index = data.columns[1:]
conn = psycopg2.connect(database='postgres', user='kline', password='kline@tushare', host='47.107.103.195', port='5432')
curs = conn.cursor()
create_table(curs, conn, 'price_new', total_index)
create_table(curs, conn, 'return_new', total_index)
'''
'''
stock_codes = np.load('stock_codes.npy')
stock_num = len(stock_codes)
conn = psycopg2.connect(database='postgres', user='kline', password='kline@tushare', host='47.107.103.195', port='5432')
curs = conn.cursor()
columns = []
for stock_code in stock_codes:
    model_path = 'models/' + stock_code[:6] + '_DA_RNN/'
    if os.path.exists(model_path + 'mean.npy'):
        columns.append(stock_code)
create_table(curs, conn, 'predict_new', columns)
'''

Tend_model = TendencyPredict(args, sql_price_name='price_new', sql_return_name='return_new', sql_predict_name='predict_new')

Tend_model.inference()