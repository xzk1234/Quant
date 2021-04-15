from TendencyPredict import *

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

Tend_model = TendencyPredict(args, sql_price_name='price_new', sql_return_name='return_new', sql_predict_name='predict_new')

stock_codes = np.load('stock_codes.npy')
code_num = len(stock_codes)
if args.gpu == '0':
    stock_codes = stock_codes[:int(code_num/4)]
elif args.gpu == '1':
    stock_codes = stock_codes[int(code_num/4):int(code_num*2/4)]
elif args.gpu == '2':
    stock_codes = stock_codes[int(code_num*2/4):int(code_num*3/4)]
else:
    stock_codes = stock_codes[int(code_num*3/4):]
wrong_code = []
count = 0
for stock_code in stock_codes:
    start_time = time.time()
    temp = Tend_model.train(args, stock_code)
    if temp is not None:
        wrong_code.append(temp)
    tf.reset_default_graph()
    count += 1
    end_time = time.time()
    print(count, end_time - start_time)
print(wrong_code)