import psycopg2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell_impl as rnn_cell
import attention_encoder
import data_process as GD
from utils import *
import pandas as pd
import time
import argparse
import datetime
from multiprocessing import Process, Manager
from load_data_from_JQuant import load_once
# from jqdatasdk import *
# auth('18795642715', 'Litianyi123')


parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--gpu', type=str, default='3')
parser.add_argument('--is_training', type=bool, default=True)
parser.add_argument('--training_iters', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--display_step', type=int, default=100)
parser.add_argument('--is_fine_tune', type=bool, default=False)
parser.add_argument('--n_steps_encoder', type=int, default=10)
parser.add_argument('--n_hidden_encoder', type=int, default=128)
parser.add_argument('--n_input_decoder', type=int, default=1)
parser.add_argument('--n_steps_decoder', type=int, default=10)
parser.add_argument('--n_hidden_decoder', type=int, default=128)
parser.add_argument('--n_classes', type=int, default=2)
args = parser.parse_args()


def prepropress(data, mean, stdev):
    inference_data = np.zeros((len(data) - 1, len(data[0])))
    for i in range(len(data) - 1):
        inference_data[i] = data[len(data) - 2 - i] / data[len(data) - 1 - i]
    inference_data = (inference_data - mean) / stdev
    return inference_data


def create_table(curs, conn, table_name, columns):
    sql = 'CREATE TABLE ' + table_name + '('
    sql += 'Time timestamp'
    for column in columns:
        sql += ', ' + column[-4:] + column[:6] + ' FLOAT8'
    sql += ')'
    curs.execute(sql)
    conn.commit()


def commit_row_data(curs, conn, pred_dict, table_name, now):
    pred_dict['time'] = now.strftime("%Y-%m-%d %H:%M:%S")
    keys = '('
    values = '('
    for item in pred_dict.keys():
        if item == 'time':
            keys += item + ', '
            values += "'" + pred_dict[item] + "', "
        else:
            keys += item[-4:] + item[:6] + ', '
            values += str(pred_dict[item]) + ', '
    keys = keys[:-2] + ')'
    values = values[:-2] + ')'
    insert_sql = 'insert into ' + table_name + keys + 'values ' + values
    curs.execute(insert_sql)
    # 提交数据
    conn.commit()


class Run(Process):
    def __init__(self, args, stock_codes, pred_dict):
        super().__init__()
        self.args = args
        self.stock_codes = stock_codes
        self.pred_dict = pred_dict
        self.conn = psycopg2.connect(database='postgres', user='kline', password='kline@tushare', host='47.107.103.195', port='5432')
        self.curs = self.conn.cursor()

    def run(self):
        for stock_code in self.stock_codes:
            pred_test = self.inference(stock_code)
            if pred_test is not None:
                self.pred_dict[stock_code] = pred_test[0, 0]
            tf.reset_default_graph()

    def inference(self, stock_code):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu
        model_path = 'models/' + stock_code[:6] + '_DA_RNN/'
        if not os.path.exists(model_path + 'mean.npy'):
            return None
        stock_dict = np.load('stock_dict.npy', allow_pickle=True).item()  # stock_dict: data from jquant dict
        index_codes = stock_dict[stock_code]
        original_codes = np.load('./columns.npy')  # codes from wukong
        load_codes = list(set(original_codes).intersection(set(index_codes)))
        load_codes.sort()
        load_codes.append(stock_code)
        sql = 'select time'
        for load_code in load_codes:
            sql += ', ' + load_code[-4:] + load_code[:6]
        sql += ' from price order by time desc limit 11'
        self.curs.execute(sql)
        data = self.curs.fetchall()
        data = np.array(data)
        data = data[:, 1:].astype('float')
        mean = np.load(model_path + 'mean.npy')
        stdev = np.load(model_path + 'stdev.npy')
        inference_data = prepropress(data, mean, stdev)
        n_steps_encoder = inference_data.shape[0]
        n_steps_decoder = inference_data.shape[0]
        n_input_encoder = inference_data.shape[1] - 1
        n_input_decoder = 1
        n_classes = 2
        n_hidden_encoder = 128
        n_hidden_decoder = 128
        encoder_input = tf.placeholder("float", [None, n_steps_encoder, n_input_encoder])
        decoder_input = tf.placeholder("float", [None, n_steps_decoder, n_input_decoder])
        decoder_gt = tf.placeholder("float", [None, n_classes])  # [1, 0] up, [0, 1] down
        encoder_attention_states = tf.placeholder("float", [None, n_input_encoder, n_steps_encoder])
        # Define weights
        weights = {'out1': tf.Variable(tf.random_normal([n_hidden_decoder, n_classes]))}
        biases = {'out1': tf.Variable(tf.random_normal([n_classes]))}

        def RNN(encoder_input, decoder_input, weights, biases, encoder_attention_states):
            # Prepare data shape to match `rnn` function requirements
            # Current data input shape: (batch_size, n_steps, n_input)
            # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

            # Prepare data for encoder
            # Permuting batch_size and n_steps
            encoder_input = tf.transpose(encoder_input, [1, 0, 2])
            # Reshaping to (n_steps*batch_size, n_input)
            encoder_input = tf.reshape(encoder_input, [-1, n_input_encoder])
            # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            encoder_input = tf.split(encoder_input, n_steps_encoder, 0)

            # Prepare data for decoder
            # Permuting batch_size and n_steps
            decoder_input = tf.transpose(decoder_input, [1, 0, 2])
            # Reshaping to (n_steps*batch_size, n_input)
            decoder_input = tf.reshape(decoder_input, [-1, n_input_decoder])
            # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            decoder_input = tf.split(decoder_input, n_steps_decoder, 0)

            # Encoder.
            with tf.variable_scope('encoder') as scope:
                encoder_cell = rnn_cell.BasicLSTMCell(n_hidden_encoder, forget_bias=1.0)
                encoder_outputs, encoder_state, attn_weights = attention_encoder.attention_encoder(encoder_input,
                                                                                                   encoder_attention_states,
                                                                                                   encoder_cell)

            # First calculate a concatenation of encoder outputs to put attention on.
            top_states = [tf.reshape(e, [-1, 1, encoder_cell.output_size]) for e in encoder_outputs]
            attention_states = tf.concat(top_states, 1)

            with tf.variable_scope('decoder') as scope:
                decoder_cell = rnn_cell.BasicLSTMCell(n_hidden_decoder, forget_bias=1.0)
                outputs, states = seq2seq.attention_decoder(decoder_input, encoder_state, attention_states,
                                                            decoder_cell)

            return tf.matmul(outputs[-1], weights['out1']) + biases['out1'], attn_weights

        pred, attn_weights = RNN(encoder_input, decoder_input, weights, biases, encoder_attention_states)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        logit_softmax = tf.nn.softmax(pred)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)
            ckpt = tf.train.latest_checkpoint(model_path)
            saver.restore(sess, ckpt)

            test_x = inference_data[:, :-1]
            test_prev_y = inference_data[:, -1]
            test_x = np.expand_dims(test_x, axis=0)
            test_prev_y = np.expand_dims(test_prev_y, axis=0)
            test_prev_y = np.expand_dims(test_prev_y, axis=-1)
            encoder_states_test = np.swapaxes(test_x, 1, 2)
            feed_dict = {encoder_input: test_x, decoder_input: test_prev_y,
                         encoder_attention_states: encoder_states_test}
            pred_test = sess.run(logit_softmax, feed_dict)
        return pred_test


# stock_codes = get_index_stocks('000300.XSHG')
# np.save('stock_codes.npy', stock_codes)
stock_codes = np.load('stock_codes.npy')
stock_num = len(stock_codes)
conn = psycopg2.connect(database='postgres', user='kline', password='kline@tushare', host='47.107.103.195', port='5432')
curs = conn.cursor()
'''
columns = []
for stock_code in stock_codes:
    model_path = 'models/' + stock_code[:6] + '_DA_RNN/'
    if os.path.exists(model_path + 'mean.npy'):
        columns.append(stock_code)
create_table(curs, conn, 'Predict', columns)
'''
last_price = None
while True:
    try:
        total_start_time = time.time()
        start_time = time.time()
        hours = datetime.timedelta(days=0, hours=4)
        now = datetime.datetime.now() - hours
        rest_time = 60 - now.second
        last_price = load_once(now, last_price)
        end_time = time.time()
        time_consuming1 = end_time - start_time
        start_time = time.time()
        if last_price is None:
            print('No data')
            end_time = time.time()
            time.sleep(60 - (end_time - start_time))
            continue
        with Manager() as manager:
            pred_dict = manager.dict()
            p_l = []
            process_num = 15
            for i in range(process_num):
                current_stock_codes = stock_codes[
                                      int(stock_num / process_num * i):int(stock_num / process_num * (i + 1))]
                p_l.append(Run(args, current_stock_codes, pred_dict))
            for p in p_l:
                p.start()
            for p in p_l:
                p.join()
            dt = datetime.timedelta(seconds=now.second, microseconds=now.microsecond)
            now = now - dt
            commit_row_data(curs, conn, pred_dict, 'Predict', now)
        end_time = time.time()
        time_consuming2 = end_time - start_time
        print(time_consuming1, time_consuming2)
        total_end_time = time.time()
        print(total_end_time - total_start_time)
        time.sleep(rest_time - (total_end_time - total_start_time))
    except KeyboardInterrupt:
        curs.close()
        break


