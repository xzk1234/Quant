import psycopg2
from jqdatasdk import *
import numpy as np
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell_impl as rnn_cell
import attention_encoder
import data_process as GD
from utils import *
import pandas as pd
import time
import datetime
auth('18795642715', 'Litianyi123')


def get_all_data(stocks, start_date, end_date, frequency):
    all_data = []
    for i in range(len(stocks)):
        df = get_price(stocks[i], start_date=start_date, end_date=end_date, frequency=frequency, fields=None,
                       skip_paused=False, fq='none')
        df.rename(columns={'close': stocks[i]}, inplace=True)
        df = df[stocks[i]]
        all_data.append(df)
    return all_data


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
    decoder_input = tf.split(decoder_input, n_steps_decoder,0 )

    # Encoder.
    with tf.variable_scope('encoder') as scope:
        encoder_cell = rnn_cell.BasicLSTMCell(n_hidden_encoder, forget_bias=1.0)
        encoder_outputs, encoder_state, attn_weights = attention_encoder.attention_encoder(encoder_input,
                                         encoder_attention_states, encoder_cell)

    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [tf.reshape(e, [-1, 1, encoder_cell.output_size]) for e in encoder_outputs]
    attention_states = tf.concat(top_states, 1)

    with tf.variable_scope('decoder') as scope:
        decoder_cell = rnn_cell.BasicLSTMCell(n_hidden_decoder, forget_bias=1.0)
        outputs, states = seq2seq.attention_decoder(decoder_input, encoder_state, attention_states, decoder_cell)

    return tf.matmul(outputs[-1], weights['out1']) + biases['out1'], attn_weights


def commit_row_data(curs, conn, df):
    df_total = df.reset_index()
    df_total.rename(columns={'index': 'Time'}, inplace=True)
    keys = '('
    values = '('
    for i in range(len(df_total.columns)):
        if i == 0:
            keys += df_total.columns[i] + ', '
            values += "'" + now.strftime("%Y-%m-%d %H:%M:%S") + "', "
        elif i == len(df_total.columns) - 1:
            keys += 'a' + df_total.columns[i].split('.')[0] + ')'
            values += str(df_total[df_total.columns[i]].values[0]) + ')'
        else:
            keys += 'a' + df_total.columns[i].split('.')[0] + ', '
            values += str(df_total[df_total.columns[i]].values[0]) + ', '
    insert_sql = 'insert into a' + stock_code.split('.')[0] + keys + 'values ' + values
    curs.execute(insert_sql)
    # 提交数据
    conn.commit()


def normalize(input, mean, stdev):
    return (input - mean) / stdev


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
stock_code = '600519.XSHG'
stock_dict = np.load('stock_dict.npy', allow_pickle=True).item()
index_codes = stock_dict[stock_code]
temp = pd.read_csv('./stock_data/600519_return_corr.csv')
total_index = temp.columns[1:]
conn = psycopg2.connect(database='postgres', user='kline', password='kline@tushare', host='47.107.103.195', port='5432')
curs = conn.cursor()
n_steps_encoder = 10
n_steps_decoder = 10
n_input_encoder = 100
n_input_decoder = 1
n_classes = 2
n_hidden_encoder = 128
n_hidden_decoder = 128
model_path = '600519_DA_RNN/'
mean = np.load('600519_DA_RNN/mean.npy')
stdev = np.load('600519_DA_RNN/stdev.npy')
# tf Graph input
encoder_input = tf.placeholder("float", [None, n_steps_encoder, n_input_encoder])
decoder_input = tf.placeholder("float", [None, n_steps_decoder, n_input_decoder])
decoder_gt = tf.placeholder("float", [None, n_classes])  # [1, 0] up, [0, 1] down
encoder_attention_states = tf.placeholder("float", [None, n_input_encoder, n_steps_encoder])
# Define weights
weights = {'out1': tf.Variable(tf.random_normal([n_hidden_decoder, n_classes]))}
biases = {'out1': tf.Variable(tf.random_normal([n_classes]))}
pred, attn_weights = RNN(encoder_input, decoder_input, weights, biases, encoder_attention_states)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
logit_softmax = tf.nn.softmax(pred)
hours = datetime.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=3, weeks=0)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(init)
    ckpt = tf.train.latest_checkpoint(model_path)
    saver.restore(sess, ckpt)
    while True:
        try:
            start_time = time.time()
            now = datetime.datetime.now() - hours
            dt = datetime.timedelta(seconds=now.second, microseconds=now.microsecond)
            now = now - dt
            all_data = get_all_data(total_index, frequency='minute', start_date=now, end_date=now)
            df_total = pd.concat(all_data, axis=1)
            end_time = time.time()
            time1 = end_time - start_time
            start_time = time.time()
            try:
                last_price
            except NameError:
                last_price = df_total.values.copy()
            temp = df_total.values.copy()
            if len(temp) == 0:
                print('No data')
                print('Get data time consuming: {}s'.format(time1))
                time.sleep(60)
                continue
            else:
                commit_row_data(curs, conn, df_total)
                temp1 = df_total.values / last_price
                df_total.iloc[:, :] = temp1
                last_price = temp
                df_total = df_total.reset_index()
                df_total.rename(columns={'index': 'Time'}, inplace=True)
                keys = '('
                values = '('
                for i in range(len(df_total.columns)):
                    if i == 0:
                        keys += df_total.columns[i] + ', '
                        values += "'" + now.strftime("%Y-%m-%d %H:%M:%S") + "', "
                    elif i == len(df_total.columns) - 1:
                        keys += 'b' + df_total.columns[i].split('.')[0] + ')'
                        values += str(df_total[df_total.columns[i]].values[0]) + ')'
                    else:
                        keys += 'b' + df_total.columns[i].split('.')[0] + ', '
                        values += str(df_total[df_total.columns[i]].values[0]) + ', '
                insert_sql = 'insert into b' + stock_code.split('.')[0] + keys + 'values ' + values
                curs.execute(insert_sql)
                # 提交数据
                conn.commit()
                sql = 'select * from b600519 order by time desc limit 10'
                curs.execute(sql)
                data = curs.fetchall()
                if len(data) < 10:
                    end_time = time.time()
                    print('Time consuming: {}s'.format(end_time - start_time))
                    time.sleep(60 - (end_time - start_time))
                else:
                    inference_data = np.zeros((len(data), len(data[0]) - 1))
                    for i in range(10):
                        for j in range(101):
                            inference_data[i][j] = float(data[9 - i][j + 1])
                    inference_data = normalize(inference_data, mean, stdev)
                    test_x = inference_data[:, :-1]
                    test_prev_y = inference_data[:, -1]
                    test_x = np.expand_dims(test_x, axis=0)
                    test_prev_y = np.expand_dims(test_prev_y, axis=0)
                    test_prev_y = np.expand_dims(test_prev_y, axis=-1)
                    encoder_states_test = np.swapaxes(test_x, 1, 2)
                    feed_dict = {encoder_input: test_x, decoder_input: test_prev_y,
                                 encoder_attention_states: encoder_states_test}
                    pred_test = sess.run(logit_softmax, feed_dict)
                    insert_sql = 'insert into predict' + stock_code.split('.')[
                        0] + '(Time, RiseProb) values ' + '(' + "'" + now.strftime("%Y-%m-%d %H:%M:%S") + "', " + str(
                        pred_test[0, 0]) + ')'
                    curs.execute(insert_sql)
                    conn.commit()
                    end_time = time.time()
                    time2 = end_time - start_time
                    print('Get data time consuming: {}s, Inference time consuming: {}s'.format(time1, time2))
                    time.sleep(60 - (end_time - start_time))
        except KeyboardInterrupt:
            curs.close()
            break
'''
sql = 'select * from b600519 order by time desc limit 10'
curs.execute(sql)
data = curs.fetchall()
stock_code = '600519.XSHG'
create_sql = 'create table predict' + stock_code.split('.')[0] + '(Time TEXT, RiseProb TEXT)'
curs.execute(create_sql)
# 提交数据
conn.commit()
inference_data = np.zeros((len(data), len(data[0]) - 1))
for i in range(10):
    for j in range(101):
        inference_data[i][j] = float(data[9-i][j+1])
print('finish')
n_steps_encoder = len(inference_data)
n_steps_decoder = len(inference_data)
n_input_encoder = inference_data.shape[1] - 1
n_input_decoder = 1
n_classes = 2
n_hidden_encoder = 128
n_hidden_decoder = 128
model_path = '600519_DA_RNN/'
# tf Graph input
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
    decoder_input = tf.split(decoder_input, n_steps_decoder,0 )

    # Encoder.
    with tf.variable_scope('encoder') as scope:
        encoder_cell = rnn_cell.BasicLSTMCell(n_hidden_encoder, forget_bias=1.0)
        encoder_outputs, encoder_state, attn_weights = attention_encoder.attention_encoder(encoder_input,
                                         encoder_attention_states, encoder_cell)

    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [tf.reshape(e, [-1, 1, encoder_cell.output_size]) for e in encoder_outputs]
    attention_states = tf.concat(top_states, 1)

    with tf.variable_scope('decoder') as scope:
        decoder_cell = rnn_cell.BasicLSTMCell(n_hidden_decoder, forget_bias=1.0)
        outputs, states = seq2seq.attention_decoder(decoder_input, encoder_state, attention_states, decoder_cell)

    return tf.matmul(outputs[-1], weights['out1']) + biases['out1'], attn_weights


pred, attn_weights = RNN(encoder_input, decoder_input, weights, biases, encoder_attention_states)
# Define loss and optimizer
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(decoder_gt, 1))
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()
logit_softmax = tf.nn.softmax(pred)
predictions = tf.argmax(pred, 1)
with tf.Session() as sess:
    sess.run(init)
    ckpt = tf.train.latest_checkpoint(model_path)
    saver.restore(sess, ckpt)
    test_x = inference_data[:, :-1]
    test_prev_y = inference_data[:, -1]
    test_x = np.expand_dims(test_x, axis=0)
    test_prev_y = np.expand_dims(test_prev_y, axis=0)
    test_prev_y = np.expand_dims(test_prev_y, axis=-1)
    encoder_states_test = np.swapaxes(test_x, 1, 2)
    feed_dict = {encoder_input: test_x, decoder_input: test_prev_y, encoder_attention_states: encoder_states_test}
    pred_test = sess.run(logit_softmax, feed_dict)
    print(pred_test[0, 0])
    now = datetime.datetime.now()
    dt = datetime.timedelta(seconds=now.second, microseconds=now.microsecond)
    now = now - dt
    insert_sql = 'insert into predict' + stock_code.split('.')[0] + '(Time, RiseProb) values ' + '(' + "'" + now.strftime("%Y-%m-%d %H:%M:%S") + "', " + str(pred_test[0, 0]) + ')'
    print(insert_sql)
    curs.execute(insert_sql)
    conn.commit()
    curs.close()
    conn.close()
'''