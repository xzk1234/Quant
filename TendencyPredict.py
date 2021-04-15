import tensorflow as tf
import numpy as np
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell_impl as rnn_cell
import attention_encoder
import data_process as GD
from utils import *
import pandas as pd
import time
import argparse
from tqdm import tqdm
import psycopg2
from load_data_from_JQuant import load_once
from multiprocessing import Process, Manager
from jqdatasdk import *
auth('18795642715', 'Litianyi123')


def move_file(src_dir, dst_dir):
    listdir = os.listdir(src_dir)  # 获取文件和子文件夹
    for dirname in listdir:
        if os.path.isfile(os.path.join(src_dir, dirname)):  # 是文件
            shutil.copyfile(os.path.join(src_dir, dirname), os.path.join(dst_dir, dirname))
            os.remove(os.path.join(src_dir, dirname))


def prepropress(data, mean, stdev):
    inference_data = np.zeros((len(data) - 1, len(data[0])))
    for i in range(len(data) - 1):
        inference_data[i] = data[len(data) - 2 - i] / data[len(data) - 1 - i]
    inference_data = (inference_data - mean) / stdev
    return inference_data


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
            if str(pred_dict[item]) == 'nan':
                values += str(0.5) + ', '
            else:
                values += str(pred_dict[item]) + ', '
    keys = keys[:-2] + ')'
    values = values[:-2] + ')'
    insert_sql = 'insert into ' + table_name + keys + 'values ' + values
    curs.execute(insert_sql)
    # 提交数据
    conn.commit()


class Run(Process):
    def __init__(self, args, stock_codes, pred_dict, sql_price_name='price_new'):
        super().__init__()
        self.args = args
        self.stock_codes = stock_codes
        self.pred_dict = pred_dict
        self.sql_price_name = sql_price_name
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
        original_codes = np.load('./columns.npy', allow_pickle=True)  # codes from wukong
        load_codes = list(set(original_codes).intersection(set(index_codes)))
        load_codes.sort()
        load_codes.append(stock_code)
        sql = 'select time'
        for load_code in load_codes:
            sql += ', ' + load_code[-4:] + load_code[:6]
        sql += ' from ' + self.sql_price_name + ' order by time desc limit 11'
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


class TendencyPredict(object):
    def __init__(self, args, sql_price_name='price', sql_return_name='return', sql_predict_name='predict'):
        self.args = args
        self.price_name = sql_price_name
        self.return_name = sql_return_name
        self.predict_name = sql_predict_name

    def train(self, stock_code, model_path=None):
        '''
        :param stock_code: the stock to be trained
        :return: if successfully trained, return None, else the stock code will be returned
        the model and related files will be stored in model_path
        '''
        if model_path is None:
            model_path = 'models/' + stock_code[:6] + '_DA_RNN'
        generate_folder(model_path)
        logger = Logger(stock_code[:6] + '_DA_RNN', model_dir='models/')
        stock_dict = np.load('stock_dict.npy', allow_pickle=True).item()
        index_codes = stock_dict[stock_code]
        data_dir = './stock_data/2020_all_stock_return.csv'
        data = pd.read_csv(data_dir, encoding='gbk')

        original_codes = data.columns[1:]
        load_codes = list(set(original_codes).intersection(set(index_codes)))
        load_codes.sort()
        load_codes.append(stock_code)
        np.save(os.path.join(model_path, 'load_codes.npy'), load_codes)
        try:
            new_data = data[load_codes]
        except KeyError:
            return stock_code
        new_data.dropna(axis=1, how='any')
        data = np.array(new_data)
        return self.train_basic(data, model_path, logger, model_path)

    def finetune(self, stock_code, model_path=None, duplication_path=None, finetune_num=12):
        '''
        :param stock_code: the stock to be finetuned
        :return: if successfully finetuned, return None, else the stock code will be returned
        the latest model and related files will be stored in model_path, also a duplication of the former model will be
        stored in duplication_path
        '''
        if model_path is None:
            model_path = 'models/' + stock_code[:6] + '_DA_RNN'
        if duplication_path is None:
            duplication_path = 'duplication_models/' + stock_code[:6] + '_DA_RNN'
        generate_folder(duplication_path)
        move_file(model_path, duplication_path)
        logger = Logger(stock_code[:6] + '_DA_RNN', model_dir='models/')
        stock_dict = np.load('stock_dict.npy', allow_pickle=True).item()
        index_codes = stock_dict[stock_code]
        data_dir = './stock_data/2020_all_stock_return.csv'
        data = pd.read_csv(data_dir, encoding='gbk')
        data_codes = data.columns[1:]
        conn = psycopg2.connect(database='postgres', user='kline', password='kline@tushare', host='47.107.103.195',
                                port='5432')
        curs = conn.cursor()
        sql = 'select time'
        for data_code in data_codes:
            sql += ', ' + data_code[-4:] + data_code[:6]
        sql += ' from ' + self.return_name + ' order by time desc limit ' + str(finetune_num)
        curs.execute(sql)
        add_data = curs.fetchall()
        add_data.reverse()
        add_data = pd.DataFrame(np.array(add_data), columns=['Time'] + list(data_codes))
        new_data = pd.concat([data, add_data])[len(add_data):]

        original_codes = new_data.columns[1:]
        load_codes = list(set(original_codes).intersection(set(index_codes)))
        load_codes.sort()
        load_codes.append(stock_code)
        np.save(os.path.join(model_path, 'load_codes.npy'), load_codes)
        try:
            new_data = new_data[load_codes]
        except KeyError:
            return stock_code
        new_data.dropna(axis=1, how='any')
        data = np.array(new_data)
        self.train_basic(data, model_path, logger, duplication_path, is_fine_tune=True, training_iters=5000)

    def inference(self, process_num=15):
        '''
        :return: This is a script function. It will read relating data from JQuant per minute and save it on the
        database. Also, it will load relating model and save the prediction on the database. If the stock market is
        closed, it will sleep 60 second and restart again.
        '''
        stock_codes = np.load('stock_codes.npy')
        stock_num = len(stock_codes)
        conn = psycopg2.connect(database='postgres', user='kline', password='kline@tushare', host='47.107.103.195',
                                port='5432')
        curs = conn.cursor()
        last_price = None
        while True:
            try:
                total_start_time = time.time()
                start_time = time.time()
                hours = datetime.timedelta(days=0, hours=0)
                now = datetime.datetime.now() - hours
                rest_time = 60 - now.second
                last_price = load_once(now, last_price, price_name=self.price_name, return_name=self.return_name)
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
                    for i in range(process_num):
                        current_stock_codes = stock_codes[
                                              int(stock_num / process_num * i):int(stock_num / process_num * (i + 1))]
                        p_l.append(Run(self.args, current_stock_codes, pred_dict))
                    for p in p_l:
                        p.start()
                    for p in p_l:
                        p.join()
                    dt = datetime.timedelta(seconds=now.second, microseconds=now.microsecond)
                    now = now - dt
                    commit_row_data(curs, conn, pred_dict, self.predict_name, now)
                end_time = time.time()
                time_consuming2 = end_time - start_time
                print(time_consuming1, time_consuming2)
                total_end_time = time.time()
                print(total_end_time - total_start_time)
                sleep_time = rest_time - (total_end_time - total_start_time)
                while sleep_time < 0:
                    sleep_time += 60
                time.sleep(sleep_time)
            except KeyboardInterrupt:
                curs.close()
                break

    def train_basic(self, data, model_path, logger, source_model_path, is_fine_tune=False, training_iters=10000):
        total_start_time = time.time()
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        # Parameters
        batch_size = self.args.batch_size
        display_step = self.args.display_step

        # Network Parameters
        # encoder parameter
        n_steps_encoder = self.args.n_steps_encoder  # time steps
        n_hidden_encoder = self.args.n_hidden_encoder  # size of hidden units

        # decoder parameter
        n_input_decoder = self.args.n_input_decoder
        n_steps_decoder = self.args.n_steps_decoder
        n_hidden_decoder = self.args.n_hidden_decoder
        n_classes = self.args.n_classes  # size of the decoder output, 2 means classification task
        Data = GD.Input_data(batch_size, n_steps_encoder, n_steps_decoder, n_hidden_encoder, data, mode='return')
        np.save(os.path.join(model_path, 'mean.npy'), Data.mean)
        np.save(os.path.join(model_path, 'stdev.npy'), Data.stdev)

        n_input_encoder = Data.n_feature  # n_feature of encoder input
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
        # Define loss and optimizer
        # Define loss and optimizer
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(decoder_gt, 1))
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        predictions = tf.argmax(pred, 1)
        actuals = tf.argmax(decoder_gt, 1)
        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(predictions)
        zeros_like_predictions = tf.zeros_like(predictions)
        tn_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, ones_like_actuals),
                    tf.equal(predictions, ones_like_predictions)
                ),
                'float'
            )
        )
        tp_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, zeros_like_actuals),
                    tf.equal(predictions, zeros_like_predictions)
                ),
                'float'
            )
        )
        fn_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, zeros_like_actuals),
                    tf.equal(predictions, ones_like_predictions)
                ),
                'float'
            )
        )
        fp_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, ones_like_actuals),
                    tf.equal(predictions, zeros_like_predictions)
                ),
                'float'
            )
        )

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=decoder_gt, logits=pred))
        global_step = tf.contrib.framework.get_or_create_global_step()
        learning_rate = 0.001
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
        init = tf.global_variables_initializer()

        # save the model
        saver = tf.train.Saver(max_to_keep=3)
        best_train_loss = 100
        best_val_loss = 100
        best_test_loss = 100
        best_train_acc = 0
        best_val_acc = 0
        best_test_acc = 0
        # Launch the graph
        start_time = time.time()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init)
            if is_fine_tune:
                ckpt = tf.train.latest_checkpoint(source_model_path)
                saver.restore(sess, ckpt)

            # Keep training until reach max iterations
            for ii in range(training_iters):
                # the shape of batch_x is (batch_size, n_steps, n_input)
                batch_x, batch_y, prev_y, encoder_states = Data.next_batch()
                feed_dict = {encoder_input: batch_x, decoder_gt: batch_y, decoder_input: prev_y,
                             encoder_attention_states: encoder_states}
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict)
                # display the result
                if ii % display_step == 0:
                    end_time = time.time()
                    time_consuming = end_time - start_time
                    logger.log('Time consuming: {}s, {} iters per second'.format(time_consuming,
                                                                                 display_step / time_consuming))
                    start_time = time.time()
                    # Calculate batch loss
                    loss = sess.run(cost, feed_dict)
                    accuracy = sess.run(acc, feed_dict)
                    logger.log('Iter: {}, Minibatch Loss: {:.3f}, Minibatch Acc: {:.3f}'.format(ii, loss, accuracy))
                    # Val
                    val_x, val_y, val_prev_y, encoder_states_val = Data.validation()
                    feed_dict = {encoder_input: val_x, decoder_gt: val_y, decoder_input: val_prev_y,
                                 encoder_attention_states: encoder_states_val}
                    loss_val1 = sess.run(cost, feed_dict)
                    val_accuracy = sess.run(acc, feed_dict)
                    tp_val, tn_val, fp_val, fn_val = sess.run([tp_op, tn_op, fp_op, fn_op], feed_dict)
                    logger.log('Val Loss: {:.3f}, Val Acc: {:.3f}, Best Val Loss: {:.3f}, Best Val Acc: {:.3f}'.
                               format(loss_val1, val_accuracy, best_val_loss, best_val_acc))
                    logger.log('Val tp: {:.3f}, tn: {:.3f}, fp: {:.3f}, fn: {:.3f}'.format(tp_val, tn_val, fp_val,
                                                                                           fn_val))

                    # Test
                    test_x, test_y, test_prev_y, encoder_states_test = Data.testing()
                    feed_dict = {encoder_input: test_x, decoder_gt: test_y, decoder_input: test_prev_y,
                                 encoder_attention_states: encoder_states_test}
                    loss_test1 = sess.run(cost, feed_dict)
                    test_accuracy = sess.run(acc, feed_dict)
                    tp_test, tn_test, fp_test, fn_test = sess.run([tp_op, tn_op, fp_op, fn_op], feed_dict)
                    logger.log('Test Loss: {:.3f}, Test Acc: {:.3f}, Best Test Loss: {:.3f}, Best Test Acc: {:.3f}'.
                               format(loss_test1, test_accuracy, best_test_loss, best_test_acc))
                    logger.log(
                        'Test tp: {:.3f}, tn: {:.3f}, fp: {:.3f}, fn: {:.3f}'.format(tp_test, tn_test, fp_test,
                                                                                     fn_test))

                    logger.log('\n')
                    # save the parameters
                    if loss_val1 <= best_val_loss:
                        # testing
                        test_x, test_y, test_prev_y, encoder_states_test = Data.testing()
                        feed_dict = {encoder_input: test_x, decoder_gt: test_y, decoder_input: test_prev_y,
                                     encoder_attention_states: encoder_states_test}
                        loss_test1 = sess.run(cost, feed_dict)
                        test_accuracy = sess.run(acc, feed_dict)
                        tp_test, tn_test, fp_test, fn_test = sess.run([tp_op, tn_op, fp_op, fn_op], feed_dict)
                        logger.log(
                            'Test Loss: {:.3f}, Test Acc: {:.3f}, Best Test Loss: {:.3f}, Best Test Acc: {:.3f}'.
                            format(loss_test1, test_accuracy, best_test_loss, best_test_acc))
                        logger.log(
                            'Test tp: {:.3f}, tn: {:.3f}, fp: {:.3f}, fn: {:.3f}'.format(tp_test, tn_test, fp_test,
                                                                                         fn_test))

                        saver.save(sess, os.path.join(model_path, 'checkpoint'), global_step=global_step)
                        best_val_loss = loss_val1
                        best_test_loss = loss_test1
                        best_val_acc = val_accuracy
                        best_test_acc = test_accuracy
                        logger.log('\n')
                        logger.log('Model Saved!')
                        logger.log(
                            'Best Val Loss: {:.6f}, Best Val Acc: {:.6f}'.format(best_val_loss, best_val_acc))
                        logger.log(
                            'Best Test Loss: {:.6f}, Best Test Acc: {:.6f}'.format(best_test_loss, best_test_acc))
                        logger.log('\n')

                logger.log("Optimization Finished!")
        total_end_time = time.time()
        logger.log('Total time consuming: {}s'.format(total_end_time - total_start_time))
        return None