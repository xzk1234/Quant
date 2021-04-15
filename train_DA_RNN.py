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
from jqdatasdk import *
auth('18795642715', 'Litianyi123')


parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--gpu', type=str, default='0')
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


def train(args, stock_code):
    total_start_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Parameters
    is_training = args.is_training  # if False, then load the model from model_path and inference the test dataset
    training_iters = args.training_iters
    batch_size = args.batch_size
    display_step = args.display_step
    model_path = 'models/' + stock_code[:6] + '_DA_RNN'
    generate_folder(model_path)
    logger = Logger(stock_code[:6] + '_DA_RNN', model_dir='models/')
    is_fine_tune = args.is_fine_tune

    # Network Parameters
    # encoder parameter
    n_steps_encoder = args.n_steps_encoder  # time steps
    n_hidden_encoder = args.n_hidden_encoder  # size of hidden units

    # decoder parameter
    n_input_decoder = args.n_input_decoder
    n_steps_decoder = args.n_steps_decoder
    n_hidden_decoder = args.n_hidden_decoder
    n_classes = args.n_classes  # size of the decoder output, 2 means classification task
    stock_dict = np.load('stock_dict.npy', allow_pickle=True).item()
    index_codes = stock_dict[stock_code]
    data_dir = './stock_data/2020_all_stock_return.csv'
    data = pd.read_csv(data_dir, encoding='gbk')
    original_codes = data.columns
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
        if is_training:
            # generate_folder(model_path)
            sess.run(init)
            if is_fine_tune:
                ckpt = tf.train.latest_checkpoint(model_path)
                saver.restore(sess, ckpt)
            step = 1
            count = 1

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
                    logger.log('Time consuming: {}s, {} iters per second'.format(time_consuming, display_step / time_consuming))
                    start_time = time.time()
                    # Calculate batch loss
                    # Calculate batch loss
                    loss = sess.run(cost, feed_dict)
                    accuracy = sess.run(acc, feed_dict)
                    logger.log('Iter: {}, Minibatch Loss: {:.3f}, Minibatch Acc: {:.3f}'.format(ii, loss, accuracy))
                    # Val
                    val_x, val_y, val_prev_y, encoder_states_val = Data.validation()
                    feed_dict = {encoder_input: val_x, decoder_gt: val_y, decoder_input: val_prev_y,
                                 encoder_attention_states: encoder_states_val}
                    pred_val = sess.run(pred, feed_dict)
                    attn_weights_val = sess.run(attn_weights, feed_dict)
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
                    pred_test = sess.run(pred, feed_dict)
                    attn_weights_test = sess.run(attn_weights, feed_dict)
                    loss_test1 = sess.run(cost, feed_dict)
                    test_accuracy = sess.run(acc, feed_dict)
                    tp_test, tn_test, fp_test, fn_test = sess.run([tp_op, tn_op, fp_op, fn_op], feed_dict)
                    logger.log('Test Loss: {:.3f}, Test Acc: {:.3f}, Best Test Loss: {:.3f}, Best Test Acc: {:.3f}'.
                          format(loss_test1, test_accuracy, best_test_loss, best_test_acc))
                    logger.log('Test tp: {:.3f}, tn: {:.3f}, fp: {:.3f}, fn: {:.3f}'.format(tp_test, tn_test, fp_test,
                                                                                      fn_test))

                    logger.log('\n')
                    #save the parameters
                    if loss_val1 <= best_val_loss:
                        # testing
                        test_x, test_y, test_prev_y, encoder_states_test = Data.testing()
                        feed_dict = {encoder_input: test_x, decoder_gt: test_y, decoder_input: test_prev_y,
                                     encoder_attention_states: encoder_states_test}
                        pred_test = sess.run(pred, feed_dict)
                        attn_weights_test = sess.run(attn_weights, feed_dict)
                        loss_test1 = sess.run(cost, feed_dict)
                        test_accuracy = sess.run(acc, feed_dict)
                        tp_test, tn_test, fp_test, fn_test = sess.run([tp_op, tn_op, fp_op, fn_op], feed_dict)
                        logger.log('Test Loss: {:.3f}, Test Acc: {:.3f}, Best Test Loss: {:.3f}, Best Test Acc: {:.3f}'.
                              format(loss_test1, test_accuracy, best_test_loss, best_test_acc))
                        logger.log('Test tp: {:.3f}, tn: {:.3f}, fp: {:.3f}, fn: {:.3f}'.format(tp_test, tn_test, fp_test,
                                                                                           fn_test))

                        saver.save(sess, os.path.join(model_path, 'checkpoint'), global_step=global_step)
                        # best_train_loss = loss_train1
                        best_val_loss = loss_val1
                        best_test_loss = loss_test1
                        # best_train_acc = train_accuracy
                        best_val_acc = val_accuracy
                        best_test_acc = test_accuracy
                        # np.save(model_path + 'attn_weights_train.npy', np.array(attn_weights_train))
                        # np.save(os.path.join(model_path, 'attn_weights_test.npy'), np.array(attn_weights_test))
                        # np.save(os.path.join(model_path, 'attn_weights.val.npy'), np.array(attn_weights_val))
                        logger.log('\n')
                        logger.log('Model Saved!')
                        # print('Best Train Loss: {:.6f}, Best Train Acc: {:.6f}'.format(best_train_loss, best_train_acc))
                        logger.log('Best Val Loss: {:.6f}, Best Val Acc: {:.6f}'.format(best_val_loss, best_val_acc))
                        logger.log('Best Test Loss: {:.6f}, Best Test Acc: {:.6f}'.format(best_test_loss, best_test_acc))
                        logger.log('\n')

            logger.log("Optimization Finished!")
        else:
            sess.run(init)
            ckpt = tf.train.latest_checkpoint(model_path)
            saver.restore(sess, ckpt)
            test_x, test_y, test_prev_y, encoder_states_test = Data.testing()
            feed_dict = {encoder_input: test_x, decoder_gt: test_y, decoder_input: test_prev_y,
                         encoder_attention_states: encoder_states_test}
            pred_test = sess.run(pred, feed_dict)
            attn_weights_test = sess.run(attn_weights, feed_dict)
            np.save(model_path + 'attn_weights_test.npy', np.array(attn_weights_test))
            loss_test1 = sess.run(cost, feed_dict)
            test_accuracy = sess.run(acc, feed_dict)
            logger.log('Test Loss: {:.6f}, Test Acc: {:.6f}'.format(loss_test1, test_accuracy))
    total_end_time = time.time()
    logger.log('Total time consuming: {}s'.format(total_end_time - total_start_time))
    return None


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
    temp = train(args, stock_code)
    if temp is not None:
        wrong_code.append(temp)
    tf.reset_default_graph()
    count += 1
    end_time = time.time()
    print(count, end_time - start_time)
print(wrong_code)