"""
@author: lilianweng
"""
import pandas as pd
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import random
import re
import shutil
import time
import tensorflow as tf
from statistics import mean
from scipy.stats import linregress
from scipy.stats import spearmanr
import datetime
import matplotlib.lines as lines
import matplotlib.pyplot as plt
from skimage.restoration._denoise import _sigma_est_dwt
from tensorflow.contrib.tensorboard.plugins import projector
import pywt
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, TensorBoard
from keras import layers, regularizers
from keras.layers.wrappers import Bidirectional
from keras.layers import Input, LSTM, Dense, Dropout, Conv1D, TimeDistributed, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import RMSprop, SGD, Adagrad, Nadam, Adadelta, Adamax,Adam
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_uniform
from custom_recurrents import AttentionDecoder
from keras.models import Model
from keras import backend as K
from sklearn.metrics import r2_score

import seaborn as sns
sns.set(style='ticks', palette='Set2')


def stock_loss(y_true, y_pred):
    loss = K.switch(K.less(y_true * y_pred, 0),
                    100 * K.abs(y_true - y_pred),
                    K.abs(y_true - y_pred)
                    )
    return K.mean(loss, axis=-1)


def plot_samples(preds, targets, figname, stock_sym=None, multiplier=1):
    def _flatten(seq):
        return np.array([x for y in seq for x in y])

    truths = _flatten(targets)[-200:]
    preds = (_flatten(preds) * multiplier)[-200:]
    days = range(len(truths))[-200:]

    plt.figure(figsize=(12, 6))
    plt.plot(days, truths, label='truth')
    plt.plot(days, preds, label='pred')
    plt.legend(loc='upper left', frameon=False)
    plt.xlabel("fig 1 : wavelet transformed ")
    plt.ylabel("normalized price")
    plt.ylim((min(truths), max(truths)))
    plt.grid(ls='--')

    if stock_sym:
        plt.title(stock_sym + " | Last %d days in test" % len(truths))

    plt.savefig(figname, format='png', bbox_inches='tight', transparent=True)
    plt.close()

    X = truths.flatten()
    y = preds.flatten()
    # second gr
    fig = plt.figure(figsize=(12, 6))
    plt.plot(X, y, 'o')
    l = lines.Line2D([0, 1], [0, 1], transform=fig.transFigure, figure=fig)
    fig.lines.extend([l])
    plt.legend(loc='upper left', frameon=False)
    plt.xlabel("Truth data")
    plt.ylabel("predicted data")
    plt.ylim((min(truths), max(truths)))
    plt.grid(ls='--')

    reg = linregress(X, y)
    plt.plot(X, X * reg.slope + reg.intercept, 'r', label='slope' + str(reg.slope))
    corr = spearmanr(X, y)

    if stock_sym:
        plt.title(
            stock_sym + " | Last %d days true vs predicted slope=%.2f corr=%.2f" % (len(truths), reg.slope, corr[0]))

    plt.savefig(figname.replace('.', '_scatter_s%.2f_c%.2f.' % (reg.slope, corr[0])), format='png', bbox_inches='tight',
                transparent=True)
    plt.close()


def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)**2) - mean(xs**2)))
    return m

class LstmRNN(object):
    def __init__(self, sess, stock_count,
                 lstm_size=128,
                 num_layers=1,
                 num_steps=30,
                 input_size=1,
                 embed_size=None,
                 logs_dir="logs",
                 plots_dir="images",
                 stats_dir="stats",
                 model_prefix="stock_rnn_lstm",
                 fwd_ret=5,
                 batch_size=64):
        """
        Construct a RNN model using LSTM cell.

        Args:
            sess:
            stock_count (int): num. of stocks we are going to train with.
            lstm_size (int)
            num_layers (int): num. of LSTM cell layers.
            num_steps (int)
            input_size (int)
            keep_prob (int): (1.0 - dropout rate.) for a LSTM cell.
            embed_size (int): length of embedding vector, only used when stock_count > 1.
            checkpoint_dir (str)
        """
        self.sess = sess
        self.stock_count = stock_count

        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.input_size = input_size

        self.use_embed = (embed_size is not None) and (embed_size > 0)
        self.embed_size = embed_size or -1

        self.logs_dir = logs_dir
        self.stats_dir = stats_dir
        self.plots_dir = plots_dir
        self.fwd_ret = fwd_ret
        self.model_prefix = model_prefix
        self.batch_size = batch_size
        self.build_graph()

    def build_graph(self):
        """
        The model asks for five things to be trained:
        - learning_rate
        - keep_prob: 1 - dropout rate
        - symbols: a list of stock symbols associated with each sample
        - input: training data X
        - targets: training label y
        """
        # inputs.shape = (number of examples, number of input, dimension of each input).
        self.learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
        self.keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")
        self.l2_reg = tf.placeholder(tf.float32, None, name="l2_reg")
        self.corr_reg = tf.placeholder(tf.float32, None, name="corr_reg")

        # Stock symbols are mapped to integers.
        self.symbols = tf.placeholder(tf.int32, [None, 1], name='stock_labels')

        self.inputs = tf.placeholder(tf.float32, [None, self.num_steps, self.input_size], name="inputs")
        self.targets = tf.placeholder(tf.float32, [None, 1], name="targets")
        self.sample_size = tf.placeholder(tf.int32, None, name="sample_size")

        def _corr_lost(name=None):
            print('computing corr lost',self.pred,self.targets)
            xy = tf.concat([tf.transpose(self.pred), tf.transpose(self.targets)], axis=0)
            mean_t = tf.reduce_mean(xy, axis=1, keep_dims=True)
            cov_t = (xy - mean_t) @ tf.transpose((xy - mean_t)) / (39699 - 1)
            cov2_t = tf.diag(1 / tf.sqrt(tf.diag_part(cov_t)+0.00001))
            corr = cov2_t @ cov_t @ cov2_t
            return tf.reduce_mean(corr,name=name)

        def _create_one_cell():
            lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_size, state_is_tuple=True, initializer=tf.contrib.layers.xavier_initializer())
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
            # lstm_cell = tf.layers.BatchNormalization(lstm_cell)
            return lstm_cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [_create_one_cell() for _ in range(self.num_layers)],
            state_is_tuple=True
        ) if self.num_layers > 1 else _create_one_cell()

        if self.embed_size > 0 and self.stock_count > 1:
            self.embed_matrix = tf.Variable(
                tf.random_uniform([self.stock_count, self.embed_size], -1.0, 1.0),
                name="embed_matrix"
            )

            # stock_label_embeds.shape = (batch_size, embedding_size)
            stacked_symbols = tf.tile(self.symbols, [1, self.num_steps], name='stacked_stock_labels')
            stacked_embeds = tf.nn.embedding_lookup(self.embed_matrix, stacked_symbols)

            # After concat, inputs.shape = (batch_size, num_steps, input_size + embed_size)
            self.inputs_with_embed = tf.concat([self.inputs, stacked_embeds], axis=2, name="inputs_with_embed")
            self.embed_matrix_summ = tf.summary.histogram("embed_matrix", self.embed_matrix)

        else:
            self.inputs_with_embed = tf.identity(self.inputs)
            self.embed_matrix_summ = None

        print( "inputs.shape:", self.inputs.shape)
        print( "inputs_with_embed.shape:", self.inputs_with_embed.shape)

        # Run dynamic RNN
        val, state_ = tf.nn.dynamic_rnn(cell, self.inputs_with_embed, dtype=tf.float32, scope="dynamic_rnn")

        # Before transpose, val.get_shape() = (batch_size, num_steps, lstm_size)
        # After transpose, val.get_shape() = (num_steps, batch_size, lstm_size)
        val = tf.transpose(val, [1, 0, 2])

        last = tf.gather(val, int(val.get_shape()[0]) - 1, name="lstm_state")
        ws = tf.get_variable(shape=[self.lstm_size, 1], name="w",
                             initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
        self.pred = tf.matmul(last, ws) + bias

        self.last_sum = tf.summary.histogram("lstm_state", last)
        self.w_sum = tf.summary.histogram("w", ws)
        self.b_sum = tf.summary.histogram("b", bias)
        self.pred_summ = tf.summary.histogram("pred", self.pred)

        self.t_vars = tf.trainable_variables()

        # self.loss = -tf.reduce_sum(targets * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
        self.lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.t_vars
                           if len(v.shape) == 2]) * self.l2_reg

        # corelation cost

        if False:
            xy = tf.multiply(self.pred, self.targets)
            wrong_sign = [tf.where(tf.greater(xy[i][0], 0),tf.constant(1,dtype=tf.float32)
                                   , tf.constant(100,dtype=tf.float32)) for i in tf.range(0,self.sample_size)]
            wrong_sign = tf.reshape(wrong_sign, shape=(self.sample_size, 1))
            x_y = self.pred - self.targets
            wrong_sign_cost = tf.multiply(x_y,wrong_sign)

        # self.loss_corr = _corr_lost(name='loss_corr') * self.corr_reg
        # multi = [1 if x >= 0 else 100 for x in tf.multiply(self.pred, self.targets)]
        # cost = tf.multiply((self.pred - self.targets), 1)
        self.loss = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse_train") \
                    + self.lossL2

        # self.loss = tf.reduce_mean(tf.square(wrong_sign_cost), name="loss_mse_train") \
        #             + self.lossL2 + self.loss_corr

        # self.optim = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, name="rmsprop_optim")
        self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, name="adamprop_optim")

        # Separated from train loss.
        self.loss_test = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse_test") \
                         + self.lossL2

        # self.loss_test = tf.reduce_mean(tf.square(wrong_sign_cost), name="loss_mse_test") \
        #                  + self.lossL2 + self.loss_corr

        self.loss_sum = tf.summary.scalar("loss_mse_train", self.loss)
        self.loss_test_sum = tf.summary.scalar("loss_mse_test", self.loss_test)
        self.learning_rate_sum = tf.summary.scalar("learning_rate", self.learning_rate)

        self.saver = tf.train.Saver()

    def train(self, dataset_list, config):
        """
        Args:
            dataset_list (<StockDataSet>)
            config (tf.app.flags.FLAGS)
        """
        assert len(dataset_list) > 0
        self.merged_sum = tf.summary.merge_all()

        # Set up the logs folder
        self.writer = tf.summary.FileWriter(os.path.join("./logs", self.model_name))
        self.writer.add_graph(self.sess.graph)

        if self.use_embed:
            # Set up embedding visualization
            # Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
            projector_config = projector.ProjectorConfig()

            # You can add multiple embeddings. Here we add only one.
            added_embed = projector_config.embeddings.add()
            added_embed.tensor_name = self.embed_matrix.name
            # Link this tensor to its metadata file (e.g. labels).

            shutil.copyfile(os.path.join(self.logs_dir, "metadata.tsv"),
                            os.path.join(self.model_logs_dir, "metadata.tsv"))
            added_embed.metadata_path = "metadata.tsv"

            # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
            # read this file during startup.
            projector.visualize_embeddings(self.writer, projector_config)

        tf.global_variables_initializer().run()

        # Merged test data of different stocks.
        merged_test_X = []
        merged_test_y = []
        merged_test_labels = []
        merged_train_X = []

        for label_, d_ in enumerate(dataset_list):
            merged_train_X += list(d_.train_X)
            merged_test_X += list(d_.test_X)
            merged_test_y += list(d_.test_y)
            merged_test_labels += [[label_]] * len(d_.test_X)

        merged_train_X = np.array(merged_train_X)
        merged_test_X = np.array(merged_test_X)
        merged_test_y = np.array(merged_test_y)
        merged_test_labels = np.array(merged_test_labels)

        print("len(merged_train_X) =", len(merged_train_X))
        print("len(merged_test_X) =", len(merged_test_X))
        print("len(merged_test_y) =", len(merged_test_y))
        print("len(merged_test_labels) =", len(merged_test_labels))

        test_data_feed = {
            self.learning_rate: 0.0,
            self.keep_prob: 1.0,
            self.inputs: merged_test_X,
            self.targets: merged_test_y,
            self.symbols: merged_test_labels,
            self.l2_reg: config.l2_reg,
            self.corr_reg: config.corr_reg,
            self.sample_size: len(merged_test_X)
        }

        global_step = 0

        num_batches = sum(len(d_.train_X) for d_ in dataset_list) // config.batch_size
        random.seed(time.time())

        # Select samples for plotting.
        sample_labels = range(min(config.sample_size, len(dataset_list)))
        sample_indices = {}
        for l in sample_labels:
            sym = dataset_list[l].stock_sym
            target_indices = np.array([
                i for i, sym_label in enumerate(merged_test_labels)
                if sym_label[0] == l])
            sample_indices[sym] = target_indices

        print("Start training for stocks:", [d.stock_sym for d in dataset_list])
        perf_log = pd.DataFrame(columns=['global_step','epoch','learning_rate','train_loss','test_loss','slope','corr','hitrate','slope_train','corr_train','hitrate_train'])
        timestamp = str(datetime.datetime.now().timestamp())
        for epoch in range(config.max_epoch):
            epoch_step = 0
            learning_rate = config.init_learning_rate * (
                config.learning_rate_decay ** max(float(epoch + 1 - config.init_epoch), 0.0)
            )
            # learning_rate = config.init_learning_rate
            for label_, d_ in enumerate(dataset_list):
                for batch_X, batch_y in d_.generate_one_epoch(config.batch_size):
                    global_step += 1
                    epoch_step += 1
                    batch_labels = np.array([[label_]] * len(batch_X))
                    train_data_feed = {
                        self.learning_rate: config.init_learning_rate ,
                        self.keep_prob: config.keep_prob,
                        self.l2_reg: config.l2_reg,
                        self.corr_reg: config.corr_reg,
                        self.inputs: batch_X,
                        self.targets: batch_y,
                        self.symbols: batch_labels,
                        self.sample_size:config.batch_size
                    }
                    train_loss, _, train_merged_sum, pred_train = self.sess.run(
                        [self.loss, self.optim, self.merged_sum, self.pred], train_data_feed)
                    self.writer.add_summary(train_merged_sum, global_step=global_step)

                    if np.mod(global_step, int(num_batches/2)) == 1:
                    # if np.mod(global_step, int(len(dataset_list) * 200 / config.input_size)) == 1:
                        test_loss, test_pred = self.sess.run([self.loss_test, self.pred], test_data_feed)

                        slope = linregress(merged_test_y.flatten(),test_pred.flatten()).slope
                        corr = spearmanr(merged_test_y.flatten(),test_pred.flatten())[0]
                        prod = np.multiply(merged_test_y.flatten(),test_pred.flatten())
                        hitrate = len(prod[prod>0])/len(prod)

                        slope_t = linregress(batch_y.flatten(), pred_train.flatten()).slope
                        corr_t = spearmanr(batch_y.flatten(), pred_train.flatten())[0]
                        prod_t = np.multiply(batch_y.flatten(), pred_train.flatten())
                        hitrate_t = len(prod_t[prod_t > 0]) / len(prod_t)

                        print("Step:%d [Epoch:%d] [Learning rate: %.6f] train_loss:%.6f test_loss:%.6f slop:%.3f corr:%.3f hitrate:%.3f" % (
                            global_step, epoch, learning_rate, train_loss, test_loss,slope,corr,hitrate))
                        print(">>>--------:[continue] slop_train:%.3f corr_train:%.3f hitrate_train:%.3f" % (
                            slope_t, corr_t, hitrate_t))

                        perf_log.loc[global_step - 1] = [global_step, epoch, learning_rate, train_loss, test_loss, slope,
                                                     corr, hitrate,slope_t,corr_t,hitrate_t]

                        try:
                            perf_path = os.path.join(self.model_stats_dir, "perf_log_{}.csv".format(timestamp))
                            perf_log.to_csv(perf_path, index=False)
                        except:
                            pass

                        # Plot samples
                        for sample_sym, indices in sample_indices.items():
                            image_path = os.path.join(self.model_plots_dir, "{}_epoch{:02d}_step{:04d}.png".format(
                                sample_sym, epoch, epoch_step))
                            sample_preds = test_pred[indices]
                            sample_truth = merged_test_y[indices]
                            multiplier = 5 #abs(np.mean(merged_test_y)/np.mean(test_pred))
                            self.plot_samples(sample_preds, sample_truth, image_path, stock_sym=sample_sym,multiplier=multiplier)

                        self.save(global_step)
        perf_path = os.path.join(self.model_stats_dir, "perf_log_{}.csv".format(timestamp))
        perf_log.to_csv(perf_path,index=False)
        final_pred, final_loss = self.sess.run([self.pred, self.loss], test_data_feed)

        # Save the final model
        self.save(global_step)
        return final_pred

    @property
    def model_name(self):
        name = "%s%d_step%d_input%d_%dlayers_%dstocks" % (self.model_prefix,
            self.lstm_size, self.num_steps, self.input_size,self.num_layers,self.stock_count)

        if self.embed_size > 0:
            name += "_embed%d" % self.embed_size

        return name

    @property
    def model_stats_dir(self):
        model_stats_dir = os.path.join(self.stats_dir, self.model_name)
        if not os.path.exists(model_stats_dir):
            os.makedirs(model_stats_dir)
        return model_stats_dir


    @property
    def model_logs_dir(self):
        model_logs_dir = os.path.join(self.logs_dir, self.model_name)
        if not os.path.exists(model_logs_dir):
            os.makedirs(model_logs_dir)
        return model_logs_dir

    @property
    def model_plots_dir(self):
        model_plots_dir = os.path.join(self.plots_dir, self.model_name)
        if not os.path.exists(model_plots_dir):
            os.makedirs(model_plots_dir)
        return model_plots_dir

    def save(self, step):
        model_name = self.model_name + ".model"
        self.saver.save(
            self.sess,
            os.path.join(self.model_logs_dir, model_name),
            global_step=step
        )

    def load(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.model_logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_logs_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter

        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


class k_LstmRNN(LstmRNN):
    def __init__(self, config,
                 logs_dir="logs",
                 plots_dir="images",
                 stats_dir="stats"
                 ):
        """
        Construct a RNN model using LSTM cell.

        Args:
            sess:
            stock_count (int): num. of stocks we are going to train with.
            lstm_size (int)
            num_layers (int): num. of LSTM cell layers.
            num_steps (int)
            input_size (int)
            keep_prob (int): (1.0 - dropout rate.) for a LSTM cell.
            embed_size (int): length of embedding vector, only used when stock_count > 1.
            checkpoint_dir (str)
        """
        self.sess = K.get_session()
        self.config = config
        self.stock_count = config.stock_count
        self.lstm_size = config.lstm_size
        self.num_layers = config.num_layers
        self.num_steps = config.num_steps
        self.input_size = config.input_size

        self.use_embed = (config.embed_size is not None) and (config.embed_size > 0)
        self.embed_size = config.embed_size or -1

        self.logs_dir = logs_dir
        self.stats_dir = stats_dir
        self.plots_dir = plots_dir
        self.fwd_ret = config.fwd_ret
        self.model_prefix = config.model_prefix
        self.batch_size = config.batch_size
        self.build_graph()

    def build_fon_graph(self):
        i = Input(shape=(self.num_steps, self.input_size))
        enc = LSTM(self.lstm_size, return_sequences=True, kernel_regularizer=regularizers.l2(self.config.l2_reg))(i)
        enc = LSTM(self.lstm_size, return_sequences=True, kernel_regularizer=regularizers.l2(self.config.l2_reg))(enc)
        # dec = AttentionDecoder(self.lstm_size, self.input_size)(enc)
        dec = LSTM(self.input_size, return_sequences=False, kernel_regularizer=regularizers.l2(self.config.l2_reg))(enc)
        dec = Dense(self.input_size,activation='linear',kernel_regularizer=regularizers.l2(self.config.l2_reg))(dec)
        dec = BatchNormalization()(dec)
        dec = LeakyReLU()(dec)
        dec = Dropout(1 - self.config.keep_prob)(dec)
        dec = Dense(self.input_size,activation='linear',kernel_regularizer=regularizers.l2(self.config.l2_reg))(dec)
        dec = BatchNormalization()(dec)
        dec = LeakyReLU()(dec)
        dec = Dropout(1 - self.config.keep_prob)(dec)

        y = Dense(1,kernel_regularizer=regularizers.l2(self.config.l2_reg),name='last_dense')(dec)
        self.model = Model(inputs=i, outputs=y)
        self.model.compile(optimizer=Adam(lr=self.config.init_learning_rate),
                                          loss='mse', metrics=['mse', 'acc'])
        self.model.summary()

    def build_bao_autoen_graph(self):
        i = Input(shape=(self.num_steps, self.input_size), name='encoder_input')
        enc = TimeDistributed(Dense(self.input_size,activation='relu',kernel_regularizer=regularizers.l2(self.config.l2_reg)))(i)
        enc = TimeDistributed(Dense(self.input_size,activation='relu',kernel_regularizer=regularizers.l2(self.config.l2_reg)))(enc)
        enc = TimeDistributed(Dense(self.input_size,activation='relu',kernel_regularizer=regularizers.l2(self.config.l2_reg)))(enc)
        enc = LSTM(self.lstm_size, return_sequences=True,kernel_regularizer=regularizers.l2(self.config.l2_reg))(enc)
        enc_out, state_h, state_c = LSTM(self.lstm_size, return_state=True,kernel_regularizer=regularizers.l2(self.config.l2_reg))(enc)
        enc_states = [state_h, state_c]
        dec_inputs = Input(shape=(self.num_steps, self.input_size), name='decoder_input')
        dec, _, _ = LSTM(self.lstm_size, return_sequences=False, return_state=True,kernel_regularizer=regularizers.l2(self.config.l2_reg))(dec_inputs, initial_state=enc_states)
        dec = Dense(self.input_size, activation='linear',kernel_regularizer=regularizers.l2(self.config.l2_reg) )(dec)
        dec = BatchNormalization()(dec)
        dec = LeakyReLU()(dec)
        dec = Dropout(1 - self.config.keep_prob)(dec)
        dec = Dense(self.input_size, activation='linear',kernel_regularizer=regularizers.l2(self.config.l2_reg))(dec)
        dec = BatchNormalization()(dec)
        dec = LeakyReLU()(dec)
        dec = Dropout(1 - self.config.keep_prob)(dec)
        y = Dense(1, name='last_dense')(dec)
        self.model = Model(inputs=[i, dec_inputs], outputs=y)
        self.model.compile(optimizer=Adam(lr=0.01), loss='mse', metrics=['mse', 'acc'])
        self.model.summary()

    def build_bao_graph(self):
        i = Input(shape=(self.num_steps, self.input_size))
        enc = TimeDistributed(Dense(self.input_size,activation='relu',kernel_regularizer=regularizers.l2(self.config.l2_reg)))(i)
        enc = TimeDistributed(Dense(self.input_size,activation='relu',kernel_regularizer=regularizers.l2(self.config.l2_reg)))(enc)
        enc = TimeDistributed(Dense(self.input_size,activation='relu',kernel_regularizer=regularizers.l2(self.config.l2_reg)))(enc)
        enc = LSTM(self.lstm_size, return_sequences=True, )(enc)
        dec = LSTM(self.lstm_size, return_sequences=False, )(enc)
        dec = Dense(self.input_size, activation='linear', )(dec)
        dec = BatchNormalization()(dec)
        dec = LeakyReLU()(dec)
        dec = Dropout(1 - self.config.keep_prob)(dec)
        dec = Dense(self.input_size, activation='linear')(dec)
        dec = BatchNormalization()(dec)
        dec = LeakyReLU()(dec)
        dec = Dropout(1 - self.config.keep_prob)(dec)

        y = Dense(1, name='last_dense')(dec)
        self.model = Model(inputs=i, outputs=y)
        self.model.compile(optimizer=Adam(lr=0.01), loss='mse', metrics=['mse', 'acc'])
        self.model.summary()

    def build_graph(self):
        if self.config.model == 'bao':
            self.build_bao_graph()
        elif self.config.model == 'bao_enc':
            self.build_bao_autoen_graph()
        else:
            self.build_fon_graph()

    def build_graph_(self):

        i = Input(shape=(self.num_steps, self.input_size))

        enc = Bidirectional(LSTM(self.lstm_size, return_sequences=True, kernel_regularizer=regularizers.l2(self.config.l2_reg)),
                            merge_mode='concat')(i)
        for _ in range(self.num_layers - 1):
            enc = Bidirectional(LSTM(self.lstm_size, return_sequences=True, kernel_regularizer=regularizers.l2(self.config.l2_reg)),
                                merge_mode='concat')(enc)
        dec = AttentionDecoder(self.lstm_size, self.input_size)(enc)
        for _ in range(self.num_layers - 1):
            dec = Bidirectional(LSTM(self.lstm_size, return_sequences=True, kernel_regularizer=regularizers.l2(self.config.l2_reg)))(dec)
            if self.config.keep_prob:
                dec = Dropout(1 - self.config.keep_prob)(dec)
        dec = Bidirectional(LSTM(self.lstm_size, return_sequences=False, kernel_regularizer=regularizers.l2(self.config.l2_reg)))(dec)
        x = dec

        x = Dense(1)(x)
        self.model = Model(inputs=i, outputs=x)
        self.model.compile(optimizer=Adam(lr=self.config.init_learning_rate),
                                          loss='mse', metrics=['mse', 'acc'])

    def train(self,dataset_list, config):
        """
        Args:
            dataset_list (<StockDataSet>)
            config (tf.app.flags.FLAGS)
        """
        assert len(dataset_list) > 0

        if self.use_embed:
            # Set up embedding visualization
            # Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
            projector_config = projector.ProjectorConfig()

            # You can add multiple embeddings. Here we add only one.
            added_embed = projector_config.embeddings.add()
            added_embed.tensor_name = self.embed_matrix.name
            # Link this tensor to its metadata file (e.g. labels).
            shutil.copyfile(os.path.join(self.logs_dir, "metadata.tsv"),
                            os.path.join(self.model_logs_dir, "metadata.tsv"))
            added_embed.metadata_path = "metadata.tsv"

            # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
            # read this file during startup.
            projector.visualize_embeddings(self.writer, projector_config)


        # Merged test data of different stocks.
        merged_test_X = []
        merged_test_y = []
        merged_test_labels = []
        merged_train_X = []
        merged_train_y = []

        for label_, d_ in enumerate(dataset_list):
            merged_train_X += list(d_.train_X)
            merged_train_y += list(d_.train_y)
            merged_test_X += list(d_.test_X)
            merged_test_y += list(d_.test_y)
            merged_test_labels += [[label_]] * len(d_.test_X)

        merged_train_X = np.array(merged_train_X)
        merged_train_y = np.array(merged_train_y)
        merged_test_X = np.array(merged_test_X)
        merged_test_y = np.array(merged_test_y)
        merged_test_labels = np.array(merged_test_labels)

        print("len(merged_train_X) =", len(merged_train_X))
        print("len(merged_train_y) =", len(merged_train_y))
        print("len(merged_test_X) =", len(merged_test_X))
        print("len(merged_test_y) =", len(merged_test_y))
        print("len(merged_test_labels) =", len(merged_test_labels))

        num_batches = sum(len(d_.train_X) for d_ in dataset_list) // config.batch_size
        random.seed(time.time())

        # Select samples for plotting.
        sample_labels = range(min(config.sample_size, len(dataset_list)))
        sample_indices = {}
        for l in sample_labels:
            sym = dataset_list[l].stock_sym
            target_indices = np.array([
                i for i, sym_label in enumerate(merged_test_labels)
                if sym_label[0] == l])
            sample_indices[sym] = target_indices

        print("Start training for stocks:", [d.stock_sym for d in dataset_list])
        perf_log = pd.DataFrame(columns=['global_step','epoch','learning_rate','train_mse','val_mse','slope','corr','hitrate','slope_train','corr_train','hitrate_train'])
        timestamp = str(datetime.datetime.now().timestamp())
        for epoch in range(config.max_epoch):
            epoch_step = 0
            learning_rate = config.init_learning_rate * (
                config.learning_rate_decay ** max(float(epoch + 1 - config.init_epoch), 0.0)
            )

            tensorboard = TensorBoard(log_dir=self.model_logs_dir, histogram_freq=1,
                                      write_graph=True, write_images=False)

            if self.config.model.find('enc') != -1:
                history = self.model.fit([merged_train_X,merged_train_X], merged_train_y,
                                epochs=1,
                                batch_size=self.batch_size, validation_split=0.05, verbose=1,callbacks=[tensorboard])
            else:
                history = self.model.fit(merged_train_X, merged_train_y,
                                epochs=1,
                                batch_size=self.batch_size, validation_split=0.05, verbose=1,callbacks=[tensorboard])

            train_mse = history.history['mean_squared_error'][0]
            val_mse = history.history['val_mean_squared_error'][0]

            if self.config.model.find('enc') != -1:
                pred_train = self.model.predict([merged_train_X,merged_train_X])
                test_pred = self.model.predict([merged_test_X,merged_test_X])
            else:
                pred_train = self.model.predict(merged_train_X)
                test_pred = self.model.predict(merged_test_X)


            slope = linregress(merged_test_y.flatten(),test_pred.flatten()).slope
            corr = spearmanr(merged_test_y.flatten(),test_pred.flatten())[0]
            prod = np.multiply(merged_test_y.flatten(),test_pred.flatten())
            hitrate = len(prod[prod>0])/len(prod)

            slope_t = linregress(merged_train_y.flatten(), pred_train.flatten()).slope
            corr_t = spearmanr(merged_train_y.flatten(), pred_train.flatten())[0]
            prod_t = np.multiply(merged_train_y.flatten(), pred_train.flatten())
            hitrate_t = len(prod_t[prod_t > 0]) / len(prod_t)

            print("Step:%d [Epoch:%d] [Learning rate: %.6f] mse:%.6f val_mse:%.6f slop:%.3f corr:%.3f hitrate:%.3f" % (
                0, epoch, learning_rate, train_mse, val_mse,slope,corr,hitrate))
            print(">>>--------:[continue] slop_train:%.3f corr_train:%.3f hitrate_train:%.3f" % (
                slope_t, corr_t, hitrate_t))

            perf_log.loc[epoch - 1] = [0, epoch, learning_rate, train_mse, val_mse, slope,
                                         corr, hitrate,slope_t,corr_t,hitrate_t]

            try:
                perf_path = os.path.join(self.model_stats_dir, "perf_log_{}.csv".format(timestamp))
                perf_log.to_csv(perf_path, index=False)
            except:
                pass

            # Plot samples
            for sample_sym, indices in sample_indices.items():
                image_path = os.path.join(self.model_plots_dir, "{}_epoch{:02d}_step{:04d}.png".format(
                    sample_sym, epoch, epoch_step))
                sample_preds = test_pred[indices]
                sample_truth = merged_test_y[indices]
                multiplier = 1 #abs(np.mean(merged_test_y)/np.mean(test_pred))
                plot_samples(sample_preds, sample_truth, image_path, stock_sym=sample_sym,multiplier=multiplier)

            self.save(epoch)
            epoch_step += 1
        perf_path = os.path.join(self.model_stats_dir, "perf_log_{}.csv".format(timestamp))
        perf_log.to_csv(perf_path,index=False)
        # Save the final model
        self.save(epoch)

    def save(self, step):
        model_name = self.model_name + '_epoc' + str(step) + ".model"
        self.model.save(filepath=os.path.join(self.model_logs_dir, model_name))

    def load(self):
        past_model = sorted([self.model_logs_dir + x for x in os.listdir(self.model_logs_dir)], key=os.path.getctime)
        if past_model:
            model_path = max(past_model)
            self.model = load_model(model_path)
            print('reloaded model ', model_path)



