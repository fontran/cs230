import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import pandas as pd
import pprint

import tensorflow as tf
import tensorflow.contrib.slim as slim
from skimage.restoration._denoise import _sigma_est_dwt

from data_model import StockDataSet
from model_rnn import LstmRNN, k_LstmRNN
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

flags = tf.app.flags
flags.DEFINE_integer("stock_count", 50, "Stock count [100]") #200
flags.DEFINE_integer("num_steps", 80, "Num of steps [30]") #80
flags.DEFINE_integer("num_layers", 2, "Num of layer [1]") #2
flags.DEFINE_integer("lstm_size", 128, "Size of one LSTM cell [128]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [64]")
flags.DEFINE_float("test_ratio", 0.05, "Ratio for testing[0.05]")
flags.DEFINE_float("keep_prob", 0.8, "Keep probability of dropout layer. [0.8]")
flags.DEFINE_float("init_learning_rate", 0.001, "Initial learning rate at early stage. [0.001]")
flags.DEFINE_float("l2_reg", 0, "for weight regularization. [0.001]")
flags.DEFINE_float("corr_reg", 0, "for weight regularization. [1]")
flags.DEFINE_float("learning_rate_decay", 1, "Decay rate of learning rate. [0.99]")
flags.DEFINE_integer("init_epoch", 2, "Num. of epoches considered as early stage. [5]")
flags.DEFINE_integer("max_epoch", 10, "Total training epoches. [50]")
flags.DEFINE_integer("embed_size", None, "If provided, use embedding vector of this size. [None]")
flags.DEFINE_string("stock_symbol", None, "Target stock symbol [None]")
flags.DEFINE_integer("sample_size", 10, "Number of stocks to plot during training. [4]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_integer("fwd_ret", 20, "Number of days ahead to forcast. [5]")
flags.DEFINE_boolean("wave", True, "whether we would use wave filter [False]")
flags.DEFINE_boolean("keras", True, "whether we would use keras model or tensorflow model [False]")
flags.DEFINE_boolean("pca", False, "whether  [False]")
flags.DEFINE_string("model", "bao", "Target stock symbol [fon]")

if flags.FLAGS.pca == True:
    flags.DEFINE_integer("input_size", 10, "Input size [10]")
else:
    flags.DEFINE_integer("input_size", 80, "Input size [80]")

model_name = flags.FLAGS.model

if flags.FLAGS.pca:
    model_name += '_pca'

if flags.FLAGS.wave:
    model_name += '_wave'

if flags.FLAGS.l2_reg > 0:
    model_name += '_reg%s_' % str(flags.FLAGS.l2_reg).replace('.','_')

model_name += '_keep%s_' % str(flags.FLAGS.keep_prob).replace('.','_')

if flags.FLAGS.corr_reg > 0:
    model_name += '_corr'

if flags.FLAGS.keras:
    model_name = 'k2_' + model_name

print("Starting model", model_name)

flags.DEFINE_string("model_prefix",model_name,"the prefix of the model name")



FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

if not os.path.exists("logs"):
    os.mkdir("logs")


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def load_sp500(input_size, num_steps, k=None, target_symbol=None, test_ratio=0.05, fwd_ret=5,pca=True,wave=True):
    if target_symbol is not None:
        return [
            StockDataSet(
                target_symbol,
                input_size=input_size,
                num_steps=num_steps,
                test_ratio=test_ratio,close_price_only=False,fwd_ret=fwd_ret,pca=pca,wave=wave)
        ]

    # Load metadata of s & p 500 stocks
    data_dir = "D:\\Users\\ftran_zim\\data_cna\\"
    info = pd.read_csv(data_dir + "constituents-financials.csv",converters={'symbol':str})
    info = info.rename(columns={col: col.lower().replace(' ', '_') for col in info.columns})
    info['file_exists'] = info['symbol'].map(lambda x: os.path.exists(data_dir + "{}.csv".format(x)))
    print (info['file_exists'].value_counts().to_dict())

    info = info[info['file_exists'] == True].reset_index(drop=True)
    info = info.sort_values(by=['market_cap'], ascending=False).reset_index(drop=True)

    #filter bad one.
    info = info[~info.symbol.isin(['XOM','JNJ','JPM'])]
    if k is not None:
        info = info.head(k)

    # Generate embedding meta file
    info[['symbol', 'sector']].to_csv(os.path.join("logs/metadata.tsv"), sep='\t', index=False)

    return [
        StockDataSet(row['symbol'],
                     input_size=input_size,
                     num_steps=num_steps,
                     test_ratio=test_ratio,close_price_only=False,fwd_ret=fwd_ret,pca=pca,wave=True)
        for _, row in info.iterrows()]


def tf_model():
    pp.pprint(flags.FLAGS.__flags)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        rnn_model = LstmRNN(
            sess,
            FLAGS.stock_count,
            lstm_size=FLAGS.lstm_size,
            num_layers=FLAGS.num_layers,
            num_steps=FLAGS.num_steps,
            input_size=FLAGS.input_size,
            embed_size=FLAGS.embed_size,
            fwd_ret=FLAGS.fwd_ret,
            model_prefix=FLAGS.model_prefix,
        )

        show_all_variables()

        stock_data_list = load_sp500(
            FLAGS.input_size,
            FLAGS.num_steps,
            k=FLAGS.stock_count,
            target_symbol=FLAGS.stock_symbol,
            fwd_ret=FLAGS.fwd_ret,
            test_ratio=FLAGS.test_ratio,
            pca=FLAGS.pca,
            wave=FLAGS.wave
        )

        if FLAGS.train:
            rnn_model.train(stock_data_list, FLAGS)
        else:
            if not rnn_model.load()[0]:
                raise Exception("[!] Train a model first, then run test mode")

def get_datalist():
    stock_data_list = load_sp500(
        FLAGS.input_size,
        FLAGS.num_steps,
        k=FLAGS.stock_count,
        target_symbol=FLAGS.stock_symbol,
        fwd_ret=FLAGS.fwd_ret,
        test_ratio=FLAGS.test_ratio,
        pca=FLAGS.pca,
        wave=FLAGS.wave
    )

    return stock_data_list

def keras_model():

    stock_data_list = load_sp500(
        FLAGS.input_size,
        FLAGS.num_steps,
        k=FLAGS.stock_count,
        target_symbol=FLAGS.stock_symbol,
        fwd_ret=FLAGS.fwd_ret,
        test_ratio=FLAGS.test_ratio,
        pca=FLAGS.pca,
        wave=FLAGS.wave
    )

    rnn_model = k_LstmRNN(FLAGS)
    rnn_model.train(stock_data_list, FLAGS)


def main(_):

    if not FLAGS.keras:
        tf_model()
    else:
        keras_model()

if __name__ == '__main__':
    pp.pprint('here we go')
    tf.app.run()
