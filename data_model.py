import numpy as np
import os
import pandas as pd
import random
import time
import datetime
from sklearn import preprocessing
random.seed(time.time())
from sklearn import decomposition
from skimage.restoration._denoise import _sigma_est_dwt
import pywt
import matplotlib.pyplot as plt
from numpy import mean, absolute

def wavedf(df, wavelet="db38", level=2, sigma_type='donoho', plot=False, title=None, mode='per'):
    x = df.reset_index()
    for c in x.columns:
        if c != 'index':
            tmp = waveletSmooth(x[c].tolist(), wavelet=wavelet, level=level, sigma_type=sigma_type, plot=plot,
                                title=title, mode=mode)
            if len(tmp) > len(x[c]):
                # print(c, len(tmp),len(x[c]))
                x[c] = tmp[:-1]
            else:
                x[c] = tmp
    if 'index' in x.columns.tolist():
        del x['index']
    x = x.reset_index(level=1, drop=True)
    return x


def mad(data, axis=None):
    return mean(absolute(data - mean(data, axis)), axis)


def waveletSmooth(x, wavelet="db4", level=None, sigma_type='donoho', plot=False, title=None, mode='per'):
    if level:
        if len(x) < 2 ** level:
            ''' you don't have enough data to proceed. no smoothing'''
            return x

    try:
        # coeff = pywt.wavedec(x, wavelet=wavelet,mode=mode)
        coeff = pywt.wavedec(x, wavelet=wavelet, level=level, mode=mode)
    except:
        ''' too little info for specific level. just default max_level and donoho'''
        coeff = pywt.wavedec(x, wavelet=wavelet, mode=mode)
        sigma_type = 'donoho'

    if sigma_type and (sigma_type == 'donoho'):
        coeff[1:] = (pywt.threshold(i, value=_sigma_est_dwt(i), mode="soft") for i in coeff[1:])
    elif sigma_type and (sigma_type == 'mad'):
        sigma = mad(coeff[1])
        uthresh = sigma * np.sqrt(2 * np.log(len(x)))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
    elif sigma_type and (sigma_type == 'SURE'):
        ''' Stein’s Unbiased Risk Estimate(SURE) '''
        n = len(x)
        uthresh = np.sqrt(2 * np.log(n * np.log2(n)))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
    else:
        coeff[1:] = (pywt.threshold(i, value=np.std(i) / 2, mode='soft') for i in coeff[1:])

    y = pywt.waverec(coeff, wavelet=wavelet, mode=mode)

    if plot:
        f, ax = plt.subplots()
        plt.plot(x, color="b", alpha=0.5)
        plt.plot(y, color="b")
        if title:
            ax.set_title(title)
        ax.set_xlim((0, len(y)))

    return y


class StockDataSet(object):
    def __init__(self,
                 stock_sym,
                 input_size=1,
                 num_steps=30,
                 test_ratio=0.1,
                 normalized=True,
                 y_normalized=False,
                 pca=True,
                 close_price_only=True,
                 fwd_ret=5,
                 factors = None,
                 wave=True):

        self.stock_sym = stock_sym
        self.input_size = input_size
        self.num_steps = num_steps
        self.test_ratio = test_ratio
        self.close_price_only = close_price_only
        self.normalized = normalized
        self.y_normalized = y_normalized
        self.fwd_ret = fwd_ret
        self.scaler = None
        self.y_scaler = None
        self.pca = pca
        self.wave = wave


        # Read csv file
        raw_df = pd.read_csv(os.path.join("D:\\Users\\ftran_zim\\data_cna\\", "%s.csv" % stock_sym))

        if 'stock' in raw_df.columns:
            del raw_df['stock']

        if factors is None:
            factors = list(raw_df.columns).copy()
            factors.remove('date')
            self.factors = factors
        else:
            self.factors = factors

        # Merge into one sequence
        if close_price_only:
            self.raw_seq = raw_df['Close'].tolist()
            self.raw_seq = np.array(self.raw_seq)
            self.train_X, self.train_y, self.test_X, self.test_y = self._prepare_data_old(self.raw_seq)

        else:
            # self.raw_seq = [price for tup in raw_df[['Volume','Open','Close','High','Low']].values for price in tup]
            # raw_df.columns = ['date','open', 'high', 'low', 'close','volume']
            raw_df.date = [datetime.datetime.strptime(x,'%Y-%m-%d').date() for x in raw_df.date]
            raw_df = raw_df.sort_values(by='date',ascending=True)
            self.raw_seq = raw_df[self.factors]
            self.train_X, self.train_y, self.test_X, self.test_y = self._prepare_data(self.raw_seq)

    def info(self):
        return "StockDataSet [%s] train: %d test: %d" % (
            self.stock_sym, len(self.train_X), len(self.test_y))


    def _prepare_data(self, seq):
        # split into items of input_size

        if False:
            Y = seq.close.shift(-self.fwd_ret) / seq.close - 1
            Y = Y[:len(Y) - self.fwd_ret]
        else:
            Y = seq['Y'].copy()
            del seq['Y']

        if self.y_normalized:
            Y = Y / np.max(Y)


        print('processing data', self.stock_sym)
        seq = seq.replace([np.inf, -np.inf], np.nan)
        seq = seq.dropna()

        if self.wave:
            seq = wavedf(seq)

        if self.normalized:
            try:
                self.scaler = preprocessing.StandardScaler().fit(seq)
                seq = self.scaler.transform(seq)
                seq = seq[:seq.shape[0] - self.fwd_ret]
            except Exception as ex:
                print('Error while preparing the data', str(ex))

        if self.pca:
            pca_ = decomposition.PCA(n_components=self.input_size)
            pca_.fit(seq)
            seq = pca_.transform(seq)



        # split into groups of num_steps
        X = np.array([seq[i: i + self.num_steps] for i in range(len(seq) - self.num_steps)])
        y = np.array([[Y[i + self.num_steps - 1]] for i in range(len(seq) - self.num_steps)])

        train_size = int(len(X) * (1.0 - self.test_ratio))
        train_X, test_X = X[:train_size], X[train_size:]
        train_y, test_y = y[:train_size], y[train_size:]
        return train_X, train_y, test_X, test_y


    def _prepare_data_old(self, seq):
        # split into items of input_size
        seq = [np.array(seq[i * self.input_size: (i + 1) * self.input_size])
               for i in range(len(seq) // self.input_size)]

        if self.normalized:
            seq = [seq[0] / seq[0][0] - 1.0] + [curr / seq[i][-1] - 1.0 for i, curr in enumerate(seq[1:])]

        # split into groups of num_steps
        # todo: change it to have multi dimension and last column for Y
        X = np.array([seq[i: i + self.num_steps] for i in range(len(seq) - self.num_steps)])
        y = np.array([seq[i + self.num_steps] for i in range(len(seq) - self.num_steps)])

        train_size = int(len(X) * (1.0 - self.test_ratio))
        train_X, test_X = X[:train_size], X[train_size:]
        train_y, test_y = y[:train_size], y[train_size:]
        return train_X, train_y, test_X, test_y

    def generate_one_epoch(self, batch_size):
        num_batches = int(len(self.train_X)) // batch_size
        if batch_size * num_batches < len(self.train_X):
            num_batches += 1

        batch_indices = list(range(num_batches))
        random.shuffle(batch_indices)
        for j in batch_indices:
            batch_X = self.train_X[j * batch_size: (j + 1) * batch_size]
            batch_y = self.train_y[j * batch_size: (j + 1) * batch_size]
            assert set(map(len, batch_X)) == {self.num_steps}
            # print('===== BATCH DATA =====')
            # print(self.stock_sym,':' ,batch_X.shape, len(batch_y))
            yield batch_X, batch_y
