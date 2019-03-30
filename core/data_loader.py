import math
import numpy as np
import pandas as pd
import json
import _pickle as pickle

class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, splitdate, cols,maxBG):
        dataframe = pickle.load(open(filename, 'rb'))
        self.maxBG = maxBG
        self.data_train = dataframe[dataframe['datetime']<splitdate].reset_index()[cols].values
        self.data_test  = dataframe[dataframe['datetime']>=splitdate].reset_index()[cols].values
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None

    def prepare_input(self, bg_list, maxlen, step=1):
        windows = []
        next_bgs = []
        for i in range(0, len(bg_list) - maxlen, step):
            windows.append(bg_list[i: i + maxlen])
            next_bgs.append(bg_list[i + maxlen])
        x = np.zeros((len(bg_list), maxlen, self.maxBG), dtype=np.bool)
        y = np.zeros((len(bg_list), self.maxBG), dtype=np.bool)
        for i, window in enumerate(windows):
            for t, bg in enumerate(window):
                x[i, t, bg] = 1
            y[i, next_bgs[i]] = 1
    
        return x,y

    def get_train_data(self, seq_len):
        
        return self.prepare_input(self.data_train, seq_len)

    
    def get_test_data(self, seq_len):
        return self.prepare_input(self.data_test, seq_len)
    
    def get_origin_test_target(self):
        return self.data_test

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i+seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        # print('window=', window)
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)