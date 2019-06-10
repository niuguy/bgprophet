from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io



from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


import pandas as pd
import json
from pprint import pprint
import _pickle as pickle


def load_json(file_name):
    with open(file_name) as file:
        j_object = json.load(file)        
    return j_object

def load_df(file):
    return pd.DataFrame(file)

def save_file():
    entries_df = load_df(load_json('data/20396154_entries__to_2018-12-20.json'))
    entries_df['datetime'] = pd.to_datetime(entries_df['dateString'])
    entries_df.sort_values(by=['datetime'], ascending=True, inplace=True)
    # print(entries_df[['datetime', 'sgv']].head())
    pickle.dump(entries_df[['datetime', 'sgv']], open('data/entries_20396154.pkl', 'wb'), -1)

def prepare_input(bg_list, maxlen, step, maxBG):
    windows = []
    next_bgs = []
    print(bg_list)
    for i in range(0, len(bg_list) - maxlen, step):
        windows.append(bg_list[i: i + maxlen])
        next_bgs.append(bg_list[i + maxlen])
    # print('len(windows):', len(windows))
    # print('len(nex_BGs):', len(next_bgs))
    x = np.zeros((len(bg_list), maxlen, maxBG), dtype=np.bool)
    y = np.zeros((len(bg_list), maxBG), dtype=np.bool)
    for i, window in enumerate(windows):
        for t, bg in enumerate(window):
            x[i, t, bg] = 1
        y[i, next_bgs[i]] = 1
    
    print('len_x=',len(x), 'len_y=',len(y))
    return x,y


def prepare_input_cnn(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def sample(preds, temperature=1.0):
# helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def temp():
    results = pickle.load(open('results/predict_2018-12-01_+12', 'rb'))
    print(results[:10])


def run_cnn():
    est_date = '2018-12-01'
    entries_df = pickle.load(open('data/entries_20396154.pkl', 'rb'))
    bg_list = entries_df['sgv']
    maxBG = np.max(bg_list)+1
    train_bg_list = entries_df[entries_df['datetime']<test_date].reset_index()['sgv']
    print('len(train_bg_list)', len(train_bg_list))
    test_bg_list = entries_df[entries_df['datetime']>=test_date].reset_index()['sgv']
    n_steps = 12
    n_features = 1
    train_x, train_y = prepare_input(train_bg_list, n_steps)
    test_x, test_y = prepare_input_cnn(train_bg_list, n_steps)
    train_x = train_x.reshape((train_x.shape[0], train.shape[1], n_features))
    print('CNN model...')
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(maxlen, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(train_x, train_y, batch_size=128,
          epochs=60)
    model.save('models/bg_predict_cnn_alpha.h5')
    print('Model saved')
    
    next_bgs = []
    for i in range(0, len(test_bg_list) - maxlen, step):
        pred_input = test_x[i:i+maxlen]
        print('pred_input', pred_input)
        pred_input = pred_input.reshape((1, n_steps, n_features))
        next_bgs.append(sample(model.predict(pred_input, verbose=0)[0]))
    
    print('len(next_bgs)', len(next_bgs))
    pickle.dump(next_bgs, open('results/predict_'+test_date+'_cnn_+'+str(maxlen), 'wb'), -1)
    

def run():
    ## load data
    maxlen = 12
    step = 1
    test_date = '2018-12-01'
    entries_df = pickle.load(open('data/entries_20396154.pkl', 'rb'))
    bg_list = entries_df['sgv']
    maxBG = np.max(bg_list)+1
    train_bg_list = entries_df[entries_df['datetime']<test_date].reset_index()['sgv']
    print('len(train_bg_list)', len(train_bg_list))
    test_bg_list = entries_df[entries_df['datetime']>=test_date].reset_index()['sgv']
    train_x,train_y = prepare_input(train_bg_list, maxlen, step, maxBG)
    test_x,test_y = prepare_input(test_bg_list, maxlen, step, maxBG)


    
#     print('Build model...')
#     model = Sequential()
#     model.add(LSTM(128, input_shape=(maxlen, maxBG)))
#     model.add(Dense(maxBG, activation='softmax'))

#     optimizer = RMSprop(lr=0.01)
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#     model.fit(train_x, train_y,
#           batch_size=128,
#           epochs=60)
#     model.save('models/bg_predict_lstm_alpha.h5')

#     print('Model saved...')
    
    
    print('CNN model...')
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(maxlen, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_x.reshape((train_x.shape[0], train.shape[1], n_features)), train_y, batch_size=128,
          epochs=60)
    model.save('models/bg_predict_cnn_alpha.h5')
    print('Save model...')


    print('Predict on test data...')
    next_bgs = []
    for i in range(0, len(test_bg_list) - maxlen, step):
        pred_input = test_x[i:i+maxlen]
        print('pred_input', pred_input)
        next_bgs.append(sample(model.predict(pred_input, verbose=0)[0]))
    
    print('len(next_bgs)', len(next_bgs))
    pickle.dump(next_bgs, open('results/predict_'+test_date+'_cnn_+'+str(maxlen), 'wb'), -1)

    # def sample(preds, temperature=1.0):
    # # helper function to sample an index from a probability array
    #     preds = np.asarray(preds).astype('float64')
    #     preds = np.log(preds) / temperature
    #     exp_preds = np.exp(preds)
    #     preds = exp_preds / np.sum(exp_preds)
    #     probas = np.random.multinomial(1, preds, 1)
    #     return np.argmax(probas)

    # def on_epoch_end(epoch, _):
    # # Function invoked at end of each epoch. Prints generated text.
    #     print('----- Generating bgs after Epoch: %d' % epoch)

    #     start_index = random.randint(0, len(bg_list) - maxlen - 1)
    #     for diversity in [0.2, 0.5, 1.0, 1.2]:
    #         print('----- diversity:', diversity)

    #         generated = []
    #         window = bg_list[start_index: start_index + maxlen]
    #         generated.append(window)
    #         print('----- Generating with seed: "', window ,'"')
    #         # sys.stdout.write(generated)

    #         for i in range(maxBG):
    #             x_pred = np.zeros((1, maxlen, maxBG))
    #             for t, bg in enumerate(window):
    #                 x_pred[0, t, bg] = 1.

    #             preds = model.predict(x_pred, verbose=0)[0]
    #             next_bg = sample(preds, diversity)

    #             generated.append(next_bg)
    #             windows = windows[1:] + next_bg

    #             print(next_bg)


    # print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

    # model.fit(x, y,
    #       batch_size=128,
    #       epochs=60,
    #       callbacks=[print_callback])

if __name__ == "__main__":
    run_cnn()
    # temp()

