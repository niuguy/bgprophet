from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
from numpy import array
import random
import sys
import io
import argparse


from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.metrics import mean_absolute_error

import pandas as pd
import json
import pickle
import mlflow
from mlflow import log_metric

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
    pickle.dump(entries_df[['datetime', 'sgv']], open('data/entries_20396154_2.pkl', 'wb'), -1)


def prepare_input(sequence, n_steps):
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


def cnn(train_x, train_y, n_steps, n_features):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(train_x, train_y, batch_size=128,
          epochs=1000)
    model.save('models/bg_predict_cnn_alpha.h5')
    print('Model saved')
    
    return model
    
def vanilla_lstm(train_x, train_y, n_steps, n_features):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_x, train_y,batch_size=128,epochs=200)
    model.save('models/bg_predict_v_lstm.h5')
    print('Model saved')
    return model

def bidirectional_lstm(train_x, train_y, n_steps, n_features):
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_x, train_y,batch_size=128,epochs=200)
    model.save('models/bg_predict_b_lstm.h5')
    print('Model saved')
    return model  

def stacked_lstm(train_x, train_y, n_steps, n_features):
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_x, train_y,batch_size=128,epochs=200)
    model.save('models/bg_predict_s_lstm.h5')
    print('Model saved')
    return model

def cnn_lstm(train_x, train_y, n_steps, n_features):
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_x, train_y,batch_size=128,epochs=200)
    model.save('models/bg_predict_s_lstm.h5')
    print('Model saved')
    return model


def run(model_name):
    with mlflow.start_run():
        test_date = '2018-12-01'
        entries_df = pickle.load(open('data/entries_20396154_2.pkl', 'rb'))
        bg_list = entries_df['sgv']
        train_bg_list = entries_df[entries_df['datetime']<test_date].reset_index()['sgv']
        print('len(train_bg_list)', len(train_bg_list))
        test_bg_list = entries_df[entries_df['datetime']>=test_date].reset_index()['sgv']
        n_steps = 12
        n_features = 1
        train_x, train_y = prepare_input(train_bg_list, n_steps)
        test_x, test_y = prepare_input(train_bg_list, n_steps)
        train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], n_features))

        if model_name == 'cnn':
            model = cnn(train_x, train_y, n_steps, n_features)    
        if model_name == 'v_lstm':
            model = vanilla_lstm(train_x, train_y, n_steps, n_features)
        if model_name == 'b_lstm':
            model = bidirectional_lstm(train_x, train_y, n_steps, n_features)
        if model_name == 's_lstm':
            model = bidirectional_lstm(train_x, train_y, n_steps, n_features)



        next_bgs = []
        pred_inputs = []
        for i in range(0, len(test_bg_list) - n_steps, 1):
            pred_input = test_x[i:i+n_steps]
            print('pred_input', pred_input)
            pred_inputs.append(pred_input)
            pred_input = pred_input.reshape((1, n_steps, n_features))
            next_bgs.append(model.predict(pred_input, verbose=0)[0])
        mae = mean_absolute_error(pred_inputs, next_bgs)    
        pickle.dump(next_bgs, open('results/predict_'+model_name+'_'+test_date+'.pkl', 'wb'), -1)
        
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("mae", mae)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="Choose from cnn,v_lstm, b_lstm, s_lstm, cnn_lstm, conv_lstm", action = 'store', default="v_lstm", type = str)
    args = parser.parse_args()
    run(args.model_name)
#     save_file()
#     run_lstm()
#     run_cnn()


