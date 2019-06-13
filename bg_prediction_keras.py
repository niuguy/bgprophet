from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import ConvLSTM2D


import numpy as np
from numpy import array
import random
import sys
import io
import argparse


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import pandas as pd
import json
import pickle
import mlflow
from mlflow import log_metric
import logging
from sklearn.metrics import mean_squared_error
from math import sqrt
import time
import math
import random

logging.basicConfig(filename='pred_results.log',level=logging.DEBUG)


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


def DTWDistance(s1, s2, w):
    DTW={}
    
    w = max(w, abs(len(s1)-len(s2)))
    
    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
  
    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
            
    return math.sqrt(DTW[len(s1)-1, len(s2)-1])

def LB_Keogh(s1,s2,r):

    LB_sum=0
    for ind,i in enumerate(s1):
        lower_bound = upper_bound = 0
        radius = s2[(ind-r if ind-r>=0 else 0):(ind+r)]
        if len(radius)!=0:
            lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
            upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2
    
    return math.sqrt(LB_sum)

def k_means_clust(data,num_clust,num_iter,w=5):
  
    centroids = random.sample(data, num_clust)
    counter=0
    for n in range(num_iter):
        print('round ', n)
        counter+=1
        assignments={}
        #assign data points to clusters
        for ind,seq_i in enumerate(data):
            min_dist=float('inf')
            closest_clust=None
            for c_ind,seq_j in enumerate(centroids):
                lb = LB_Keogh(seq_i,j,5)
                if lb<min_dist:
                    cur_dist=DTWDistance(seq_i,seq_j,w)
                    if cur_dist<min_dist:
                        min_dist=cur_dist
                        closest_clust=c_ind            
            assignments.setdefault(closest_clust,[])
            assignments[closest_clust].append(ind)
    
        #recalculate centroids of clusters
        for key in assignments:
            clust_sum=np.zeros(len(data[0]))
            for k in assignments[key]:
                clust_sum=clust_sum+data[k]
                
            centroids[key]=[m/len(assignments[key]) for m in clust_sum]
    
    return centroids,assignments


def cluster_input(sequence, window, y_step=0, cluster_id = 0):
    print('Start clustering')
    bg_segs_x, bg_segs_y = prepare_input(sequence, window, y_step)
    centroids, assignments = k_means_clust(bg_segs_x, 1, 1)
    pickle.dump(assignments, open('results/cluster_2_10.pkl', 'wb'), -1)
#     centroids, assignments = pickle.load(open('results/cluster_4.pkl', 'rb'))
        
    X= []
    y= []

    for i in assignments[cluster_id]:
        X.append(bg_segs_x[i])
        y.append(bg_segs_y[i])

    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.33, random_state=42) 

    print('End clustering')
    print('X_train.shape={0}'.format(X_train.shape))
    return np.array(X_train), np.array(X_test), y_train, y_test

def prepare_input(sequence, window, y_step=0):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + window
        if end_ix+y_step > len(sequence)-1:break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix+y_step]
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


def cnn(train_x, train_y, window, n_features):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(window, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(train_x, train_y, batch_size=128,
          epochs=150)
    model.save('models/bg_predict_cnn_alpha.h5')
    print('Model saved')
    
    return model
    
def vanilla_lstm(train_x, train_y, window, n_features):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(window, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_x, train_y,batch_size=128,epochs=200)
    model.save('models/bg_predict_v_lstm.h5')
    print('Model saved')
    return model

def bi_lstm(train_x, train_y, window, n_features):
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(window, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_x, train_y,batch_size=128,epochs=200)
    model.save('models/bg_predict_b_lstm.h5')
    print('Model saved')
    return model  

def stacked_lstm(train_x, train_y, window, n_features):
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(window, n_features)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_x, train_y,batch_size=128,epochs=200)
    model.save('models/bg_predict_s_lstm.h5')
    print('Model saved')
    return model

def cnn_lstm(train_x, train_y, window,n_seq,  n_features):
    
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None,int(window/n_seq), n_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_x, train_y,batch_size=128,epochs=200)

    print('Model saved')
    return model

def conv_lstm(train_x, train_y, window,n_seq, n_features):
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, int(window/n_seq), n_features)))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_x, train_y,batch_size=128,epochs=500)

    print('Model saved')
    return model
    


def run(model_name, window, y_step):
    with mlflow.start_run():
        print('start training {0}'.format(model_name))
                
        n_features = 1
        n_seq =2 
        entries_df = pickle.load(open('data/entries_20396154_2.pkl', 'rb'))
        bg_list = entries_df['sgv']
        
        test_date = '2018-12-01'
        train_bg_list = entries_df[entries_df['datetime']<test_date].reset_index()['sgv']
        print('len(train_bg_list)', len(train_bg_list))
        test_bg_list = entries_df[entries_df['datetime']>=test_date].reset_index()['sgv']
        print('len(test_bg_list)', len(test_bg_list))
        train_x, train_y = prepare_input(train_bg_list, window, y_step)
        test_x, test_y = prepare_input(test_bg_list, window, y_step)
        
#         train_x, test_x, train_y, test_y = cluster_input(bg_list, window, y_step)
        
        
        if model_name == 'cnn_lstm':
            train_x = train_x.reshape((train_x.shape[0], n_seq, int(window/n_seq), n_features))
        elif model_name == 'conv_lstm':
            train_x = train_x.reshape((train_x.shape[0], n_seq, 1, int(window/n_seq), n_features))
        else:
            train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], n_features))

        if model_name == 'cnn':
            model = cnn(train_x, train_y, window, n_features)    
        if model_name == 'v_lstm':
            model = vanilla_lstm(train_x, train_y, window, n_features)
        if model_name == 'b_lstm':
            model = bi_lstm(train_x, train_y, window, n_features)
        if model_name == 's_lstm':
            model = stacked_lstm(train_x, train_y, window, n_features)
        if model_name == 'cnn_lstm':
            model = cnn_lstm(train_x, train_y, window, n_seq, n_features)
        if model_name == 'conv_lstm':
            model = conv_lstm(train_x, train_y, window, n_seq, n_features)
        

        next_bgs = []
        pred_inputs = []
        for i in range(0, len(test_x), 1):
            pred_input = test_x[i]
            pred_inputs.append(pred_input[0])
            if model_name == 'cnn_lstm':
                pred_input = pred_input.reshape((1, n_seq, int(window/n_seq), n_features))
            elif model_name == 'conv_lstm':
                pred_input = pred_input.reshape((1, n_seq, 1, int(window/n_seq), n_features))
            else:
                pred_input = pred_input.reshape((1, window, n_features))
            next_bgs.append(model.predict(pred_input, verbose=0)[0])
        mae = mean_absolute_error(test_y, next_bgs)
        rms = sqrt(mean_squared_error(test_y, next_bgs))

        logging.info('model={0},window = {1}, y_tesp={2}, mae={3}, rms={4}'.format(model_name,window, y_step, mae, rms))
        
        
        pred_results = pd.DataFrame(data={'y_true':test_y, 'y_pred':next_bgs})
        pickle.dump(pred_results, open('results/predict_'+model_name+'_'+str(window)+'_'+str(y_step)+'.pkl', 'wb'), -1)
        
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("mae", mae)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", help="Choose from cnn,v_lstm, b_lstm, s_lstm, cnn_lstm, conv_lstm", action = 'store', default="v_lstm", type = str)
    parser.add_argument("--s", help="prediction steps", action = 'store', default="0", type = int)
    parser.add_argument("--w", help="window size", action = 'store', default="12", type = int)

    args = parser.parse_args()
    target_step = args.s
    window = args.w
    for y_step in range(6, -1, -1):
        for m in [ "v_lstm"]:
            run(m, window, y_step)
    
    
#     save_file()
#     run_lstm()
#     run_cnn()

#  "cnn","v_lstm", "b_lstm", "s_lstm", "cnn_lstm"


