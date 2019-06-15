from __future__ import print_function

import argparse
import json
import logging
import math
import pickle
import random
from math import sqrt

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from keras.layers import Bidirectional
from keras.layers import ConvLSTM2D
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from numpy import array
from sklearn.model_selection import train_test_split

logging.basicConfig(filename='pred_results.log', level=logging.INFO)

INPUT_DIM = 2
TIME_STEPS = 20
# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False


def load_json(file_name):
    with open(file_name) as file:
        j_object = json.load(file)
    return j_object


def load_df(file):
    return pd.DataFrame(file)


def preprocess_file(json_path):
    # entries_df = load_df(load_json('data/20396154_entries__to_2018-12-20.json'))
    entries_df = load_df(load_json(json_path))
    entries_df['datetime'] = pd.to_datetime(entries_df['dateString'])
    entries_df.sort_values(by=['datetime'], ascending=True, inplace=True)
    # print(entries_df[['datetime', 'sgv']].head())
    pickle.dump(entries_df[['datetime', 'sgv']], open(json_path+'.pkl', 'wb'), -1)


def DTWDistance(s1, s2, w):
    DTW = {}

    w = max(w, abs(len(s1) - len(s2)))

    for i in range(-1, len(s1)):
        for j in range(-1, len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i - w), min(len(s2), i + w)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return math.sqrt(DTW[len(s1) - 1, len(s2) - 1])


def LB_Keogh(s1, s2, r):
    LB_sum = 0
    for ind, i in enumerate(s1):
        lower_bound = upper_bound = 0
        radius = s2[(ind - r if ind - r >= 0 else 0):(ind + r)]
        if len(radius) != 0:
            lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
            upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])

        if i > upper_bound:
            LB_sum = LB_sum + (i - upper_bound) ** 2
        elif i < lower_bound:
            LB_sum = LB_sum + (i - lower_bound) ** 2

    return math.sqrt(LB_sum)


def k_means_clust(data, num_clust, num_iter, w=5):
    centroids = random.sample(data, num_clust)
    counter = 0
    for n in range(num_iter):
        print('round ', n)
        counter += 1
        assignments = {}
        # assign data points to clusters
        for ind, seq_i in enumerate(data):
            min_dist = float('inf')
            closest_clust = None
            for c_ind, seq_j in enumerate(centroids):
                lb = LB_Keogh(seq_i, seq_j, 5)
                if lb < min_dist:
                    cur_dist = DTWDistance(seq_i, seq_j, w)
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_clust = c_ind
            assignments.setdefault(closest_clust, [])
            assignments[closest_clust].append(ind)

        # recalculate centroids of clusters
        for key in assignments:
            clust_sum = np.zeros(len(data[0]))
            for k in assignments[key]:
                clust_sum = clust_sum + data[k]

            centroids[key] = [m / len(assignments[key]) for m in clust_sum]

    return centroids, assignments


def cluster_input(sequence, window, y_step=0, cluster_id=0):
    print('Start clustering')
    bg_segs_x, bg_segs_y = prepare_input(sequence, window, y_step)
    centroids, assignments = k_means_clust(bg_segs_x, 1, 1)
    pickle.dump(assignments, open('results/cluster_2_10.pkl', 'wb'), -1)
    #     centroids, assignments = pickle.load(open('results/cluster_4.pkl', 'rb'))

    X = []
    y = []

    for i in assignments[cluster_id]:
        X.append(bg_segs_x[i])
        y.append(bg_segs_y[i])

    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.33, random_state=42)

    print('End clustering')
    print('X_train.shape={0}'.format(X_train.shape))
    return np.array(X_train), np.array(X_test), y_train, y_test


def prepare_input(seq, window, y_step=0):
    X, y = list(), list()
    in_start = 0
    for i in range(len(seq)):
        in_end = in_start + window
        out_end = in_end + y_step
        if out_end < len(seq):
            x_input = seq[in_start:in_end]
            # x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(seq[in_end:out_end])
        in_start += 1
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


def cnn(train_x, train_y, window, y_step,  n_features):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(window, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(y_step))
    model.compile(optimizer='adam', loss='mse')

    model.fit(train_x, train_y, batch_size=128,
              epochs=150)
    model.save('models/bg_predict_cnn_alpha.h5')
    print('Model saved')

    return model


def vanilla_lstm(train_x, train_y, window, y_step, n_features):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(window, n_features)))
    model.add(Dense(y_step))
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_x, train_y, batch_size=128, epochs=200)
    model.save('models/bg_predict_v_lstm.h5')
    print('Model saved')
    return model


def bi_lstm(train_x, train_y, window, y_step, n_features):
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(window, n_features)))
    model.add(Dense(y_step))
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_x, train_y, batch_size=128, epochs=200)
    model.save('models/bg_predict_b_lstm.h5')
    print('Model saved')
    return model


def stacked_lstm(train_x, train_y, window, y_step, n_features):
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(window, n_features)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(y_step))
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_x, train_y, batch_size=128, epochs=200)
    model.save('models/bg_predict_s_lstm.h5')
    print('Model saved')
    return model


def cnn_lstm(train_x, train_y, window, n_seq, y_step, n_features):
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),
                              input_shape=(None, int(window / n_seq), n_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(y_step))
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_x, train_y, batch_size=128, epochs=100)

    print('Model saved')
    return model


def conv_lstm(train_x, train_y, window, n_seq, y_step, n_features):
    model1 = Sequential()
    model1.add(ConvLSTM2D(filters=64, kernel_size=(1, 2), activation='relu',
                          input_shape=(n_seq, 1, int(window / n_seq), n_features)))
    model1.add(Flatten())
    model1.add(Dense(y_step))
    model1.compile(optimizer='adam', loss='mse')
    model1.fit(train_x, train_y, batch_size=128, epochs=100)
    #
    #
    #
    #
    # model2 = Sequential()                            # input_shape  = (batch, step, input_dim)
    # model2.add(Lambda(lambda x: K.mean(x, axis=2)))  # output_shape = (batch, step)
    # model2.add(Activation('softmax'))                # output_shape = (batch, step)model.fit(train_x, train_y,batch_size=64,epochs=10)
    # model2.add(RepeatVector(32))                 # output_shape = (batch, hidden, step)
    # model2.add(Permute(2, 1))                        # output_shape = (batch, step, hidden)print('Model saved')
    #
    # #The final model which gives the weighted sum:
    # model = Sequential()
    # model.add(merge([model1, model2], 'mul'))  # Multiply each element with corresponding weight a[i][j][k] * b[i][j]
    # model.add(TimeDistributed(merge('sum'))) # Sum the weighted elements.
    #
    # inputs = Input(shape=(n_seq, 1, int(window/n_seq), n_features))
    # lstm_units = 32
    # lstm_out = ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu')(inputs)
    # attention_mul = attention_3d_block(lstm_out)
    # attention_mul = Flatten()(attention_mul)
    # output = Dense(1, activation='sigmoid')(attention_mul)
    # model = Model(input=[inputs], output=output)
    # model.compile(optimizer='adam', loss='mse')

    print('conv_lstm complete')
    return model1


#
#
# def attention_3d_block(inputs):
#     # inputs.shape = (batch_size, time_steps, input_dim)
#     input_dim = int(inputs.shape[3])
#     a = Permute((3, 2, 1))(inputs)
#     # a = Reshape((input_dim, inputs.shape[0]))(a) # this line is not useful. It's just to know which dimension is what.
#     a = Dense(n, activation='softmax')(a)
#     if SINGLE_ATTENTION_VECTOR:
#         a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
#         a = RepeatVector(input_dim)(a)
#     a_probs = Permute((3, 2, 1), name='attention_vec')(a)
#     output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
#     return output_attention_mul
#

def evaluate_forcasts(actual, predicted):
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score

def flattern_result(actual, predicted):
    actual_out = [s[-1] for s in actual]
    pred_out = [s[-1] for s in predicted]
    return actual_out, pred_out
    # Pick the last one of each sets




def clarke_error_grid(ref_values, pred_values, y_period):
    # Checking to see if the lengths of the reference and prediction arrays are the same
    assert (len(ref_values) == len(pred_values)), "Unequal number of values (reference : {}) (prediction : {}).".format(
        len(ref_values), len(pred_values))

    # Checks to see if the values are within the normal physiological range, otherwise it gives a warning
    if max(ref_values) > 400 or max(pred_values) > 400:
        print(
            "Input Warning: the maximum reference value {} or the maximum prediction value {} exceeds the normal physiological range of glucose (<400 mg/dl).".format(
                max(ref_values), max(pred_values)))
    if min(ref_values) < 0 or min(pred_values) < 0:
        print(
            "Input Warning: the minimum reference value {} or the minimum prediction value {} is less than 0 mg/dl.".format(
                min(ref_values), min(pred_values)))

    # Clear plot
    plt.clf()

    # Set up plot
    plt.scatter(ref_values, pred_values, marker='o', color='black', s=8)
    plt.title(y_period + " minutes Clarke Error Grid")
    plt.xlabel("Reference Concentration (mg/dl)")
    plt.ylabel("Prediction Concentration (mg/dl)")
    plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.gca().set_facecolor('white')

    # Set axes lengths
    plt.gca().set_xlim([0, 400])
    plt.gca().set_ylim([0, 400])
    plt.gca().set_aspect((400) / (400))

    # Plot zone lines
    plt.plot([0, 400], [0, 400], ':', c='black')  # Theoretical 45 regression line
    plt.plot([0, 175 / 3], [70, 70], '-', c='black')
    # plt.plot([175/3, 320], [70, 400], '-', c='black')
    plt.plot([175 / 3, 400 / 1.2], [70, 400], '-',
             c='black')  # Replace 320 with 400/1.2 because 100*(400 - 400/1.2)/(400/1.2) =  20% error
    plt.plot([70, 70], [84, 400], '-', c='black')
    plt.plot([0, 70], [180, 180], '-', c='black')
    plt.plot([70, 290], [180, 400], '-', c='black')
    # plt.plot([70, 70], [0, 175/3], '-', c='black')
    plt.plot([70, 70], [0, 56], '-', c='black')  # Replace 175.3 with 56 because 100*abs(56-70)/70) = 20% error
    # plt.plot([70, 400],[175/3, 320],'-', c='black')
    plt.plot([70, 400], [56, 320], '-', c='black')
    plt.plot([180, 180], [0, 70], '-', c='black')
    plt.plot([180, 400], [70, 70], '-', c='black')
    plt.plot([240, 240], [70, 180], '-', c='black')
    plt.plot([240, 400], [180, 180], '-', c='black')
    plt.plot([130, 180], [0, 70], '-', c='black')

    # Add zone titles
    plt.text(30, 15, "A", fontsize=15)
    plt.text(370, 260, "B", fontsize=15)
    plt.text(280, 370, "B", fontsize=15)
    plt.text(160, 370, "C", fontsize=15)
    plt.text(160, 15, "C", fontsize=15)
    plt.text(30, 140, "D", fontsize=15)
    plt.text(370, 120, "D", fontsize=15)
    plt.text(30, 370, "E", fontsize=15)
    plt.text(370, 15, "E", fontsize=15)

    # Statistics from the data
    zone = [0] * 5
    for i in range(len(ref_values)):
        if (ref_values[i] <= 70 and pred_values[i] <= 70) or (
                pred_values[i] <= 1.2 * ref_values[i] and pred_values[i] >= 0.8 * ref_values[i]):
            zone[0] += 1  # Zone A

        elif (ref_values[i] >= 180 and pred_values[i] <= 70) or (ref_values[i] <= 70 and pred_values[i] >= 180):
            zone[4] += 1  # Zone E

        elif ((ref_values[i] >= 70 and ref_values[i] <= 290) and pred_values[i] >= ref_values[i] + 110) or (
                (ref_values[i] >= 130 and ref_values[i] <= 180) and (pred_values[i] <= (7 / 5) * ref_values[i] - 182)):
            zone[2] += 1  # Zone C
        elif (ref_values[i] >= 240 and (pred_values[i] >= 70 and pred_values[i] <= 180)) or (
                ref_values[i] <= 175 / 3 and pred_values[i] <= 180 and pred_values[i] >= 70) or (
                (ref_values[i] >= 175 / 3 and ref_values[i] <= 70) and pred_values[i] >= (6 / 5) * ref_values[i]):
            zone[3] += 1  # Zone D
        else:
            zone[1] += 1  # Zone B

    return plt, zone


def run(model_name, window, y_step):
    with mlflow.start_run():
        print('start training {0}'.format(model_name))

        n_features = 1
        n_seq = 2
        entries_df = pickle.load(open('data/entries_20396154_2.pkl', 'rb'))
        bg_list = entries_df['sgv']

        # test_date = '2018-12-01'
        # train_bg_list = entries_df[entries_df['datetime'] < test_date].reset_index()['sgv']
        # print('len(train_bg_list)', len(train_bg_list))
        # test_bg_list = entries_df[entries_df['datetime'] >= test_date].reset_index()['sgv']
        # print('len(test_bg_list)', len(test_bg_list))
        lenf = len(entries_df)
        split_point = lenf*2/3
        train_bg_list = entries_df[:split_point]['sgv']
        test_bg_list = entries_df[split_point:]['sgv']
        train_x, train_y = prepare_input(train_bg_list, window, y_step)
        test_x, test_y = prepare_input(test_bg_list, window, y_step)

        #         train_x, test_x, train_y, test_y = cluster_input(bg_list, window, y_step)

        if model_name == 'cnn_lstm':
            train_x = train_x.reshape((train_x.shape[0], n_seq, int(window / n_seq), n_features))
        elif model_name == 'conv_lstm':
            train_x = train_x.reshape((train_x.shape[0], n_seq, 1, int(window / n_seq), n_features))
        else:
            train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], n_features))

        if model_name == 'cnn':
            model = cnn(train_x, train_y, window, y_step, n_features)
        if model_name == 'v_lstm':
            model = vanilla_lstm(train_x, train_y, window, y_step, n_features)
        if model_name == 'b_lstm':
            model = bi_lstm(train_x, train_y, window, y_step, n_features)
        if model_name == 's_lstm':
            model = stacked_lstm(train_x, train_y, window, y_step, n_features)
        if model_name == 'cnn_lstm':
            model = cnn_lstm(train_x, train_y, window, n_seq, y_step, n_features)
        if model_name == 'conv_lstm':
            model = conv_lstm(train_x, train_y, window, n_seq, y_step, n_features)

        next_bgs = []
        pred_inputs = []
        for i in range(0, len(test_x), 1):
            pred_input = test_x[i]
            pred_inputs.append(pred_input[0])
            if model_name == 'cnn_lstm':
                pred_input = pred_input.reshape((1, n_seq, int(window / n_seq), n_features))
            elif model_name == 'conv_lstm':
                pred_input = pred_input.reshape((1, n_seq, 1, int(window / n_seq), n_features))
            else:
                pred_input = pred_input.reshape((1, window, n_features))
            next_bgs.append(model.predict(pred_input, verbose=0)[0])

        pred_y = np.array(next_bgs)
        rms = evaluate_forcasts(test_y, pred_y)

        test_y_last = [s[-1] for s in test_y]
        pred_y_last = [s[-1] for s in pred_y]

        plt, zone = clarke_error_grid(test_y_last, pred_y_last, '30 minutes CEG')
        plt.show()
        ylen = float(len(test_y_last))
        print("zones number:", zone)
        logging.info('model={0}, window = {1}, y_step={2}, rms={3}'.format(model_name, window, y_step, rms))
        logging.info('model={0}, zoneA = {1}, zoneB ={2}, zoneC={3}, zoneD={4}, zongE={5}'.format(model_name, float(zone[0])/ylen, float(zone[1])/ylen, float(zone[2])/ylen, float(zone[3])/ylen, float(zone[4])/ylen))

        # pred_results = pd.DataFrame(data={'y_true':test_y, 'y_pred':next_bgs})
        # pickle.dump(pred_results, open('results/predict_'+model_name+'_'+str(window)+'_'+str(y_step)+'.pkl', 'wb'), -1)

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("rms", rms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", help="Choose from cnn,v_lstm, b_lstm, s_lstm, cnn_lstm, conv_lstm", action = 'store', default="v_lstm", type = str)
    parser.add_argument("--s", help="prediction steps", action = 'store', default="0", type = int)
    parser.add_argument("--w", help="window size", action = 'store', default="12", type = int)

    args = parser.parse_args()
    target_step = args.s
    window = args.w
    for y_step in range(12, 5, -3):
        for m in [ "cnn", "v_lstm", "b_lstm","s_lstm", "cnn_lstm", "conv_lstm"]:
            run(m, window, y_step)



