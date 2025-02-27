__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import tensorflow as tf
import os
import json
import time
import math
# import matplotlib.pyplot as plt
from core.data_loader import DataLoader
from core.model import Model

import mlflow
from mlflow import log_metric
import _pickle as pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import math


def record_configs_mlflow(configs):
    mlflow.log_param("finename", configs['data']['filename'])  
    mlflow.log_param("sequence_length", configs['data']['sequence_length'])
    mlflow.log_param("epochs", configs['training']['epochs'])
    mlflow.log_param("batch_size", configs['training']['epochs'])
    mlflow.log_param("loss", configs['model']['loss'])
    mlflow.log_param("optimizer", configs['model']['optimizer'])
    mlflow.log_param("layers", configs['model']['layers'])

def evaluate_results(preds, targets):
    rmse = math.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    print('rmse', rmse)
    print('mae', mae)
    print('r2', r2)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    

# def plot_results(predicted_data, true_data):
#     fig = plt.figure(facecolor='white')
#     ax = fig.add_subplot(1)
#     ax.plot(true_data, label='True Data')
#     plt.plot(predicted_data, label='Prediction')
#     plt.legend()
#     plt.show()


# def plot_results_multiple(predicted_data, true_data, prediction_len):
#     fig = plt.figure(facecolor='white')
#     ax = fig.add_subplot(111)
#     ax.plot(true_data, label='True Data')
# 	# Pad the list of predictions to shift it in the graph to it's correct start
#     for i, data in enumerate(predicted_data):
#         padding = [None for p in range(i * prediction_len)]
#         plt.plot(padding + data, label='Prediction')
#         plt.legend()
#     plt.show()


def main():
    with mlflow.start_run():
        configs = json.load(open('config.json', 'r'))
        record_configs_mlflow(configs)
        if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

        data = DataLoader(
            os.path.join('data', configs['data']['filename']),
            configs['data']['split_date'],
            configs['data']['columns'],
            configs['data']['maxBG']
        )


        model = Model()
        model.build_model(configs)
        x, y = data.get_train_data(
            seq_len=configs['data']['sequence_length'],
        )

        
        # in-memory training
        model.train(
            x,
            y,
            epochs = configs['training']['epochs'],
            batch_size = configs['training']['batch_size'],
            save_dir = configs['model']['save_dir']
        )
        # model.load_model('saved_models/15032019-164754-e60.h5')
        ''' 
        # out-of memory generative training
        steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
        model.train_generator(
            data_gen=data.generate_train_batch(
                seq_len=configs['data']['sequence_length'],
                batch_size=configs['training']['batch_size'],
                normalise=configs['data']['normalise']
            ),
            epochs=configs['training']['epochs'],
            batch_size=configs['training']['batch_size'],
            steps_per_epoch=steps_per_epoch,
            save_dir=configs['model']['save_dir']
        )
        '''

        x_test, y_test = data.get_test_data(
            seq_len=configs['data']['sequence_length'],
        )

        predictions = model.predict_sequence(x_test, configs['data']['sequence_length'])
        
        pickle.dump(predictions, open('results/predict_'+str(time.time())+'.pkl', 'wb'))


        y_true = data.get_origin_test_target()
        print('predictions[0]',predictions[0] )
        print('y_test[0]',y_true[0] )

        evaluate_results(predictions, y_true[configs['data']['sequence_length']:])

        # plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
        # plot_results(predictions, y_test[configs['data']['sequence_length']-1:])

if __name__ == '__main__':
    main()

