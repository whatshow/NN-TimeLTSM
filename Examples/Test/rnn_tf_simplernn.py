import warnings
import os
import sys 
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pandas
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras

class RNN_TF_SimpleRNN:
    folder = "rnn_simple_rnn/"
    eps                     = 1e-8;             # minimal loss
    patience                = 5;                # stop when loss stops decrease than 5 epo
    model_file_path = "simplernn.h5";
    
    
    def __call__(self, path, device, lstm_layer_neuron_num, lstm_in_feature_num, time_step, epoch_iter, epoch_size, batch_size, learning_rate, data_train_x, data_train_y, data_test_x):
        # build the folder
        self.folder = path + self.folder;
        if not os.path.exists(self.folder):
            os.makedirs(self.folder);
        self.model_file_path = self.folder + self.model_file_path;
        
        # build the model
        is_model_exist = False
        model = None;
        try:
            model = keras.models.load_model(self.model_file_path);
            is_model_exist = True;
        except:
            model = keras.models.Sequential();
            model.add(keras.layers.InputLayer(input_shape=(time_step, 1)));
            model.add(keras.layers.SimpleRNN(time_step, activation='tanh'));
            model.add(keras.layers.Dense(units=1, activation='sigmoid'));
            model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse']);
            earlystopping = keras.callbacks.EarlyStopping(
                monitor             = 'loss',           # monitor the train loss
                mode                = "auto",           # select the mode based on the monitor ('min' for loss, 'max' for accuracy)
                patience            = 5,                # if the monitored metric does not change in this number of consecutive epoches, we stop
                start_from_epoch    = 20                # the top 20 epoch are not counted
            );
        
        print("- RNN_TF_SimpleRNN");
        # train
        if not is_model_exist:
            data_train_x = np.squeeze(data_train_x, -1);
            data_train_y = np.squeeze(data_train_y, -1);
            # train
            model.fit(data_train_x,
                      data_train_y,
                      batch_size          = batch_size,
                      epochs              = epoch_iter,
                      verbose             = 2,                                    # show epochs, like Epoch 1/10            
                      callbacks           = [earlystopping]);
            # save
            model.save(self.model_file_path);
        
        # predict
        predict_data = model.predict(data_test_x);
        return predict_data;