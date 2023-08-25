import warnings
import os
import sys 
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pandas
from sklearn.preprocessing import MinMaxScaler

class DataLoader:
    test_percent = 0.2;     # 20% data is used to test
    # config - data url
    sunspots_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv'
    #
    pred_step = 1;
    
    
    def __call__(self, time_step):
        data_frame = pandas.read_csv(self.sunspots_url, usecols=[1], engine='python')
        data = np.array(data_frame.values.astype('float32'));
        # normalization
        scaler = MinMaxScaler(feature_range=(0, 1));
        data = scaler.fit_transform(data).flatten();
        # Point for splitting data into train and test
        data_split_point = int(len(data)*(1-self.test_percent));
        data_train = data[range(data_split_point)];
        data_test = data[data_split_point:];
        # build the data into the form (the top `time_step` won't be inside `y`)
        # X: [n, xm]
        # Y: [n, 1]
        # data build - train
        data_train_x = [];
        data_train_y = [];
        for data_train_id in range(0, len(data_train) - time_step - self.pred_step + 1):
            data_train_x.append(data_train[data_train_id : data_train_id + time_step]);
            data_train_y.append(data_train[data_train_id + time_step: data_train_id + time_step + self.pred_step]);
        data_train_x = np.asarray(data_train_x);
        data_train_y = np.asarray(data_train_y);
        # data build - test
        data_test_x = [];
        data_test_y = [];
        for data_test_id in range(0, len(data_test) - time_step - self.pred_step + 1):
            data_test_x.append(data_test[data_test_id : data_test_id + time_step]);
            data_test_y.append(data_test[data_test_id + time_step : data_test_id + time_step + self.pred_step]);
        data_test_x = np.asarray(data_test_x);
        data_test_y = np.asarray(data_test_y);
        # add feature dimension (in our case, xm = 1)
        # x -> (total_len, 12, xm)
        # y -> (total_len, 1,  xm) 
        data_train_x = np.expand_dims(data_train_x, -1);
        data_train_y = np.expand_dims(data_train_y, -1);
        data_test_x = np.expand_dims(data_test_x, -1);
        data_test_y = np.expand_dims(data_test_y, -1);
        
        return data_train_x, data_train_y, data_test_x, data_test_y;