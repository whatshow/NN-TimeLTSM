import sys 
sys.path.append("../..");
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pandas
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from TimeLSTMLayer import TimeLSTMLayer

# config
test_percent = 0.2;     # 20% data is used to test
batch_size = 64;
epoch_size = 100;       # this is the maximal epoch size (will change when the data is more or less) 
time_step = 12;         # 12 data -> 1 data
pred_step = 1;
learning_rate = 0.0005;
lstm_layer_neuron_num = 128;

# load data
sunspots_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv'
data_frame = pandas.read_csv(sunspots_url, usecols=[1], engine='python')
data = np.array(data_frame.values.astype('float32'));
# normalization
scaler = MinMaxScaler(feature_range=(0, 1));
data = scaler.fit_transform(data).flatten();
# Point for splitting data into train and test
data_split_point = int(len(data)*(1-test_percent));
data_train = data[range(data_split_point)];
data_test = data[data_split_point:];

# build the data into the form (the top `time_step` won't be inside `y`)
# X: [n, xm]
# Y: [n, 1]
# data build - train
data_train_x = [];
data_train_y = [];
for data_train_id in range(0, len(data_train) - time_step - pred_step + 1):
    data_train_x.append(data_train[data_train_id : data_train_id + time_step]);
    data_train_y.append(data_train[data_train_id + time_step: data_train_id + time_step + pred_step]);
data_train_x = np.asarray(data_train_x);
data_train_y = np.asarray(data_train_y);
# data build - test
data_test_x = [];
data_test_y = [];
for data_test_id in range(0, len(data_test) - time_step - pred_step + 1):
    data_test_x.append(data_test[data_test_id : data_test_id + time_step]);
    data_test_y.append(data_test[data_test_id + time_step : data_test_id + time_step + pred_step]);
data_test_x = np.asarray(data_test_x);
data_test_y = np.asarray(data_test_y);
# add feature dimension
data_train_x = np.expand_dims(data_train_x, -1);
data_train_y = np.expand_dims(data_train_y, -1);
data_test_x = np.expand_dims(data_test_x, -1);
data_test_y = np.expand_dims(data_test_y, -1);

# train
# USE GPU if available
device = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu');
if torch.cuda.is_available():
    torch.cuda.set_device(0)
# 

# create the neuron netwok
ttl = TimeLSTMLayer(lstm_layer_neuron_num, TimeLSTMLayer.NN_TYPE_LSTM_ALEX_GRAVES, TimeLSTMLayer.NN_INIT_TYPE_RANDN, 1, 1);
    
# set criterion
criterion = nn.MSELoss(device);
optimizer = torch.optim.Adam([{'params': ttl.parameters()}
                              ], lr=learning_rate);