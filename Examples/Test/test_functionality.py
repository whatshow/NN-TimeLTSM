import warnings
import os
import sys 
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pandas
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
# import self defined models
sys.path.append("../..");
from TimeLSTMLayer import TimeLSTMLayer

# config
# config - paths & folders
folder_rnn      = "./_build/rnn";
folder_lstm_ag  = "./_build/lstm_alex_graves";
# config - nn parameters
test_percent = 0.2;     # 20% data is used to test
batch_size = 64;
epoch_size = 100;       # this is the maximal epoch size (will change when the data is more or less) 
time_step = 12;         # 12 data -> 1 data
pred_step = 1;
learning_rate = 0.0005;
lstm_layer_neuron_num = 128;
lstm_in_feature_num = 1;
lstm_out_feature_num = 4;
# config - data url
sunspots_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv'

# create the folder
if not os.path.exists(folder_rnn):
    os.makedirs(folder_rnn);
if not os.path.exists(folder_lstm_ag):
    os.makedirs(folder_lstm_ag);

# load data
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
# add feature dimension (in our case, xm = 1)
# x -> (total_len, 12, xm)
# y -> (total_len, 1,  xm) 
data_train_x = np.expand_dims(data_train_x, -1);
data_train_y = np.expand_dims(data_train_y, -1);
data_test_x = np.expand_dims(data_test_x, -1);
data_test_y = np.expand_dims(data_test_y, -1);


# train
# adjust epoch_size
epoch_size = len(data_train_x)//batch_size;
# USE GPU if available
device = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu');
if torch.cuda.is_available():
    torch.cuda.set_device(0)
# create the neuron netwok
# model = nn.Sequential(
#     TimeLSTMLayer(lstm_layer_neuron_num, TimeLSTMLayer.NN_TYPE_LSTM_ALEX_GRAVES, TimeLSTMLayer.NN_INIT_TYPE_RANDN, lstm_in_feature_num, lstm_out_feature_num),
#     nn.Linear(lstm_layer_neuron_num, 1)
#     );
model = TimeLSTMLayer(lstm_layer_neuron_num, TimeLSTMLayer.NN_TYPE_LSTM_ALEX_GRAVES, TimeLSTMLayer.NN_INIT_TYPE_RANDN, lstm_in_feature_num, lstm_out_feature_num);
model_linear_mcs = nn.Linear(lstm_layer_neuron_num, 1);
model_linear_rssi = nn.Linear(lstm_out_feature_num, 1);
# model move to a certain device (include submodules and nn.parameters)
model = model.to(device);
# set criterion (MSE loss by means)
criterion = nn.MSELoss(reduction='mean').to(device);
# set the gradient optimizer (using adam algorithm)
gradient_optimizer = torch.optim.Adam([{'params': model.parameters()}
                              ], lr=learning_rate);
# set the learning rate optimizer (print the update)
lr_optimizer = torch.optim.lr_scheduler.ReduceLROnPlateau(gradient_optimizer, factor=0.91, patience=0, threshold=1e-6, verbose=True);
# train in epoch
for epo_id in range(epoch_size):
    # enter train
    model.train();
    # generate data
    data_train_x_cur = torch.from_numpy(data_train_x[epo_id*batch_size : (epo_id+1)*batch_size, :, :]).to(device);
    data_train_y_cur = torch.from_numpy(data_train_y[epo_id*batch_size : (epo_id+1)*batch_size, :, :]).to(device);
    # forward
    data_train_y_cur_pred = model.forward(data_train_x_cur, time_step);
    
    # calculate the loss
    loss = criterion(data_train_y_cur_pred, data_train_y_cur);
    # backward
    gradient_optimizer.zero_grad();
    loss.backward();
    gradient_optimizer.step();
