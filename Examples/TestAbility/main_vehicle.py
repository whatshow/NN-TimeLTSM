
import os
import sys 
sys.path.append("../..");
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pandas
import torch
from torch import nn
from data_loader_ns3 import DataLoaderNS3 as DataLoader
from Scaler import ZScoreScaler
from TimeLSTM_v3 import TimeLSTM_v3 as TimeLSTM
# load test
from rnn_lstm_ag import RNN_LSTM_AG
from rnn_lstm_ag_cm import RNN_LSTM_AG_CM
from rnn_lstm_time1_cm import RNN_LSTM_TIME1_CM
from rnn_lstm_time2_cm import RNN_LSTM_TIME2_CM
from rnn_lstm_time3_cm import RNN_LSTM_TIME3_CM

# config
# config - nn
epoch_iter = 300;               # we train 300 times at most 
time_step = 12;
lstm_layer_neuron_num = 128;
lstm_in_feature_num = 1;
learning_rate = 0.005;

# USE GPU if available
device = torch.device('cpu');
# the folder
folder = "./_build/vehicle/";
path_folder ="./_dist/vehicle/";
if not os.path.exists(folder):
    os.makedirs(folder);
if not os.path.exists(path_folder):
    os.makedirs(path_folder);

# load data
dl = DataLoader(data_type=1);
data_train_x, data_train_y, data_test_x, data_test_y = dl(time_step);

# scale data
zss = ZScoreScaler();
zss.fit(data_train_x, data_train_y);
data_train_x = zss.transform(data_train_x);
data_train_y = zss.transform(data_train_y);
data_test_x = zss.transform(data_test_x);

# test different neural networks
print("Data loaded");
rnn1 = RNN_LSTM_AG();
RNN_LSTMAGV1_pred = rnn1(folder, device, lstm_layer_neuron_num, lstm_in_feature_num, time_step, epoch_iter, learning_rate, data_train_x, data_train_y, data_test_x);
rnn2 = RNN_LSTM_AG_CM();
RNN_LSTMAGV2_pred = rnn2(folder, device, lstm_layer_neuron_num, lstm_in_feature_num, time_step, epoch_iter, learning_rate, data_train_x, data_train_y, data_test_x);
rnn3 = RNN_LSTM_TIME1_CM();
RNN_LSTM_TIME1_CM_pred = rnn3(folder, device, lstm_layer_neuron_num, lstm_in_feature_num, time_step, epoch_iter, learning_rate, data_train_x, data_train_y, data_test_x);
rnn4 = RNN_LSTM_TIME2_CM();
RNN_LSTM_TIME2_CM_pred = rnn4(folder, device, lstm_layer_neuron_num, lstm_in_feature_num, time_step, epoch_iter, learning_rate, data_train_x, data_train_y, data_test_x);
rnn5 = RNN_LSTM_TIME3_CM();
RNN_LSTM_TIME3_CM_pred = rnn5(folder, device, lstm_layer_neuron_num, lstm_in_feature_num, time_step, epoch_iter, learning_rate, data_train_x, data_train_y, data_test_x);

# unscale
RNN_LSTMAGV1_pred = zss.inverse_transform(RNN_LSTMAGV1_pred);
RNN_LSTMAGV2_pred = zss.inverse_transform(RNN_LSTMAGV2_pred);
RNN_LSTM_TIME1_CM_pred = zss.inverse_transform(RNN_LSTM_TIME1_CM_pred);
RNN_LSTM_TIME2_CM_pred = zss.inverse_transform(RNN_LSTM_TIME2_CM_pred);
RNN_LSTM_TIME3_CM_pred = zss.inverse_transform(RNN_LSTM_TIME3_CM_pred);

# to linear
sta_id = 0;
for file_id in range(len(data_test_y)):
    data_test_y[file_id] = data_test_y[file_id][:, sta_id, 0];
    RNN_LSTMAGV1_pred[file_id] = RNN_LSTMAGV1_pred[file_id][:, sta_id, 0];
    RNN_LSTMAGV2_pred[file_id] = RNN_LSTMAGV2_pred[file_id][:, sta_id, 0];
    RNN_LSTM_TIME1_CM_pred[file_id] = RNN_LSTM_TIME1_CM_pred[file_id][:, sta_id, 0];
    RNN_LSTM_TIME2_CM_pred[file_id] = RNN_LSTM_TIME2_CM_pred[file_id][:, sta_id, 0];
    RNN_LSTM_TIME3_CM_pred[file_id] = RNN_LSTM_TIME3_CM_pred[file_id][:, sta_id, 0];

# plot
titles = ['vehicle'];
for file_id in range(len(data_test_y)):
    plt.figure(figsize=(15, 6), dpi=200)
    legend_labels = [];
    plt.plot(range(len(data_test_y[file_id])), data_test_y[file_id]);
    legend_labels.append('Actual');
    plt.plot(range(len(RNN_LSTMAGV1_pred[file_id])), RNN_LSTMAGV1_pred[file_id]);
    legend_labels.append('LSTM (Alex Graves)');
    plt.plot(range(len(RNN_LSTMAGV2_pred[file_id])), RNN_LSTMAGV2_pred[file_id]);
    legend_labels.append('LSTM (Alex Graves - Cell Memory)');
    plt.plot(range(len(RNN_LSTM_TIME1_CM_pred[file_id])), RNN_LSTM_TIME1_CM_pred[file_id]);
    legend_labels.append('LSTM (Time Gate 1 - Cell Memory)');
    plt.plot(range(len(RNN_LSTM_TIME2_CM_pred[file_id])), RNN_LSTM_TIME2_CM_pred[file_id]);
    legend_labels.append('LSTM (Time Gate 2 - Cell Memory)');
    plt.plot(range(len(RNN_LSTM_TIME3_CM_pred[file_id])), RNN_LSTM_TIME3_CM_pred[file_id]);
    legend_labels.append('LSTM (Time Gate 3 - Cell Memory)');
    # show config
    plt.legend(legend_labels);
    plt.xlabel('Time');
    plt.ylabel('Rx Signal Power(dBm)')
    plt.title(titles[file_id]);
    plt.grid();
    plt.savefig(path_folder + titles[file_id] + '.jpg');
    
    print(titles[file_id]);
    print("Loss-AG.LSTM: %.4f"%(sum((RNN_LSTMAGV1_pred[file_id] - data_test_y[file_id])**2)));
    print("Loss-AG.LSTM.CM: %.4f"%(sum((RNN_LSTMAGV2_pred[file_id] - data_test_y[file_id])**2)));
    print("Loss-T1.LSTM.CM: %.4f"%(sum((RNN_LSTM_TIME1_CM_pred[file_id] - data_test_y[file_id])**2)));
    print("Loss-T2.LSTM.CM: %.4f"%(sum((RNN_LSTM_TIME2_CM_pred[file_id] - data_test_y[file_id])**2)));
    print("Loss-T3.LSTM.CM: %.4f"%(sum((RNN_LSTM_TIME3_CM_pred[file_id] - data_test_y[file_id])**2)));

