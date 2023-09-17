
import os
import sys 
sys.path.append("../..");
import matplotlib.pyplot as plt
import numpy as np
import torch
from data_loader_ns3 import DataLoaderNS3 as DataLoader
from Scaler import ZScoreScaler
# load test
from rnn_lstm1 import RNN_LSTM1
from rnn_lstm2 import RNN_LSTM2


# config
# config - nn
epoch_iter = 300;               # we train 300 times at most 
time_step = 12;
lstm_layer_neuron_num = 128;
lstm_in_feature_num = 1;
learning_rate = 0.0005;

# USE GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
rnn1 = RNN_LSTM1();
RNN_LSTM1_pred = rnn1(folder, device, lstm_layer_neuron_num, lstm_in_feature_num, time_step, epoch_iter, learning_rate, data_train_x, data_train_y, data_test_x);
rnn2 = RNN_LSTM2();
RNN_LSTM2_pred = rnn2(folder, device, lstm_layer_neuron_num, lstm_in_feature_num, time_step, epoch_iter, learning_rate, data_train_x, data_train_y, data_test_x);


# unscale
RNN_LSTM1_pred = zss.inverse_transform(RNN_LSTM1_pred);
RNN_LSTM2_pred = zss.inverse_transform(RNN_LSTM2_pred);


# to linear
loss_lstm1 = [];
loss_lstm2 = [];

for sta_id in range(data_test_y[0].shape[1]):
    for file_id in range(len(data_test_y)):
        d0 = data_test_y[file_id][:, sta_id, 0];
        d1 = RNN_LSTM1_pred[file_id][:, sta_id, 0];
        d2 = RNN_LSTM2_pred[file_id][:, sta_id, 0];

    # plot
    titles = ['vehicle'];
    #titles = ['vehicle'];
    for file_id in range(len(data_test_y)):
        plt.figure(figsize=(15, 6), dpi=200)
        legend_labels = [];
        plt.plot(range(len(d0)), d0);
        legend_labels.append('Actual');
        plt.plot(range(len(d1)), d1);
        legend_labels.append('LSTM (1 layer)');
        plt.plot(range(len(d2)), d2);
        legend_labels.append('LSTM (2 layer)');
        # show config
        plt.legend(legend_labels);
        plt.xlabel('Time');
        plt.ylabel('Rx Signal Power(dBm)')
        plt.title(titles[file_id]);
        plt.grid();
        plt.savefig(path_folder + titles[file_id] + "_" + str(sta_id) +  '.jpg');
        
        # record the loss
        data_len = len(d0);
        loss_lstm1.append(sum((d1 - d0)**2)/data_len);
        loss_lstm2.append(sum((d2 - d0)**2)/data_len);
        
print("Loss-LSTM.L1: %.4f"%(np.sqrt(np.mean(loss_lstm1))));
print("Loss-LSTM.L2: %.4f"%(np.sqrt(np.mean(loss_lstm2))));
