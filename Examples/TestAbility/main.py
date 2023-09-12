
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


# config
# config - nn
epoch_iter = 100;               # we train 100 times at most 
time_step = 12;
lstm_layer_neuron_num = 128;
lstm_in_feature_num = 1;
learning_rate = 0.0005;

# USE GPU if available
device = torch.device('cpu');
# the folder
folder = "./_build/";

# load data
dl = DataLoader();
data_train_x, data_train_y, data_test_x, data_test_y = dl(time_step);

# scale data

print("Data loaded");

rnn1 = RNN_LSTM_AG();
RNN_LSTMAGV1_pred = rnn1(folder, device, lstm_layer_neuron_num, lstm_in_feature_num, time_step, epoch_iter, learning_rate, data_train_x, data_train_y, data_test_x);
    


# plot
a = RNN_LSTMAGV1_pred[0];
b = data_test_y[0];


