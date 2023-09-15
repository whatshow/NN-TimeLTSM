
import os
import sys 
sys.path.append("../..");
import matplotlib.pyplot as plt
import torch
from data_loader_ns3 import DataLoaderNS3 as DataLoader
from Scaler import ZScoreScaler
# load test
from rnn_lstm1 import RNN_LSTM1
from rnn_lstm2 import RNN_LSTM2


# config
# config - nn
epoch_iter = 20;               # we train 300 times at most 
time_step = 12;
lstm_layer_neuron_num = 128;
lstm_in_feature_num = 1;
learning_rate = 0.005;

# USE GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# the folder
folder = "./_build/";
path_folder ="./_dist/2l/";
if not os.path.exists(folder):
    os.makedirs(folder);
if not os.path.exists(path_folder):
    os.makedirs(path_folder);

# load data
dl = DataLoader(debug=True);
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
sta_id = 0;
for file_id in range(len(data_test_y)):
    data_test_y[file_id] = data_test_y[file_id][:, sta_id, 0];
    RNN_LSTM1_pred[file_id] = RNN_LSTM1_pred[file_id][:, sta_id, 0];
    RNN_LSTM2_pred[file_id] = RNN_LSTM2_pred[file_id][:, sta_id, 0];

# plot
titles = ['human', 'vehicle', 'uav'];
#titles = ['vehicle'];
for file_id in range(len(data_test_y)):
    plt.figure(figsize=(15, 6), dpi=200)
    legend_labels = [];
    plt.plot(range(len(data_test_y[file_id])), data_test_y[file_id]);
    legend_labels.append('Actual');
    plt.plot(range(len(RNN_LSTM1_pred[file_id])), RNN_LSTM1_pred[file_id]);
    legend_labels.append('LSTM (1 layer)');
    plt.plot(range(len(RNN_LSTM2_pred[file_id])), RNN_LSTM2_pred[file_id]);
    legend_labels.append('LSTM (2 layer)');
    # show config
    plt.legend(legend_labels);
    plt.xlabel('Time');
    plt.ylabel('Rx Signal Power(dBm)')
    plt.title(titles[file_id]);
    plt.grid();
    plt.savefig(path_folder + titles[file_id] + '.jpg');
    
    print(titles[file_id]);
    print("Loss-LSTM (L2): %.4f"%(sum((RNN_LSTM1_pred[file_id] - data_test_y[file_id])**2)));

