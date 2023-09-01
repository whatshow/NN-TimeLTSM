import numpy as np
import matplotlib.pyplot as plt
import torch
from data_loader import DataLoader
from rnn_lstm_ag_v1 import RNN_LSTMAGV1
from rnn_lstm_ag_v1_epo100 import RNN_LSTMAGV1_epo100
from rnn_lstm_ag_v2 import RNN_LSTMAGV2
from rnn_lstm_ag_v3 import RNN_LSTMAGV3
from rnn_lstm_ag_v4 import RNN_LSTMAGV4
from rnn_lstm_ag_v5 import RNN_LSTMAGV5
from rnn_lstm_ag_v6 import RNN_LSTMAGV6 
from rnn_lstm_ag_v7 import RNN_LSTMAGV7
from rnn_lstm_v1 import RNN_LSTM_v1
from rnn_lstm_v1_epo100 import RNN_LSTM_v1_epo100
from rnn_lstm_self_v1_epo100 import RNN_LSTM_self_v1_epo100
from rnn_tf_simplernn import RNN_TF_SimpleRNN

# config
# config - paths & folders
folder = "./_build/";
# nn settings
test_percent = 0.2;     # 20% data is used to test
batch_size = 64;
epoch_iter = 100;       # we train 100 times at most 
epoch_size = None;      # this is determined by the data size
time_step = 12;         # 12 data -> 1 data
learning_rate = 0.0005;
lstm_layer_neuron_num = 128;
lstm_in_feature_num = 1;

# get data
dl = DataLoader();
data_train_x, data_train_y, data_test_x, data_test_y = dl(time_step);

# adjust epoch_size
epoch_size = len(data_train_x)//batch_size;
# USE GPU if available
device = torch.device('cpu');

# test NN

# RNN - LSTM Alex Graves v1
# feature x 1, MSE loss, DNN->Sigmoid 
rnn1 = RNN_LSTMAGV1();
RNN_LSTMAGV1_pred = rnn1(folder, device, lstm_layer_neuron_num, lstm_in_feature_num, time_step, epoch_size, batch_size, learning_rate, data_train_x, data_train_y, data_test_x);
rnn1_epo100 = RNN_LSTMAGV1_epo100();
RNN_LSTMAGV1_epo100_pred = rnn1_epo100(folder, device, lstm_layer_neuron_num, lstm_in_feature_num, time_step, epoch_iter, epoch_size, batch_size, learning_rate, data_train_x, data_train_y, data_test_x);

# RNN - LSTM Alex Graves v2
# feature x 1, MSE loss, DNN
rnn2 =  RNN_LSTMAGV2();
RNN_LSTMAGV2_pred = rnn2(folder, device, lstm_layer_neuron_num, lstm_in_feature_num, time_step, epoch_size, batch_size, learning_rate, data_train_x, data_train_y, data_test_x);
# RNN -LSTM Alex Graves v3
# feature x 1, MSE loss, DNN->DNN
rnn3 =  RNN_LSTMAGV3();
RNN_LSTMAGV3_pred = rnn3(folder, device, lstm_layer_neuron_num, lstm_in_feature_num, time_step, epoch_size, batch_size, learning_rate, data_train_x, data_train_y, data_test_x);
# RNN -LSTM Alex Graves v4 DNN->tanh->DNN
rnn4 = RNN_LSTMAGV4();
RNN_LSTMAGV4_pred = rnn4(folder, device, lstm_layer_neuron_num, lstm_in_feature_num, time_step, epoch_size, batch_size, learning_rate, data_train_x, data_train_y, data_test_x);
# RNN - LSTM Alex Graves v5 no separate neurons->DNN->Sig
rnn_lstm_ag_5 = RNN_LSTMAGV5();
RNN_LSTMAGV5_pred = rnn_lstm_ag_5(folder, device, lstm_layer_neuron_num, lstm_in_feature_num, time_step, epoch_iter, epoch_size, batch_size, learning_rate, data_train_x, data_train_y, data_test_x);
# RNN - LSTM Alex Graves v6 no separate neurons->DNN->Tanh->DNN
rnn_lstm_ag_6 = RNN_LSTMAGV6();
rnn_lstm_ag_6_pred = rnn_lstm_ag_6(folder, device, lstm_layer_neuron_num, lstm_in_feature_num, time_step, epoch_iter, epoch_size, batch_size, learning_rate, data_train_x, data_train_y, data_test_x);
# RNN - LSTM Alex Graves v7 no separate neurons->DNN->Tanh->DNN
rnn_lstm_ag_7 = RNN_LSTMAGV7();
rnn_lstm_ag_7_pred = rnn_lstm_ag_7(folder, device, lstm_layer_neuron_num, lstm_in_feature_num, time_step, epoch_iter, epoch_size, batch_size, learning_rate, data_train_x, data_train_y, data_test_x);

# torch LSTM
# RNN - LSTM v1: RNN -> DNN -> tanh -> DNN
rnn5 = RNN_LSTM_v1();
RNN_LSTM_v1_pred = rnn5(folder, device, lstm_layer_neuron_num, lstm_in_feature_num, time_step, epoch_size, batch_size, learning_rate, data_train_x, data_train_y, data_test_x);
# RNN - LSTM v1 (100 epos): RNN -> DNN -> tanh -> DNN
rnn6 = RNN_LSTM_v1_epo100();
RNN_LSTM_v1_epo100_pred = rnn6(folder, device, lstm_layer_neuron_num, lstm_in_feature_num, time_step, epoch_iter, epoch_size, batch_size, learning_rate, data_train_x, data_train_y, data_test_x);
# RNN - LSTM self v1 (100 epos): RNN -> DNN -> tanh -> DNN
rnn_LSTM_self_v1_epo100 = RNN_LSTM_self_v1_epo100();
RNN_LSTM_self_v1_epo100_pred = rnn_LSTM_self_v1_epo100(folder, device, lstm_layer_neuron_num, lstm_in_feature_num, time_step, epoch_iter, epoch_size, batch_size, learning_rate, data_train_x, data_train_y, data_test_x);

# SimpleRNN
# tensorflow
rnn7 = RNN_TF_SimpleRNN();
RNN_TF_SimpleRNN_pred = rnn7(folder, device, lstm_layer_neuron_num, lstm_in_feature_num, time_step, epoch_iter, epoch_size, batch_size, learning_rate, data_train_x, data_train_y, data_test_x);
 

# transfer the result to 1D
data_test_y_plot = np.squeeze(data_test_y);

RNN_TF_SimpleRNN_pred = np.squeeze(RNN_TF_SimpleRNN_pred);

RNN_LSTM_v1_pred = np.squeeze(RNN_LSTM_v1_pred.numpy());
RNN_LSTM_v1_epo100_pred = np.squeeze(RNN_LSTM_v1_epo100_pred);
RNN_LSTM_self_v1_epo100_pred = np.squeeze(RNN_LSTM_self_v1_epo100_pred);

RNN_LSTMAGV1_pred = np.squeeze(RNN_LSTMAGV1_pred.numpy());
RNN_LSTMAGV1_epo100_pred = np.squeeze(RNN_LSTMAGV1_epo100_pred.numpy());
RNN_LSTMAGV2_pred = np.squeeze(RNN_LSTMAGV2_pred.numpy());
RNN_LSTMAGV3_pred = np.squeeze(RNN_LSTMAGV3_pred.numpy());
RNN_LSTMAGV4_pred = np.squeeze(RNN_LSTMAGV4_pred.numpy());
RNN_LSTMAGV5_pred = np.squeeze(RNN_LSTMAGV5_pred.numpy());
RNN_LSTMAGV6_pred = np.squeeze(rnn_lstm_ag_6_pred.numpy());
RNN_LSTMAGV7_pred = np.squeeze(rnn_lstm_ag_7_pred.numpy());

# plot
plt.figure(figsize=(15, 6), dpi=200)
legend_labels = [];
plt.plot(range(len(data_test_y_plot)), data_test_y_plot);
legend_labels.append('Actual');
plt.plot(range(len(RNN_TF_SimpleRNN_pred)), RNN_TF_SimpleRNN_pred);
legend_labels.append('SimpleRNN(100 epo)');
# plt.plot(range(len(RNN_LSTM_v1_pred)), RNN_LSTM_v1_pred);
# legend_labels.append('LSTM');
plt.plot(range(len(RNN_LSTM_v1_epo100_pred)), RNN_LSTM_v1_epo100_pred);
legend_labels.append('LSTM(100 epo)');
plt.plot(range(len(RNN_LSTM_self_v1_epo100_pred)), RNN_LSTM_self_v1_epo100_pred);
legend_labels.append('LSTM self(100 epo)');
# plt.plot(range(len(RNN_LSTMAGV1_pred)), RNN_LSTMAGV1_pred);
# legend_labels.append('LSTM Alex Graves v1');
# plt.plot(range(len(RNN_LSTMAGV1_epo100_pred)), RNN_LSTMAGV1_epo100_pred);
# legend_labels.append('LSTM Alex Graves v1(100epo)');
# plt.plot(range(len(RNN_LSTMAGV2_pred)), RNN_LSTMAGV2_pred);
# legend_labels.append('LSTM Alex Graves v2');
# plt.plot(range(len(RNN_LSTMAGV3_pred)), RNN_LSTMAGV3_pred);
# legend_labels.append('LSTM Alex Graves v3');
# plt.plot(range(len(RNN_LSTMAGV4_pred)), RNN_LSTMAGV4_pred);
# legend_labels.append('LSTM Alex Graves v4');
# plt.plot(range(len(RNN_LSTMAGV5_pred)), RNN_LSTMAGV5_pred);
# legend_labels.append('LSTM Alex Graves v5 (100 epo)');
plt.plot(range(len(RNN_LSTMAGV6_pred)), RNN_LSTMAGV6_pred);
legend_labels.append('LSTM Alex Graves v6 (100 epo)');
plt.plot(range(len(RNN_LSTMAGV7_pred)), RNN_LSTMAGV7_pred);
legend_labels.append('LSTM Alex Graves v7 (100 epo)');


#plt.legend(['Actual', 'SimpleRNN(100 epo)', 'LSTM', 'LSTM(100 epo)', 'LSTM Alex Graves v1', 'LSTM Alex Graves v1(100epo)', 'LSTM Alex Graves v2', 'LSTM Alex Graves v3', 'LSTM Alex Graves v4', 'LSTM Alex Graves v5 (100 epo)', 'LSTM Alex Graves v6 (100 epo)']);
plt.legend(legend_labels);
plt.xlabel('Time');
plt.ylabel('Normalised Rx Signal Power')
plt.title('Validation Report');
plt.show();

print("Loss-SimpleRNN: %.4f"%(sum((RNN_TF_SimpleRNN_pred - data_test_y_plot)**2)));
print("Loss-Torch.LSTM: %.4f"%(sum((RNN_LSTM_v1_epo100_pred - data_test_y_plot)**2)));
print("Loss-My.LSTM: %.4f"%(sum((RNN_LSTM_v1_epo100_pred - data_test_y_plot)**2)));
print("Loss-AG.LSTM: %.4f"%(sum((RNN_LSTMAGV6_pred - data_test_y_plot)**2)));
print("Loss-AG.LSTM: %.4f"%(sum((RNN_LSTMAGV7_pred - data_test_y_plot)**2)));