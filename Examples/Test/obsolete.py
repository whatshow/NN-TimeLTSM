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
import torch
from torch import nn
# import self defined models
sys.path.append("../..");
from TimeLSTMLayer import TimeLSTMLayer

# config
# config - paths & folders
folder = "./_build/";
modelfile_rnn          = folder + "rnn.h";
modelfile_lstm_tf      = folder + "lstm_tf.h5";
modelfile_lstm_ag      = folder + "lstm_alex_graves.pkl";
modelfile_dnn_msc      = folder + "dnn_mcs.pkl";
modelfile_dnn_rssi     = folder + "dnn_rssi.pkl";
# config - nn parameters
test_percent = 0.2;     # 20% data is used to test
batch_size = 64;
epoch_size = 100;       # this is the maximal epoch size (will change when the data is more or less) 
time_step = 12;         # 12 data -> 1 data
pred_step = 1;
learning_rate = 1e-5;
lstm_layer_neuron_num = 128;
lstm_in_feature_num = 1;
lstm_out_feature_num = 4;
# config - data url
sunspots_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv'

# create the folder
if not os.path.exists(folder):
    os.makedirs(folder);

# load data
if 'data_frame' not in dir():
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


# model define
# model define - tensorflow lstm
tflstm = keras.models.Sequential();
tflstm.add(keras.layers.InputLayer(input_shape=(time_step, lstm_in_feature_num)));
tflstm.add(keras.layers.LSTM(lstm_layer_neuron_num));
tflstm.add(keras.layers.Dense(1, activation='sigmoid'));
tflstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse']);

# model define - alex grave lstm
# adjust epoch_size
epoch_size = len(data_train_x)//batch_size;
# USE GPU if available
device = torch.device('cpu');
# device = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu');
# if torch.cuda.is_available():
#     torch.cuda.set_device(0)
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
model_linear_mcs = model_linear_mcs.to(device);
model_linear_mcs_act = nn.Tanh();
model_linear_rssi = model_linear_rssi.to(device);
model_linear_rssi_act = nn.Sigmoid();
# set criterion (MSE loss by means)
criterion = nn.MSELoss(reduction='mean').to(device);
# set the gradient optimizer (using adam algorithm)
gradient_optimizer = torch.optim.Adam([{'params': model.parameters()},
                                       {'params': model_linear_mcs.parameters()},
                                       {'params': model_linear_rssi.parameters()}
                              ], lr=learning_rate);
# set the learning rate optimizer (print the update)
lr_optimizer = torch.optim.lr_scheduler.ReduceLROnPlateau(gradient_optimizer, factor=0.91, patience=0, threshold=1e-8, verbose=True);
# try to load the model
lstm_ag_has_trained = False;
if os.path.exists(modelfile_lstm_ag) and os.path.exists(modelfile_dnn_msc) and os.path.exists(modelfile_dnn_rssi):
    lstm_ag_has_trained = True;
    model.load_state_dict(torch.load(modelfile_lstm_ag));
    model_linear_mcs.load_state_dict(torch.load(modelfile_dnn_msc));
    model_linear_rssi.load_state_dict(torch.load(modelfile_dnn_rssi));

# train in epoch
if not lstm_ag_has_trained:
    model.train();
    model_linear_mcs.train();
    model_linear_rssi.train();
    for epo_id in range(epoch_size):
        # generate data
        data_train_x_cur = torch.from_numpy(data_train_x[epo_id*batch_size : (epo_id+1)*batch_size, :, :]).to(device);
        data_train_y_cur = torch.from_numpy(data_train_y[epo_id*batch_size : (epo_id+1)*batch_size, :, :]).to(device);
        # forward (output shape [batch_size, lstm_layer_neuron_num, lstm_out_feature_num])
        # e.g., [64, 128, 4]
        data_train_y_cur_pred = model.forward(data_train_x_cur, time_step);
        # transpose to [batch_size, lstm_out_feature_num, lstm_layer_neuron_num])
        # e.g., [64, 4, 128]
        data_train_y_cur_pred_mcss = torch.transpose(data_train_y_cur_pred, -1, -2);
        # merge the neurons results into one [batch_size, lstm_out_feature_num, 1]
        # e.g., [64, 4, 1]
        data_train_y_cur_pred_mcs = model_linear_mcs_act(model_linear_mcs(data_train_y_cur_pred_mcss));
        # transpose to [batch_size, 1, lstm_out_feature_num])
        # e.g., [64, 1 ,4]
        data_train_y_cur_pred_features = torch.transpose(data_train_y_cur_pred_mcs, -1, -2);
        # merge the features into one [batch_size, 1, 1]
        # e.g., [64, 1, 1]
        data_train_y_cur_pred_rssi = model_linear_rssi(data_train_y_cur_pred_features);
        # calculate the loss
        loss = criterion(data_train_y_cur, data_train_y_cur_pred_rssi);
        # backward
        gradient_optimizer.zero_grad();
        loss.backward();
        gradient_optimizer.step();
        # show the progress
        print("Training epo %d/%d: the loss is %.6f"%(epo_id+1, epoch_size, loss));
    # save the model
    torch.save(model.state_dict(), modelfile_lstm_ag);
    torch.save(model_linear_mcs.state_dict(), modelfile_dnn_msc);
    torch.save(model_linear_rssi.state_dict(), modelfile_dnn_rssi);
# train the tensorflow lstm


# test
model.eval();
model_linear_mcs.eval();
model_linear_rssi.eval();
# predict
with torch.no_grad():
    # load data
    data_test_x_cur = torch.from_numpy(data_test_x).to(device);
    # predict
    data_test_y_cur_pred = model.forward(data_test_x_cur, time_step);
    data_test_y_cur_pred_mcss = torch.transpose(data_test_y_cur_pred, -1, -2);
    data_test_y_cur_pred_mcs = model_linear_mcs_act(model_linear_mcs(data_test_y_cur_pred_mcss));
    data_test_y_cur_pred_features = torch.transpose(data_test_y_cur_pred_mcs, -1, -2);
    data_test_y_cur_pred_rssi = model_linear_rssi(data_test_y_cur_pred_features);
# transfer the result to 1D
data_test_y_plot = np.squeeze(data_test_y);
data_test_y_cur_pred_rssi_plot = np.squeeze(data_test_y_cur_pred_rssi.numpy());

# plot
plt.figure(figsize=(15, 6), dpi=80)
plt.plot(range(len(data_test_y_plot)), data_test_y_plot);
plt.plot(range(len(data_test_y_cur_pred_rssi_plot)), data_test_y_cur_pred_rssi_plot);
plt.legend(['Actual', 'Prediction']);
plt.xlabel('Time');
plt.ylabel('Normalised Rx Signal Power')
plt.title('Validation Report');
plt.show();
    