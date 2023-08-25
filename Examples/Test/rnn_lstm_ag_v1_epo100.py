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

# 1 feature
class RNN_LSTMAGV1_epo100:
    
    folder = "rnn_lstm_ag_v1_epo100/"
    modelfile_lstm_ag      = "lstm_alex_graves.pkl";
    modelfile_dnn1         = "dnn1.pkl";
    
    eps                     = 1e-8;             # minimal loss
    patience                = 5;                # stop when loss stops decrease than 5 epo
    
    def __call__(self, path, device, lstm_layer_neuron_num, lstm_in_feature_num, time_step, epoch_iter, epoch_size, batch_size, learning_rate, data_train_x, data_train_y, data_test_x):
        # build the folder
        self.folder = path + self.folder;
        if not os.path.exists(self.folder):
            os.makedirs(self.folder);
        self.modelfile_lstm_ag = self.folder + self.modelfile_lstm_ag;
        self.modelfile_dnn1 = self.folder + self.modelfile_dnn1;
        # build the model
        model = TimeLSTMLayer(lstm_layer_neuron_num, TimeLSTMLayer.NN_TYPE_LSTM_ALEX_GRAVES, TimeLSTMLayer.NN_INIT_TYPE_RANDN, lstm_in_feature_num, 1);
        model_dnn1 = nn.Linear(lstm_layer_neuron_num, 1);
        model_dnn1_act = nn.Sigmoid();
        # to device
        model = model.to(device);
        model_dnn1 = model_dnn1.to(device);
        # set criterion (MSE loss by means)
        criterion = nn.MSELoss(reduction='mean').to(device);
        # set the gradient optimizer (using adam algorithm)
        gradient_optimizer = torch.optim.Adam([{'params': model.parameters()},
                                               {'params': model_dnn1.parameters()},
                                              # {'params': model_linear_rssi.parameters()}
                                      ], lr=learning_rate);
        # set the learning rate optimizer (print the update)
        lr_optimizer = torch.optim.lr_scheduler.ReduceLROnPlateau(gradient_optimizer, factor=0.91, patience=0, threshold=1e-8, verbose=True);
        # try to load the model
        has_trained = False;
        if os.path.exists(self.modelfile_lstm_ag) and os.path.exists(self.modelfile_dnn1):
            has_trained = True;
            model.load_state_dict(torch.load(self.modelfile_lstm_ag));
            model_dnn1.load_state_dict(torch.load(self.modelfile_dnn1));
        
        print("- RNN_LSTM_AlexGraves_V1_epo100");
        # try to train
        if not has_trained:
            print("  - train");
            model.train();
            model_dnn1.train();
            
            patience_past = 0;
            loss_total_prev = 0.0;
            for epo_id in range(epoch_iter):
                
                loss_total = 0.0;
                for batch_id in range(epoch_size):
                    # generate data
                    data_train_x_cur = torch.from_numpy(data_train_x[batch_id*batch_size : (batch_id+1)*batch_size, :, :]).to(device);
                    data_train_y_cur = torch.from_numpy(data_train_y[batch_id*batch_size : (batch_id+1)*batch_size, :, :]).to(device);
                    data_train_y_cur = torch.squeeze(data_train_y_cur, -1);
                    # forward (output shape [batch_size, lstm_layer_neuron_num, lstm_out_feature_num])
                    # e.g., [64, 128, 4]
                    data_train_y_cur_pred = model.forward(data_train_x_cur, time_step);
                    dnn1_in = torch.squeeze(data_train_y_cur_pred, -1);
                    dnn1_out = model_dnn1_act(model_dnn1(dnn1_in));
                    # calculate the loss
                    loss = criterion(dnn1_out, data_train_y_cur);
                    # # transpose to [batch_size, lstm_out_feature_num, lstm_layer_neuron_num])
                    # # e.g., [64, 4, 128]
                    # data_train_y_cur_pred_mcss = torch.transpose(data_train_y_cur_pred, -1, -2);
                    # # merge the neurons results into one [batch_size, lstm_out_feature_num, 1]
                    # # e.g., [64, 4, 1]
                    # data_train_y_cur_pred_mcs = model_dnn1(data_train_y_cur_pred_mcss));
                    # # transpose to [batch_size, 1, lstm_out_feature_num])
                    # # e.g., [64, 1 ,4]
                    # data_train_y_cur_pred_features = torch.transpose(data_train_y_cur_pred_mcs, -1, -2);
                    # # merge the features into one [batch_size, 1, 1]
                    # # e.g., [64, 1, 1]
                    # data_train_y_cur_pred_rssi = model_linear_rssi(data_train_y_cur_pred_features);
                    # # calculate the loss
                    # loss = criterion(data_train_y_cur, data_train_y_cur_pred_rssi);
                    # backward
                    gradient_optimizer.zero_grad();
                    loss.backward();
                    gradient_optimizer.step();
                    # accumulate the loss
                    loss_total = loss_total + loss;
                # calulate the average loss
                loss_aver = loss_total/epoch_size;
                # adjust the learning rate for each epo
                lr_optimizer.step(loss_aver);
                #show the progress
                print("    - epo %d/%d: the loss is %.6f"%(epo_id+1, epoch_iter, loss));
                # if the loss is bigger than the past, we pass a patience
                if loss_aver - loss_total_prev >= self.eps:
                    patience_past = patience_past + 1;
                # repord the previous loss
                loss_total_prev = loss_aver;
                # stop when patience reaches the most
                if patience_past >= self.patience:
                    print("    - stop because the loss stops decreasing for %d epos"%patience_past);
                    break;
                # stop when loss reaches the minimal
                if loss_total_prev <= self.eps:
                    print("    - stop because the loss has reached its minimal"%patience_past);
                
            # save the model
            torch.save(model.state_dict(), self.modelfile_lstm_ag);
            torch.save(model_dnn1.state_dict(), self.modelfile_dnn1);
        # test
        model.eval();
        model_dnn1.eval();
        # predict
        with torch.no_grad():
            # load data
            data_test_x_cur = torch.from_numpy(data_test_x).to(device);
            # predict
            data_test_y_cur_pred = model.forward(data_test_x_cur, time_step);
            dnn1_test_in = torch.squeeze(data_test_y_cur_pred, -1);
            dnn1_test_out = model_dnn1_act(model_dnn1(dnn1_test_in));
        # return
        return dnn1_test_out;

# data_test_y_cur_pred_mcss = torch.transpose(data_test_y_cur_pred, -1, -2);
# data_test_y_cur_pred_mcs = model_linear_mcs_act(model_linear_mcs(data_test_y_cur_pred_mcss));
# data_test_y_cur_pred_features = torch.transpose(data_test_y_cur_pred_mcs, -1, -2);
# data_test_y_cur_pred_rssi = model_linear_rssi(data_test_y_cur_pred_features);