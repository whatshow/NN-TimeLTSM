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
from TimeLSTM_v1 import TimeLSTM_v1 as TimeLSTM

# 1 feature
class RNN_LSTMAGV6:
    
    folder      = "rnn_lstm_ag_v6/"
    modelfile   = "lstm_alex_graves.pkl";
    eps                     = 1e-8;             # minimal loss
    patience                = 10;                # stop when loss stops decrease than 5 epo
    
    def __call__(self, path, device, lstm_layer_neuron_num, lstm_in_feature_num, time_step, epoch_iter, epoch_size, batch_size, learning_rate, data_train_x, data_train_y, data_test_x):
        # build the folder
        self.folder = path + self.folder;
        if not os.path.exists(self.folder):
            os.makedirs(self.folder);
        self.modelfile = self.folder + self.modelfile;
        
        # build the model
        model = nn.Sequential(
            TimeLSTM(TimeLSTM.NN_INIT_TYPE_RANDN, lstm_in_feature_num, lstm_layer_neuron_num),
            nn.Linear(lstm_layer_neuron_num, lstm_layer_neuron_num),
            nn.Tanh(),
            nn.Linear(lstm_layer_neuron_num, 1)
            );
        # to device
        model = model.to(device);
        # set criterion (MSE loss by means)
        criterion = nn.MSELoss(reduction='mean').to(device);
        # set the gradient optimizer (using adam algorithm)
        gradient_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate);
        # set the learning rate optimizer (print the update)
        lr_optimizer = torch.optim.lr_scheduler.ReduceLROnPlateau(gradient_optimizer, factor=0.91, patience=0, threshold=0.0001, verbose=True);
        # try to load the model
        has_trained = False;
        if os.path.exists(self.modelfile):
            has_trained = True;
            model.load_state_dict(torch.load(self.modelfile));
            
        print("- RNN_LSTM_AlexGraves_V6 (100 epo)");
        # try to train
        if not has_trained:
            print("  - train");
            model.train();
            
            patience_past = 0;
            loss_total_prev = 0.0;
            for epo_id in range(epoch_iter):
                loss_total = 0.0;
                for batch_id in range(epoch_size):
                    # generate data
                    data_train_x_cur = torch.from_numpy(data_train_x[batch_id*batch_size : (batch_id+1)*batch_size, :, :]).to(device);
                    data_train_y_cur = torch.from_numpy(data_train_y[batch_id*batch_size : (batch_id+1)*batch_size, :, :]).to(device);
                    # exchange the 1st and the 2nd dimension of x
                    data_train_y_cur = torch.squeeze(data_train_y_cur, -1);
                    out_tmp = model.forward(data_train_x_cur);
                    # calculate the loss
                    loss = criterion(out_tmp, data_train_y_cur);
                    # backward
                    gradient_optimizer.zero_grad();
                    loss.backward();
                    gradient_optimizer.step();
                    # append the loss to the total loss
                    loss_total = loss_total + loss;
                # calulate the average loss
                loss_aver = loss_total/epoch_size;
                # adjust the learning rate for each epo
                lr_optimizer.step(loss_aver);
                #show the progress
                print("    - epo %d/%d: the loss is %.6f"%(epo_id+1, epoch_iter, loss_aver));
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
            torch.save(model.state_dict(), self.modelfile);
        # test
        model.eval();
        # predict
        with torch.no_grad():
            # load data
            data_test_x_cur = torch.from_numpy(data_test_x).to(device);
            # predict
            out_tmp_test = model.forward(data_test_x_cur);
        # return
        return out_tmp_test;