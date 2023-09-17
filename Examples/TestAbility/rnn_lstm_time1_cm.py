import warnings
import os
import sys 
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
# import self defined models
sys.path.append("../..");
from TimeLSTM_v3 import TimeLSTM_v3 as TimeLSTM

class RNN_LSTM_TIME1_CM:
    split_per               = 0.2;
    eps                     = 1e-8;             # minimal loss
    patience                = 30;                # stop when loss stops decrease than 5 epo
    folder = "rnn_lstm_time1_cm/"
    modelfile_lstm_ag      = "lstm.pkl";
    modelfile_dnn1         = "dnn1.pkl";
    modelfile_dnn2         = "dnn2.pkl";
    
    
    def __call__(self, path, device, lstm_layer_neuron_num, lstm_in_feature_num, time_step, epoch_iter, learning_rate, data_train_x, data_train_y, data_test_x):
        # build the folder
        self.folder = path + self.folder;
        if not os.path.exists(self.folder):
            os.makedirs(self.folder);
        self.modelfile_lstm_ag = self.folder + self.modelfile_lstm_ag;
        self.modelfile_dnn1 = self.folder + self.modelfile_dnn1;
        self.modelfile_dnn2 = self.folder + self.modelfile_dnn2;
        
        # split into train and validate set
        train_data_len = len(data_train_x);
        valid_data_len = int(train_data_len*self.split_per);
        if valid_data_len <= 0:
            raise Exception("not enough data for valid");
        valid_data_ids = np.random.choice(train_data_len, valid_data_len);
        valid_data_ids = valid_data_ids.tolist();
        train_data_x = [];
        train_data_y = [];
        valid_data_x = [];
        valid_data_y = [];
        for idx in range(train_data_len):
            if idx in valid_data_ids:
                valid_data_x.append(data_train_x[idx]);
                valid_data_y.append(data_train_y[idx]);
            else:
                train_data_x.append(data_train_x[idx]);
                train_data_y.append(data_train_y[idx]);
        data_train_x = train_data_x;
        data_train_y = train_data_y;
        
        # build the model
        model = TimeLSTM(TimeLSTM.NN_INIT_TYPE_RANDN, lstm_in_feature_num, lstm_layer_neuron_num, nn_type=TimeLSTM.NN_TYPE_LSTM_TIME1);
        model_dnn1 = nn.Linear(lstm_layer_neuron_num, lstm_layer_neuron_num);
        model_dnn2 = nn.Linear(lstm_layer_neuron_num, 1);
        
        # to device
        model = model.to(device);
        model_dnn1 = model_dnn1.to(torch.float64).to(device);
        model_dnn1_act = torch.tanh;
        model_dnn2 = model_dnn2.to(torch.float64).to(device);
        # set criterion (MSE loss by means)
        criterion = nn.MSELoss(reduction='none').to(device);
        # set the gradient optimizer (using adam algorithm)
        gradient_optimizer = torch.optim.Adam([{'params': model.parameters()},
                                               {'params': model_dnn1.parameters()},
                                               {'params': model_dnn2.parameters()}
                                      ], lr=learning_rate);
        # set the learning rate optimizer (print the update)
        lr_optimizer = torch.optim.lr_scheduler.ReduceLROnPlateau(gradient_optimizer, factor=0.91, patience=0, threshold=1e-8, verbose=True);
        # try to load the model
        has_trained = False;
        if os.path.exists(self.modelfile_lstm_ag) and os.path.exists(self.modelfile_dnn1):
            has_trained = True;
            model.load_state_dict(torch.load(self.modelfile_lstm_ag));
            model_dnn1.load_state_dict(torch.load(self.modelfile_dnn1));
            model_dnn2.load_state_dict(torch.load(self.modelfile_dnn2));
            
        print("- RNN_LSTM_TIME1_CM");
        # try to train
        if not has_trained:
            print("  - train");
            model.train();
            model_dnn1.train();
            model_dnn2.train();
            
            patience_past = 0;
            loss_total_prev = 0.0;
            for epo_id in range(epoch_iter):
                loss_total = 0.0;
                loss_total_trials = 0;
                filelen = len(data_train_x);
                for idx_file in range(filelen):
                    pred_cm = None;
                    beacon_len = len(data_train_x[idx_file]);
                    for idx_beacon in range(beacon_len):
                        x_all = data_train_x[idx_file][idx_beacon];
                        x = torch.from_numpy(x_all[:, :, 0:1]).to(device);
                        t = torch.from_numpy(x_all[:, :, 1:2]).to(device);
                        y = torch.from_numpy(data_train_y[idx_file][idx_beacon]).to(device);
                        y_nonzero_ids = torch.where(y != 0);
                        pred_h, pred_cm = model.forward(x, cm=pred_cm, t=t);
                        dnn_in = torch.squeeze(pred_h, -1);
                        dnn1_out = model_dnn1_act(model_dnn1(dnn_in));
                        dnn2_out = model_dnn2(dnn1_out);
                        # calculate the loss
                        loss = criterion(dnn2_out, y);
                        # only count on the loss with non zeros in y
                        loss = torch.mean(loss[y_nonzero_ids]);
                        # backward
                        gradient_optimizer.zero_grad();
                        loss.backward();
                        gradient_optimizer.step();
                        # append the loss to the total loss
                        loss_total = loss_total + loss;
                        loss_total_trials = loss_total_trials + 1;
                        # detach pred_cm
                        pred_cm = pred_cm.detach();
                # calulate the average loss
                loss_aver = loss_total/loss_total_trials;
                #show the progress
                print("    - epo %d/%d: the loss is %.6f"%(epo_id+1, epoch_iter, loss_aver));
                
                # valid
                if (epo_id+1)%5 == 0: 
                    model.eval();
                    model_dnn1.eval();
                    model_dnn2.eval();
                    loss_total = 0.0;
                    loss_total_trials = 0;
                    filelen = len(valid_data_x);
                    for idx_file in range(filelen):
                        pred_cm = None;
                        beacon_len = len(valid_data_x[idx_file]);
                        for idx_beacon in range(beacon_len):
                            x_all = valid_data_x[idx_file][idx_beacon];
                            x = torch.from_numpy(x_all[:, :, 0:1]).to(device);
                            t = torch.from_numpy(x_all[:, :, 1:2]).to(device);
                            y = torch.from_numpy(valid_data_y[idx_file][idx_beacon]).to(device);
                            y_nonzero_ids = torch.where(y != 0);
                            pred_h, pred_cm = model.forward(x, cm=pred_cm, t=t);
                            dnn_in = torch.squeeze(pred_h, -1);
                            dnn1_out = model_dnn1_act(model_dnn1(dnn_in));
                            dnn2_out = model_dnn2(dnn1_out);
                            # calculate the loss
                            loss = criterion(dnn2_out, y);
                            # only count on the loss with non zeros in y
                            loss = torch.mean(loss[y_nonzero_ids]);
                            # append the loss to the total loss
                            loss_total = loss_total + loss;
                            loss_total_trials = loss_total_trials + 1;
                            # detach pred_cm
                            pred_cm = pred_cm.detach();
                    # calulate the average loss
                    loss_aver = loss_total/loss_total_trials;
                    # adjust the learning rate for each epo
                    lr_optimizer.step(loss_aver);
                    #show the progress
                    print("      - valid: the loss is %.6f"%loss_aver);
                    model.train();
                    model_dnn1.train();
                    model_dnn2.train();
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
            torch.save(model_dnn2.state_dict(), self.modelfile_dnn2);
        # test
        model.eval();
        model_dnn1.eval();
        model_dnn2.eval();
        # predict
        data_test_y = [];
        with torch.no_grad():
            filelen = len(data_test_x);
            for idx_file in range(filelen):
                pred_cm = None;
                data_test_y_beacon = [];
                beacon_len = len(data_test_x[idx_file]);
                for idx_beacon in range(beacon_len):
                    x_all = data_test_x[idx_file][idx_beacon];
                    x = torch.from_numpy(x_all[:, :, 0:1]).to(device);
                    t = torch.from_numpy(x_all[:, :, 1:2]).to(device);
                    pred_h, pred_cm = model.forward(x, cm=pred_cm, t=t);
                    dnn_in = torch.squeeze(pred_h, -1);
                    dnn1_out = model_dnn1_act(model_dnn1(dnn_in));
                    dnn2_out = model_dnn2(dnn1_out);
                    # attach batched data into beacon
                    dnn2_out = dnn2_out.to('cpu');
                    data_test_y_beacon.append(dnn2_out.numpy());
                    # detach pred_cm
                    pred_cm = pred_cm.detach();
                # to numpy
                data_test_y_beacon = np.asarray(data_test_y_beacon);
                # attach beacon data into the 
                data_test_y.append(data_test_y_beacon);
                
        # return
        return data_test_y;