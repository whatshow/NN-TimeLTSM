
import sys 
sys.path.append("../..");
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pandas
import torch
from torch import nn
from data_loader_ns3_stawise import DataLoaderNS3Stawise as DataLoader
from TimeLSTM_v3 import TimeLSTM_v3 as TimeLSTM

# config
# config - data extra information
simu_time = 240 + 4;                                                    # 240s for simulation, but we have 4 seconds as extra
beacon_interval = 0.5;                                                  # beacon interval (seconds)
# config - nn
nn_lstm_len = 10;                                                       # nn_lstm length
nn_vocab_size = 26;                                                     # MCS number
nn_predict_max_time = 8;                                                # maximal nn prediction duration
nn_predict_max_beacon_interval = nn_predict_max_time/beacon_interval;   # maximal nn prediction beacon interval num
lstm_layer_neuron_num = 128;
lstm_in_feature_num = 1;
learning_rate = 0.0005;

# prepare for dataloader
dl = DataLoader(debug=False);
is_end = False;
features = None;
targets = None;
last_features = None;
last_targets = None;
trials = 0;

# USE GPU if available
device = torch.device('cpu');
# the folder
folder = "./_build/";

# create the neuron network
#model = TimeLSTM(TimeLSTM.NN_INIT_TYPE_RANDN, lstm_in_feature_num, lstm_layer_neuron_num, nn_type=TimeLSTM.NN_TYPE_LSTM_ALEX_GRAVES);
model = TimeLSTM(TimeLSTM.NN_INIT_TYPE_RANDN, lstm_in_feature_num, lstm_layer_neuron_num, nn_type=TimeLSTM.NN_TYPE_LSTM_TIME1);
model_dnn1 = nn.Linear(lstm_layer_neuron_num, lstm_layer_neuron_num);
model_dnn2 = nn.Linear(lstm_layer_neuron_num, 1);
# to device
model = model.to(device);
model_dnn1 = model_dnn1.to(device);
model_dnn2 = model_dnn2.to(device);
# set criterion (MSE loss by means)
criterion = nn.MSELoss(reduction='mean').to(device);
# set the gradient optimizer (using adam algorithm)
gradient_optimizer = torch.optim.Adam([{'params': model.parameters()},
                                       {'params': model_dnn1.parameters()},
                                       {'params': model_dnn2.parameters()}
                              ], lr=learning_rate);
# set the learning rate optimizer (print the update)
lr_optimizer = torch.optim.lr_scheduler.ReduceLROnPlateau(gradient_optimizer, factor=0.91, patience=0, threshold=1e-8, verbose=True);
# train
model.train();
model_dnn1.train();
model_dnn2.train();

while not is_end:
    #is_end, features, targets = dl(12, tar_type=DataLoader.TAR_TYPE_PERIOD, data_uniform_type=DataLoader.DATA_UNIFORM_TYPE_ZERO_PADDING);
    #is_end, features, targets = dl(12, tar_type=DataLoader.TAR_TYPE_BEACONS);
    is_end, features, targets = dl(12, tar_type=DataLoader.TAR_TYPE_NEXT_BEACON);
    features_rssi = torch.from_numpy(features[:, :, 0:1].astype(np.float32)).to(device);
    features_t = torch.from_numpy(features[:, :, 1:2].astype(np.float32)).to(device);
    
    
    if not is_end:
        last_features = features;
        last_targets = targets;
        pred_h, pred_cm = model.forward(features_rssi, t=features_t);
        trials = trials + 1;
        break;

# build the data holder for each station
# time_step = 12;
# stanum = len(staids);
# stalist_rssilist = np.asarray([[[0.0, 0.0]]*12]*124);

# load data

# merge train filenames

# ERR_SetDevice2GPU_gpu_idx_WRONG_DIM = "[gpu_idx] can only be a scalar or a vector";
# ERR_SetDevice2GPU_gpu_idx_OVERFLOW = "[gpu_idx] selects not existing GPU";
# def SetDevice(self, device, *, gpu_idx=[0], gpu_all = False):
#     # GPU(cuda) is not available
#     if not torch.cuda.is_available():
#         return False;
#     # use all GPU
#     if gpu_all:
#         self.device = torch.device("cuda");
#         return True;
#     # use some GPU
#     gpu_idx = np.asarray(gpu_idx);
#     gpu_num = torch.cuda.device_count();
#     if gpu_idx.ndim > 1:
#         raise Exception(ERR_SetDevice2GPU_gpu_idx_WRONG_DIM);
#     if gpu_idx.ndim == 0:
#         if gpu_idx >= gpu_num:
#             raise Exception(ERR_SetDevice2GPU_gpu_idx_OVERFLOW);
#     if gpu_idx.ndim == 1:
#         for cur_gpu_idx in gpu_idx:
#             if cur_gpu_idx >= gpu_num:
#                 raise Exception(ERR_SetDevice2GPU_gpu_idx_OVERFLOW);
#     # gpu correct