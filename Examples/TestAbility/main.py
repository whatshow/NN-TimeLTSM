
import sys 
sys.path.append("../..");
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pandas
import torch
from torch import nn

# config
# config - data extra information
beacon_interval = 0.5;                                                  # beacon interval (seconds)
# config - nn
nn_lstm_len = 10;                                                       # nn_lstm length
nn_vocab_size = 26;                                                     # MCS number
nn_predict_max_time = 8;                                                # maximal nn prediction duration
nn_predict_max_beacon_interval = nn_predict_max_time/beacon_interval;   # maximal nn prediction beacon interval num

# config - seeds (used in the file system)
seeds = np.asarray([5, 6, 7]);
# config - station id (6 + 15 + 16 + 7 + 2 + 14 + 2 + 14 + 10 + 19 + 19)
staids = ["0a", "0b", "0c", "0d", "0e", "0f",
          "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "ca", "cb", "cc", "cd", "ce", "cf", 
          "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "da", "db", "dc", "dd", "de", "df", 
          "01", "1a", "1b", "1c", "1d", "1e", "1f",
          "02", 
          "03",
          "04", "4a", "4b", "4c", "4d", "4e", "4f",
          "05", "5a", "5b", "5c", "5d", "5e", "5f",
          "06",
          "07",
          "08", "8a", "8b", "8c", "8d", "8e", "8f",
          "09", "9a", "9b", "9c", "9d", "9e", "9f",
          "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
          "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59",
          "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99"];
# config - filenames for human, vehicle, uav
filename_human = ["NNData_STA128_C00_rec_human_1",
                  "NNData_STA128_C00_rec_human_2",
                  "NNData_STA128_C00_rec_human_3",
                  "NNData_STA128_C00_rec_human_4",
                  "NNData_STA128_C00_rec_human_5",
                  ];
filename_vehicle = ["NNData_STA128_C00_rec_vehicle_1",
                    "NNData_STA128_C00_rec_vehicle_2",
                    "NNData_STA128_C00_rec_vehicle_3",
                    "NNData_STA128_C00_rec_vehicle_4",
                    "NNData_STA128_C00_rec_vehicle_5"];
filename_uav = ["NNData_STA128_C00_rec_uav_1",
                "NNData_STA128_C00_rec_uav_2",
                "NNData_STA128_C00_rec_uav_3",
                "NNData_STA128_C00_rec_uav_4",
                "NNData_STA128_C00_rec_uav_5"];
filename_human = np.asarray(filename_human);
filename_vehicle = np.asarray(filename_vehicle);
filename_uav = np.asarray(filename_uav);

# load data
# select 3 files to test for huanm, vehicle and uav respectively. The rest are for training and valid
test_file_id_human = np.random.choice(len(filename_human), 1);
test_file_id_vehicle = np.random.choice(len(filename_vehicle), 1);
test_file_id_uav = np.random.choice(len(filename_uav), 1);
# separate train files and test files
test_filename_human = filename_human[test_file_id_human];
train_filename_human = np.delete(filename_human, test_file_id_human);
test_filename_vehicle = filename_vehicle[test_file_id_vehicle];
train_filename_vehicle = np.delete(filename_vehicle, test_file_id_vehicle);
test_filename_uav = filename_uav[test_file_id_uav];
train_filename_uav = np.delete(filename_uav, test_file_id_uav);
# merge train filenames
train_filename = np.concatenate((train_filename_human, train_filename_vehicle, train_filename_uav), axis=-1);
# load data
for filename in train_filename:
    for seed in seeds:
        for staid in staids:
            data_frame = pandas.read_csv("data/" + filename + "/log/seed_000000000" + str(seed) + "/mac_rec/0000000000" + str(staid)  + ".csv", header=None);
            data = data_frame.values;




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