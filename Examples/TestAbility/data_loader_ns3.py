import numpy as np
import pandas
from sklearn.preprocessing import MinMaxScaler

# this data loader requires NN to output 0s when take 0s in targets
class DataLoaderNS3:

    # the column meaning of the data file
    FILE_COL_START_TIME = 0;
    FILE_COL_END_TIME = 1;
    FILE_COL_MAC_SIZE = 2;
    FILE_COL_PHY_SIZE = 3;
    FILE_COL_SNR = 4;               # linear SNR
    FILE_COL_RSSI = 5;              # linear RSSI
    FILE_COL_MCS = 6;               # the predicted MCS for this transmission from the previous beacon
    
    def __init__(self, *, data_type=3, debug=False):
        # beacon interval
        self.beacon_interval = 0.5;
        self.next_beacon_interval_from_this_beacon_start = 3*self.beacon_interval;
        # config - seeds (used in the file system)
        self.staids = np.concatenate([[str(item).rjust(5, '0') for item in np.arange(1, 32)],
                                [str(item).rjust(5, '0') for item in np.arange(65, 96)],
                                [str(item).rjust(5, '0') for item in np.arange(129, 160)],
                                [str(item).rjust(5, '0') for item in np.arange(193, 224)]], axis=-1);
        if debug:
            self.staids = np.array(['00001', '00002']);
        # config - filenames for human, vehicle, uav
        filename_human = ["NNData_STA128_C00_rec_human_3",
                          # "NNData_STA128_C00_rec_human_4",
                          # "NNData_STA128_C00_rec_human_5",
                          ];
        filename_vehicle = [#"NNData_STA128_C00_rec_vehicle_3",
                            # "NNData_STA128_C00_rec_vehicle_4",
                            "NNData_STA128_C00_rec_vehicle_5"
                            ];
        filename_uav = ["NNData_STA128_C00_rec_uav_3",
                        # "NNData_STA128_C00_rec_uav_4",
                        # "NNData_STA128_C00_rec_uav_5"
                        ];
        seeds = np.arange(1, 11);
        # config - holding time
        if debug:
            self.get_train_test_files(None, filename_vehicle, None, seeds);
        else:
            if data_type == 3:
                self.get_train_test_files(filename_human, filename_vehicle, filename_uav, seeds);
            if data_type == 0:
                self.get_train_test_files(filename_human, None, None, seeds);
            if data_type == 1:
                self.get_train_test_files(None, filename_vehicle, None, seeds);
            if data_type == 2:
                self.get_train_test_files(None, None, filename_uav, seeds);
        
    '''
    get the train and test files
    '''
    def get_train_test_files(self, filename_human, filename_vehicle, filename_uav, seeds):
        # create folder paths
        seed_str_len = 10;
        folderpaths_human = [];
        if filename_human is not None:
            for foldername in filename_human:
                for seed in seeds:
                    seed = str(seed);
                    seed = seed.zfill(seed_str_len);
                    folderpaths_human.append("../../../NN-TimeLTSM-Data/" + foldername + "/log/seed_" + seed + "/ap_rec/");
        folderpaths_vehicle = [];
        if filename_vehicle is not None:
            for foldername in filename_vehicle:
                for seed in seeds:
                    seed = str(seed);
                    seed = seed.zfill(seed_str_len);
                    folderpaths_vehicle.append("../../../NN-TimeLTSM-Data/" + foldername + "/log/seed_" + seed + "/ap_rec/");
        folderpaths_uav = [];
        if filename_uav is not None:
            for foldername in filename_uav:
                for seed in seeds:
                    seed = str(seed);
                    seed = seed.zfill(seed_str_len);
                    folderpaths_uav.append("../../../NN-TimeLTSM-Data/" + foldername + "/log/seed_" + seed + "/ap_rec/");
        # select one to test
        if filename_human is not None:
            test_file_id_human = int(np.random.choice(len(folderpaths_human), 1));
        if filename_vehicle is not None:
            test_file_id_vehicle = int(np.random.choice(len(folderpaths_vehicle), 1)); 
        if filename_uav is not None:
            test_file_id_uav = int(np.random.choice(len(folderpaths_uav), 1)); 
        # separate - test files
        if filename_human is not None:
            test_filename_human = folderpaths_human[test_file_id_human];
        if filename_vehicle is not None:
            test_filename_vehicle = folderpaths_vehicle[test_file_id_vehicle];
        if filename_uav is not None:
            test_filename_uav = folderpaths_uav[test_file_id_uav];
        # separate - train files
        if filename_human is not None:
            train_filename_human = np.delete(folderpaths_human, test_file_id_human);
        if filename_vehicle is not None:
            train_filename_vehicle = np.delete(folderpaths_vehicle, test_file_id_vehicle);
        if filename_uav is not None:
            train_filename_uav = np.delete(folderpaths_uav, test_file_id_uav);
        # we build a bigger train & test filename
        train_filenames = [];
        if filename_human is not None:
            train_filenames.append(train_filename_human);
        if filename_vehicle is not None:
            train_filenames.append(train_filename_vehicle);
        if filename_uav is not None:
            train_filenames.append(train_filename_uav);
        test_filenames = [];
        if filename_human is not None:
            test_filenames.append(test_filename_human);
        if filename_vehicle is not None:
            test_filenames.append(test_filename_vehicle);
        if filename_uav is not None:
            test_filenames.append(test_filename_uav);
        # to numpy array
        train_filenames = np.asarray(train_filenames);
        test_filenames = np.asarray(test_filenames);
        
        # merge
        train_filenames = np.concatenate(train_filenames, axis=0);
        # set
        self.train_filenames = train_filenames;
        self.train_files_len = len(train_filenames);
        self.test_filenames = test_filenames;
        self.test_files_len = len(test_filenames);
    
    '''
    get the last index from a given end time
    <INPUT>
    @tar_type:                  the type of the target                 
    @pred_period:               the maximal prediction periods(from the last known point)
    @memory_len:                the memory length
    @start_time:                the start time
    @column_end_time:           the end time of a packet
    @column_rssi:               the rssi
    <OUTPUT>
    DataLoaderNS3Stawise.TAR_TYPE_PERIOD gives unequal length target
    output shape is 
    DataLoaderNS3Stawise.TAR_TYPE_PERIOD:                  [target_len, (rssi, time)]
    DataLoaderNS3Stawise.TAR_TYPE_BEACONS:                 [pred_beacon_num, (average rssi, beacon_start, beacon_end)]
    DataLoaderNS3Stawise.TAR_TYPE_NEXT_BEACON:             [average rssi]
    '''
    def get_memory_data(self, memory_len, start_time, column_end_time, column_rssi):
        
        # parameters settings
        data_len = len(column_rssi);
        idx_tar = None;                         # target  index
        # we set return values
        is_filled = False;                      # whether we have enough data
        next_start_time = -1;                   # next beacon start time we should look at
        features = np.zeros((memory_len, 2));   # the meory data [memory_len, (rssi, time)] 
        targets = None;                          # the target data
        
        # load data, every time we load a beacon data
        # `cur_start_time` tells the start time for each iteration
        cur_start_time = 0;
        cur_end_time = start_time;
        while True:
            # move to the end of the previous beacon
            cur_start_time = cur_end_time;
            # find the end of this beacon period
            cur_end_time = cur_start_time + self.beacon_interval;
            #------------ jump point (when the beacon end time not inside this time period)
            if cur_end_time <= min(column_end_time):
                continue;
            #------------ jump point (when the beacon end time overflows)
            if cur_end_time not in column_end_time:
                break;
            
            # find the data packets between this beacon interval
            features_ids = np.where((column_end_time > cur_start_time) & (column_end_time < cur_end_time));
            features_ids = features_ids[0]; # only take 1st dimension
            for feature_id in features_ids:
                features = np.vstack((features, [10*np.log10(column_rssi[feature_id]*1000), column_end_time[feature_id]]));
                features = np.delete(features, 0, axis=0);
            # if we still have 0s in features (in the early stage, we haven't loaded enough data) 
            if np.any(features == 0):
                continue;
            # change the time points to intervals
            feature_start = np.min(features[:, 1]);
            features[:, 1] = features[:, 1] - feature_start;
            # now we have enough features, we need to the targets
            feature_last_time = features[-1, 1];
            feature_beacon_end_time = cur_end_time;
            target_time_start = None;
            target_time_end = None;
            targets_segment_num = None;     # only used in `beacons` data
            targets_ids = None;
            targets = [];
            # target
            # target - period: `last known time` + `pred_periods`
            # set the end of prediction
            target_time_start = feature_beacon_end_time + self.next_beacon_interval_from_this_beacon_start;
            target_time_end = target_time_start + self.beacon_interval;
            # find the targets
            targets_ids = np.where((column_end_time > target_time_start) & (column_end_time <= target_time_end) & (column_rssi != 0));
            if len(targets_ids[0]) > 0:
                targets = [np.mean(column_rssi[targets_ids])];
                targets[0] = 10*np.log10(targets[0]*1000);
            else:
                # if this sta does not send any data, we fill 0
                targets = [0];
            #------------ jump point (when targets is empty)
            if len(targets) == 0:
                break;
            # transfer the target to numpy
            targets = np.asarray(targets);
            #------------ jump point (we fill in the targets)
            is_filled = True;
            next_start_time = feature_beacon_end_time;
            break;
        
        # return
        return is_filled, next_start_time, features, targets;
    
    
    '''
    get x, y from a folder
    '''
    def get_x_y_from_folder(self, memory_len, folder):
        #print("- %s"%folder);
        data_train_x_beacon = [];
        data_train_y_beacon = [];
        # load the data for the sta
        stadata = [];
        sta_beacon_time = np.zeros(len(self.staids));
        for staid in self.staids:
            filepath = folder + staid  + ".csv";
            data_frame_tmp = pandas.read_csv(filepath, header=None);
            data_tmp = data_frame_tmp.values.astype('float32');
            stadata.append(data_tmp);
        #print("  - all files are loaded");
        # format the data into the format of features and targets
        trials = 1;
        is_data_end = False;
        while not is_data_end:
            #print("  - try to create batched data: %d"%trials);
            data_train_x_beacon_batch = [];
            data_train_y_beacon_batch = [];
            for sta_id in range(len(self.staids)):
                cur_data = stadata[sta_id];
                cur_sta_beacon_time = sta_beacon_time[sta_id];  
                is_filled, next_start_time, memory_data, future_data = self.get_memory_data(memory_len,
                                                                                            cur_sta_beacon_time,
                                                                                            cur_data[:, DataLoaderNS3.FILE_COL_END_TIME],
                                                                                            cur_data[:, DataLoaderNS3.FILE_COL_RSSI]);
                if is_filled:
                    # store this sta data
                    data_train_x_beacon_batch.append(memory_data);
                    data_train_y_beacon_batch.append(future_data);
                    # move time to next
                    sta_beacon_time[sta_id] = next_start_time;
                    # if we view all stas, we create a piece of batched data
                    if sta_id == len(self.staids) - 1:
                        # append
                        data_train_x_beacon.append(np.asarray(data_train_x_beacon_batch));
                        data_train_y_beacon.append(np.asarray(data_train_y_beacon_batch));

                        # clean
                        data_train_x_beacon_batch.clear();
                        data_train_y_beacon_batch.clear();
                        #print("    - create a batch");
                else:
                    #print("    - not fill for sta id: %d, start time: %f"%(sta_id, cur_sta_beacon_time));
                    is_data_end = True;
                    break;
            
            trials = trials + 1;
        #print("  - batch data is done");
        return data_train_x_beacon, data_train_y_beacon;
    
    
    '''
    generate data
    <INPUT>
    @memory_len:                the memory length 
    '''
    def __call__(self, memory_len):
        data_train_x = [];
        data_train_y = [];
        data_test_x = [];
        data_test_y = [];
        # load train data
        print("- start to load data for training");
        idx_max = len(self.train_filenames);
        idx=1;
        for folder in self.train_filenames:
            x, y = self.get_x_y_from_folder(memory_len, folder);
            x = np.asarray(x);
            y = np.asarray(y);
            # attech the data into total train
            data_train_x.append(x);
            data_train_y.append(y);
            print("\r%.4f%%"%(idx*100/idx_max),end="")
            idx = idx + 1;
        print();
        
        print("- start to load data for testing");
        idx_max = len(self.test_filenames);
        idx=1;
        for folder in self.test_filenames:
            x, y = self.get_x_y_from_folder(memory_len, folder);
            # y to numpy
            x = np.asarray(x);
            y = np.asarray(y);
            # attech the data into total train
            data_test_x.append(x);
            data_test_y.append(y);
            print("\r%.4f%%"%(idx*100/idx_max),end="")
            idx = idx + 1;
        print();
        
        return data_train_x, data_train_y, data_test_x, data_test_y;
            