import numpy as np
import pandas


class DataLoaderNS3:
    # `last known time` + `pred_periods`
    # [batch_size, target_len, (rssi, time)]
    TAR_TYPE_PERIOD = 1;
    # `last beacon` + `pred_periods//beacon_interval`
    # [batch_size, pred_beacon_num, (average rssi, beacon_start, beacon_end)]
    TAR_TYPE_BEACONS = 2;
    # `last beacon` + `next available beacon` in `last beacon` + `pred_periods`
    # [batch_size, average rssi]
    TAR_TYPE_NEXT_BEACON = 3;
    TAR_TYPES = [TAR_TYPE_PERIOD, TAR_TYPE_BEACONS, TAR_TYPE_NEXT_BEACON];
    
    '''
    @hold_time_min:             the minimal holdtime (minimum has a higher priority than maximum)
    @hold_time_max:             the maximal holdtime
    @is_human:                  we consider humans
    @is_vehicle:                we consider vehicle
    @is_uav:                    we consider uav
    @debug:                     whether we debug
    '''
    def __init__(self, *, hold_time_min=1, hold_time_max=5, is_human=True, is_vehicle=True, is_uav=True, debug=False):
        # beacon interval
        self.beacon_interval = 0.5;
        self.next_beacon_interval_from_this_beacon_start = 4*self.beacon_interval;
        # config - seeds (used in the file system)
        self.seeds = np.asarray([5, 6, 7]);
        if debug:
            self.seeds = np.asarray([5]);
        self.seeds_len = len(self.seeds);
        # config - station id (6 + 15 + 16 + 7 + 2 + 14 + 2 + 14 + 10 + 19 + 19)
        self.staids = ["0a", "0b", "0c", "0d", "0e", "0f",
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
        if debug:
            self.staids = self.staids[0:1];
        self.staids_len = len(self.staids);
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
        self.filename_human = np.asarray(filename_human);
        self.filename_vehicle = np.asarray(filename_vehicle);
        self.filename_uav = np.asarray(filename_uav);
        # config - holding time
        self.hold_time_min = 1;
        self.hold_time_max = 5;
        hold_time_min = int(hold_time_min);
        hold_time_max = int(hold_time_max);
        if hold_time_min < self.hold_time_min:
            hold_time_min = self.hold_time_min;
        if hold_time_max < self.hold_time_max:
            hold_time_max = self.hold_time_max;
        if hold_time_min > hold_time_max:
            hold_time_max = hold_time_min;
        self.hold_time_min = hold_time_min;
        self.hold_time_max = hold_time_max;
        # get train, test files
        if debug:
            self.train_filenames = ["NNData_STA128_C00_rec_human_1"];
            self.train_files_len = 1;
            self.test_filenames = ["NNData_STA128_C00_rec_vehicle_2"];
            self.test_files_len = 1;
        else:
            self.get_train_test_files(hold_time_min, hold_time_max, is_human, is_vehicle, is_uav);
        # reset the condition
        self.reset();
        
    '''
    get the train and test files
    '''
    def get_train_test_files(self, hold_time_min, hold_time_max, is_human, is_vehicle, is_uav):
        # if we have set the attributes, we quit
        if hasattr(self, 'train_filenames') and hasattr(self, 'test_filenames'):
            return;
        
        # we haven't set
        # calculate the relative indices for holding time
        hold_time_range = np.asarray(range(5));
        hold_time_idx = np.where((hold_time_range >= hold_time_min-1) & (hold_time_range <= hold_time_max-1));
        # select data with the hold time
        filename_human = self.filename_human[hold_time_idx];
        filename_vehicle = self.filename_vehicle[hold_time_idx];
        filename_uav = self.filename_uav[hold_time_idx];
        # now we select each file from 3 vessels to test
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
        # we build a bigger train & test filename
        train_filenames = [];
        test_filenames = [];
        if is_human:
            train_filenames.append(train_filename_human);
            test_filenames.append(test_filename_human);
        if is_vehicle:
            train_filenames.append(train_filename_vehicle);
            test_filenames.append(test_filename_vehicle);
        if is_uav:
            train_filenames.append(train_filename_uav);
            test_filenames.append(test_filename_uav);
        # merge
        train_filenames = np.concatenate(train_filenames, axis=-1);
        test_filenames = np.concatenate(test_filenames, axis=-1);
        # shuffle for the training file
        np.random.shuffle(train_filenames);
        # set
        self.train_filenames = train_filenames;
        self.train_files_len = len(train_filenames);
        self.test_filenames = test_filenames;
        self.test_files_len = len(test_filenames);
    
    '''
    reset our condition to the initial (while an epoch starts)
    '''
    def reset(self):
        # whether we have load all data
        self.is_all_data_loaded = False;
        # data loader index
        self.filename_cur_id = 0;
        self.seed_cur_id = 0;
        self.file_data = None;
        
    
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
    DataLoaderNS3.TAR_TYPE_PERIOD gives unequal length target
    output shape is 
    DataLoaderNS3.TAR_TYPE_PERIOD:                  [target_len, (rssi, time)]
    DataLoaderNS3.TAR_TYPE_BEACONS:                 [pred_beacon_num, (average rssi, beacon_start, beacon_end)]
    DataLoaderNS3.TAR_TYPE_NEXT_BEACON:             [average rssi]
    '''
    def get_memory_data(self, tar_type, pred_period, memory_len, start_time, column_end_time, column_rssi):
        if tar_type not in DataLoaderNS3.TAR_TYPES:
            raise Exception("The target type is not included in the ");
        
        # parameters settings
        data_len = len(column_rssi);
        idx_tar = None;                         # target  index
        cur_start_time = start_time;            # feature start time
        # we set return values
        is_filled = False;                      # whether we have enough data
        next_start_time = -1;                   # next beacon start time we should look at
        features = np.zeros((memory_len, 2));   # the meory data [memory_len, (rssi, time)] 
        target = None;                          # the target data
        
        # load data, every time we load a beacon data
        # `cur_start_time` tells the start time for each iteration
        while True:
            # find the end of this beacon period
            cur_end_time = cur_start_time + self.beacon_interval;
            #------------ jump point (when the beacon end time overflows)
            if cur_end_time not in column_end_time:
                break;
            
            # find the data packets between this beacon interval
            features_ids = np.where((column_end_time > cur_start_time) & (column_end_time < cur_end_time));
            features_ids = features_ids[0]; # only take 1st dimension
            for feature_id in features_ids:
                features = np.vstack(features, [column_rssi[feature_id], column_end_time[feature_id]]);
                features = np.delete(features, 0, axis=0);
            # we move the start time to the next beacon
            cur_start_time = cur_end_time;
            # if we still have 0s in features (in the early stage, we haven't loaded enough data) 
            if np.any(features == 0):
                continue;
            # now we have enough features, we need to the targets
            cur_pred_end = None;
            # target
            # target - `last known time` + `pred_periods`
            # target - [batch_size, target_len, (rssi, time)]
            # target - [target_len, (rssi, time)]
            if tar_type == DataLoaderNS3.TAR_TYPE_PERIOD:
                # set parameters
                cur_pred_end = cur_end_time + pred_period;
                # find the targets
                targets_ids = np.where();
                    # end time exists, it is a packet
                    if column_end_time[idx_tar] != 0: 
                        # add into the target
                        target.append([column_rssi[idx_tar], column_end_time[idx_tar]]);
                        # move to the next time point
                        idx_tar = idx_tar + 1;
                # take 1 step back to align with the actual data end 
                idx_tar = idx_tar - 1;
                # to numpy
                target = np.asarray(target);
                
            # target - `last beacon` + `pred_periods//beacon_interval`
            # target - [batch_size, pred_beacon_num, (average rssi, beacon_start, beacon_end)]
            # target - [pred_beacon_num, (average rssi, beacon_start, beacon_end)]
            # if tar_type == DataLoaderNS3.TAR_TYPE_BEACONS:
            #     # set parameters
            #     target_len = int(pred_periods//self.beacon_interval);           # this is just the maximal, we may not be able to fill beacuse our `pred_periods` is out of the data scope
            #     end_time_target = last_beacon_end_time + target_len*self.beacon_interval;
            #     target = np.zeros((target_len, 3));
            #     target_beacon_start = np.arange(last_beacon_end_time, last_beacon_end_time, self.beacon_interval);
                
            #     while column_end_time[idx_tar] == 0 and column_start_time[idx_tar] <= end_time_target and idx_tar < data_len:
            #         # if we start at a beacon, we clean everything
            #         if column_end_time[idx_tar] == 0:
            #             pass
                        
                        
                
            # target - `last beacon` + `next available beacon` in `last beacon` + `pred_periods`
            # target - [batch_size, average rssi]
            # target - [average rssi]
            # if tar_type == DataLoaderNS3.TAR_TYPE_NEXT_BEACON:
            #     # set parameters
            #     end_time_target = last_beacon_end_time + pred_periods;
            #     tar_sum = 0;
            #     tar_num = 0;
            #     target = np.zeros(2);
            
            # if our target is full, we jump
            # if our target is empty and `end_time_target` not outstripes, we continue
            # if our target is empty and `end_time_target` outstripes, we jump
        
        
        # return
        return is_filled, next_start_time, features, target;
    
    '''
    generate data
    @memory_len:                the memory length 
    @pred_periods:              5s by default, the maximal prediction periods(from the last known point)
    @tar_type:                  the type of target
    '''
    def __call__(self, memory_len, *, pred_periods=5):
        pred_beacon_num = int(pred_periods//self.beacon_interval);
        if pred_beacon_num <= 0:
            raise Exception("the prediction won't last a beacon interval");
        
        # load the temperary data for this turn
        if self.file_data is None:
            # reset the data
            self.file_data_idx = np.zeros((self.staids_len, 1));
            self.file_beacon_time = np.zeros((self.staids_len, 1));
            # refill the data
            self.file_data = [];
            for staid in self.staids:
                file_path_cur = "../../../NN-TimeLTSM-Data/" + self.train_filenames[self.filename_cur_id] + "/log/seed_000000000" + str(self.seeds[self.seed_cur_id]) + "/mac_rec/0000000000" + str(staid)  + ".csv"
                data_frame_tmp = pandas.read_csv(file_path_cur, header=None);
                data_tmp = data_frame_tmp.values;
                # retrieve the data
                self.file_data.append(data_tmp);
        # now we have temporary data
        # give the batched data
        # features  [batch_size, (rssi, time)]
        features = np.zeros((self.staids_len, memory_len, 2));
        # targets  
        targets_period = np.zeros((self.staids_len, 2));                      # [batch_size, (average rssi, end_time)]
        targets_beacons = np.zeros((self.staids_len, pred_beacon_num, 3));    # [batch_size, pred_beacon_num, (average rssi, beacon_start, beacon_end)]
        targets_next_beacon = np.zeros((self.staids_len, 1));                 # [batch_size, average rssi]
        
        # we build the data for each station
        for sta_id in range(self.staids_len):
            loaded_memory = 0;
            cur_data = self.file_data[sta_id];
            cur_data_beacon_time = self.file_beacon_time[sta_id];
            
            # split data into useful tags
            start_time = cur_data[:, 0];
            end_time = cur_data[:, 1];
            #mac_size = cur_data[:, 2];
            #phy_size = cur_data[:, 3];
            #snr_linear = cur_data[:, 4];
            rssi_linear = cur_data[:, 5];
            #mcs = data = cur_data[:, 6];
            
            # 1st we need to calibrate to the start of 
            
            
            
            is_filled, next_start_time, memory_data, future_data = self.get_memory_data(DataLoaderNS3.TAR_TYPE_PERIOD,
                                                                                        pred_periods,
                                                                                        memory_len,
                                                                                        cur_data_beacon_time,
                                                                                        end_time,
                                                                                        rssi_linear);
            print();
            
            
            
                
        
        
        
        # we have given all data in this file, we need to move to the next
        self.file_data = None;
        # now we move to the next seed
        self.seed_cur_id = self.seed_cur_id + 1;
        # seed is full, we need to move to the next file
        if self.seed_cur_id < self.seeds_len:
            self.seed_cur_id = 0;
            # move to the file
            self.filename_cur_id = self.filename_cur_id + 1;
            # file is full, we need to stop giving data
            if self.filename_cur_id >= self.train_files_len:
                self.is_all_data_loaded = True;
        