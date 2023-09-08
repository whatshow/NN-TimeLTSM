import numpy as np
import pandas

# this data loader requires NN to output 0s when take 0s in targets
class DataLoaderNS3Stawise:
    # constants
    # `last known time` + `pred_periods`
    # [batch_size, target_len, (rssi, time)]
    TAR_TYPE_PERIOD = 1;
    # `last beacon` + `pred_periods//beacon_interval`
    # [batch_size, pred_beacon_num, (average rssi, beacon_start, beacon_end)]
    TAR_TYPE_BEACONS = 2;
    # `last beacon` + `next available beacon` in `last beacon` + `pred_periods`
    # [batch_size, (average rssi, beacon_start, beacon_end)]
    TAR_TYPE_NEXT_BEACON = 3;
    TAR_TYPES = [TAR_TYPE_PERIOD, TAR_TYPE_BEACONS, TAR_TYPE_NEXT_BEACON];
    # the column meaning of the data file
    FILE_COL_START_TIME = 0;
    FILE_COL_END_TIME = 1;
    FILE_COL_MAC_SIZE = 2;
    FILE_COL_PHY_SIZE = 3;
    FILE_COL_SNR = 4;               # linear SNR
    FILE_COL_RSSI = 5;              # linear RSSI
    FILE_COL_MCS = 6;               # the predicted MCS for this transmission from the previous beacon
    # the way to uniform data length
    DATA_UNIFORM_TYPE_ZERO_PADDING = 1;
    DATA_UNIFORM_TYPE_CROP = 2;
    DATA_UNIFORM_TYPES = [DATA_UNIFORM_TYPE_ZERO_PADDING, DATA_UNIFORM_TYPE_CROP];
    
    
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
        self.next_beacon_interval_from_this_beacon_start = 3*self.beacon_interval;
        # config - seeds (used in the file system)
        #self.seeds = np.asarray([5, 6, 7]);
        self.seeds = np.arange(1, 10);
        self.seeds_len = len(self.seeds);
        # config - station id (6 + 15 + 16 + 7 + 2 + 14 + 2 + 14 + 10 + 19 + 19)
        # self.staids = ["0a", "0b", "0c", "0d", "0e", "0f",
        #           "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "ca", "cb", "cc", "cd", "ce", "cf", 
        #           "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "da", "db", "dc", "dd", "de", "df", 
        #           "01", "1a", "1b", "1c", "1d", "1e", "1f",
        #           "02", 
        #           "03",
        #           "04", "4a", "4b", "4c", "4d", "4e", "4f",
        #           "05", "5a", "5b", "5c", "5d", "5e", "5f",
        #           "06",
        #           "07",
        #           "08", "8a", "8b", "8c", "8d", "8e", "8f",
        #           "09", "9a", "9b", "9c", "9d", "9e", "9f",
        #           "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
        #           "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59",
        #           "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99"];
        self.staids = np.concatenate([[str(item).rjust(5, '0') for item in np.arange(1, 32)],
                                      [str(item).rjust(5, '0') for item in np.arange(65, 96)],
                                      [str(item).rjust(5, '0') for item in np.arange(129, 160)],
                                      [str(item).rjust(5, '0') for item in np.arange(193, 224)]], axis=-1);
        self.staids_len = len(self.staids);
        # config - filenames for human, vehicle, uav
        # filename_human = ["NNData_STA128_C00_rec_human_1",
        #                   "NNData_STA128_C00_rec_human_2",
        #                   "NNData_STA128_C00_rec_human_3",
        #                   "NNData_STA128_C00_rec_human_4",
        #                   "NNData_STA128_C00_rec_human_5",
        #                   ];
        # filename_vehicle = ["NNData_STA128_C00_rec_vehicle_1",
        #                     "NNData_STA128_C00_rec_vehicle_2",
        #                     "NNData_STA128_C00_rec_vehicle_3",
        #                     "NNData_STA128_C00_rec_vehicle_4",
        #                     "NNData_STA128_C00_rec_vehicle_5"];
        # filename_uav = ["NNData_STA128_C00_rec_uav_1",
        #                 "NNData_STA128_C00_rec_uav_2",
        #                 "NNData_STA128_C00_rec_uav_3",
        #                 "NNData_STA128_C00_rec_uav_4",
        #                 "NNData_STA128_C00_rec_uav_5"];
        filename_human = ["NNData_STA128_C00_rec_human_3",
                          "NNData_STA128_C00_rec_human_4",
                          "NNData_STA128_C00_rec_human_5",
                          ];
        filename_vehicle = ["NNData_STA128_C00_rec_vehicle_3",
                            "NNData_STA128_C00_rec_vehicle_4",
                            "NNData_STA128_C00_rec_vehicle_5"];
        filename_uav = ["NNData_STA128_C00_rec_uav_3",
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
            self.train_filenames = ["NNData_STA128_C00_rec_human_3"];
            self.train_files_len = 1;
            self.test_filenames = ["NNData_STA128_C00_rec_vehicle_3"];
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
        # hold_time_range = np.asarray(range(5));
        # hold_time_idx = np.where((hold_time_range >= hold_time_min-1) & (hold_time_range <= hold_time_max-1));
        
        # select data with the hold time
        # filename_human = self.filename_human[hold_time_idx];
        # filename_vehicle = self.filename_vehicle[hold_time_idx];
        # filename_uav = self.filename_uav[hold_time_idx];
        
        # now we select each file from 3 vessels to test
        # test_file_id_human = np.random.choice(len(filename_human), 1);       
        # test_file_id_vehicle = np.random.choice(len(filename_vehicle), 1);
        # test_file_id_uav = np.random.choice(len(filename_uav), 1);
        
        # separate train files and test files
        # test_filename_human = filename_human[test_file_id_human];
        # train_filename_human = np.delete(filename_human, test_file_id_human);
        # test_filename_vehicle = filename_vehicle[test_file_id_vehicle];
        # train_filename_vehicle = np.delete(filename_vehicle, test_file_id_vehicle);
        # test_filename_uav = filename_uav[test_file_id_uav];
        # train_filename_uav = np.delete(filename_uav, test_file_id_uav);
        
        # we build a bigger train & test filename
        train_filenames = [];
        test_filenames = [];
        if is_human:
            train_filenames.append(self.filename_human);
            #test_filenames.append(test_filename_human);
        if is_vehicle:
            train_filenames.append(self.filename_vehicle);
            #test_filenames.append(test_filename_vehicle);
        if is_uav:
            train_filenames.append(self.filename_uav);
            #test_filenames.append(test_filename_uav);
        # merge
        train_filenames = np.concatenate(train_filenames, axis=-1);
        #test_filenames = np.concatenate(test_filenames, axis=-1);
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
    DataLoaderNS3Stawise.TAR_TYPE_PERIOD gives unequal length target
    output shape is 
    DataLoaderNS3Stawise.TAR_TYPE_PERIOD:                  [target_len, (rssi, time)]
    DataLoaderNS3Stawise.TAR_TYPE_BEACONS:                 [pred_beacon_num, (average rssi, beacon_start, beacon_end)]
    DataLoaderNS3Stawise.TAR_TYPE_NEXT_BEACON:             [average rssi]
    '''
    def get_memory_data(self, tar_type, pred_period, memory_len, start_time, column_end_time, column_rssi):            
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
                features = np.vstack((features, [column_rssi[feature_id], column_end_time[feature_id]]));
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
            if tar_type == DataLoaderNS3Stawise.TAR_TYPE_PERIOD:
                # set the end of prediction
                target_time_start = feature_last_time;
                target_time_end = target_time_start + pred_period;
                # find the targets
                targets_ids = np.where((column_end_time > target_time_start) & (column_end_time <= target_time_end) & (column_rssi != 0));
                targets_ids = targets_ids[0];
                for target_id in targets_ids:
                    targets.append([column_rssi[target_id], column_end_time[target_id]]);                    
            # target - beacons: `last beacon` + `pred_periods//beacon_interval`
            if tar_type == DataLoaderNS3Stawise.TAR_TYPE_BEACONS:
                # set the end of prediction
                target_time_start = feature_beacon_end_time;
                targets_segment_num = int(pred_period//self.beacon_interval);
                target_time_end = target_time_start + targets_segment_num*self.beacon_interval;
                # find the targets
                for targets_segment_id in range(targets_segment_num):
                    targets_segment_start = target_time_start + targets_segment_id*self.beacon_interval;
                    targets_segment_end = target_time_start + (targets_segment_id+1)*self.beacon_interval;
                    targets_ids = np.where((column_end_time > targets_segment_start) & (column_end_time <= targets_segment_end) & (column_rssi != 0));
                    # add a segment
                    if len(targets_ids[0]) > 0:
                        targets.append([np.mean(column_rssi[targets_ids]), targets_segment_start, targets_segment_end]);
            # target - `last beacon` + `next available beacon` in `last beacon` + `pred_periods`
            # target - [batch_size, average rssi]
            # target - [average rssi]
            if tar_type == DataLoaderNS3Stawise.TAR_TYPE_NEXT_BEACON:
                # set the end of prediction
                target_time_start = feature_beacon_end_time + self.next_beacon_interval_from_this_beacon_start;
                target_time_end = target_time_start + self.beacon_interval;
                # find the targets
                targets_ids = np.where((column_end_time > target_time_start) & (column_end_time <= target_time_end) & (column_rssi != 0));
                if len(targets_ids[0]) > 0:
                    targets = [np.mean(column_rssi[targets_ids])];
                else:
                    # if this sta does not send any data, we fill 0
                    targets = [0];
            #------------ jump point (when targets is empty)
            if not targets:
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
    generate data
    <INPUT>
    @memory_len:                the memory length 
    @pred_periods:              5s by default, the maximal prediction periods(from the last known point)
    @tar_type:                  the type of target
    @data_uniform_type:         the way to uniform data length
    <OUTPUT>
    @features
    @targets
    <WARN>
    zero padding data fits the loss sum.
    crop data fits the loss sum or mean.
    '''
    def __call__(self, memory_len, *, pred_periods=5, tar_type=TAR_TYPE_PERIOD, data_uniform_type=DATA_UNIFORM_TYPE_CROP):
        # input check
        if tar_type not in DataLoaderNS3Stawise.TAR_TYPES:
            raise Exception("The target type is not included.");
        if data_uniform_type not in DataLoaderNS3Stawise.DATA_UNIFORM_TYPES:
            raise Exception("The data uniform type is not included.");
        
        # return values
        features = None;
        targets = None;
        
        # iteratively loading data
        while not self.is_all_data_loaded:
            features = [];
            targets = [];
            # load the temperary data for this turn
            if self.file_data is None:
                # reset the data
                self.file_beacon_time = np.zeros(self.staids_len);
                self.file_data = [];
                # refill the data
                for staid in self.staids:
                    file_path_cur = "../../../NN-TimeLTSM-Data/" + self.train_filenames[self.filename_cur_id] + "/log/seed_000000000" + str(self.seeds[self.seed_cur_id]) + "/ap_rec/" + staid  + ".csv"
                    data_frame_tmp = pandas.read_csv(file_path_cur, header=None);
                    data_tmp = data_frame_tmp.values;
                    # retrieve the data
                    self.file_data.append(data_tmp);
            # now we have temporary data
            is_this_file_not_enough = False;
            targets_2nd_dim_len_max = 0;
            targets_2nd_dim_len_min = -1;
            # we build the data for each station
            for sta_id in range(self.staids_len):
                cur_data = self.file_data[sta_id];
                cur_data_beacon_time = self.file_beacon_time[sta_id];  
                is_filled, next_start_time, memory_data, future_data = self.get_memory_data(tar_type,
                                                                                            pred_periods,
                                                                                            memory_len,
                                                                                            cur_data_beacon_time,
                                                                                            cur_data[:, DataLoaderNS3Stawise.FILE_COL_END_TIME],
                                                                                            cur_data[:, DataLoaderNS3Stawise.FILE_COL_RSSI]);
                # not enough for one station, we notify that this file is not enough
                if not is_filled:
                    is_this_file_not_enough = True;
                    break;
                else:
                    # update
                    self.file_beacon_time[sta_id] = next_start_time;
                    features.append(memory_data);
                    targets.append(future_data);
                    if tar_type == DataLoaderNS3Stawise.TAR_TYPE_PERIOD or DataLoaderNS3Stawise.TAR_TYPE_BEACONS:
                        future_data_len = len(future_data);
                        if targets_2nd_dim_len_max < future_data_len:
                            targets_2nd_dim_len_max = future_data_len;
                        if targets_2nd_dim_len_min == -1 or targets_2nd_dim_len_min > future_data_len:
                            targets_2nd_dim_len_min = future_data_len;
            
            # if there is not enough data, we go to the next file
            if is_this_file_not_enough:
                self.file_data = None;                                          # clean the space for the temporary data 
                self.seed_cur_id = self.seed_cur_id + 1;                        # now we move to the next seed
                # seed is full, we need to move to the next file
                if self.seed_cur_id >= self.seeds_len:
                    self.seed_cur_id = 0;                                       # we start from the 1st seed
                    self.filename_cur_id = self.filename_cur_id + 1;            # move to the file
                    # file is full, we need to stop giving data
                    if self.filename_cur_id >= self.train_files_len:
                        self.is_all_data_loaded = True;
                continue;                                                       # we need to try again
            else:
                # make the targets the same length
                if tar_type == DataLoaderNS3Stawise.TAR_TYPE_PERIOD or DataLoaderNS3Stawise.TAR_TYPE_BEACONS:
                    # zero padding base
                    zero_padding_element = None;
                    if tar_type == DataLoaderNS3Stawise.TAR_TYPE_PERIOD:
                        zero_padding_element = [[0, 0]];
                    if tar_type == DataLoaderNS3Stawise.TAR_TYPE_BEACONS:
                        zero_padding_element = [[0]];
                    # uniform data length
                    for sta_id in range(self.staids_len):
                        cur_target_len = len(targets[sta_id]);
                        # padding zeros
                        if data_uniform_type == DataLoaderNS3Stawise.DATA_UNIFORM_TYPE_ZERO_PADDING:
                            if cur_target_len < targets_2nd_dim_len_max:
                                zero_paddings = zero_padding_element*(targets_2nd_dim_len_max - cur_target_len);
                                targets[sta_id] = np.concatenate((targets[sta_id], zero_paddings));
                        # crop
                        if data_uniform_type == DataLoaderNS3Stawise.DATA_UNIFORM_TYPE_CROP:
                            if cur_target_len > targets_2nd_dim_len_min:
                                crop_elements_num = cur_target_len - targets_2nd_dim_len_min;
                                crop_elements_ids = np.random.permutation(cur_target_len);
                                crop_elements_ids = np.take(crop_elements_ids, np.arange(crop_elements_num));
                                targets[sta_id] = np.delete(targets[sta_id], crop_elements_ids, axis=0);
                # to numpy
                features = np.asarray(features);
                targets = np.asarray(targets);
                # we have enough data
                break;
        # return
        return self.is_all_data_loaded, features, targets;
    
    
    '''
    get the test data
    @sta_type: 0 human, 1 vehicle, 2 uav
    '''
    def get_test_data(self, memory_len, sta_type):
        pass
    
        
        