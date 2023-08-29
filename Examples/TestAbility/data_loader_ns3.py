import numpy as np
import pandas


class DataLoaderNS3:
    '''
    @hold_time_min:             the minimal holdtime (minimum has a higher priority than maximum)
    @hold_time_max:             the maximal holdtime
    @is_human:                  we consider humans
    @is_vehicle:                we consider vehicle
    @is_uav:                    we consider uav
    '''
    def __init__(self, *, hold_time_min=1, hold_time_max=5, is_human=True, is_vehicle=True, is_uav=True):
        # beacon interval
        self.beacon_interval = 0.5;
        # config - seeds (used in the file system)
        self.seeds = np.asarray([5, 6, 7]);
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
    reset our condition to the initial
    '''
    def reset(self):
        # whether we have load all data
        self.is_all_data_loaded = False;
        # data loader index
        self.filename_cur_id = 0;
        self.seed_cur_id = 0;
        self.file_data = None;
    
    '''
    generate data
    @memory_len:                the memory length 
    @pred_periods:              5s by default, the maximal prediction periods(from the last known point)
    @is_target_period:          [batch_size, (average rssi, end_time)], last known time + `pred_periods`
    @is_target_beacons:         [batch_size, pred_beacon_num, (average rssi, beacon_start, beacon_end)], last beacon + `pred_periods`
    @is_target_next_beacon:     [batch_size, average rssi], next beacon average rssi. If not data, we move until there is data
    '''
    def __call__(self, memory_len, *, pred_periods=5):
        pred_beacon_num = int(pred_periods//self.beacon_interval);
        if pred_beacon_num <= 0:
            raise Exception("the prediction won't last a beacon interval");
        
        # load the temperary data for this turn
        if self.file_data is None:
            # reset the data
            self.file_data_idx = np.zeros((self.staids_len, 1));
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
            cur_data_idx = self.file_data_idx[sta_id];
            # start_time = data[:, 0];
            # end_time = data[:, 1];
            # mac_size = data[:, 2];
            # phy_size = data[:, 3];
            # snr_linear = data[:, 4];
            # rssi_linear = data[:, 5];
            # mcs = data = data[:, 6];
            # load the training features
            while loaded_memory < memory_len:
                # has end data, we put
                if cur_data[cur_data_idx, 1] > 0:
                    features[sta_id];
                
                cur_data_idx = cur_data_idx + 1;
                loaded_memory = loaded_memory + 1;
            
            
                
        
        
        
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
        