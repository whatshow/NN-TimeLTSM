import numpy as np

class ZScoreScaler:
    mean = 0;
    variance = 0;
    st_deviation = 0;
    
    
    '''
    find the mean and the variance of the data
    '''
    def fit(self, data_x, data_y):        
        # merge
        # merge - x
        data_x_merged = np.concatenate(data_x, axis=0);
        data_x_merged_rssi = data_x_merged[:, :, :, 0];
        # merge - y
        data_y_merged = np.concatenate(data_y, axis=0);
        # merge - x & y
        data_x_merged_rssi = data_x_merged_rssi.flatten();
        data_y_merged = data_y_merged.flatten();
        data = np.concatenate([data_x_merged_rssi, data_y_merged], axis=0);
        # gate the mean and the variance
        self.mean = np.mean(data);
        self.variance = np.var(data);
        self.st_deviation = np.sqrt(self.variance);
    
    
    def transform(self, data):
        data_len = len(data);
        for data_id in range(data_len):
            data[data_id] = (data[data_id] - self.mean)/self.st_deviation;
        return data;
    
    def inverse_transform(self, data):
        data_len = len(data);
        for data_id in range(data_len):
            data[data_id] = data[data_id]*self.st_deviation + self.mean;
        return data;