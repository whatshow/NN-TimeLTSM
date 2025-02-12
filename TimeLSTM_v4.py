import math
import numpy as np
import torch
from torch import nn

# macros
ERR_BUILD_NN_INIT_FUNCTION_WRONG_TYPE = "`nn_init_func` is not a recognised a type";
ERR_BUILD_NN_IN_FEATURE_NUM_ILLEGAL = "`nn_in_feature_num` must be a positive integer";
ERR_BUILD_NN_OUT_FEATURE_NUM_ILLEGAL = "`nn_out_feature_num` must be a positive integer";
ERR_INIT_NN_TYPE_WRONG = "`hidden_neuron_type` is not a recognised type";
ERR_INIT_NN_INIT_FUNC_TYPE_WRONG = "`nn_init_func_type` is not a recognised type"
ERR_FORWARD_X_ILLEGAL = "`x` can have 3 or 4 dimensions, [batch, n, xm] or [batch, n, xm, 1]: xm is the features number of x";
# TimeLSTM takes two inputs
# the neurons are registered as https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
# [batch, n, xm]: xm is the features we measure for x
# [batch, n, tm]: tm is the features we measure for time, in our case tm=3
class TimeLSTM_v3(nn.Module):
    # constants
    # nn types
    NN_TYPE_LSTM_ALEX_GRAVES = 0;
    NN_TYPE_LSTM_PHASE = 1;
    NN_TYPE_LSTM_TIME1 = 2;
    NN_TYPE_LSTM_TIME2 = 3;
    NN_TYPE_LSTM_TIME3 = 4;
    NN_TYPES = [NN_TYPE_LSTM_ALEX_GRAVES, NN_TYPE_LSTM_PHASE, NN_TYPE_LSTM_TIME1, NN_TYPE_LSTM_TIME2, NN_TYPE_LSTM_TIME3];
    # NN init types
    NN_INIT_TYPE_ZERO = 0;
    NN_INIT_TYPE_ONE = 1;
    NN_INIT_TYPE_RANDN = 9;
    NN_INIT_TYPES = [NN_INIT_TYPE_ZERO, NN_INIT_TYPE_ONE, NN_INIT_TYPE_RANDN];
    
    
    '''
    build the parameters
    @nn_init_func:                  the function to init nn parameters 
    @nn_in_feature_num:             input feature number
    @nn_out_feature_num:            output feature number
    @nn_type:                       
    @nn_time_feature_num:           input feature number (time)
    '''
    def build(self, nn_init_func, nn_in_feature_num, nn_out_feature_num, *, nn_type=NN_TYPE_LSTM_ALEX_GRAVES, nn_time_feature_num=1):
        # input check
        if nn_init_func not in [torch.zeros, torch.ones, torch.randn]:
            raise Exception(ERR_BUILD_NN_INIT_FUNCTION_WRONG_TYPE);
        if not isinstance(nn_in_feature_num, int):
            raise Exception(ERR_BUILD_NN_IN_FEATURE_NUM_ILLEGAL);
        elif nn_in_feature_num <= 0:
            raise Exception(ERR_BUILD_NN_IN_FEATURE_NUM_ILLEGAL);
        if not isinstance(nn_out_feature_num, int):
            raise Exception(ERR_BUILD_NN_OUT_FEATURE_NUM_ILLEGAL);
        elif nn_out_feature_num <= 0:
            raise Exception(ERR_BUILD_NN_OUT_FEATURE_NUM_ILLEGAL);
        
        # create nn parameters
        # NN parameters
        # weight:                   [gate namte]_w_[input name] 
        # bias:                     [gate name]_b
        # nolinearity function:     [gate name]_func
        # forget gate
        self.fg_w_c = nn.Parameter(nn_init_func(nn_out_feature_num, 1));
        self.fg_w_h = nn.Parameter(nn_init_func(nn_out_feature_num, nn_out_feature_num));
        self.fg_w_x = nn.Parameter(nn_init_func(nn_out_feature_num, nn_in_feature_num));
        self.fg_b = nn.Parameter(nn_init_func(nn_out_feature_num, 1));
        # input gate
        self.ig_w_c = nn.Parameter(nn_init_func(nn_out_feature_num, 1));
        self.ig_w_h = nn.Parameter(nn_init_func(nn_out_feature_num, nn_out_feature_num));
        self.ig_w_x = nn.Parameter(nn_init_func(nn_out_feature_num, nn_in_feature_num));
        self.ig_b = nn.Parameter(nn_init_func(nn_out_feature_num, 1));
        # input node
        self.in_w_h = nn.Parameter(nn_init_func(nn_out_feature_num, nn_out_feature_num));
        self.in_w_x = nn.Parameter(nn_init_func(nn_out_feature_num, nn_in_feature_num));
        self.in_b = nn.Parameter(nn_init_func(nn_out_feature_num, 1));
        # output gate
        self.og_w_cn = nn.Parameter(nn_init_func(nn_out_feature_num, 1));
        self.og_w_h = nn.Parameter(nn_init_func(nn_out_feature_num, nn_out_feature_num));
        self.og_w_x = nn.Parameter(nn_init_func(nn_out_feature_num, nn_in_feature_num));
        self.og_b = nn.Parameter(nn_init_func(nn_out_feature_num, 1));
        
        # remove the forget gate (for time lstm 3)
        # output gate
        self.og_w_t = nn.Parameter(nn_init_func(nn_out_feature_num, 1));
        # time gate 1
        self.tg1_w_x = nn.Parameter(nn_init_func(nn_out_feature_num, nn_in_feature_num));
        self.tg1_w_t = nn.Parameter(nn_init_func(nn_out_feature_num, nn_time_feature_num));
        self.tg1_b = nn.Parameter(nn_init_func(nn_out_feature_num, 1));
        self.tg1_func_t = torch.tanh;
        self.tg1_func = torch.sigmoid;
        # time gate 2
        self.tg2_w_x = nn.Parameter(nn_init_func(nn_out_feature_num, nn_in_feature_num));
        self.tg2_w_t = nn.Parameter(nn_init_func(nn_out_feature_num, nn_time_feature_num));
        self.tg2_b = nn.Parameter(nn_init_func(nn_out_feature_num, 1));
        self.tg2_func_t = torch.tanh;
        self.tg2_func = torch.sigmoid;
        
    
    '''
    init
    @nn_init_func_type:     the initial function for learnable parameters 
    @nn_in_feature_num:     the input feature number
    @nn_out_feature_num:    the output feature number we want
    '''
    def __init__(self, nn_init_func_type, nn_in_feature_num, nn_out_feature_num):
        super(TimeLSTM_v4, self).__init__();
        # input check
        if nn_init_func_type not in self.NN_INIT_TYPES:
            raise Exception(ERR_INIT_NN_INIT_FUNC_TYPE_WRONG);
        # create nn init functions
        nn_init_func = None;
        if nn_init_func_type == self.NN_INIT_TYPE_ZERO:
            nn_init_func = torch.zeros;
        if nn_init_func_type == self.NN_INIT_TYPE_ONE:
            nn_init_func = torch.ones;
        if nn_init_func_type == self.NN_INIT_TYPE_RANDN:
            nn_init_func = torch.randn;
        
        # record input & output feature number
        self.nn_in_feature_num = nn_in_feature_num;
        self.nn_out_feature_num = nn_out_feature_num;
        
        # build
        self.build(nn_init_func, nn_in_feature_num, nn_out_feature_num);
        
        # average the weight
        stdv = 1.0/math.sqrt(nn_out_feature_num);
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv);
        
    '''
    forward
    <INSTRUCTION>
    The time has 3 features (tm<=3) at most
        1) tm1 is the time point or the time difference
        2) tm2 is the predict start
        3) tm3 is the predict end
    <INPUT>
    @x: the feature input [batch, n, xm] (xm is the feature number)
    @t: the time input [batch, n, 1] (the time point)
    @pred_start: the prediction start [batch, 1] (could be None, then we use the )
    @pred_end: the prediction end [batch, 1]
    <OUTPUT>
    [batch, (rssi_pred, last_time)], `last_time` could be a specific time point or calibrated to an end of a beacon
    [batch, (rssi_pred)], `rssi_pred` is the for the next available beacon
    <LOSS CALCULATION>
    The target could be zero, we have to set `reduction='none'`, manually remove the losses with 0 targets and calculate the mean of the loss
    '''
    def forward(self, x, *, t=None, pred_start=None, pred_end=None):
        # input check
        if not (x.ndim == 3 or x.ndim == 4 and x.shape[-1] != 1):
            raise Exception(ERR_FORWARD_X_ILLEGAL);
        # extend the dimension of x from [batch, n, xm] to [batch, n, xm, 1]
        if x.ndim == 3:
            x = torch.unsqueeze(x, -1);
            
        # get the memory length
        memory_len = x.shape[-3];
        # get batch size
        batch_size = x.shape[0];
        # get the current device
        device_cur = self.og_b.device;
        # fill in the initial C and H if not given
        cm = torch.zeros(batch_size, self.nn_out_feature_num, 1, dtype=torch.float32).to(device_cur);
        h = torch.zeros(batch_size, self.nn_out_feature_num, 1, dtype=torch.float32).to(device_cur);
        
        # iteratively train data
        for memory_id in range(memory_len):
            x_cur = x[:, memory_id, :, :];
            
            # input gate
            ig = torch.sigmoid(self.ig_w_c*cm + self.ig_w_h @ h + self.ig_w_x @ x_cur + self.ig_b);
            # forget gate
            fg = torch.sigmoid(self.fg_w_c*cm + self.fg_w_h @ h + self.fg_w_x @ x_cur + self.fg_b);
            # input node
            inn = torch.tanh(self.in_w_h @ h + self.in_w_x @ x_cur + self.in_b); 
            # cell memory node
            cm = fg*cm + ig*inn;
            # output gate
            og = torch.sigmoid(self.og_w_cn*cm + self.og_w_h @ h + self.og_w_x @ x_cur + self.og_b);
            # hidden state
            h = og*torch.tanh(cm);
        
        # remove the last dimension for cm and h
        # e.g., they should be [batch, ?]  ? is the output feature number 
        cm = torch.squeeze(cm, -1);
        h = torch.squeeze(h, -1);
        # return
        return h;