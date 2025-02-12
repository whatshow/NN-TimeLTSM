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
ERR_FORWARD_T_ILLEGAL = "`t` can have 3 or 4 dimensions, [batch, n, 1] or [batch, n, 1, 1]";
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
    
    # the type of this layer
    nn_type = None;
    # the precision of data
    precision = torch.float64;
    
    '''
    build the parameters
    @nn_init_func:                  the function to init nn parameters 
    @nn_in_feature_num:             input feature number
    @nn_out_feature_num:            output feature number
    @nn_type:                       
    '''
    def build(self, nn_init_func, nn_in_feature_num, nn_out_feature_num):
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
        # only when not time gate type 3 is chosen
        if self.nn_type != TimeLSTM_v3.NN_TYPE_LSTM_TIME3:
            self.fg_w_c = nn.Parameter(nn_init_func(nn_out_feature_num, 1, dtype=self.precision));
            self.fg_w_h = nn.Parameter(nn_init_func(nn_out_feature_num, nn_out_feature_num, dtype=self.precision));
            self.fg_w_x = nn.Parameter(nn_init_func(nn_out_feature_num, nn_in_feature_num, dtype=self.precision));
            self.fg_b = nn.Parameter(nn_init_func(nn_out_feature_num, 1, dtype=self.precision));
        # input gate
        self.ig_w_c = nn.Parameter(nn_init_func(nn_out_feature_num, 1, dtype=self.precision));
        self.ig_w_h = nn.Parameter(nn_init_func(nn_out_feature_num, nn_out_feature_num, dtype=self.precision));
        self.ig_w_x = nn.Parameter(nn_init_func(nn_out_feature_num, nn_in_feature_num, dtype=self.precision));
        self.ig_b = nn.Parameter(nn_init_func(nn_out_feature_num, 1, dtype=self.precision));
        # input node
        self.in_w_h = nn.Parameter(nn_init_func(nn_out_feature_num, nn_out_feature_num, dtype=self.precision));
        self.in_w_x = nn.Parameter(nn_init_func(nn_out_feature_num, nn_in_feature_num, dtype=self.precision));
        self.in_b = nn.Parameter(nn_init_func(nn_out_feature_num, 1, dtype=self.precision));
        # output gate
        self.og_w_cn = nn.Parameter(nn_init_func(nn_out_feature_num, 1, dtype=self.precision));
        self.og_w_h = nn.Parameter(nn_init_func(nn_out_feature_num, nn_out_feature_num, dtype=self.precision));
        self.og_w_x = nn.Parameter(nn_init_func(nn_out_feature_num, nn_in_feature_num, dtype=self.precision));
        self.og_b = nn.Parameter(nn_init_func(nn_out_feature_num, 1, dtype=self.precision));
        # output gate - time interval weight
        # all 3 time gate types are chosen
        if self.nn_type in [TimeLSTM_v3.NN_TYPE_LSTM_TIME1, TimeLSTM_v3.NN_TYPE_LSTM_TIME2, TimeLSTM_v3.NN_TYPE_LSTM_TIME3]:
            self.og_w_t = nn.Parameter(nn_init_func(nn_out_feature_num, 1, dtype=self.precision));
        # time gate 1
        # all 3 time gate types are chosen
        if self.nn_type in [TimeLSTM_v3.NN_TYPE_LSTM_TIME1, TimeLSTM_v3.NN_TYPE_LSTM_TIME2, TimeLSTM_v3.NN_TYPE_LSTM_TIME3]:
            self.tg1_w_x = nn.Parameter(nn_init_func(nn_out_feature_num, nn_in_feature_num, dtype=self.precision));
            self.tg1_w_t = nn.Parameter(nn_init_func(nn_out_feature_num, 1, dtype=self.precision));
            self.tg1_b = nn.Parameter(nn_init_func(nn_out_feature_num, 1, dtype=self.precision));
        # time gate 2
        # all time gate type 2, 3 are chosen
        if self.nn_type in [TimeLSTM_v3.NN_TYPE_LSTM_TIME2, TimeLSTM_v3.NN_TYPE_LSTM_TIME3]:
            self.tg2_w_x = nn.Parameter(nn_init_func(nn_out_feature_num, nn_in_feature_num, dtype=self.precision));
            self.tg2_w_t = nn.Parameter(nn_init_func(nn_out_feature_num, 1, dtype=self.precision));
            self.tg2_b = nn.Parameter(nn_init_func(nn_out_feature_num, 1, dtype=self.precision));
    
    '''
    init
    @nn_init_func_type:     the initial function for learnable parameters 
    @nn_in_feature_num:     the input feature number
    @nn_out_feature_num:    the output feature number we want
    @nn_type:               the tpye of the neural network
    '''
    def __init__(self, nn_init_func_type, nn_in_feature_num, nn_out_feature_num, *, nn_type=NN_TYPE_LSTM_ALEX_GRAVES, precision=torch.float64):
        super(TimeLSTM_v3, self).__init__();
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
        # record the type
        self.nn_type = nn_type;
        # record the presion
        self.precision = precision;
        
        # build
        self.build(nn_init_func, nn_in_feature_num, nn_out_feature_num);
        
        # average the weight
        stdv = 1.0/math.sqrt(nn_out_feature_num);
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv);
        
    '''
    forward
    <INPUT>
    @x:     the feature input [batch, n, xm] (xm is the feature number)
    @t:     the time input [batch, n, 1] (the time interval, the 1st interval is 0)
    @cm:    the cell memory from the previous training
    '''
    def forward(self, x, *, t=None, cm=None):
        # input check
        if not (x.ndim == 3 or x.ndim == 4 and x.shape[-1] != 1):
            raise Exception(ERR_FORWARD_X_ILLEGAL);
        if t is not None:
            if not (x.ndim == 3 or x.ndim == 4 and x.shape[-1] != 1):
                raise Exception(ERR_FORWARD_T_ILLEGAL);
        # extend the dimension of x from [batch, n, xm] to [batch, n, xm, 1]
        if x.ndim == 3:
            x = torch.unsqueeze(x, -1);
        if t is not None:
            if t.ndim == 3:
                t = torch.unsqueeze(t, -1);
            
        # get the memory length
        memory_len = x.shape[-3];
        # get batch size
        batch_size = x.shape[0];
        # get the current device
        device_cur = self.og_b.device;
        # fill in the initial C
        if cm is None:
            cm = torch.zeros(batch_size, self.nn_out_feature_num, 1, dtype=self.precision).to(device_cur);
        # the 1st input always has not hidden states
        h = torch.zeros(batch_size, self.nn_out_feature_num, 1, dtype=self.precision).to(device_cur);
        
        # iteratively train data
        for memory_id in range(memory_len):
            x_cur = x[:, memory_id, :, :];
            if t is not None:
                t_cur = t[:, memory_id, :, :];
            # input gate
            ig = torch.sigmoid(self.ig_w_c*cm + self.ig_w_h @ h + self.ig_w_x @ x_cur + self.ig_b);
            # forget gate (not TimeLSTM type 3)
            if self.nn_type != TimeLSTM_v3.NN_TYPE_LSTM_TIME3:
                fg = torch.sigmoid(self.fg_w_c*cm + self.fg_w_h @ h + self.fg_w_x @ x_cur + self.fg_b);
            # input node
            inn = torch.tanh(self.in_w_h @ h + self.in_w_x @ x_cur + self.in_b); 
            # others
            if self.nn_type == TimeLSTM_v3.NN_TYPE_LSTM_ALEX_GRAVES:
                # cell memory node
                cm = fg*cm + ig*inn;
                # output gate
                og = torch.sigmoid(self.og_w_cn*cm + self.og_w_h @ h + self.og_w_x @ x_cur + self.og_b);
                # hidden state
                h = og*torch.tanh(cm);
            elif self.nn_type == TimeLSTM_v3.NN_TYPE_LSTM_TIME1:
                # time gates
                tm1 = torch.sigmoid(self.tg1_w_x @ x_cur + torch.tanh(self.tg1_w_t @ t_cur) + self.tg1_b);
                # cell memory node
                cm = fg*cm + ig*tm1*inn;
                # output gate
                og = torch.sigmoid(self.og_w_cn*cm + self.og_w_t @ t_cur + self.og_w_h @ h + self.og_w_x @ x_cur + self.og_b);
                # hidden state
                h = og*torch.tanh(cm);
            elif self.nn_type == TimeLSTM_v3.NN_TYPE_LSTM_TIME2:
                # time gates
                tm1 = torch.sigmoid(self.tg1_w_x @ x_cur + torch.tanh(self.tg1_w_t @ t_cur) + self.tg1_b);
                tm2 = torch.sigmoid(self.tg2_w_x @ x_cur + torch.tanh(self.tg2_w_t @ t_cur) + self.tg2_b);
                # cell memory node
                cm_hat = fg*cm + ig*tm1*inn;
                cm = fg*cm + ig*tm2*inn;
                # output gate
                og = torch.sigmoid(self.og_w_cn*cm_hat + self.og_w_t @ t_cur + self.og_w_h @ h + self.og_w_x @ x_cur + self.og_b);
                # hidden state
                h = og*torch.tanh(cm_hat);
            elif self.nn_type == TimeLSTM_v3.NN_TYPE_LSTM_TIME3:
                # time gates
                tm1 = torch.sigmoid(self.tg1_w_x @ x_cur + torch.tanh(self.tg1_w_t @ t_cur) + self.tg1_b);
                tm2 = torch.sigmoid(self.tg2_w_x @ x_cur + torch.tanh(self.tg2_w_t @ t_cur) + self.tg2_b);
                # cell memory node
                cm_hat = (1 - ig*tm1)*cm + ig*tm1*inn;
                cm = (1 - ig*tm2)*cm + ig*tm2*inn;
                # output gate
                og = torch.sigmoid(self.og_w_cn*cm_hat + self.og_w_t @ t_cur + self.og_w_h @ h + self.og_w_x @ x_cur + self.og_b);
                # hidden state
                h = og*torch.tanh(cm_hat);

        # remove the last dimension for h
        # e.g., they should be [batch, ?]  ? is the output feature number 
        h = torch.squeeze(h, -1);
        # cm is [batch, ?, 1] as the long memory staying inside this cell
    
        # return
        return h, cm;