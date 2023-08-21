import numpy as np
import torch
from torch import nn

ERR_INIT_BUILD_NN_INIT_FUNCTION_WRONG_TYPE = "`nn_init_func` is not a recognised a type";
ERR_INIT_BUILD_NN_IN_FEATURE_NUM_ILLEGAL = "`nn_in_feature_num` must be a positive integer";
ERR_INIT_BUILD_NN_OUT_FEATURE_NUM_ILLEGAL = "`nn_out_feature_num` must be a positive integer";
ERR_forward_x_WRONG_DIM = "`x` can have 3 or 4 dimensions, [batch, n, xm] or [batch, n, xm, 1]: xm is the features number of x";
ERR_forward_n_OVERFLOW = "`n` outstripes the time series length in x (2nd dimesion)";
ERR_forward_C_MISSING = "`C` must exist starting from the 2nd time series data";
ERR_forward_H_MISSING = "`H` must exist starting from the 2nd time series data";

# LSTM
# [batch, n, xm]: xm is the features we measure for x
# the name of gates and nodes are inspired from https://d2l.ai/chapter_recurrent-modern/lstm.html
class LSTM_AlexGraves(nn.Module):
    # constants
    # NN dimensions
    NN_FEATURE_NUM = 1;             # xm=1, so is the output feature number
    
    # properties
    nn_type = None;                 # the type of NN
    nn_in_feature_num = None;       # input feature number
    nn_out_feature_num = None;      # output feature number
    
    
    '''
    build the parameters
    @nn_init_func:                  the function to init nn parameters 
    @nn_in_feature_num:             input feature number
    nn_out_feature_num:             output feature number
    '''
    def build(self, nn_init_func, nn_in_feature_num, nn_out_feature_num):
        # input check
        if nn_init_func not in [torch.zeros, torch.ones, torch.randn]:
            raise Exception(ERR_INIT_BUILD_NN_INIT_FUNCTION_WRONG_TYPE);
        if not isinstance(nn_in_feature_num, int):
            raise Exception(ERR_INIT_BUILD_NN_IN_FEATURE_NUM_ILLEGAL);
        elif nn_in_feature_num <= 0:
            raise Exception(ERR_INIT_BUILD_NN_IN_FEATURE_NUM_ILLEGAL);
        if not isinstance(nn_out_feature_num, int):
            raise Exception(ERR_INIT_BUILD_NN_OUT_FEATURE_NUM_ILLEGAL);
        elif nn_out_feature_num <= 0:
            raise Exception(ERR_INIT_BUILD_NN_OUT_FEATURE_NUM_ILLEGAL);
        
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
        self.fg_func = nn.Sigmoid();
        # input gate
        self.ig_w_c = nn.Parameter(nn_init_func(nn_out_feature_num, 1));
        self.ig_w_h = nn.Parameter(nn_init_func(nn_out_feature_num, nn_out_feature_num));
        self.ig_w_x = nn.Parameter(nn_init_func(nn_out_feature_num, nn_in_feature_num));
        self.ig_b = nn.Parameter(nn_init_func(nn_out_feature_num, 1));
        self.ig_func = nn.Sigmoid();
        # input node
        self.in_w_h = nn.Parameter(nn_init_func(nn_out_feature_num, nn_out_feature_num));
        self.in_w_x = nn.Parameter(nn_init_func(nn_out_feature_num, nn_in_feature_num));
        self.in_b = nn.Parameter(nn_init_func(nn_out_feature_num, 1));
        self.in_func = nn.Tanh();
        # output gate
        self.og_w_cn = nn.Parameter(nn_init_func(nn_out_feature_num, 1));
        self.og_w_h = nn.Parameter(nn_init_func(nn_out_feature_num, nn_out_feature_num));
        self.og_w_x = nn.Parameter(nn_init_func(nn_out_feature_num, nn_in_feature_num));
        self.og_b = nn.Parameter(nn_init_func(nn_out_feature_num, 1));
        self.og_func = nn.Sigmoid();
        # hidden state (output)
        self.ho_func = nn.Tanh();
    
    
    '''
    init one LSTM node
    @nn_init_func:              the function to init nn parameters 
    @nn_in_feature_num:         input feature number
    @nn_out_feature_num:        output feature number
    '''
    def __init__(self, *, nn_init_func=torch.randn, nn_in_feature_num=NN_FEATURE_NUM, nn_out_feature_num=NN_FEATURE_NUM):
        # init the model
        super(LSTM_AlexGraves, self).__init__();
        
        # record input & output feature number
        self.nn_in_feature_num = nn_in_feature_num;
        self.nn_out_feature_num = nn_out_feature_num;
        
        # build
        self.build(nn_init_func, nn_in_feature_num, nn_out_feature_num);
        
        
        # memory cell node          (only used for this NN node, abbr. cn)
        # memory cell node esti     (only used for this NN node, abbr. cne)
        # time gate 1 
        # time gate 2
    
    '''
    forward
    @x: the input [batch, n, xm]: xm is the features we measure for x
    @n: the selected index for time series x
    @C: memory cell  [batch, ?, 1]: ? in the output feature number (from the last iteration, only missing when n=0)
    @H: hidden state [batch, ?, 1]: ? in the output feature number (from the last iteration, only missing when n=0)
    '''
    def forward(self, x, n, *, C = None, H = None):
        # input check
        if not (x.ndim == 3 or x.ndim == 4 and x.shape[-1] != 1):
            raise Exception(ERR_forward_x_WRONG_DIM);
        if n >= x.shape[1]:
            raise Exception(ERR_forward_n_OVERFLOW);
        if n > 0 and C is None:
            raise Exception(ERR_forward_C_MISSING);
        if n > 0 and H is None:
            raise Exception(ERR_forward_H_MISSING);
        # extend the dimension of x from [batch, n, xm] to [batch, n, xm, 1]
        if x.ndim == 3:
            x = torch.unsqueeze(x, -1);
        # get the current x from the time series x
        # x_cur: [batch, xm, 1]: xm is the features we measure for x
        x_cur = x[:, n, :, :];
        # get batch size
        batch_size = x.shape[0];
        # fill in the initial C and H if not given
        device_cur = self.og_b.device;
        if C is None:
            C = torch.zeros(batch_size, self.nn_out_feature_num, 1).to(device_cur);
        if H is None:
            H = torch.zeros(batch_size, self.nn_out_feature_num, 1).to(device_cur);
        
        # forget gate
        fg = self.fg_func(self.fg_w_c*C + torch.matmul(self.fg_w_h, H) + torch.matmul(self.fg_w_x, x_cur) + self.fg_b);
        # input gate
        ig = self.ig_func(self.ig_w_c*C + torch.matmul(self.ig_w_h, H) + torch.matmul(self.ig_w_x, x_cur) + self.ig_b);
        # input node
        inn = self.in_func(torch.matmul(self.in_w_h, H) + torch.matmul(self.in_w_x, x_cur) + self.in_b); 
        # cell memory node
        cm = fg*C + ig*inn;
        # output gate
        og = self.og_func(self.og_w_cn*cm + torch.matmul(self.og_w_h, H) + torch.matmul(self.og_w_x, x_cur) + self.og_b);
        # hidden state
        h = og*self.ho_func(cm);
        
        # return
        return cm, h;