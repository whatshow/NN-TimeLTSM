import numpy as np
import torch
from torch import nn

ERR_SetDevice_device_WRONG_TYPE = "TimeLSTM/SetDevice(): [device] is not a torch device"
ERR_init_nn_type_WRONG_TYPE = "TimeLSTM/init(): [nn_type] is not a recognised a type";
ERR_init_nn_init_type_WRONG_TYPE = "TimeLSTM/init(): [nn_init_type] is not a recognised a type";
ERR_init_nn_feature_num_WRONG_TYPE = "TimeLSTM/init(): [nn_feature_num] must be an integer";
ERR_forward_x_WRONG_DIM = "TimeLSTM/forward(): [x] can only have 2 dimensions, i.e., [batch, x]: x=[x1, x2 ... xm] is the features we measure for x";

# TimeLSTM takes two inputs
# [batch, x]: x=[x1, x2 ... xm] is the features we measure for x
# [batch, 1]: time point for the current x
# the name of gates and nodes are inspired from https://d2l.ai/chapter_recurrent-modern/lstm.html
class TimeLSTM(nn.Module):
    # constants
    # NN dimensions
    NN_FEATURE_NUM = 1;             # [batch, t, x1, x2 ... xm]: x=1
    # NN init types
    NN_INIT_TYPE_0 = 0;             # all parameters are intialised into 0
    NN_INIT_TYPE_1 = 1;             # all parameters are initialised into 1
    NN_INIT_TYPE_RAND = 4;          # all parameters are initialsed randomly
    NN_INIT_TYPES = [NN_INIT_TYPE_0, NN_INIT_TYPE_1, NN_INIT_TYPE_RAND];
    # NN types
    NN_TYPE_LSTM_ALEX_GRAVES = 1;   # from Alex Graves in `Generating Sequences With RNN` (2013)
    NN_TYPE_LSTM_PHASE = 2;
    NN_TYPE_LSTM_TIME_1 = 3;        # TimeLSTM1
    NN_TYPE_LSTM_TIME_2 = 4;        # TimeLSTM2
    NN_TYPE_LSTM_TIME_3 = 5;        # TimeLSTM3
    NN_TYPES = [NN_TYPE_LSTM_ALEX_GRAVES, NN_TYPE_LSTM_PHASE, NN_TYPE_LSTM_TIME_1, NN_TYPE_LSTM_TIME_2, NN_TYPE_LSTM_TIME_3];
    # device
    
    # properties
    nn_type = None;                 # the type of NN
    nn_in_feature_num = None;       # input feature number
    nn_out_feature_num = None;      # output feature number
    
    # NN parameters
    # weight:                   [gate namte]_w_[input name] 
    # bias:                     [gate name]_b
    # nolinearity function:     [gate name]_func
    # forget gate
    fg_w_c = None;
    fg_w_h = None;
    fg_w_x = None;
    fg_b = None;
    fg_func = nn.Sigmoid;
    # input gate
    ig_w_c = None;
    ig_w_h = None;
    ig_w_x = None;
    ig_b = None;
    ig_func = nn.Sigmoid;
    # input node
    in_w_h = None;
    in_w_x = None;
    in_b = None;
    in_func = nn.Tanh;
    # memory cell node          (only used for this NN node, abbr. cn)
    # memory cell node esti     (only used for this NN node, abbr. cne)
    # output gate 
    og_w_cn = None;              # cell node weight
    og_w_h = None;
    og_w_x = None;
    og_b = None;
    og_func = nn.Sigmoid;
    # hidden state (output)
    ho_func = nn.Tanh;
    # time gate 1 
    # time gate 2
    
    
    #nn.Parameter(torch.tensor([1.0]).to(device=self.device);
    
    
    def SetDevice(self, device):
        if not isinstance(device, torch.device):
            raise Exception(ERR_SetDevice_device_WRONG_TYPE);
        else:
            self.device = device;
    
    '''
    init one LSTM node
    @nn_type:                   the type of nn
    @nn_init_type:              the way to init nn parameters 
    @nn_in_feature_num:         input feature number
    @nn_out_feature_num:        output feature number
    '''
    def __init__(self, *, nn_type=NN_TYPE_LSTM_ALEX_GRAVES, nn_init_type=NN_INIT_TYPE_RAND, nn_in_feature_num=NN_FEATURE_NUM, nn_out_feature_num=NN_FEATURE_NUM):
        # input check
        if nn_type not in self.NN_TYPES:
            raise Exception(ERR_init_nn_type_WRONG_TYPE);
        if nn_init_type not in self.NN_INIT_TYPES:
            raise Exception();
        if not isinstance(nn_in_feature_num, int):
            raise Exception(ERR_init_nn_feature_num_WRONG_TYPE);
        # set NN type
        self.nn_type = nn_type;
        # set the init function
        init_func = None;
        if nn_init_type == self.NN_INIT_TYPE_0:
            init_func = torch.zeros;
        if nn_init_type == self.NN_INIT_TYPE_1:
            init_func = torch.ones;
        if nn_init_type == self.NN_INIT_TYPE_RAND:
            init_func = torch.randn;
        # record input & output feature number
        self.nn_in_feature_num = nn_in_feature_num;
        self.nn_out_feature_num = nn_out_feature_num;
        # create nn parameters
        # forget gate
        self.fg_w_c = nn.Parameter(init_func(nn_out_feature_num, 1));
        self.fg_w_h = nn.Parameter(init_func(nn_out_feature_num, nn_out_feature_num));
        self.fg_w_x = nn.Parameter(init_func(nn_out_feature_num, nn_in_feature_num));
        self.fg_b = nn.Parameter(init_func(nn_out_feature_num, 1));
        # input gate
        self.ig_w_c = nn.Parameter(init_func(nn_out_feature_num, 1));
        self.ig_w_h = nn.Parameter(init_func(nn_out_feature_num, nn_out_feature_num));
        self.ig_w_x = nn.Parameter(init_func(nn_out_feature_num, nn_in_feature_num));
        self.ig_b = nn.Parameter(init_func(nn_out_feature_num, 1));
        # input node
        self.in_w_h = nn.Parameter(init_func(nn_out_feature_num, nn_out_feature_num));
        self.in_w_x = nn.Parameter(init_func(nn_out_feature_num, nn_in_feature_num));
        self.in_b = nn.Parameter(init_func(nn_out_feature_num, 1));
        # memory cell node (only used for this NN node, abbr. cn)
        # output gate
        self.og_w_cn = nn.Parameter(init_func(nn_out_feature_num, 1));
        self.og_w_h = nn.Parameter(init_func(nn_out_feature_num, nn_out_feature_num));
        self.og_w_x = nn.Parameter(init_func(nn_out_feature_num, nn_in_feature_num));
        self.og_b = nn.Parameter(init_func(nn_out_feature_num, 1));
        # time gate 1 
        # time gate 2
    
    '''
    forward
    @x: the input [batch, x]: x=[x1, x2 ... xm] is the features we measure for x
    @C: memory cell  [batch, ?], ? in the output feature number
    @H: hidden state [batch, ?], ? in the output feature number
    '''
    def forward(self, x, *, C = None, H = None):
        # input check
        if x.ndim != 2:
            raise Exception(ERR_forward_x_WRONG_DIM);
        
        # extend the dimension of x from [batch, x] to [batch, x, 1]
        
        # fill in the initial C and H if not given
        
        # forget gate
        fg = self.fg_func(self.fg_w_c*C + torch.matmul(self.fg_w_h, H) + torch.matmul(self.fg_w_x, x) + self.fg_b);
        # input gate
        ig = self.ig_func(self.ig_w_c*C + torch.matmul(self.ig_w_h, H) + torch.matmul(self.ig_w_x, x) + self.ig_b);
        # input node
        inn = self.in_func(torch.matmul(self.in_w_h, H) + torch.matmul(self.in_w_x, x) + self.in_b); 
        # cell memory node
        cm = fg*C + ig*inn;
        # output gate
        og = self.og_func(self.og_w_cn*cm + torch.matmul(self.og_w_h, H) + torch.matmul(self.og_w_x, x) + self.og_b);
        # hidden state
        h = og*self.ho_func(cm);
        
        # return
        return cm, h;