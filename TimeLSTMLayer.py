import numpy as np
import torch
from torch import nn
from LSTM_AlexGraves import LSTM_AlexGraves

# macros
ERR_INIT_NN_TYPE_WRONG = "`hidden_neuron_type` is not a recognised type";
ERR_INIT_NN_INIT_FUNC_TYPE_WRONG = "`nn_init_func_type` is not a recognised type"

# TimeLSTM takes two inputs
# the neurons are registered as https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
# [batch, n, xm]: xm is the features we measure for x
# [batch, n, tm]: tm is the features we measure for time, in our case tm=3
class TimeLSTMLayer(nn.Module):
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
    init
    @hidden_neuron_size:    the number of neurons in this layer
    @hidden_neuron_type:    the type of hidden neuron
    @nn_init_func_type:     the initial function for learnable parameters 
    @nn_in_feature_num:     the input feature number
    @nn_out_feature_num:    the output feature number we want
    '''
    def __init__(self, hidden_neuron_size, hidden_neuron_type, nn_init_func_type, nn_in_feature_num, nn_out_feature_num):
        super(TimeLSTMLayer, self).__init__();
        # input check
        if hidden_neuron_type not in self.NN_TYPES:
            raise Exception(ERR_INIT_NN_TYPE_WRONG);
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
        
        # set device type (cpu)
        self.device = torch.device('cpu');
        # record the hidden neuron type
        self.hidden_neuron_type = hidden_neuron_type;
        # create neurons
        self.hidden_neuron_size = hidden_neuron_size;
        if hidden_neuron_type == self.NN_TYPE_LSTM_ALEX_GRAVES:
            self.hidden_neurons = nn.ModuleList([LSTM_AlexGraves(nn_init_func=nn_init_func, nn_in_feature_num=nn_in_feature_num, nn_out_feature_num=nn_out_feature_num) for nn_id in range(hidden_neuron_size)]);
        # create cell memory and hidden states
        self.hidden_neurons_C = [];                 # to hold hidden neurons cell memory
        self.hidden_neurons_CN = [];                # to hold hidden neurons cell memory node (only used for this NN node)    
        self.hidden_neurons_H = [];                 # to hold hidden neurons hidden states
        for hns_id in range(0, hidden_neuron_size):
            self.hidden_neurons_C.append(None);
            self.hidden_neurons_CN.append(None);
            self.hidden_neurons_H.append(None);
    
    '''
    forward
    <input>
    @x: the input [batch, n, xm]: xm is the features we measure for x
    @t: the input [batch, n, tm]: tm is the features we measure for time, in our case tm=3
    @n: the selected index for time series x
    <output>
    the concatenated hidden states of all hidden neurons
    [batch, hidden_neuron_size, ?]: ? is the output feature number
    '''
    def forward(self, x, n, *, t = None):
        # go through all data
        for n_id in range(n):
            # go through the all neurons to update the hidden state
            for nn_id, nn_cur in enumerate(self.hidden_neurons):
                C_cur = self.hidden_neurons_C[nn_id];
                H_cur = self.hidden_neurons_H[nn_id];
                # forward & update
                # Alex Graves's LSTM
                if self.hidden_neuron_type == self.NN_TYPE_LSTM_ALEX_GRAVES:
                    C_update, H_update = nn_cur.forward(x, n_id, C=C_cur, H=H_cur);
                    self.hidden_neurons_C[nn_id] = C_update;
                    self.hidden_neurons_H[nn_id] = H_update;
        # now each hidden state is [batch, ?, 1]
        # we convert it into [batch, 1, ?, 1] (create a new aix at `-3`)
        # so the layer output will be [batch, hidden_neuron_size, ?, 1]
        for nn_id, nn_cur in enumerate(self.hidden_neurons):
            self.hidden_neurons_H[nn_id] = torch.unsqueeze(self.hidden_neurons_H[nn_id], -3);
        # merge the hidden states of all neurons to create the layer output
        layer_hidden_state = torch.cat(self.hidden_neurons_H, dim=-3);
        # remove the last redundant dimension
        layer_hidden_state = torch.squeeze(layer_hidden_state, dim=-1);
        
        return layer_hidden_state;

