# NN-TimeLSTM
## Introduction
### Development History
* `TimeLSTMLayer.py` and `LSTM_AlexGraves.py` are 1st version of codes. However, it is obsolete because it ignores the information exchange among different neurons. 
* `LSTM_v1.py` is a self-written LSTM to test our ability and to prove that weight initialisation is important.
* `TimeLSTM_v1.py` is our replication of `Alex Graves` LSTM
* `TimeLSTM_v2.py` uses less parameters
* `TimeLSTM_v3.py` uses 3 types of time gates 
* `TimeLSTM_v4.py` introduces the prediction end time as an additive output

### Device Management
* In our model `nn.Parameter` is kept on the CPU while only computation is performed on the GPU in a differentiable manner. That way, it will automatically accumulate back onto the CPU parameters. See [the explanation by albanD](https://discuss.pytorch.org/t/keeping-only-part-of-model-parameters-on-gpu/71308)
* Also, our model supports multiple GPUs as in [the stackoverflow explanation](https://stackoverflow.com/questions/54216920/how-to-use-multiple-gpus-in-pytorch)



### hidden states -> predicted RSSI
* 1st solution is to use a DNN to exact the predicted RSSI for the next available beacon interval
    Hidden states features at `n` features
    ```
    unknown_feature
    ...
    unknown_feature
    rssi_last,
    left_available_time,
    ```
    We make another `time hidden states` at `n` features. It is created by another DNN from `the difference between the estimated time period and the last available time point (two difference)`.
* 2nd solution is to take `the difference between the estimated time period and the last available time point (two difference)`, 

### Others
* [why use `loss.backward(retain_graph=True)`](https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method)
* [How to write your own LSTM](https://blog.csdn.net/junbaba_/article/details/106135219)