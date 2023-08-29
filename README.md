# NN-TimeLSTM
## Introduction
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