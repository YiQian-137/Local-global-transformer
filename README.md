# Local-global-transformer
This repository includes our model code and dataset for process monitoring.
### Model
This structure of LGT is shown in the following illustration:
![](https://github.com/YiQian-137/Local-global-transformer/blob/main/img/LGT.png)
The experiments were performed on a computer with the following specifications:
* CPU: Intel(R) Core(TM) i5-6500 @3.20GHz; RAM: 16.0 GB;
* GPU: NVIDIA GeForce GTX 1070;
* Operating System: Ubuntu 18.04 LTS 64-bit computing (Python 3.8).

dataset: Tennessee Eastman process (TEP)
### Requirements
* torch==1.9.0(cuda 11.2)
* scipy==1.5.3
* scikit-learn==0.24.2
* numpy==1.19.5

transformer_para_1800 is the set of parameters that our model eventually saves.
