# HybridCompFL: Model-Heterogeneous Federated Learning via Data-free Hybrid Model Compression

Authors:


To run experiments and visualize results, install the required packages using Python (version 3.10 or higher):

```python
pip install -r requirements.txt
```





<p align="center">
  <img src="figures/DML.jpeg" alt="Static Diagram" width="45.5%" style="max-width: 100%; height: auto; margin-right: 2%;">
  <img src="figures/FL.jpeg" alt="Animated Demo" width="52.3%" style="max-width: 100%; height: auto;">
  <em>  <br><be> Figure 2: The left shows the DML. The right shows the FL. </em>
</p>


| Hyperparamaters                  | Options   | Alternatives                               |
|----------------------------------|-----------|--------------------------------------------|
| MODEL_NAME                       | VGG16     | LeNet \| AlexNet \| VGG16                  |
| DATASET_NAME                     | CIFAR10   | MNIST \| FashionMNIST \| CIFAR10           |
| AGGREGATION_METHOD               | avg       | avg \| avg_momentum \| adagrad             |
| GLOBAL_ROUND                     | 100       | —                                          |
| TOTAL_DEVICES_IN_THE_SYSTEM      | 100       | —                                          |
| PARTICIPATION_RATE               | 0.3       | —                                          |
| PERCENTAGE_OF_CAPABLE_DEVICES    | 0.2       | —                                          |
| LOCAL_EPOCH                      | 1         | —                                          |
| BATCH_SIZE                       | 32        | —                                          |
| OPTIMIZER                        | Adam      | —                                          |
| LEARNING_RATE                    | 0.01      | —                                          |




<img src="figures/model_selection.jpeg" alt="Model Selection" width="80%"/>



![HybridCompFL](figures/HybridCompFL.jpeg)


> [!NOTE]
>
> > In the current version of the code, min_nonzero=0.01 this cause a high sparsification result during quantization (While sparsification amount is 0.1, and sparsification % of sparsed model is 10%, for the quantized model it can increase to about 60%, due to many weights in the VGG16 trained with CIFAR10 to be smaller then 0.01).
