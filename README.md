**⚠️ <span style="color:red">This README is currently under construction. Please check back later for updates. We appreciate your patience as we finalize the documentation. </span>**

# HybridCompFL: Model-Heterogeneous Federated Learning via Data-free Hybrid Model Compression

This repository introduces the source code of the "Resource-aware Models via Pruning and Quantization in Heterogeneous Federated Learning" paper.

**Authors:** [Zeyneddin Oez](https://orcid.org/0000-0002-4216-9854), [Farshad Taebi], [Saeedeh Ghanadbashi](https://orcid.org/0000-0003-0983-301X), [Muhammad Farooq], [Abdollah Malekjafarian](https://orcid.org/0000-0003-1358-1943), [Kristof Van Laerhoven](https://orcid.org/0000-0001-5296-5347), and [Fatemeh Golpayegani](https://orcid.org/0000-0002-3712-6550)

**Affiliation:** [**University of Siegen**](https://www.uni-siegen.de/start/), and [**University College Dublin**](https://www.ucd.ie/)






<p align="center">
  <img src="simulation_codes/figures/DML.jpeg" alt="Static Diagram" width="45.5%" style="max-width: 100%; height: auto; margin-right: 2%;">
  <img src="simulation_codes/figures/FL.jpeg" alt="Animated Demo" width="52.3%" style="max-width: 100%; height: auto;">
  <em>  <br><be> Figure 2: The left shows the DML. The right shows the FL. </em>
</p>


| Model & Dataset   | Aggregation Strategy | Global Round | Total Devices in the System | Participation Rate | Percentage of Capable Devices | Local Epoch | Batch Size | Optimizer | Learning Rate |
|-------------------|----------------------|--------------|-----------------------------|--------------------|-------------------------------|-------------|------------|-----------|---------------|
| LeNet & MNIST     | FedAVG               | 50           | 50                          | 0.6                | 0.1                           | 5           | 64         | Adam      | 0.001         |
| AlexNet & FMNIST  | FedAVG               | 50           | 50                          | 0.6                | 0.1                           | 5           | 64         | Adam      | 0.001         |
| VGG16 & CIFAR10   | FedAVG               | 100          | 50                          | 0.6                | 0.2                           | 5           | 64         | Adam      | 0.001         |






<img src="simulation_codes/figures/model_selection.jpeg" alt="Model Selection" width="80%"/>



![HybridCompFL](figures/HybridCompFL.jpeg)


> [!NOTE]
>
> > In the current version of the code, min_nonzero=0.01, this causes a high sparsification result during quantization (While sparsification amount is 0.1, and sparsification % of sparsed model is 10%, for the quantized model it can increase to about 60%, due to many weights in the VGG16 trained with CIFAR10 being smaller than 0.01).
