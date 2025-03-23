# Context-Aware Frequency-Embedding Networksfor Spatio-Temporal Portfolio Selection
Recent developments in the applications of deep reinforcement learning methods to portfolio selection have achieved superior performance to conventional methods. However, two major challenges remain unaddressed in these models and inevitably lead to the deterioration of model performance. First, asset characteristics often suffer from low and unstable signal-to-noise ratios, leading to poor learning robustness of the predictive feature representations. Second, existing literature fails to consider the complexity and diversity in long-term and short-term spatio-temporal predictive relations between the feature sequences and portfolio objectives. To tackle these problems, we propose a novel **Context-Aware Frequency-Embedding Graph Convolution Network (Cafe-GCN)** for spatio-temporal portfolio selection. It contains three important modules: 

* frequency-embedding block that explicitly captures the short-term and long-term predictive information embedded in asset characteristics meanwhile filtering out noise; 

* context-aware block that learns multiscale temporal dependencies in the feature space; 

* multi-relation graph convolutional block that exploits both static and dynamic spatial relations among assets. Extensive experiments on two real-world datasets demonstrate that Cafe-GCN consistently outperforms proposed techniques in the literature.

## Usage
### Requirement
```bash
python >= 3.8
pytorch >= 1.12.1
einops >= 0.8.0
```
### Train the model
1. Download CSI100 and NASDAQ100 Datasets, which are all some common datasets, and we do not provide the download links here.
2. Train models using command:
```bash
# bash run.sh
```

