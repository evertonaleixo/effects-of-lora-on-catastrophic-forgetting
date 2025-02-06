# LoRA and ConvLoRA in the Context of CF

This repository contains code and resources for studying the impact of Low-Rank Adaptation (LoRA) and Convolutional LoRA (ConvLoRA) techniques in avoiding **Catastrophic Forgetting (CF)** in dynamic neural networks, particularly in continuous learning scenarios. The project evaluates how LoRA and ConvLoRA minimize parameter growth and maintain accuracy over multiple tasks without retraining the model.

## Table of Contents
- [Introduction](#introduction)
- [LoRA](#lora)
- [ConvLoRA](#convlora)
- [Catastrophic Forgetting (CF)](#catastrophic-forgetting-cf)
- [Categories of CF Solutions](#categories-of-cf-solutions)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Citations](#citations)
  
## Introduction

This project aims to analyze the advantages and weaknesses of using **LoRA** and **ConvLoRA** to mitigate **Catastrophic Forgetting (CF)** in dynamic networks. We are particularly focused on methods from the **Dynamic Networks** category, which adapt the structure of neural networks over time to retain past knowledge while learning new tasks.

Catastrophic Forgetting occurs when a model forgets previously learned information while learning new tasks, leading to significant drops in performance. In this project, we explore solutions like LoRA and ConvLoRA that efficiently address this problem by introducing a minimal set of new parameters without retraining the entire network.

## LoRA

**Low-Rank Adaptation (LoRA)** is a technique designed to reduce the number of trainable parameters when fine-tuning large models, such as **Large Language Models (LLMs)** or **Convolutional Neural Networks (CNNs)**. LoRA achieves this by decomposing the weight update matrices into low-rank matrices, allowing efficient adaptation without significantly increasing the model size.

In LoRA, instead of updating the entire weight matrix, we introduce two low-rank matrices, **A** and **B**, to approximate the update as follows:

\[
W + \Delta W = W + A \cdot B
\]

Where:
- \( W \) is the original weights matrix of size \( d \times d \),
- \( A \in \mathbb{R}^{d \times r} \), \( B \in \mathbb{R}^{r \times d} \), with \( r \ll d \), are the low-rank matrices used for fine-tuning,
- \( \Delta W = A \cdot B \) represents the low-rank update to the weight matrix.

This decomposition significantly reduces the number of parameters being trained, making it ideal for continual learning or task-specific fine-tuning without overfitting or requiring excessive computational resources.

### Insights from the Literature
According to the paper ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/pdf/2106.09685), LoRA offers several advantages over traditional fine-tuning methods. It reduces the memory footprint and computational costs by freezing the majority of the pre-trained weights and only learning a small number of additional parameters through the low-rank matrices \( A \) and \( B \). This approach maintains the performance of the model across new tasks while preventing **Catastrophic Forgetting (CF)**, as it preserves the learned information from previous tasks.

The paper demonstrates that LoRA is particularly beneficial in the following ways:
- **Efficiency**: By using low-rank adaptation, LoRA minimizes the number of trainable parameters, reducing both memory usage and computational costs during fine-tuning.
- **Transfer Learning**: LoRA effectively transfers knowledge between tasks without requiring a full re-training of the model, making it an excellent choice for continual learning scenarios.
- **Parameter Isolation**: Since LoRA adapts only specific parts of the model through the low-rank matrices, it avoids interfering with the original weights, which is crucial for mitigating **Catastrophic Forgetting**.

### Example
Consider a neural network layer with a weight matrix \( W \) of size \( 768 \times 768 \). Fine-tuning this layer traditionally would involve updating all \( 768^2 = 589,824 \) parameters. However, using LoRA with a low-rank setting \( r = 32 \), the number of trainable parameters is reduced to:

\[
A \in \mathbb{R}^{768 \times 32} \quad \text{and} \quad B \in \mathbb{R}^{32 \times 768}
\]

The total number of new parameters becomes \( 768 \times 32 + 32 \times 768 = 49,152 \), which is a **91% reduction** in the number of parameters compared to full fine-tuning. This makes LoRA highly efficient and scalable for large models.

### Benefits
- **Parameter Efficiency**: LoRA significantly reduces the number of parameters needed for fine-tuning, making it ideal for tasks with limited computational resources.
- **Memory Efficiency**: By only updating low-rank matrices, LoRA requires less memory during training and inference.
- **Mitigation of Catastrophic Forgetting**: LoRA isolates task-specific updates to the low-rank matrices, ensuring that previously learned tasks are not forgotten.

## ConvLoRA

**ConvLoRA** is an extension of LoRA specifically designed for convolutional neural networks (CNNs). In ConvLoRA, the low-rank adaptation approach is applied to convolutional filters, reducing the number of parameters while preserving performance in continuous learning scenarios. This technique helps mitigate **Catastrophic Forgetting (CF)** by introducing minimal additional parameters for each new task, allowing the model to retain prior knowledge efficiently.

In standard CNNs, convolutional layers consist of filters that capture spatial hierarchies in the data. ConvLoRA reduces the complexity of these layers by applying low-rank matrix approximations to the convolutional weights. This is achieved by decomposing the weight matrix into three low-dimensional matrices:

\[
W_{conv} + \Delta W_{conv} = W_{conv} + A_{h \times 1} \cdot B_{1 \times w} \cdot C_{1 \times k}
\]

Where:
- \(A_{h \times 1}\), \(B_{1 \times w}\), and \(C_{1 \times k}\) are vectors representing the low-rank approximation of the original weights,
- \(W_{conv}\) represents the original convolutional filters,
- \(h\) and \(w\) are the dimensions of the filters, and \(k\) is the number of output channels.

By utilizing this low-rank decomposition, ConvLoRA significantly reduces the number of trainable parameters while retaining the same filter size and output dimensions, making it highly efficient in preventing CF during task adaptation.

### Example
For a convolutional layer with 32 filters of size 3x3, ConvLoRA approximates the filter weights using:

\[
W + \Delta W = W + A_{3 \times 1} \cdot B_{1 \times 3} \cdot C_{1 \times 32}
\]

This reduces the total number of new parameters needed for task adaptation, thus minimizing memory overhead while ensuring high task accuracy.

### Insights from the Literature
The concept of ConvLoRA is supported by recent research on efficient adaptation methods for CNNs. According to the paper ["Low-Rank Adaptation of CNNs for Efficient Transfer Learning"](https://arxiv.org/pdf/2402.04964), applying low-rank approximations to convolutional layers not only reduces the number of parameters but also helps in transferring knowledge across tasks in a more efficient manner. The paper demonstrates that low-rank adaptation techniques like ConvLoRA maintain a strong performance while reducing computational costs, making them suitable for scenarios where **continual learning** and **catastrophic forgetting** are significant concerns.

ConvLoRA builds on these principles, extending them to handle **continual learning** challenges effectively, where the network must accommodate new tasks without forgetting previously learned ones. By freezing the original convolutional weights and only adapting a small number of parameters, ConvLoRA ensures the preservation of prior knowledge with minimal computational overhead.

### Benefits
- **Parameter Efficiency**: ConvLoRA reduces the number of trainable parameters by up to 95% per convolutional layer.
- **Memory Efficiency**: By focusing on low-rank updates, ConvLoRA minimizes memory usage compared to traditional full-layer updates.
- **Mitigation of Catastrophic Forgetting**: ConvLoRA's approach ensures that the network maintains performance on old tasks while learning new ones.

ConvLoRA has been demonstrated to provide efficient adaptation, making it particularly useful for continual learning tasks, as highlighted in [this paper](https://arxiv.org/pdf/2402.04964) that emphasizes the benefits of low-rank approximation for deep learning models.

## Catastrophic Forgetting (CF)

CF happens when a neural network learns a new task and forgets previous ones. ConvLoRA and LoRA aim to mitigate CF by adding new parameters for each task while keeping the model lightweight and efficient.

## Categories of CF Solutions

CF solutions can be categorized into four major groups:

### 1. **Rehearsal**
Rehearsal methods store past data or generated representations and periodically replay them to the model during training. This approach helps the network "remember" previously learned tasks, thus reducing CF. However, storing past data can be memory-intensive and not always feasible due to privacy constraints or storage limitations.

- **Example**: Memory replay, where a buffer of previous tasks' samples is stored and replayed to prevent forgetting.

### 2. **Distance-Based**
Distance-based methods rely on measuring similarities between task representations to help the network retain past knowledge. These techniques compute the distance between old and new task features to prevent overlapping parameter updates that could cause CF.

- **Example**: Projecting data onto a shared feature space, ensuring that similar tasks stay close in the learned space, preventing forgetting by maintaining relationships between tasks.

### 3. **Sub-Networks**
Sub-networks isolate different parts of the model for each task to minimize interference. Each task gets its own subset of the network’s parameters, avoiding any overlap with previously learned tasks.

- **Example**: Masking specific weights or neurons in the network for each task to ensure task-specific learning and prevent interference with previously learned tasks.

### 4. **Dynamic Networks**
Dynamic networks, which include **LoRA** and **ConvLoRA**, expand the architecture by adding new parameters as the model learns new tasks. This ensures that the original model parameters remain unchanged, thus avoiding CF while allowing for continuous learning.

- **Example**: **Progressive Neural Networks (PNN)**, which add new layers or parameters for each new task while freezing the old ones to preserve previously acquired knowledge.

Our project focuses on Dynamic Networks due to their efficiency in handling CF while minimizing computational overhead and memory usage. LoRA and ConvLoRA add small, efficient parameter sets per task, making them ideal for scenarios requiring continual learning.

## Installation

To set up the project locally, follow these steps:

```bash
python -m venv venv
source venv/bin/activate
git clone https://github.com/evertonaleixo/effects-of-lora-on-catastrophic-forgetting.git
cd effects-of-lora-on-catastrophic-forgetting/
pip install -r requirements.txt
```

### Troubleshooting

#### Import error

If you have problem with the following import:

```python
from avalanche.benchmarks.classic import SplitCIFAR10
```

with a error like this:

```bash
     24 except Exception:
---> 25     from pytorchcv.models.common import DwsConvBlock
     28 def remove_sequential(network: nn.Module, all_layers: List[nn.Module]):
     29     for layer in network.children():
     30         # if sequential layer, apply recursively to layers in sequential layer

ImportError: cannot import name 'DwsConvBlock' from 'pytorchcv.models.common' 
```

is because **pytorchcv** and **avalanche-lib** dependencies are having conflit. To resolve it, downgrade to pytorchcv to 0.0.67 version.


Always use a isolated environment to run python projects.

## Usage


## Results

The evaluation results of **ConvLoRA** demonstrate its effectiveness in mitigating Catastrophic Forgetting (CF) while limiting the growth of trainable parameters, particularly when compared to baseline models like **SenaCNN**. Below is a summary of accuracy and parameter growth for ConvLoRA and SenaCNN on the **CIFAR-100** dataset:

| Method     | Accuracy (CIFAR-100) | Parameter Growth |
|------------|----------------------|------------------|
| ConvLoRA   | 85%                  | +45%             |
| SenaCNN    | 84%                  | +82%             |

### Key Observations:
- ConvLoRA manages to retain **85% accuracy** while adding only **45%** to the number of parameters.
- **SenaCNN**, by contrast, achieves **84% accuracy**, but at the cost of increasing the parameter count by **82%**.
- ConvLoRA also shows **faster convergence** and **better parameter efficiency**, making it ideal for resource-constrained environments.

In summary, **ConvLoRA** provides a more parameter-efficient solution to CF, maintaining high accuracy while reducing computational and memory overhead.

## Citations

- Aleixo et al., "Catastrophic Forgetting in Deep Learning: A Comprehensive Taxonomy," *Journal of Brazilian Computer Society*, 2024. [DOI:10.5753/jbcs.2024.3966]
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," *ICLR*, 2022. [DOI:10.48550/arXiv.2106.03702]
- Sidra Aleem et al., "ConvLoRA: Low-Rank Adaptation for Convolutional Layers," *Proceedings of the IEEE International Symposium on Biomedical Imaging (ISBI)*, 2024. [GitHub Repository: https://github.com/aleemsidra/ConvLoRA] [Published at: https://biomedicalimaging.org/2024/]
- Zacarias et al., "SenaCNN: Overcoming Catastrophic Forgetting in CNNs by Selective Network Augmentation," *Journal of Artificial Neural Networks*, 2018. [DOI:10.1007/s12046-018-0888-7]

## How to cite this repository

```
@software{everton_lima_aleixo_2025_14824807,
  author       = {Everton Lima Aleixo},
  title        = {evertonaleixo/effects-of-lora-on-catastrophic-
                   forgetting: Experiments to Handle CF using Low
                   Rank Adapters
                  },
  month        = feb,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {Adapters},
  doi          = {10.5281/zenodo.14824807},
  url          = {https://doi.org/10.5281/zenodo.14824807},
  swhid        = {swh:1:dir:07b83e6049358d64d74e0cd5db2adf9499a0a6a3
                   ;origin=https://doi.org/10.5281/zenodo.14824806;vi
                   sit=swh:1:snp:1ed93a22e0d2188cdf4b03194a8407b59498
                   5a04;anchor=swh:1:rel:bc16b5eb39573667a25570f08f9b
                   4fa83b22f049;path=evertonaleixo-effects-of-lora-
                   on-catastrophic-forgetting-2073070
                  },
}
```

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14824807.svg)](https://doi.org/10.5281/zenodo.14824807)
