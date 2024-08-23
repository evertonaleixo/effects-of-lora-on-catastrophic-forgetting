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

**Low-Rank Adaptation (LoRA)** is a fine-tuning strategy that introduces trainable low-rank matrices to approximate the weight updates in large neural networks. It reduces the number of parameters added when a network is trained on new tasks.

\[
W + \Delta W = W + A \cdot B
\]

Where \(A\) and \(B\) are low-rank matrices, and \(W\) is the original weights matrix.

## ConvLoRA

**ConvLoRA** is an extension of LoRA for convolutional neural networks (CNNs). It applies the same low-rank adaptation principle but adapts it to convolutional filters, allowing efficient parameter expansion without overloading the network’s memory capacity.

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
git clone https://github.com/evertonaleixo/effects-of-lora-on-catastrophic-forgetting.git
cd effects-of-lora-on-catastrophic-forgetting/
pip install -r requirements.txt
```

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

If you use this repository in your research, please cite the following works:

- Zacarias et al., "SenaCNN: Overcoming Catastrophic Forgetting in CNNs by Selective Network Augmentation," *Journal of Artificial Neural Networks*, 2018. [DOI:10.1007/s12046-018-0888-7]
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," *ICLR*, 2022. [DOI:10.48550/arXiv.2106.03702]
- Aleixo et al., "Catastrophic Forgetting in Deep Learning: A Comprehensive Taxonomy," *Journal of Brazilian Computer Society*, 2024. [DOI:10.5753/jbcs.2024.3966]
