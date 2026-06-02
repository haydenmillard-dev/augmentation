# Data Augmentation Strategies for Improving DCNN Outputs
The quality of input data plays a major role in the quality of outputs of neural networks.

## What is data augmentation?
These are all techniques that modify input data to aid in the generalisation of a learning algorithm.
By changing the inputs, we aim to highlight generalisable features that reduce overfitting. 

## ResNet
Academic resource: https://arxiv.org/abs/1512.03385
## U-Net
Academic resource: https://arxiv.org/abs/1505.04597
## Representation Learning Theory (RLT)
Academic resource: https://arxiv.org/abs/1206.5538

## Explored Architectures
- U-Net
- U-Net with a ResNet backbone for the encoder module
- U-Net with a ResNet backbone for the encoder module and a single layer adapter
- U-Net with a ResNet backbone for the encoder module and a multi-layer adapter
- U-Net with a ResNet backbone for the encoder module and a deep layer adapter

## Multi-input Augmentaiton & Adapters
Multi-input augmentation is an idea that stems from data augmentation:
- Data augmentation aims to modify input representations to assist neural nets to generalise to the underlying data's distribution.
- As a secondary effect, data augmenation also increases the effective size of the training data, thereby improving model accuracy.
- RLT emphasises that neural networks should learn to "disentangle the factors of variation" to converge on meaningful representations.
- Multi-input data augmentation, like standard data augmenation, hopes to promote this philosophy by providing multiple representations of the same input.

## The Heuristic
- Standard data augmentation regimes provide one variant of the input per epoch.
- Conversely, multi-input data augmenation provide multiple variants of the input per epoch.
- This means that for each epoch, there are N! more variations of the inputs provided.
