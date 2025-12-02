# Deep CNN Architectures for CIFAR‑10

This project implements and compares several convolutional architectures for image classification on the CIFAR‑10 dataset using PyTorch.  It showcases modular CNN building blocks (plain convolutional, residual, bottleneck, and Inception-style) and a custom network (`MyNetwork`) integrated into a full training and evaluation pipeline.

## Project overview

- Task: Supervised image classification on CIFAR‑10 (10 classes, 32×32 RGB images).
- Goal: Implement core CNN building blocks from scratch and study their behavior within a unified training pipeline.
- Implementation: All models are implemented in a single Jupyter notebook, including data loading, augmentation, training loop, evaluation, and TensorBoard logging.

## Implemented architectures

The notebook defines the following building blocks and networks:

- **MLP baseline (`MLPBlock`, `MyNetworkExample`)**  
  - Fully connected baseline operating on flattened images, with batch normalization and ReLU activations.
  - Used as a reference non-convolutional architecture.

- **Convolutional block (`ConvBlock`)**  
  - 3×3 convolution, batch normalization, and ReLU, parameterized by kernel size, stride, and padding.
  - Serves as the basic feature extractor for a plain CNN.

- **Plain residual block (`ResBlockPlain`)**  
  - Two 3×3 convolutions with batch normalization and ReLU, plus an identity skip connection.
  - Implements a simple ResNet-style residual mapping without channel change.

- **Bottleneck residual block (`ResBlockBottleneck`)**  
  - 1×1 convolution for channel reduction, 3×3 convolution, and 1×1 projection back to the original channel dimension, all with batch normalization and ReLU, plus residual skip.
  - Follows the bottleneck design used in deeper ResNet variants for parameter efficiency.

- **Inception-style block (`InceptionBlock`)**  
  - Four parallel branches: 1×1 conv, 1×1 + 3×3 conv, 1×1 + 5×5 conv, and max‑pool + 1×1 conv.
  - Outputs are concatenated along the channel dimension to capture multi‑scale features.

- **Custom network (`MyNetwork`)**  
  - Front-end `ConvBlock` followed by a `ResBlockPlain`, `ResBlockBottleneck`, and an `InceptionBlock`.
  - Adaptive average pooling and a final linear classifier produce logits for the 10 CIFAR‑10 classes.

## Dataset and preprocessing

- Dataset: CIFAR‑10 from `torchvision.datasets.CIFAR10`, automatically downloaded under `dataset/cifar10`.
- Train transforms: random crop with padding, random horizontal flip, tensor conversion, and per‑channel normalization.
- Test transforms: tensor conversion and the same normalization (no augmentation).
- Data loaders: batched using `DataLoader` with shuffling for training and deterministic ordering for testing.

## Training pipeline

The notebook contains a complete training and evaluation pipeline:

- Configuration: all hyperparameters are stored in an `EasyDict` (`args`), including learning rate, batch size, epochs, and block type.
- Optimization: cross‑entropy loss, SGD optimizer, and learning‑rate scheduler are used for all experiments.
- Logging: TensorBoard is used to track train/test loss and accuracy over time, with logs written to a configurable directory.
- Checkpointing: model weights are periodically saved, and the best checkpoint (by test accuracy) is stored per block type.

To run training for a given block type:

1. Set `args.block_type` to one of `["mlp", "conv", "resPlain", "resBottleneck", "inception"]`.
2. Adjust `args.num_blocks`, `args.lr`, `args.epoch`, and `args.batch_size` as needed.
3. Execute the notebook cells to train and evaluate the model, monitoring TensorBoard for learning curves.

## Experimental results

Initial experiments on CIFAR‑10 produced the following best test accuracies:

| Model type     | Best test accuracy | Approx. training time |
|----------------|--------------------|------------------------|
| MLP network    | 0.627              | 2400.959 s             |
| Conv network   | 0.109              | 2918.439 s             |
| ResPlain net   | 0.106              | 2858.259 s             |

These results highlight that the current CNN configurations underperform compared to the MLP baseline, suggesting room for improvement in architecture design (e.g., depth, channel widths, pooling strategy) and hyperparameters (learning rate schedules, regularization).

## How to use this project

- **As a learning resource**:  
  - Study each block implementation to understand how modern CNN components are built from low‑level PyTorch layers.
  - Inspect the training loop to see a complete, reproducible pipeline for supervised image classification.

- **As a starting point for further experiments**:  
  - Improve the CNN variants (e.g., add more layers, refine pooling, or adjust channel counts).  
  - Tune hyperparameters and compare new test accuracies to the baseline table.  
  - Extend the project to additional datasets or more advanced architectures.

## Future work

Planned or possible extensions:

- Refine the `MyNetwork` architecture and the CNN variants to achieve higher accuracy than the MLP baseline.  
- Add experiment scripts (`train.py`) to complement the notebook and allow command‑line training.  
- Include plots (learning curves, confusion matrices) and a more extensive results section documenting ablation studies.
