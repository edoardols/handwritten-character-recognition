# Handwritten Character Recognition

A Machine Learning project developed in Python. The core objective of this project is to implement, train, and evaluate multi-layer feedforward neural networks using the classic MNIST dataset under various levels of synthetic noise.

This software was developed as a project for the course *Machine Learning* during the MSc in *Artificial Intelligence and Automation Engineering* at the *University of Siena*.

## Architectural & Project Characteristics

### 1. Neural Network Topologies
The project evaluates and compares two distinct structural configurations built from scratch (implementing manual forward and backpropagation passes):
* **Single-Layer Topology:** A baseline network configuration consisting solely of direct input-to-output mapping.
* **Multi-Layer Topology (Two Hidden Layers):** An architecture integrating two hidden processing layers, each containing 16 hidden neurons, designed to learn complex, non-linear pixel boundary representations.
* **XOR Verification:** Includes isolated sanity-check implementations (located in the XOR module) to benchmark the backpropagation algorithm's capability to converge on non-linearly separable functions.

### 2. Robustness Analysis & Noised Datasets
To evaluate model generalization and degradation profiles, the baseline MNIST training and validation sets are subjected to five distinct synthetic degradation models:
* **Blob:** Clusters of structural distortions.
* **Brightness:** Global illumination changes.
* **Obscure:** Block-wise occlusions.
* **Salt & Pepper:** High-frequency impulse noise.
* **Thickness:** Structural thinning and widening of structural handwritten pen strokes.

### 3. Analytics & Visualization Engine
The evaluation suite captures extensive execution data to generate direct comparative performance graphs, evaluating:
* Accuracy degradation across scaling noise filters.
* Convergence rate comparison between topologies.
* Comparative plotting logic isolated within dedicated validation modules.

## Repository Structure

+-- dataset/                      # Baseline MNIST train/validation sets along with noised datasets
+-- src/
    +-- handwrittencharacter/
        +-- backpropagation/      # Multi-layer networks with backpropagation training artifacts
        +-- forward/              # Implementation of forward propagation variants
        +-- main.py               # Main execution entry point for configuring hyperparameters
        +-- validation_filter.py  # Plotting suite containing comparative analytical functions
        +-- XOR/                  # Minimal neural network validation testing on the XOR problem


## Contributors

This project was developed collaboratively by:
* **Edoardo Lascala** - [@edoardols](https://github.com/edoardols)
* **Tiberio Di Pisello** - [TiberioDiPisello](https://github.com/TiberioDiPisello)
* **gsalvatore99** - [gsalvatore99](https://github.com/gsalvatore99)

## License

This project is licensed under the **GPL-3.0 License**. See the `LICENSE` file in the root repository for complete terms and details.
