# Bridging Realms: Physics Tensors vs Machine Learning Tensors 🌌🤖

## Overview

This project explores the **concept of tensors** from two perspectives:
1. **Physics** — focusing on stress-energy tensors in **General Relativity** and their transformations under Lorentz boosts.
2. **Machine Learning** — demonstrating how tensors are used in **Deep Learning**, particularly in the manipulation of high-dimensional data.

The goal is to provide a hands-on comparison and understanding of tensor operations across these two realms.

## Table of Contents
- [Project Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Tensor Operations](#tensor-operations)
- [Visualizations](#visualizations)
- [Contributions](#contributions)
- [License](#license)

## Installation

To run this project, you'll need to have Python installed along with a few libraries. The following steps will guide you through the setup:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/bridging-realms.git
    cd bridging-realms
    ```

2. **Create a virtual environment** (optional, but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate   # For Mac/Linux
    venv\Scripts\activate      # For Windows
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once you've set up the environment, you can explore the demo notebooks and scripts:

1. **Physics Tensor Demo**: `notebooks/physics_tensor_demo.ipynb`
    - Demonstrates Lorentz transformations of stress-energy tensors.
   
2. **ML Tensor Demo**: `notebooks/ml_tensor_demo.ipynb`
    - Shows tensor operations (reshaping, transposing) commonly used in machine learning.

3. **Comparison Visualizer**: `notebooks/comparison_visualizer.ipynb`
    - Compares tensor transformations between physics and ML with visualizations.

To run the entire pipeline, simply execute:

```bash
python main.py
```

## Tensor Operations

### Physics Tensor: Stress-Energy Tensor
The project begins by defining a simple 2x2 stress-energy tensor in general relativity. We apply a Lorentz transformation to simulate the effect of high-speed motion on physical systems.

```python
# Stress-Energy Tensor Transformation
def lorentz_transform(tensor, velocity, speed_of_light=1):
    # Apply Lorentz transformation to tensor
    gamma = 1 / np.sqrt(1 - (velocity ** 2) / (speed_of_light ** 2))
    lorentz_matrix = np.array([[gamma, -gamma * velocity],
                               [-gamma * velocity, gamma]])
    transformed_tensor = np.dot(lorentz_matrix, np.dot(tensor, lorentz_matrix.T))
    return transformed_tensor
```

### Machine Learning Tensor: Rank-M Tensor Operations
We manipulate rank-4 tensors (e.g., batches of images in deep learning). Operations like reshaping and transposing are applied to demonstrate tensor transformations in machine learning tasks.

```python
# ML Tensor Operations
def ml_tensor_operations(tensor):
    reshaped_tensor = np.reshape(tensor, (-1, tensor.shape[-1]))  # Reshaping
    transposed_tensor = np.transpose(tensor, axes=(0, 3, 1, 2))  # Transposing
    return reshaped_tensor, transposed_tensor
```

## Visualizations

### Physics Tensor Visualization
We visualize the original and transformed stress-energy tensors as heatmaps, showing how the components change under Lorentz transformation.

### Machine Learning Tensor Visualization
We visualize slices of rank-4 ML tensors (representing batches of RGB images) using heatmaps and annotate each pixel's value.

```python
# Visualizing ML tensor slices
def plot_ml_tensor_slices(tensor):
    for b in range(batch):
        for c in range(channels):
            ax.imshow(tensor[b, c], cmap='viridis')
            for (j, i), val in np.ndenumerate(tensor[b, c]):
                ax.text(i, j, f"{val:.2f}", ha='center', va='center', color='white')
```

## Contributions

Contributions are welcome! If you'd like to add new features, improve code quality, or provide suggestions, feel free to open a pull request or issue.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
