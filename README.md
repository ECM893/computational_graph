# Computational Graph Polynomial Fit

This repository demonstrates a minimal, extensible computational graph framework in Python, supporting forward and backward (autodiff) passes, parameter optimization, and visualization. The main example fits a quadratic polynomial to noisy data using gradient descent. This methodology is unpacked further in my blog posts on Neural Net Architecture.

## Features

- **Custom computational graph**: Nodes for addition, multiplication, power, and more.
- **Automatic differentiation**: Backpropagation for gradients.
- **Parameter optimization**: Simple SGD optimizer.
- **Visualization**: 2D/3D data/model plots, loss/parameter tracking, and graph structure.
- **Testing**: Pytest-based unit tests for all core node operations.

## Example: Polynomial Fit

The main script (`example_polynomial_fit.py`) fits a function of the form:

```
y = c0 * x1^2 + c1 * x1 + c2 * x2^2 + c3 * x2 + c4
```

to noisy synthetic data, using a computational graph built from basic node classes. The graph supports any number of input features and quadratic terms.

### How it works

1. **Data Generation**: Synthetic 2D data is generated with a known quadratic relationship and noise.
2. **Graph Construction**: Nodes are created for each operation and parameter. The graph is built to represent the polynomial equation.
3. **Forward Pass**: Computes predictions and mean squared error loss.
4. **Backward Pass**: Computes gradients for all parameters using autodiff.
5. **Optimization**: Parameters are updated using SGD with gradient clipping and learning rate decay.
6. **Visualization**: 
   - 3D plot of data and learned surface.
   - Loss and parameter values over epochs.
   - DOT-format graph visualization.

### Running the Example

1. Install dependencies (using [uv](https://github.com/astral-sh/uv) or pip):

   ```
   uv pip install -r requirements.txt
   ```

2. Run the main example:

   ```
   python example_polynomial_fit.py
   ```

   This will print training progress, plot the data/model, and show loss/parameter curves.

### Testing

Unit tests for all node operations and simple backpropagation are provided in `test_node.py`. To run all tests:

```
pytest
```

### File Overview

- `example_polynomial_fit.py` — Main example script.
- `src/node.py` — Node classes for graph operations and autodiff.
- `src/graph.py` — Graph management and topological traversal.
- `src/optimizer.py` — SGD optimizer.
- `src/visualization.py` — Plotting and graph visualization utilities.
- `test_node.py` — Unit tests for node operations and gradients.
- `requirements.txt` — Python dependencies.

### Notes

- The framework is extensible. You can add new node types for more complex operations.
- The code is educational and minimal. This is not optimized for production or large-scale deep learning.

---

**Have fun with computational graphs and autodiff!**
