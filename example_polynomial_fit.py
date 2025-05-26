import numpy as np
from src.node import InputNode, ConstantNode, MulNode, AddNode, PowNode, MeanNode
from src.graph import ComputationGraph
from src.optimizer import SGD
from src.visualization import (
    plot_training_progress,
    save_training_progress_plotly,
    Model3DPlotter,
)


DEBUG = False  # Set to True to enable debug output
# Hyper parameters
# (these work fine)
MAX_GRAD = 1000
MIN_LR = 1e-3
LR_DECAY = 0.999
EPOCHS = 300
LR = 0.01

# # These dont work
# max_grad = 1000
# min_lr = 1e-3
# decay = 0.999
# epochs = 300
# lr = 0.1


def main() -> None:
    """
    Example script to fit a quadratic polynomial to noisy data using a custom computational graph.
    Demonstrates forward and backward passes, optimization, and visualization.
    """
    # Generate synthetic data for 2D input
    x1 = np.linspace(-2, 2, 30)
    x2 = np.linspace(-2, 2, 30)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    X = np.stack([x1_grid.ravel(), x2_grid.ravel()], axis=1)  # shape (N, 2)
    # True function: y = 1.5*X1^2 - 0.5*X1 + 2.0*X2^2 - 1.0*X2 + 2 + noise
    y = (
        1.5 * X[:, 0] ** 2
        - 0.5 * X[:, 0]
        + 0 * X[:, 1] ** 2
        - 1.0 * X[:, 1]
        + 2
        + 0.2 * np.random.randn(X.shape[0])
    )

    def model_equation(X_mesh, params):
        # For 5 params: c0*x1^2 + c1*x1 + c2*x2^2 + c3*x2 + c4
        c0, c1, c2, c3, c4 = params
        return c0 * X_mesh[:, 0] ** 2 + c1 * X_mesh[:, 0] + c2 * X_mesh[:, 1] ** 2 + c3 * X_mesh[:, 1] + c4

    # Define trainable parameters: c0 (X1^2), c1 (X1), c2 (X2^2), c3 (X2), c4 (bias)
    constants = [
        ConstantNode(np.random.randn(), name="c0"),
        ConstantNode(np.random.randn(), name="c1"),
        ConstantNode(np.random.randn(), name="c2"),
        ConstantNode(np.random.randn(), name="c3"),
        ConstantNode(np.random.randn(), name="c4"),
    ]
    c0, c1, c2, c3, c4 = constants

    print("Initial parameters: " + ", ".join(f"{p.name}={p.value}" for p in constants))

    # Build the computational graph for y_pred = c0*X1^2 + c1*X1 + c2*X2^2 + c3*X2 + c4
    graph = ComputationGraph()
    x1_node = InputNode(X[:, 0], name="x1")
    x2_node = InputNode(X[:, 1], name="x2")
    x1_sq = PowNode([x1_node], 2, name="x1_sq")
    x2_sq = PowNode([x2_node], 2, name="x2_sq")
    c0_x1_sq = MulNode([c0, x1_sq], name="c0_x1_sq")
    c1_x1 = MulNode([c1, x1_node], name="c1_x1")
    c2_x2_sq = MulNode([c2, x2_sq], name="c2_x2_sq")
    c3_x2 = MulNode([c3, x2_node], name="c3_x2")
    y_pred = AddNode([c0_x1_sq, c1_x1, c2_x2_sq, c3_x2, c4], name="y_pred")
    y_true = InputNode(y, name="y_true")
    neg_one = ConstantNode(-1, name="neg_one")
    neg_y_true = MulNode([neg_one, y_true], name="neg_y_true")
    diff = AddNode([y_pred, neg_y_true], name="diff")
    loss = PowNode([diff], 2, name="loss")
    mse_mean = MeanNode([loss], name="mse")

    # Add nodes to graph
    for node in [
        x1_node,
        x2_node,
        x1_sq,
        x2_sq,
        c0_x1_sq,
        c1_x1,
        c2_x2_sq,
        c3_x2,
        neg_y_true,
        diff,
        y_pred,
        loss,
        mse_mean,
    ]:
        graph.add_node(node)
    graph.set_head(mse_mean)


    # Initialize optimizer
    optimizer = SGD(constants, lr=LR)
    losses = []
    param_vals = [[] for _ in constants]

    # Training loop
    for epoch in range(EPOCHS):
        graph.forward()
        if np.isnan(mse_mean.value):
            print("NaN detected in loss at epoch", epoch)
            print("y_pred:", y_pred.value)
            print("loss:", loss.value)
            break
        graph.backward(mse_mean)
        for p in constants:
            p.grad = np.clip(p.grad, -MAX_GRAD, MAX_GRAD)
        losses.append(mse_mean.value)
        for i, p in enumerate(constants):
            param_vals[i].append(p.value.copy())
        if DEBUG:
            print(
                f"Epoch {epoch} grads: {', '.join(f'{p.name}={p.grad:.3f}' for p in constants)}"
            )
            print(f"Epoch {epoch} learning rate: {optimizer.lr:.3f}")
            print(
                f"Epoch {epoch} params: {', '.join(f'{p.name}={p.value:.3f}' for p in constants)}"
            )
        optimizer.step()
        optimizer.lr = max(optimizer.lr * LR_DECAY, MIN_LR)
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {mse_mean.value}")

    # Visualization
    # For 2D input, plot 3D surface

    plotter = Model3DPlotter(model_equation)
    plotter.plot(X=X,
                 y=y,
                 params=[p.value for p in constants],
                 save_path="./figures/model_surface.png",
                 plotly_json_path="./figures/model_surface.json",
                 plotly_path="./figures/model_surface.html"
                 )
    
    plot_training_progress(losses, *param_vals, save_path="./figures/losses.png")
    save_training_progress_plotly(
        losses, *param_vals, json_path="./figures/losses.json", save_path="./figures/losses.html"
    )


    # Print out final Parameters
    print("Final parameters: " + ", ".join(f"{p.name}={p.value:.1f}" for p in constants))
    # Print out actual Parameters
    print(
        "Actual parameters: c0=1.5, c1=-0.5, c2=0.0, c3=-1.0, c4=2.0 (for y = 1.5*x1^2 - 0.5*x1 + 0.0*x2^2 - 1.0*x2 + 2)"
    )


if __name__ == "__main__":
    main()
