"""
Visualization utilities for computational graph experiments.

This module provides:
- plot_training_progress: Plot loss and parameter values over epochs.
- Model3DPlotter: Class for 3D visualization of data and model surfaces.
- (Other plotting helpers for 1D data.)

All plots use matplotlib and seaborn for styling.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
import plotly.graph_objs as go
import plotly.io as pio


def plot_training_progress(losses, *param_lists, save_path=None):
    """
    Plot loss (log scale) and parameter values over epochs using seaborn style.

    Args:
        losses (list or np.ndarray): Loss values for each epoch.
        *param_lists: Any number of lists/arrays of parameter values (one per parameter).
        save_path (str, optional): If provided, save the plot to this path.
    """
    sns.set_theme(style="darkgrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(losses, label="Loss", color="red")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="red")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor="red")
    ax2 = ax1.twinx()
    colors = [
        "blue",
        "green",
        "purple",
        "orange",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]
    for i, param_vals in enumerate(param_lists):
        label = f"c{i}"
        color = colors[i % len(colors)]
        ax2.plot(param_vals, label=label, color=color)
    ax2.set_ylabel("Parameter values", color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    plt.title("Loss (log scale) and Parameter Values Over Time")
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def save_training_progress_plotly(losses, *param_lists, save_path=None, json_path=None):
    """
    Save a Plotly interactive plot of loss and parameter values over epochs.

    Args:
        losses (list or np.ndarray): Loss values for each epoch.
        *param_lists: Any number of lists/arrays of parameter values (one per parameter).
        save_path (str, optional): If provided, save the plot to this path (as HTML).
        json_path (str, optional): If provided, save the plot as a JSON file for use with plotly.js.
    """

    epochs = list(range(len(losses)))
    fig = go.Figure()

    # Loss (log scale)
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=losses,
            mode="lines",
            name="Loss",
            yaxis="y1",
            line=dict(color="red"),
        )
    )

    # Parameters
    colors = [
        "blue",
        "green",
        "purple",
        "orange",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]
    for i, param_vals in enumerate(param_lists):
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=param_vals,
                mode="lines",
                name=f"c{i}",
                yaxis="y2",
                line=dict(color=colors[i % len(colors)]),
            )
        )

    # Layout with dual y-axes
    fig.update_layout(
        title="Loss (log scale) and Parameter Values Over Time",
        xaxis=dict(title="Epoch"),
        yaxis={
            "title": {"text": "Loss", "font": {"color": "red"}},
            "type": "log",
            "tickfont": {"color": "red"},
        },
        yaxis2=dict(
            title={"text": "Parameter values", "font": {"color": "black"}},
            overlaying="y",
            side="right",
            tickfont={"color": "black"},
        ),
        legend=dict(
            x=1,
            y=1,
            xanchor="right",
            yanchor="top"
        ),
        template="plotly_white",
    )

    if save_path is not None:
        pio.write_html(fig, save_path, full_html=True, include_plotlyjs='cdn', config={'responsive': True})
    if json_path is not None:
        with open(json_path, "w") as f:
            f.write(pio.to_json(fig))
    return fig


class Model3DPlotter:
    """
    Object for plotting 3D data and a model surface.

    Args:
        equation_func (callable): Function(X_mesh, params) -> y_mesh, computes model surface.
        param_labels (list, optional): List of parameter names for legend (not used in plot).

    Usage:
        plotter = Model3DPlotter(equation_func)
        plotter.plot(X, y, params)
    """

    def __init__(self, equation_func, param_labels=None):
        self.equation_func = equation_func
        self.param_labels = param_labels

    def plot(
        self,
        X,
        y,
        params,
        title="3D Data and Model Surface",
        save_path=None,
        plotly_path=None,
        plotly_json_path=None,
    ):
        """
        Plot 3D scatter of data and the model surface.

        Args:
            X (np.ndarray): Input data of shape (N, 2).
            y (np.ndarray): Target values of shape (N,).
            params (list): List of parameter values for the model equation.
            title (str): Plot title.
            save_path (str, optional): If provided, save the matplotlib plot to this path.
            plotly_path (str, optional): If provided, save a Plotly interactive plot to this path (as HTML).
            plotly_json_path (str, optional): If provided, save the Plotly figure as JSON for plotly.js.
        """
        # Matplotlib plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X[:, 0], X[:, 1], y, c="blue", label="Data")
        x1_grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
        x2_grid = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)
        x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)
        X_mesh = np.stack([x1_mesh.ravel(), x2_mesh.ravel()], axis=1)
        y_mesh = self.equation_func(X_mesh, params)
        surf = ax.plot_trisurf(
            X_mesh[:, 0],
            X_mesh[:, 1],
            y_mesh,
            color="red",
            alpha=0.5,
            linewidth=0,
            antialiased=True,
            label="Predicted Surface",
        )
        proxy_surface = Line2D(
            [0],
            [0],
            linestyle="none",
            marker="s",
            color="red",
            alpha=0.5,
            label="Predicted Surface",
        )
        handles, labels = ax.get_legend_handles_labels()
        handles.append(proxy_surface)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("y")
        ax.legend(handles, labels)
        ax.set_title(title)
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.show()

        # Plotly plot (optional)
        if plotly_path is not None or plotly_json_path is not None:
            # Prepare mesh for surface
            x1_grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
            x2_grid = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)
            x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)
            X_mesh = np.stack([x1_mesh.ravel(), x2_mesh.ravel()], axis=1)
            y_mesh = self.equation_func(X_mesh, params)
            y_mesh_grid = y_mesh.reshape(x1_mesh.shape)

            fig_plotly = go.Figure()
            # Data points
            fig_plotly.add_trace(
                go.Scatter3d(
                    x=X[:, 0],
                    y=X[:, 1],
                    z=y,
                    mode="markers",
                    marker=dict(size=3, color="blue"),
                    name="Data",
                )
            )
            # Surface
            fig_plotly.add_trace(
                go.Surface(
                    x=x1_mesh,
                    y=x2_mesh,
                    z=y_mesh_grid,
                    colorscale="Reds",
                    opacity=0.5,
                    name="Predicted Surface",
                    showscale=False,
                )
            )

            fig_plotly.update_layout(
                title=title,
                scene=dict(
                    xaxis_title="x1",
                    yaxis_title="x2",
                    zaxis_title="y",
                    aspectmode="manual",
                    aspectratio=dict(
                        x=1,
                        y=1,
                        z=1,
                    ),
                ),
                legend=dict(x=0.01, y=0.99),
            )
            if plotly_path is not None:
                os.makedirs(os.path.dirname(plotly_path), exist_ok=True)
                pio.write_html(fig_plotly, plotly_path, full_html=True, include_plotlyjs='cdn', config={'responsive': True})
            if plotly_json_path is not None:
                os.makedirs(os.path.dirname(plotly_json_path), exist_ok=True)
                with open(plotly_json_path, "w") as f:
                    f.write(pio.to_json(fig_plotly))
