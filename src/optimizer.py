from typing import List
import numpy as np


class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer.

    Args:
        parameters (List): List of parameters (nodes) to optimize.
        lr (float): Learning rate.
    """

    def __init__(self, parameters: List, lr: float = 0.01) -> None:
        self.parameters = parameters
        self.lr = lr

    def step(self) -> None:
        """
        Performs a single optimization step (parameter update).
        """
        for p in self.parameters:
            # Only update parameters that require gradients
            if getattr(p, "requires_grad", False):
                # Debug: print before/after values
                # print(f"Before: {p.name} value={p.value}, grad={p.grad}")
                p.value -= self.lr * p.grad
                # print(f"After: {p.name} value={p.value}")
                p.grad = np.zeros_like(p.grad)
