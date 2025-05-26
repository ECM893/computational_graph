"""
Node classes for computational graph and autodiff.

This module defines the core node types for building a computational graph:
- Node: Base class for all nodes (values, parameters, inputs).
- ConstantNode: Trainable parameter node.
- InputNode: Data input node.
- ComputationalNode: Base class for operation nodes.
- AddNode, MulNode, PowNode: Operation nodes for addition, multiplication, and power.
- MeanNode: Node for computing the mean of its input.

Each node supports forward and backward passes for automatic differentiation.
"""

import numpy as np


class Node:
    """
    Base class for all nodes in the computational graph.

    Args:
        value (object): The value held by the node (scalar or array-like).
        name (str, optional): Optional name for the node.
        requires_grad (bool, optional): Whether this node requires gradient computation.
    """

    def __init__(
        self, value: object, name: str = None, requires_grad: bool = False
    ) -> None:
        self.value: np.ndarray = np.array(value, dtype=float)
        self.grad: np.ndarray = np.zeros_like(self.value)
        self.name: str = name
        self.requires_grad: bool = requires_grad
        self.parents: list["Node"] = []
        self.children: list["Node"] = []

    def zero_grad(self) -> None:
        """
        Resets the gradient to zero.
        """
        self.grad = np.zeros_like(self.value)


class ConstantNode(Node):
    """
    Node representing a trainable parameter (constant to be learned).

    Args:
        value (object): Initial value of the parameter (scalar or array-like).
        name (str, optional): Optional name for the parameter node.
    """

    def __init__(self, value, name: str = None):
        super().__init__(value, name=name, requires_grad=True)


class InputNode(Node):
    """
    Node representing an input (data fed into the graph).

    Args:
        value (object): Input data (scalar or array-like).
        name (str, optional): Optional name for the input node.
    """

    def __init__(self, value, name: str = None):
        super().__init__(value, name=name, requires_grad=False)


class ComputationalNode(Node):
    """
    Node representing an operation in the graph.

    Args:
        inputs (list[Node]): List of input nodes for this operation.
        name (str, optional): Optional name for the operation node.
    """

    def __init__(self, inputs: list[Node], name: str = None) -> None:
        super().__init__(None, name=name)
        self.inputs: list[Node] = inputs
        for inp in inputs:
            inp.children.append(self)
            self.parents.append(inp)

    def forward(self) -> None:
        """
        Computes the forward pass for this node.
        """
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> None:
        """
        Computes the backward pass for this node.
        To be implemented in subclasses or via op-specific logic.
        """
        raise NotImplementedError


def _accumulate_grad(param: Node, grad: np.ndarray):
    """
    Accumulate gradient into param.grad, handling broadcasting for scalars/vectors.
    """
    if param.grad.shape == grad.shape:
        param.grad += grad
    elif param.grad.shape == ():  # param is scalar, grad is vector
        param.grad += np.sum(grad)
    else:
        param.grad += np.broadcast_to(grad, param.grad.shape)


class AddNode(ComputationalNode):
    """
    Node representing element-wise addition.
    Supports two or more input nodes.

    Args:
        inputs (list[Node]): List of nodes to add together.
        name (str, optional): Optional name for the addition node.
    """

    def forward(self) -> None:
        """
        Computes the forward pass for addition.
        """
        # Broadcast all input values to a common shape before summing
        values = [np.asarray(inp.value) for inp in self.inputs]
        broadcasted = np.broadcast_arrays(*values)
        self.value = np.add.reduce(broadcasted)

    def backward(self, grad_output: np.ndarray) -> None:
        """
        Computes the backward pass for addition.
        """
        for inp in self.inputs:
            _accumulate_grad(inp, grad_output)
            # Only call backward if this node is not a leaf (i.e., not ConstantNode or InputNode)
            if isinstance(inp, ComputationalNode):
                inp.backward(grad_output)


class MulNode(ComputationalNode):
    """
    Node representing element-wise multiplication.

    Args:
        inputs (list[Node]): List of nodes to multiply together.
        name (str, optional): Optional name for the multiplication node.
    """

    def forward(self) -> None:
        """
        Computes the forward pass for multiplication.
        """
        result = self.inputs[0].value
        for inp in self.inputs[1:]:
            result = np.multiply(result, inp.value)
        self.value = result

    def backward(self, grad_output: np.ndarray) -> None:
        """
        Computes the backward pass for multiplication.
        """
        for i, inp in enumerate(self.inputs):
            prod = np.ones_like(self.value)
            for j, other_inp in enumerate(self.inputs):
                if i != j:
                    prod = prod * other_inp.value
            local_grad = grad_output * prod
            _accumulate_grad(inp, local_grad)
            if isinstance(inp, ComputationalNode):
                inp.backward(local_grad)


class PowNode(ComputationalNode):
    """
    Node representing element-wise power with a fixed exponent (default 2).

    Args:
        inputs (list[Node]): List with a single base node.
        power (float, optional): The exponent to raise the base to (default: 2).
        name (str, optional): Optional name for the power node.
    """

    def __init__(
        self, inputs: list["Node"], power: float = 2.0, name: str = None
    ) -> None:
        super().__init__(inputs, name=name)
        self.power = power

    def forward(self) -> None:
        base = self.inputs[0]
        self.value = np.power(base.value, self.power)

    def backward(self, grad_output: np.ndarray) -> None:
        base = self.inputs[0]
        grad_base = grad_output * self.power * np.power(base.value, self.power - 1)
        _accumulate_grad(base, grad_base)
        if isinstance(base, ComputationalNode):
            base.backward(grad_base)


class MeanNode(AddNode):
    """
    Node representing the mean of its input vector.

    Args:
        inputs (list[Node]): List with a single node whose mean is to be computed.
        name (str, optional): Optional name for the mean node.
    """

    def __init__(self, inputs: list, name: str = None) -> None:
        super().__init__(inputs, name=name)

    def forward(self) -> None:
        self.value = np.mean(self.inputs[0].value)

    def backward(self, grad_output: np.ndarray) -> None:
        # grad_output is scalar, distribute equally to all elements
        n = self.inputs[0].value.size
        grad = np.ones_like(self.inputs[0].value) * grad_output / n
        # Always call backward on the input node to ensure propagation
        if hasattr(self.inputs[0], "backward"):
            self.inputs[0].backward(grad)
        else:
            self.inputs[0].grad += grad
