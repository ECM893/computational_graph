"""
Graph classes for managing computational graphs and automatic differentiation.

This module provides:
- DirectedGraph: A simple directed graph structure for holding nodes.
- ComputationGraph: A directed graph with topological sorting, forward, and backward passes for autodiff.

The ComputationGraph supports depth-first topological ordering for correct forward and backward computation.
"""

from src.node import Node


class DirectedGraph:
    """
    Simple directed graph for holding nodes.

    Attributes:
        nodes (list): List of nodes in the graph.
    """

    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        """
        Add a node to the graph.

        Args:
            node (Node): The node to add.
        """
        self.nodes.append(node)


class ComputationGraph(DirectedGraph):
    """
    Computational graph supporting forward and backward passes.

    Attributes:
        head (Node): The output node (head of the graph).
        _topo_order (list): Cached topological order of nodes.
    """

    def __init__(self):
        super().__init__()
        self.head = None  # The output node (head of the graph)
        self._topo_order = None  # Store topological order after DFS

    def set_head(self, node):
        """
        Set the head (output) node of the graph.

        Args:
            node (Node): The output node.
        """
        self.head = node

    def _compute_topological_order(self):
        """
        Computes and stores the topological order of nodes (post-order DFS from head).
        Ensures all dependencies are visited before a node.
        """
        # NOTE: I know i probably should solve this with recursion from the head node, but I started and got lazy doing it this way.
        visited = set()
        order = []

        def dfs(node):
            if id(node) in visited:
                return
            for parent in getattr(node, "parents", []):
                dfs(parent)
            if node not in order:
                order.append(node)
            visited.add(id(node))

        if self.head is None:
            raise ValueError("Head node is not set for the computation graph.")
        dfs(self.head)
        self._topo_order = order

    def forward(self):
        """
        Perform a depth-first post-order traversal (topological order) from the head node
        to compute all values in correct order. This ensures all dependencies are computed
        before a node's forward pass.
        """
        self._compute_topological_order()
        for node in self._topo_order:
            if type(node) is not Node:
                if hasattr(node, "forward"):
                    node.forward()

    def backward(self, loss_node):
        """
        Perform the backward pass (autodiff) through the graph.

        Args:
            loss_node (Node): The loss/output node to start backpropagation from.
        """
        # Zero all gradients
        for node in self._topo_order:
            node.zero_grad()
        # Set the initial gradient for the loss node
        # It should be 1.0 (not the value of the loss), because dL/dL = 1 for backprop
        loss_node.grad = 1.0
        # Backpropagate in reverse topological order
        for node in reversed(self._topo_order):
            if hasattr(node, "backward"):
                node.backward(node.grad)
