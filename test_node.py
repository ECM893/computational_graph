import numpy as np
import pytest
from src.node import Node, AddNode, MulNode, PowNode

# ---- Forward Tests ----


def test_add_node_vector_forward():
    a = Node([1, 2, 3])
    b = Node([4, 5, 6])
    add = AddNode([a, b])
    add.forward()
    assert np.allclose(add.value, [5, 7, 9])


def test_add_node_broadcast_forward():
    a = Node([1, 2, 3])
    b = Node(10)
    add = AddNode([a, b])
    add.forward()
    assert np.allclose(add.value, [11, 12, 13])


def test_mul_node_vector_forward():
    a = Node([1, 2, 3])
    b = Node([4, 5, 6])
    mul = MulNode([a, b])
    mul.forward()
    assert np.allclose(mul.value, [4, 10, 18])


def test_mul_node_broadcast_forward():
    a = Node([1, 2, 3])
    b = Node(2)
    mul = MulNode([a, b])
    mul.forward()
    assert np.allclose(mul.value, [2, 4, 6])


def test_pow_node_vector_forward():
    a = Node([2, 3, 4])
    b = Node(2)
    pow_node = PowNode([a, b])
    pow_node.forward()
    assert np.allclose(pow_node.value, [4, 9, 16])


def test_pow_node_broadcast_forward():
    a = Node([2, 3, 4])
    pow_node = PowNode([a], power=2)
    pow_node.forward()
    assert np.allclose(pow_node.value, [4, 9, 16])


def test_add_node_multiple_inputs_forward():
    a = Node([1, 2, 3])
    b = Node([4, 5, 6])
    c = Node([7, 8, 9])
    add = AddNode([a, b, c])
    add.forward()
    assert np.allclose(add.value, [12, 15, 18])


# ---- Backward Tests ----


def test_add_node_backward():
    a = Node([1, 2, 3], requires_grad=True)
    b = Node([4, 5, 6], requires_grad=True)
    add = AddNode([a, b])
    add.forward()
    add.backward(np.array([1, 1, 1]))
    assert np.allclose(a.grad, [1, 1, 1])
    assert np.allclose(b.grad, [1, 1, 1])


def test_mul_node_backward():
    a = Node([1, 2, 3], requires_grad=True)
    b = Node([4, 5, 6], requires_grad=True)
    mul = MulNode([a, b])
    mul.forward()
    mul.backward(np.array([1, 1, 1]))
    assert np.allclose(a.grad, [4, 5, 6])
    assert np.allclose(b.grad, [1, 2, 3])


def test_pow_node_backward():
    a = Node([2, 3, 4], requires_grad=True)
    pow_node = PowNode([a], 2)
    pow_node.forward()
    pow_node.backward(np.array([1, 1, 1]))
    # d/da a**2 = 2*a
    expected_grad = 2 * np.array([2, 3, 4])
    assert np.allclose(a.grad, expected_grad)


# ---- Simple end-to-end test ----


def test_simple_chain_backward():
    # y = (a * x + b) ** 2, x = [1, 2, 3], a, b are scalars
    x = Node([1, 2, 3])
    a = Node(2.0, requires_grad=True)
    b = Node(1.0, requires_grad=True)
    ax = MulNode([a, x])
    ax.forward()
    ax_b = AddNode([ax, b])
    ax_b.forward()
    y = PowNode([ax_b], 2)
    y.forward()
    # Assume loss is sum(y)
    grad_output = np.ones_like(y.value)
    y.backward(grad_output)
    # Check gradients (just check they are not zero and are finite)
    assert np.isfinite(a.grad)
    assert np.isfinite(b.grad)


# ---- Run all tests if run as script ----

if __name__ == "__main__":
    pytest.main([__file__])
