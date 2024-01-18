import numpy as np
import torch
import cupy as cp
import tensorflow as tf
import time

# Function to perform the test
def perform_test():
    # Create random matrices
    size = 1000
    np_matrix_a = np.random.rand(size, size)
    np_matrix_b = np.random.rand(size, size)

    torch_matrix_a = torch.rand(size, size)
    torch_matrix_b = torch.rand(size, size)

    cupy_matrix_a = cp.asarray(np_matrix_a)
    cupy_matrix_b = cp.asarray(np_matrix_b)

    # tf_matrix_a = tf.constant(np_matrix_a)
    # tf_matrix_b = tf.constant(np_matrix_b)

    # NumPy dot product
    start_time = time.time()
    np_dot_result = np.dot(np_matrix_a, np_matrix_b)
    np_time = time.time() - start_time

    # PyTorch matrix multiplication using torch.mm
    start_time = time.time()
    torch_mm_result = torch.mm(torch_matrix_a, torch_matrix_b)
    torch_mm_time = time.time() - start_time

    # PyTorch matrix multiplication using torch.matmul
    start_time = time.time()
    torch_matmul_result = torch.matmul(torch_matrix_a, torch_matrix_b)
    torch_matmul_time = time.time() - start_time

    # CuPy dot product
    start_time = time.time()
    cupy_dot_result = cp.dot(cupy_matrix_a, cupy_matrix_b)
    cupy_time = time.time() - start_time

    # TensorFlow matrix multiplication
    start_time = time.time()
    tf_matmul_result = tf.matmul(tf_matrix_a, tf_matrix_b)
    tf_time = time.time() - start_time

    # Return timing information
    return np_time, torch_mm_time, torch_matmul_time, cupy_time, tf_time

# Repeat the test for 600 iterations
num_iterations = 600
total_np_time, total_torch_mm_time, total_torch_matmul_time, total_cupy_time, total_tf_time = 0, 0, 0, 0, 0

for _ in range(num_iterations):
    np_time, torch_mm_time, torch_matmul_time, cupy_time, tf_time = perform_test()
    total_np_time += np_time
    total_torch_mm_time += torch_mm_time
    total_torch_matmul_time += torch_matmul_time
    total_cupy_time += cupy_time
    total_tf_time += tf_time

# Calculate average timing
average_np_time = total_np_time / num_iterations
average_torch_mm_time = total_torch_mm_time / num_iterations
average_torch_matmul_time = total_torch_matmul_time / num_iterations
average_cupy_time = total_cupy_time / num_iterations
average_tf_time = total_tf_time / num_iterations

# Print average timing information
print("Average NumPy Dot Product Time:", average_np_time)
print("Average PyTorch Matrix Multiplication (torch.mm) Time:", average_torch_mm_time)
print("Average PyTorch Matrix Multiplication (torch.matmul) Time:", average_torch_matmul_time)
print("Average CuPy Dot Product Time:", average_cupy_time)
print("Average TensorFlow Matrix Multiplication Time:", average_tf_time)

