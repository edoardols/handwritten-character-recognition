import numpy as np
import torch
import time

# Function to perform the test
def perform_test():
    # Create random matrices
    size = 100
    np_matrix_a = np.random.rand(size, size)
    np_matrix_b = np.random.rand(size, size)

    torch_matrix_a = torch.rand(size, size)
    torch_matrix_b = torch.rand(size, size)

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

    # Return timing information
    return np_time, torch_mm_time, torch_matmul_time

# Repeat the test for 600 iterations
num_iterations = 60000
total_np_time, total_torch_mm_time, total_torch_matmul_time = 0, 0, 0

for _ in range(num_iterations):
    np_time, torch_mm_time, torch_matmul_time = perform_test()
    total_np_time += np_time
    total_torch_mm_time += torch_mm_time
    total_torch_matmul_time += torch_matmul_time

# Calculate average timing
average_np_time = total_np_time / num_iterations
average_torch_mm_time = total_torch_mm_time / num_iterations
average_torch_matmul_time = total_torch_matmul_time / num_iterations

# Print average timing information
print("Average NumPy Dot Product Time:", average_np_time)
print("Average PyTorch Matrix Multiplication (torch.mm) Time:", average_torch_mm_time)
print("Average PyTorch Matrix Multiplication (torch.matmul) Time:", average_torch_matmul_time)
