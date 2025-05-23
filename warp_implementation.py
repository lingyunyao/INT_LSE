import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import warp as wp

# Initialize Warp
wp.init()

# Constants
FRAC_BITS = 16
SCALE = 1 << FRAC_BITS
FRAC_MASK = SCALE - 1

# Lookup table (64 entries)
lut_floats = np.array([
    -0.00710867, -0.01231277, -0.01569946, -0.0161281,  -0.01534673, -0.01578892,
    -0.01221189, -0.00391352,  0.00185403,  0.00285605,  0.00580598,  0.01069435,
     0.01808678,  0.0276791,   0.03971929,  0.0548892,   0.06403098,  0.06596849,
     0.06870083,  0.07221534,  0.07660637,  0.0820575,   0.08814302,  0.0952751,
     0.10349645,  0.11241663,  0.12237417,  0.13240126,  0.14425048,  0.15718189,
     0.17098776,  0.18599182,  0.18599182,  0.18599182,  0.18599182,  0.18599182,
     0.18599182,  0.18599182,  0.18599182,  0.18599182,  0.18599182,  0.18599182,
     0.18599182,  0.18599182,  0.18599182,  0.18599182,  0.18599182,  0.18599182,
     0.18599182,  0.18599182,  0.18599182,  0.18599182,  0.18599182,  0.18599182,
     0.18599182,  0.18599182,  0.18599182,  0.18599182,  0.18599182,  0.18599182,
     0.18599182,  0.18599182,  0.18599182,  0.18599182
])
lut_fixed = (lut_floats * SCALE).astype(np.int32)
lookup_table_wp = wp.array(lut_fixed, dtype=wp.int32, device="cuda")

# PyTorch reference
def logsumexp_shape_preserving_cpu(x, y):
    return torch.logsumexp(torch.stack((x, y), dim=0), dim=0)

# Warp kernel
@wp.kernel
def lse_warp_kernel(x: wp.array(dtype=wp.int32), 
                    y: wp.array(dtype=wp.int32),
                    lookup_table: wp.array(dtype=wp.int32),
                    out: wp.array(dtype=wp.int32)):

    tid = wp.tid()
    x_val = x[tid]
    y_val = y[tid]
    x_max = wp.max(x_val, y_val)
    y_min = wp.min(x_val, y_val)
    sub = y_min - x_max

    Ey = -(sub >> FRAC_BITS)
    if (sub & FRAC_MASK) != 0:
        Ey += 1

    My = sub & FRAC_MASK
    M = (SCALE + My) >> Ey
    index = (M >> 10) & 0x3F
    correction = lookup_table[index]
    out[tid] = x_max + M + correction

### Accuracy Test ###
print("=== Accuracy Test ===")
num_points = 100000
x_float = torch.rand(num_points, dtype=torch.float32)
y_float = torch.rand(num_points, dtype=torch.float32)
true_lse = torch.logsumexp(torch.stack([x_float, y_float], dim=0), dim=0).numpy()

x_int = (x_float.numpy() * SCALE).astype(np.int32)
y_int = (y_float.numpy() * SCALE).astype(np.int32)

x_wp = wp.array(x_int, dtype=wp.int32, device="cuda")
y_wp = wp.array(y_int, dtype=wp.int32, device="cuda")
out_wp = wp.empty_like(x_wp)

wp.launch(
    kernel=lse_warp_kernel,
    dim=num_points,
    inputs=[x_wp, y_wp, lookup_table_wp, out_wp]
)
wp.synchronize()

approx_int = out_wp.numpy()
approx_float = approx_int.astype(np.float32) / SCALE
error = np.abs(approx_float - true_lse)

print(f"Max error: {np.max(error):.6f}")
print(f"Mean error: {np.mean(error):.6f}")

### Performance Benchmark ###
print("\n=== Performance Benchmark ===")
data_size = 100000
trials = 100
elapsed_time_float = []
elapsed_time_warp = []

for _ in range(trials):
    funcs = ['float', 'warp']
    random.shuffle(funcs)
    func = funcs[0]

    if func == 'float':
        xf = torch.rand(data_size, dtype=torch.float32)
        yf = torch.rand(data_size, dtype=torch.float32)
        start_time = time.time()
        result_float = logsumexp_shape_preserving_cpu(xf, yf)
        elapsed_time_float.append((time.time() - start_time) * 1000)

    elif func == 'warp':
        xi = torch.randint(0, 2**31, (data_size,), dtype=torch.int32).numpy()
        yi = torch.randint(0, 2**31, (data_size,), dtype=torch.int32).numpy()

        x_wp = wp.array(xi, dtype=wp.int32, device="cuda")
        y_wp = wp.array(yi, dtype=wp.int32, device="cuda")
        out_wp = wp.empty_like(x_wp)

        start_time = time.time()
        wp.launch(kernel=lse_warp_kernel, dim=data_size, inputs=[x_wp, y_wp, lookup_table_wp, out_wp])
        wp.synchronize()
        elapsed_time_warp.append((time.time() - start_time) * 1000)

# Discard warm-up run
elapsed_time_float = elapsed_time_float[1:]
elapsed_time_warp = elapsed_time_warp[1:]

def mean(lst):
    return sum(lst) / len(lst) if lst else float('nan')

print(f"Mean Float LSE (PyTorch): {mean(elapsed_time_float):.4f} ms")
print(f"Mean Warp LSE (GPU): {mean(elapsed_time_warp):.4f} ms")
