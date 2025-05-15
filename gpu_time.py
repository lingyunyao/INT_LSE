import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Constants for fixed-point Q16.16
FRAC_BITS = 16
SCALE = 1 << FRAC_BITS
FRAC_MASK = SCALE - 1


# GPU timing utility
def measure_op_cuda(op, runs=1):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(runs):
        op()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / runs  # ms

# GPU warm-up
def warm_up_gpu(x, y, rounds=10):
    for _ in range(rounds):
        _ = x * y
# Int warm-up
def warm_up_int(x, y, rounds=10):
    for _ in range(rounds):
        _ = x * y

# Float LogSumExp (GPU)
def logsumexp_shape_preserving_gpu(x, y):
    timings = {}
    timings['max'] = measure_op_cuda(lambda: torch.maximum(x, y))
    max_xy = torch.maximum(x, y)
    timings['min'] = measure_op_cuda(lambda: torch.minimum(x, y))
    min_xy = torch.minimum(x, y)
    timings['sub'] = measure_op_cuda(lambda: min_xy - max_xy)
    sub = min_xy - max_xy
    timings['exp'] = measure_op_cuda(lambda: torch.exp(sub))
    exp_sub = torch.exp(sub)
    timings['sum'] = measure_op_cuda(lambda: exp_sub + 1.0)
    sum_exp = exp_sub + 1.0
    timings['log'] = measure_op_cuda(lambda: torch.log(sum_exp))
    log_sum = torch.log(sum_exp)
    timings['add'] = measure_op_cuda(lambda: max_xy + log_sum)
    timings['total'] = (
        timings['max'] + timings['min'] + timings['sub'] +timings['exp'] + timings['sum'] + timings['log'] + timings['add']
    )
    return max_xy + log_sum, timings

# Integer LogSumExp approximation (GPU)
def lse_pe_int_gpu(x: torch.Tensor, y: torch.Tensor):
    timings = {}
    timings['max'] = measure_op_cuda(lambda: torch.maximum(x, y))
    x_max = torch.maximum(x, y)
    timings['min'] = measure_op_cuda(lambda: torch.minimum(x, y))
    y_min = torch.minimum(x, y)
    timings['sub'] = measure_op_cuda(lambda: y_min - x_max)
    sub = y_min - x_max
    timings['bit_com'] = measure_op_cuda(lambda: (sub >> FRAC_BITS, sub & FRAC_MASK))
    I = (sub >> FRAC_BITS).clamp(min=-31, max=0)
    F = sub & FRAC_MASK
    timings['approx'] = measure_op_cuda(lambda: (SCALE + F) >> (-I))
    one_plus_F = SCALE + F
    shift_amount = -I
    approx = one_plus_F >> shift_amount
    timings['clut'] = measure_op_cuda(lambda: torch.zeros_like(approx))  # placeholder
    correction = torch.zeros_like(approx)
    timings['add'] = measure_op_cuda(lambda: x_max + approx + correction)
    timings['total'] = (
        timings['max'] + timings['min'] + timings['sub'] + timings['bit_com'] + timings['approx'] + timings['clut'] + timings['add']
    )
    return x_max + approx + correction, timings

trials = 10
# Lists to store timing data
float_timings_record = {k: [] for k in ['max', 'min', 'sub', 'exp', 'sum', 'log', 'add', 'total']}
int_timings_record = {k: [] for k in ['max', 'min', 'sub', 'bit_com', 'approx', 'clut', 'add', 'total']}
# Run trials
for _ in range(trials):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Warm-up
    # Generate random inputs
    xf = torch.rand(1, dtype=torch.float32)
    yf = torch.rand(1, dtype=torch.float32)
    warm_up_gpu(xf, yf)
    _, float_timings = logsumexp_shape_preserving_gpu(xf, yf)
    for k in float_timings:
        float_timings_record[k].append(float_timings[k])


    x_int = (xf * SCALE).to(torch.int32)
    y_int = (yf * SCALE).to(torch.int32)
    warm_up_int(x_int, y_int)
    _, int_timings = lse_pe_int_gpu(x_int, y_int)
    for k in int_timings:
        int_timings_record[k].append(int_timings[k])
# Convert to DataFrames
df_float = pd.DataFrame(float_timings_record)
df_int = pd.DataFrame(int_timings_record)
print("GPU: Float32 logsumexp timings in 10 times run experiments:")
print(df_float)
print("GPU: Four Int32 logsumexp timings in 10 times run experiments:")
print(df_int)







# Extract step orders
float_steps = df_float.columns.tolist()
int_steps = df_int.columns.tolist()

# Extract aligned timing values
float_vals = df_float.iloc[0]
int_vals = df_int.iloc[0]
float_aligned = [float_vals.get(step, 0) for step in float_steps]
int_aligned = [int_vals.get(step, 0) for step in int_steps]

# Y positions
y_pos_float = np.arange(len(float_steps))
y_pos_int = np.arange(len(int_steps))

# Shared x-axis range
x_max = max(max(float_aligned), max(int_aligned)) * 1.3

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Float32 plot
ax1.barh(y_pos_float, float_aligned, color='skyblue')
ax1.set_title("Float32 Step Timings")
ax1.set_xlim(0, x_max)
ax1.set_yticks(y_pos_float)
ax1.set_yticklabels(float_steps)
ax1.invert_yaxis()
ax1.grid(True, axis='x')
for i, v in enumerate(float_aligned):
    ax1.text(v + x_max * 0.01, i, f"{v:.3f} ms", va='center')

# Int32 plot
ax2.barh(y_pos_int, int_aligned, color='lightgreen')
ax2.set_title("Int32 Step Timings")
ax2.set_xlim(0, x_max)
ax2.set_yticks(y_pos_int)
ax2.set_yticklabels(int_steps)
ax2.invert_yaxis()
ax2.set_xlabel("Time (ms)")
ax2.grid(True, axis='x')
for i, v in enumerate(int_aligned):
    ax2.text(v + x_max * 0.01, i, f"{v:.3f} ms", va='center')

# Final layout
plt.suptitle("GPU Step-wise Timing Comparison: Float32 vs Int32", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('gpu_timings.png', dpi=300)
