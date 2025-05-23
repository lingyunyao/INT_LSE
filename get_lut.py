import torch
import numpy as np
import warp as wp

# Constants for fixed-point Q16.16
FRAC_BITS = 16
SCALE = 1 << FRAC_BITS
FRAC_MASK = SCALE - 1

# Initialize Warp
wp.init()

# --- Warp Kernel for LSE ---
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

# --- GPU Approximation ---
def fast_lse_gpu(x_np, y_np, lookup_table):
    x_wp = wp.array(x_np, dtype=wp.int32, device="cuda")
    y_wp = wp.array(y_np, dtype=wp.int32, device="cuda")
    lut_wp = wp.array(lookup_table, dtype=wp.int32, device="cuda")
    out_wp = wp.empty_like(x_wp)

    wp.launch(kernel=lse_warp_kernel, dim=x_np.shape[0], inputs=[x_wp, y_wp, lut_wp, out_wp])
    wp.synchronize()

    return out_wp.numpy()

# --- Interpolation for Missing LUT Entries ---
def interpolate_missing_lut(lut):
    lut_filled = lut.copy()
    nonzero_indices = np.nonzero(lut)[0]
    for i in range(64):
        if lut[i] == 0:
            lower = max([j for j in nonzero_indices if j < i], default=None)
            upper = min([j for j in nonzero_indices if j > i], default=None)
            if lower is not None and upper is not None:
                lut_filled[i] = (lut[lower] + lut[upper]) / 2
            elif lower is not None:
                lut_filled[i] = lut[lower]
            elif upper is not None:
                lut_filled[i] = lut[upper]
    return lut_filled

# --- Generate Corrected LUT ---
def compute_corrected_lut(x_int, y_int, true_lse, approx_float):
    error_dict = {i: [] for i in range(64)}

    for i in range(len(x_int)):
        x_val = x_int[i]
        y_val = y_int[i]
        x_max = max(x_val, y_val)
        y_min = min(x_val, y_val)
        sub = y_min - x_max

        Ey = -(sub >> FRAC_BITS)
        if (sub & FRAC_MASK) != 0:
            Ey += 1

        My = sub & FRAC_MASK
        M_fixed = (SCALE + My) >> Ey
        index = (M_fixed & FRAC_MASK) >> 10

        err = (true_lse[i] - approx_float[i]) * SCALE
        error_dict[index].append(err)

    lut_floats = np.zeros(64, dtype=np.float32)
    for i in range(64):
        if error_dict[i]:
            avg_error_fixed = np.mean(error_dict[i])
            lut_floats[i] = avg_error_fixed / SCALE

    return interpolate_missing_lut(lut_floats), (interpolate_missing_lut(lut_floats) * SCALE).astype(np.int32)

# --- Main Process ---
if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    num_points = 10000

    # Wider input coverage
    x_base = torch.rand(num_points, dtype=torch.float32) * 10 - 5
    noise = torch.randn(num_points, dtype=torch.float32) * 2
    x_float = x_base
    y_float = x_base + noise

    true_lse = torch.logsumexp(torch.stack([x_float, y_float], dim=0), dim=0).numpy()

    x_int = (x_float.numpy() * SCALE).astype(np.int32)
    y_int = (y_float.numpy() * SCALE).astype(np.int32)

    # First pass: zero LUT
    zero_lut_fixed = np.zeros(64, dtype=np.int32)
    approx_int = fast_lse_gpu(x_int, y_int, zero_lut_fixed)
    approx_float = approx_int.astype(np.float32) / SCALE

    print(f"Max error: {np.max(np.abs(approx_float - true_lse)):.6f}")
    print(f"Mean error: {np.mean(np.abs(approx_float - true_lse)):.6f}")

    # Generate better LUT
    lut_floats, lut_fixed = compute_corrected_lut(x_int, y_int, true_lse, approx_float)

    # Second pass: updated LUT
    approx_int_updated = fast_lse_gpu(x_int, y_int, lut_fixed)
    approx_float_updated = approx_int_updated.astype(np.float32) / SCALE

    updated_error = np.abs(approx_float_updated - true_lse)
    print(f"Updated Max error: {np.max(updated_error):.6f}")
    print(f"Updated Mean error: {np.mean(updated_error):.6f}")

    print("Corrected LUT (float):")
    print(np.round(lut_floats, 8))
