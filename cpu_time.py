import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import random
# Constants for fixed-point Q16.16
FRAC_BITS = 16
SCALE = 1 << FRAC_BITS
FRAC_MASK = SCALE - 1



# Float LSE
def logsumexp_shape_preserving_cpu(x, y):
    max_xy = torch.maximum(x, y)
    min_xy = torch.minimum(x, y)   
    sub = torch.sub(min_xy, max_xy)
    exp_sub = torch.exp(sub)    
    sum_exp = torch.add(exp_sub, 1.0)   
    log_sum = torch.log(sum_exp)  
    result=torch.add(max_xy, log_sum)  
    return result

# Integer LSE
def lse_pe_int(x, y):
    x_max = torch.maximum(x, y)    
    y_min = torch.minimum(x, y)
    sub = torch.sub(y_min, x_max)
    shift = torch.clamp(-(sub >> FRAC_BITS), min=-31, max=0)
    approx = torch.bitwise_right_shift(SCALE + (sub & FRAC_MASK), shift)
    result = torch.add(x_max, approx)
    return result


trials = 10000
elapsed_time_float = []
elapsed_time_int = []
# Lists to store timing data
for i in range(trials):
    funcs = ['int', 'float']
    random.shuffle(funcs)
    if funcs[0] == 'float':
        xf = torch.randint(0, 1, (1000,), dtype=torch.float32)
        yf = torch.randint(0,1, (1000,), dtype=torch.float32)
        start_time = time.time()
        result_float = logsumexp_shape_preserving_cpu(xf, yf)
        elapsed_time_float.append((time.time() - start_time) * 1000)  # Convert to milliseconds

    else:

        xi = torch.randint(0, 2**31, (1000,), dtype=torch.int32)
        yi = torch.randint(0, 2**31, (1000,), dtype=torch.int32)
        start_time = time.time()
        result_int = lse_pe_int(xi, yi)
        elapsed_time_int.append((time.time() - start_time) * 1000)  # Convert to milliseconds

#calculate mean
mean_float = sum(elapsed_time_float) / len(elapsed_time_float)
mean_int = sum(elapsed_time_int) / len(elapsed_time_int)
print("mean float:", mean_float)
print("mean int:", mean_int)


