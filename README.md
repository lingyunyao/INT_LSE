# INT_LSE
see warp implementation for now; 
python3 warp_implementation.py 
Warp 1.7.1 initialized:
   CUDA Toolkit 12.8, Driver 12.8
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA A100-SXM4-80GB" (79 GiB, sm_80, mempool enabled)
   Kernel cache:
     /home/yaol4/.cache/warp/1.7.1
=== Accuracy Test ===
Module __main__ 9ddb08f load on device 'cuda:0' took 374.65 ms  (compiled)
Max error: 0.299756
Mean error: 0.002642

=== Performance Benchmark ===
Mean Float LSE (PyTorch): 0.6051 ms
Mean Warp LSE (GPU): 0.0499 ms