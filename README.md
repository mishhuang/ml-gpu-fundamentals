# ML GPU Fundamentals

A hands-on exploration of GPU computing fundamentals relevant to 
AI/ML infrastructure. Built to understand the hardware-software 
interface that powers modern AI training and inference workloads.

---

## Project 1: CUDA Matrix Multiplication

### What and why
Matrix multiplication is the core mathematical operation of every 
neural network — every forward pass, every training step, every 
inference call ultimately reduces to multiplying large matrices 
together. Understanding how GPUs accelerate this operation is 
fundamental to understanding AI infrastructure.

This project implements matrix multiplication three ways and 
benchmarks each one to understand the performance tradeoffs at 
different levels of the stack.

### Implementations
- **CPU (NumPy):** Baseline sequential implementation
- **GPU (PyTorch):** GPU-accelerated via PyTorch, which calls 
  NVIDIA's cuBLAS library under the hood
- **GPU (Custom Triton Kernel):** Direct GPU kernel implementation 
  where each GPU core independently computes one block of the 
  output matrix

### Results
1000×1000 matrices, averaged over 10 runs on NVIDIA T4 GPU:

| Implementation | Time | Speedup vs CPU |
|---------------|------|----------------|
| CPU (NumPy) | 21.58 ms | 1x |
| GPU (PyTorch / cuBLAS) | 0.85 ms | 25.4x |
| GPU (Custom Kernel) | 1.98 ms | 10.9x |

![Benchmark Results](results/benchmark_results.png)

### Key observations
The custom Triton kernel runs ~11x faster than CPU, 
confirming that GPU parallelism works as expected — each GPU core 
independently computes its assigned block of the output matrix 
simultaneously.

PyTorch's implementation runs ~25x faster than CPU because 
it calls cuBLAS under the hood — a heavily optimized library that 
uses advanced techniques including shared memory tiling (loading 
matrix chunks into fast on-chip memory to avoid repeated VRAM 
access) and vectorized memory operations. The gap between the custom 
kernel and cuBLAS illustrates exactly why optimized library layers 
exist and why the software stack between hardware and ML frameworks 
actually matters.

### What I learned
- GPU operations are **asynchronous** — Python queues instructions 
  without waiting for execution. `torch.cuda.synchronize()` is 
  required to measure actual GPU execution time, not queue time
- GPUs require a **warmup run** before benchmarking — the first 
  operation carries initialization overhead that would skew results
- CPU and GPU have **completely separate memory** — data must be 
  explicitly transferred to VRAM via `.cuda()` before the GPU can 
  work on it, and this transfer cost is a real performance 
  consideration in production systems
- Correctness verification is as important as performance 
  measurement — the custom kernel was validated against PyTorch's 
  output to confirm mathematical accuracy before interpreting 
  benchmark results

### Environment
- Hardware: NVIDIA T4 GPU (Google Colab)
- Python 3.x
- PyTorch, Triton, NumPy

### How to run
Open in Google Colab and set runtime to T4 GPU:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1f-K29Qmoe7J_K_9gpPOqXbgjL46a-0UF?usp=sharing)


---

*Projects 2 (Model Quantization) and 3 (GPU Profiling) coming soon*
