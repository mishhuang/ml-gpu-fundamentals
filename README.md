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
The custom Triton kernel runs ~11x faster than CPU, confirming that
GPU parallelism works as expected — each GPU core independently
computes its assigned block of the output matrix simultaneously.

PyTorch's implementation runs ~25x faster than CPU because it calls
cuBLAS under the hood — a heavily optimized library that uses
advanced techniques including shared memory tiling (loading matrix
chunks into fast on-chip memory to avoid repeated VRAM access) and
vectorized memory operations. The gap between the custom kernel and
cuBLAS illustrates exactly why optimized library layers exist and
why the software stack between hardware and ML frameworks actually
matters.

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

## Project 2: Model Quantization

### What and why
Once a model is trained, the goal shifts from learning to deploying
efficiently. Quantization reduces model precision from 32-bit
floating point (FP32) to 8-bit integer (INT8), theoretically making
models 4x smaller and faster to run — one of the primary tools for
making inference economically viable at scale.

This project applies dynamic quantization to ResNet-18 and
systematically measures what actually changes — and investigates
why results differ from theoretical expectations.

### Model
ResNet-18 pretrained on ImageNet — 11.7M parameters, 1,000 output
classes, standard vision model for image classification.

### What quantization does in theory
- **4x smaller:** FP32 = 4 bytes per parameter, INT8 = 1 byte
- **Faster inference:** Integer math is simpler for hardware than
  floating point math
- **Minimal accuracy loss:** Models learn to be robust to small
  weight variations during training

### What actually happened

| Metric | FP32 (Original) | INT8 (Quantized) |
|--------|----------------|-----------------|
| Model size | 44.7 MB | 43.2 MB |
| Inference time | 93.85 ms | 93.85 ms |
| Size reduction | — | 1.03x smaller |
| Speedup | — | effectively none |
| Top-1 accuracy | matched | matched |
| Top-5 predictions | [107, 611, 4, 65, 845] | [107, 611, 4, 65, 845] |
| Max output diff | — | 0.0888 |

![Quantization Results](results/quantization_results.png)

### Key observations

**Why there was no meaningful speedup or size reduction —
investigation findings:**

**1. Only the fc layer was quantized.**
Dynamic quantization in modern PyTorch versions only affects Linear
layers. Conv2d dynamic quantization was dropped due to inconsistent
performance across hardware. ResNet-18 has a single Linear layer
(fc) representing ~4% of parameters (512K out of 11.7M) — not
enough to produce measurable size reduction.

**2. The fc layer is too small to benefit from dynamic quantization.**
At 512×1000 parameters, the overhead of dynamically converting
between FP32 and INT8 at runtime offsets any computational savings,
resulting in effectively identical inference times.

**3. Dynamic quantization is the wrong tool for vision models.**
Dynamic quantization produces meaningful results on models dominated
by large Linear layers — for example BERT's 768×3072 attention
layers. For vision models dominated by Conv2d layers, static
quantization or quantization-aware training are the appropriate
approaches.

**Accuracy was fully preserved.**
Top-1 and top-5 predictions were identical between FP32 and INT8
models. Max output difference of 0.0888 across 1,000 class scores
confirms negligible numerical impact from the rounding.

**What this demonstrates:**
Quantization strategy must be matched to model architecture. The
investigation process — applying quantization, measuring results,
diagnosing why they differed from expectations, and identifying the
correct approach — reflects real production ML infrastructure work.

### Three types of quantization
- **Dynamic (used here):** Weights pre-converted to INT8, activations
  converted on the fly. No calibration data needed. Best for
  transformer models with large Linear layers.
- **Static:** Both weights and activations pre-converted. Requires
  calibration data to determine scale factors per layer. Better
  suited for vision models.
- **Quantization-aware training (QAT):** Model trained with
  simulated quantization from the start. Best accuracy preservation.
  Most complex — requires retraining.

### What I learned
- Quantization strategy must match model architecture — applying
  the wrong type produces no benefit and can introduce overhead
- Dynamic quantization in modern PyTorch does not support Conv2d
  layers, limiting it to transformer-style architectures
- The crossover point for dynamic quantization: layers must be
  large enough that INT8 computation savings exceed the runtime
  conversion overhead
- Scale factors are the core mechanism of quantization — each
  layer maps its specific value range onto -128 to 127, requiring
  per-layer calibration for static quantization
- `torch.no_grad()` is required during inference — disables
  gradient tracking that is only needed during training

### Environment
- Hardware: CPU (intentional — dynamic quantization targets
  CPU inference)
- Python 3.x
- PyTorch, torchvision

### How to run
Open in Google Colab (CPU runtime is sufficient):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11R0644KGLPRREgdB2Tm-RJBjJdxCuXR9?usp=sharing)

---

*Project 3 (GPU Profiling) coming soon*
