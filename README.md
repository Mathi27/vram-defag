# VRAM Defragmenter

This Project is a pluggable memory allocator for PyTorch and JAX designed to mitigate external VRAM fragmentation in large-scale model training and inference. 

By implementing a Spatio-Temporal allocation strategy, the system isolates tensors based on their predicted lifespan, preventing the "interleaving" of persistent and transient memory blocks that leads to Out-of-Memory (OOM) errors.

In production environments running Long-Context LLMs or Mixture-of-Experts (MoE) models, ST-Allocator can recover up to 25% of effectively "lost" VRAM, allowing for larger batch sizes and extended context windows on existing hardware.

### The Problem: External Fragmentation

Standard CUDA memory allocators (such as the default caching allocator in PyTorch) use a general-purpose strategy that does not account for the lifecycle of specific tensors.

## In a typical LLM inference cycle:

- **Persistent Tensors**: Model weights and KV-caches remain in memory for the duration of the session.

- **Ephemeral Tensors**: Intermediate activations and gradients are allocated and freed rapidly.

When these two types are interleaved in the same physical memory space, the freeing of ephemeral tensors creates small, non-contiguous "holes." Over time, the heap becomes highly fragmented.

A request for a large contiguous block (e.g., a new KV-cache layer) may fail even if the total free VRAM exceeds the requested size.

## Solution 

This project implements a dual-pool architecture that segregates memory based on temporal characteristics
### 1. Physical Separation
The allocator maintains distinct virtual memory regions for different tensor categories. This ensures that the churn in the activation pool never impacts the continuity of the weight and cache pools.

### 2. Lifespan Heuristics
Using the PyTorch Pluggable Allocator API, this project inspects tensor metadata at the moment of allocation to predict its lifespan:

**Persistent:** Parameters, buffers, and tensors with high rank/size that persist across iterations.

**Transient:** Tensors generated during the forward pass with a high probability of being freed during the backward pass