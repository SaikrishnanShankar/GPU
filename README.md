# CS 8803 — GPU Architecture & Programming

**Saikrishnan Sankar | Georgia Institute of Technology**

A collection of assignments exploring GPU architecture, CUDA optimization, cycle-accurate simulation, and compiler analysis.

---

## Assignment 1 — Tiled Matrix Multiplication (CUDA)

Implemented tiled matrix multiplication in CUDA using shared memory to reduce global memory traffic.

- `TILE_WIDTH = 16` — each block handles a 16×16 tile of the output matrix
- Each thread loads one element into shared memory, syncs, then computes a partial dot product
- Tiling reduces global memory loads by 16× compared to naive implementation
- Benchmarked GPU vs CPU time using CUDA events

**Key concepts:** thread/block/grid hierarchy, shared memory, `__syncthreads()`, memory coalescing

---

## Assignment 2 — Bitonic Sort Optimization (CUDA)

Implemented and optimized GPU bitonic sort on the H100, achieving **1049 Million Elements Per Second** on 100M integers (191× faster than CPU `std::sort`).

**Optimizations applied:**
1. **Shared memory** — batch multiple merge steps in shared memory (up to 8192 elements per block), reducing kernel launches from O(log²N) to O(log N)
2. **Elements per thread** — each thread processes 4 elements for better SM occupancy
3. **Pinned memory** — `cudaHostRegister()` on existing `malloc` allocations for direct DMA transfers (H2D: 207ms → 20ms)
4. **Async transfers** — `cudaMemcpyAsync()` to overlap memory transfers with compute

**Nsight Compute findings:**
- Global memory kernel: memory bound (75% memory throughput > 72% compute) due to large non-sequential strides
- Shared memory kernels: compute bound, data stays on-chip
- Occupancy limited by shared memory (32KB/block = 1 block/SM on H100)
- Register spilling occurs beyond 24 registers/thread

---

## Assignment 3 — Warp Scheduling Simulator (C++)

Extended the MacSim cycle-accurate GPU simulator to implement three warp scheduling policies and measured their impact on cache performance across ML workloads (LavaMD, Backprop, NN, Crystal, Hotspot, Pathfinder).

**Policies implemented:**
- **RR (Round Robin)** — rotates through dispatch queue each cycle
- **GTO (Greedy Then Oldest)** — sticks with current warp until memory stall, then picks oldest
- **CCWS (Cache-Conscious Warp Scheduling)** — throttles cache-thrashing warps using a Victim Tag Array (VTA) and Live Liveness Score (LLS)

**Key result on LavaMD benchmark:**
| Policy | Cache Hit Rate | IPC |
|--------|---------------|-----|
| RR     | 30.49%        | 7.90 |
| GTO    | 79.03%        | 7.90 |
| CCWS   | 30.98%        | 7.90 |

GTO achieves 2.6× better cache hit rate than RR by preserving temporal locality — each warp reuses its own cached data before being evicted by other warps.

**Key concepts:** warp stalls, latency hiding, cache thrashing, taint-based scheduling, VTA

---

## Assignment 4 — Tensor Core Pipeline Extension (C++)

Extended MacSim to model tensor core instruction execution with register dependency checking.

**Task 1 — Compute Execution Buffer:**
- Tracks in-flight tensor core operations with completion timestamps
- Buffer capacity = execution width (configurable 2–16)
- Stalls warp when buffer full, preventing RAW (Read-After-Write) hazards
- `get_latency()` returns tensor latency (32–128 cycles) for H-prefixed opcodes

**Task 2 — Register Dependency Checking:**
- Before scheduling a warp, checks if its next instruction's source registers are still being computed
- Iterates through all warps to find a dependency-free one
- Stalls core only if all warps are blocked

**Key findings:**
| Metric | Task-1 (scalar) | Task-2 (tensor) |
|--------|----------------|----------------|
| Peak IPC | 3.70 | 3.70 |
| Stall ratio (Width=2) | 13% | 71% |
| Stall ratio (Width=16) | 13% | 13% |
| Performance variation | 0.4% | 36% |

- Width ≥ 8 captures 95% of peak performance — warp interleaving hides 128-cycle tensor latency
- RR outperforms GTO for tensor cores (opposite of Assignment 3) — interleaving hides compute latency better than greedy scheduling

**Key concepts:** RAW hazards, execution width, latency hiding, tensor core vs CUDA core

---

## Assignment 5 — GPU Assembly Analysis (Python)

Built a static analysis tool that parses NVIDIA SASS (GPU assembly) and detects branch divergence points — branches whose outcome depends on thread ID, causing warps to serialize.

**Pipeline:**
```
.sass file → Parser → CFG (basic blocks) → Type Analysis → XMAD→IMAD → Branch Divergence Detection → output
```

**Components:**
- **Parser** — parses SASS into instruction objects with opcodes, operands, predicate flags
- **XMAD → IMAD transform** — replaces 3-instruction XMAD multiplication patterns (SM52/SM35) with single IMAD instruction (SM75+), reducing instruction count
- **Branch divergence detection** — iterative taint analysis propagating thread ID dependency through the CFG:
  - Source: `S2R` instruction loading `SR_TID.X`
  - Propagation: any instruction using a tainted register taints its output
  - Detection: tainted branch instruction = divergence point
  - Fixed-point iteration handles loops where taint propagates across back edges

**Key concepts:** taint analysis, control flow graphs, dataflow fixed-point iteration, branch divergence, SASS

---

## Results Summary

| Assignment | Key Metric | Result |
|-----------|-----------|--------|
| 1 — Tiled MatMul | Memory load reduction | 16× fewer global loads |
| 2 — Bitonic Sort | Throughput | 1049 MEPS (191× CPU speedup) |
| 3 — Warp Scheduler | GTO cache hit rate (LavaMD) | 79% vs 30% for RR |
| 4 — Tensor Pipeline | Latency hiding at Width=8 | 128-cycle op → 3.5% overhead |
| 5 — SASS Analysis | Branch divergence detection | Iterative taint analysis on CFG |
