# NVIDIA Interview Prep — Saikrishnan Sankar

---

## GPU Fundamentals

### Thread Hierarchy
- **Thread**: single execution unit, runs one instance of the kernel, has unique `threadIdx`
- **Warp**: 32 threads executing the same instruction in lockstep (SIMT). Hardware unit — not programmer-defined
- **Block**: programmer-defined group of threads (max 1024 on H100). Threads in a block share shared memory and can sync via `__syncthreads()`
- **Grid**: collection of blocks launched for one kernel call

```
Grid → Blocks → Warps (hardware) → Threads
```

### Memory Hierarchy (fastest → slowest)
| Level | Latency | Scope | Size |
|-------|---------|-------|------|
| Registers | 1 cycle | Per thread | 65536 per SM |
| Shared Memory | ~5 cycles | Per block | 48KB per SM (H100) |
| L1 Cache | ~30 cycles | Per SM | Shared with shared mem |
| L2 Cache | ~100 cycles | Chip-wide | 50MB (H100) |
| Global Memory (HBM) | ~200 cycles | All threads | 80GB (H100) |

### Warp Execution
- All 32 threads in a warp execute the same instruction simultaneously
- **Latency hiding**: SM holds many warps. When one warp stalls (memory load), SM switches to another — zero cost because all warp state lives on-chip in registers
- **Occupancy**: % of SM warp slots actually filled. Limited by registers/thread, shared memory/block, block size
- **Branch divergence**: when threads in a warp take different paths, GPU serializes both paths — halves throughput

### Key H100 Numbers
- 132 SMs, 64 warps/SM max, 2048 threads/SM max
- Peak FP16 Tensor Core: 989 TFLOPS
- Peak FP32 CUDA Core: 60 TFLOPS
- HBM3 bandwidth: 3.35 TB/s
- Shared memory: 48KB/SM (configurable up to 228KB with L1 reduction)
- Max threads/block: 1024
- Warp size: 32

---

## Assignment 1 — Matrix Multiply & CUDA Basics

### What Was Built
Tiled matrix multiplication in CUDA using shared memory. Each block computes a 16×16 tile of the output matrix C. Threads collaboratively load tiles of A and B into shared memory, sync, compute partial dot products, advance to the next tile, repeat.

### Key Implementation (kernel.cu)
```cpp
#define TILE_WIDTH 16

__global__ void MatrixMulCUDA(float* C, float* A, float* B, int matrixWidth) {
    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int tiles = matrixWidth / TILE_WIDTH;
    float value = 0;

    for (int t = 0; t < tiles; t++) {
        sharedA[threadRow][threadCol] = A[row * matrixWidth + t * TILE_WIDTH + threadCol];
        sharedB[threadRow][threadCol] = B[(t * TILE_WIDTH + threadRow) * matrixWidth + col];
        __syncthreads();                          // wait for all loads

        for (int k = 0; k < TILE_WIDTH; k++)
            value += sharedA[threadRow][k] * sharedB[k][threadCol];
        __syncthreads();                          // wait before overwriting shared mem
    }
    C[row * matrixWidth + col] = value;
}
```

### Key Concepts
- **Tiling**: divide matrix into TILE_WIDTH×TILE_WIDTH blocks. Each block loads one tile into shared memory and reuses it for all computations in that tile
- **Memory load reduction**: naive = W loads/element from global. Tiled = W/T loads/element. For T=16: **16× fewer global memory loads**
- **Two `__syncthreads()`**: first ensures all threads finished loading before any computes. Second ensures all threads finished computing before next tile overwrites shared memory
- **Thread indexing**: `row = blockIdx.y * TILE_WIDTH + threadIdx.y` — each thread owns exactly one output element C[row][col]
- **cudaMallocManaged**: unified memory — CPU and GPU share same pointer. Simpler but slower than explicit cudaMalloc + cudaMemcpy

### What Breaks With Non-Power-of-TILE_WIDTH Sizes
```cpp
int tiles = matrixWidth / TILE_WIDTH;  // integer division truncates
```
If matrixWidth=18, TILE_WIDTH=4: tiles=4, covers only 16 columns. Last 2 columns never computed. Silent wrong answer — needs boundary checks or padding.

### Timing Setup
Used `cudaEventRecord` / `cudaEventElapsedTime` for GPU timing, `chrono::high_resolution_clock` for CPU. Computes speedup = cpu_time / gpu_time.

### Important Numbers
- TILE_WIDTH = 16 → 256 threads/block = 8 warps/block
- 16× reduction in global memory loads vs naive
- Block loads 16×16×2 = 512 elements from global memory, does 16×16×16 = 4096 MACs → arithmetic intensity = 4096/512 = 8 FLOPS/byte

### 5 Likely Interview Questions

**Q1: Why do you need two `__syncthreads()` instead of one?**
The first `__syncthreads()` after loading ensures all 256 threads have finished writing to shared memory before any thread starts reading — prevents a fast thread from computing with stale data. The second `__syncthreads()` after computing ensures all threads finish using the current tile before anyone overwrites it with the next tile's data. Without the second, a fast thread on tile t+1 could corrupt shared_A/B that a slow thread is still reading for tile t.

**Q2: What is the arithmetic intensity of this kernel and is it compute or memory bound?**
Arithmetic intensity = FLOPS / bytes loaded = (2 × W³) / (2 × W² × 4 bytes) = W/4 FLOPS/byte. For W=1024, that's 256 FLOPS/byte. H100 roofline crossover is ~295 FLOPS/byte (60 TFLOPS / 3.35 TB/s). So at W=1024 this kernel is on the edge — memory bound for small W, compute bound for large W. Tiling pushes the effective intensity up by a factor of TILE_WIDTH.

**Q3: How does the tiled version reduce global memory traffic?**
In naive matmul, each thread independently loads W elements of A and W elements of B from global memory. In the tiled version, a block of TILE_WIDTH² threads collaboratively loads one tile of A (TILE_WIDTH² elements) and one tile of B into shared memory — each element is loaded once but used by TILE_WIDTH different threads. This reduces global memory loads by a factor of TILE_WIDTH (16× for TILE_WIDTH=16).

**Q4: What is shared memory bank conflict and could it appear here?**
Shared memory is divided into 32 banks. If multiple threads in a warp access different addresses in the same bank, they serialize (bank conflict). In this kernel, `sharedA[threadRow][k]` — all threads in a warp read the same k from different rows, which maps to different banks. `sharedB[k][threadCol]` — all threads read the same row k but different columns, also different banks. So the access pattern here is conflict-free.

**Q5: What's the difference between `cudaMallocManaged` and `cudaMalloc + cudaMemcpy`?**
`cudaMallocManaged` allocates unified memory accessible from both CPU and GPU with the same pointer — CUDA runtime handles data migration automatically via page faults. Simpler to code but has overhead from page migration and may not achieve peak bandwidth. `cudaMalloc + cudaMemcpy` explicitly allocates GPU memory and transfers data in bulk — more control, can overlap transfers with compute using async APIs, generally faster for production. For this assignment, unified memory was fine since we don't need to optimize transfers.

---

## Assignment 2 — Kernel Optimization & Nsight

### What Was Built
GPU bitonic sort on H100, optimized from a naive global-memory-only implementation to a hybrid shared/global memory approach with pinned memory and async transfers.

**Final result: 1049 MEPS on 100M integers, 191× faster than CPU `std::sort`**

### Bitonic Sort Algorithm
- A sorting network based on compare-and-swap operations
- For each stage `i` (sequence size doubles: 2, 4, 8, ..., N) and step `j` (distance halves):
  - Each thread compares element `idx` with partner `idx XOR 2^j`
  - Ascending or descending based on which group: `(idx & (1<<i)) == 0`
- All comparisons at each (i,j) step are **independent** → perfectly parallel

**Partner finding**: `b = idx ^ dist` where `dist = 1 << j`
Only lower-index thread does the swap: `if (idx >= b) return`

**Direction**: `bool ascending = (idx & (1 << i)) == 0`
Groups threads into blocks of size 2^i — even groups ascending, odd descending.

### Optimizations Applied

**1. Shared Memory Batching (biggest win)**
When comparison distance < block size, both elements are in same block. Load into shared memory once, do all remaining j steps without leaving kernel, write back once. Reduces kernel launches from O(log²N) ≈ 378 to ~50.
```
LOG_MAX_SHARED = 13 → handle all j steps where dist < 2^13 = 8192 in shared memory
8192 elements × 4 bytes = 32KB ← exactly at H100's 48KB limit
```

**2. Elements Per Thread (4 strides)**
Each thread processes 4 elements instead of 1:
```cpp
int elements_per_thread = max((1 << (LOG_MAX_SHARED - 1)) / THREADS_PER_BLOCK, 1);
// = (8192/2) / 1024 = 4
```
More work per thread → threads live longer → better SM occupancy → better latency hiding.

**3. Pinned Memory**
`cudaMallocHost()` not possible (array already `malloc`'d). Used `cudaHostRegister()` to retroactively pin:
```cpp
cudaHostRegister(arrCpu, size * sizeof(DTYPE), cudaHostRegisterDefault);
```
Eliminates intermediate copy step in H2D transfer. DMA accesses host memory directly.
Result: H2D 207ms → 20ms, D2H 101ms → 7ms.

**4. Async Transfers**
```cpp
cudaMemcpyAsync(d_arr, arrCpu, size * sizeof(DTYPE), cudaMemcpyHostToDevice);
// removed cudaDeviceSynchronize() calls
```
Overlaps H2D transfer with CPU work. Non-blocking — returns immediately.

**5. Optimized Index Calculation (eliminates redundant work)**
```cpp
int idx1 = thread_id + (thread_id & ~((1 << j) - 1));
int idx2 = idx1 + (1 << j);
```
Masks out lower j bits of thread_id, grouping threads into unique comparison pairs. Every thread does useful work, no thread compares the same pair twice. Also achieves coalesced memory access since consecutive threads access nearby addresses.

**6. Loop Unrolling**
```cpp
#pragma unroll
for (int j = j_start; j >= 0; j--) { ... }
```
Compiler unrolls loops known at compile time. Eliminates loop counter overhead, branch instructions, enables better register allocation and instruction-level parallelism.

### Nsight Compute Results

**Three kernels profiled:**

| Kernel | Compute Throughput | Memory Throughput | Bound |
|--------|-------------------|-------------------|-------|
| `bitonic_sort_shared_initial` | Higher | Lower | Compute |
| `bitonic_merge_global` | 72% | 75% | Memory |
| `bitonic_sort_shared_final` | Higher | Lower | Compute |

**Occupancy analysis:**
- Shared memory is the binding constraint: 32KB/block → only 1 block fits per SM (48KB total)
- 1 block × 1024 threads = 32 warps → ~50% theoretical occupancy
- Cannot go higher: 2^14 = 64KB exceeds SM limit → occupancy drops to 0
- Register spilling: performance degrades hard after 24 registers/thread
- Chosen: 1024 threads/block for maximum occupancy within constraints

**Why global kernel is memory bound:**
Large stride patterns (e.g. j=25, dist=33M) → thread 0 accesses element 0, thread 1 accesses element 33M+1 → completely uncoalesced → 32 separate memory transactions per warp instead of 1 → wasted bandwidth.

### Baseline vs Optimized

| Metric | Baseline | Optimized |
|--------|----------|-----------|
| MEPS | 251 | 1049 |
| Kernel Time | 101ms | 66ms |
| H2D Time | 207ms | 20ms |
| D2H Time | 101ms | 7ms |
| CPU Speedup | 43× | 191× |

### 5 Likely Interview Questions

**Q1: What was limiting occupancy in your bitonic sort kernel and how did you handle it?**
Shared memory was the primary limiter. I used 8192 elements per block at 4 bytes each = 32KB of shared memory. The H100 has 48KB per SM, so only 1 block could fit per SM (48/32 = 1.5, rounds to 1). This gave ~50% theoretical occupancy. I couldn't increase it because going to 2^14 = 64KB would exceed the SM limit and occupancy would drop to zero. So 32KB/8192 elements was the sweet spot — right at the constraint boundary. I compensated by using 1024 threads/block (maximum) and 4 elements/thread to maximize work per active warp.

**Q2: Your global memory kernel was memory bound — how did Nsight tell you that and what did you do?**
Nsight showed memory throughput at 75% and compute throughput at 72% — memory was higher, so the GPU was spending more time waiting on data than doing math. The cause was uncoalesced access: bitonic sort's large strides (up to 33M elements apart) meant consecutive threads accessed completely non-adjacent memory addresses, forcing 32 separate transactions per warp instead of one. My fix was to push as many comparison steps as possible into shared memory. Steps where distance < 8192 (LOG_MAX_SHARED=13) are handled entirely in shared memory in a single kernel call, avoiding global memory almost completely for those steps.

**Q3: Why does `cudaHostRegister` help and why couldn't you use `cudaMallocHost`?**
`cudaMallocHost` allocates page-locked memory from scratch, enabling DMA directly from host to GPU without an intermediate copy. But `arrCpu` was already allocated with `malloc()` in code I couldn't modify. `cudaHostRegister` retrofits an existing allocation — it page-locks the memory in place, giving the same DMA benefit without reallocating. The speedup is from eliminating the hidden intermediate copy that `cudaMemcpy` normally does: regular malloc → temporary pinned buffer → GPU. Pinned memory → GPU directly. H2D went from 207ms to 20ms.

**Q4: Walk me through why RR would be terrible for bitonic sort and you had to redesign the kernel launch strategy.**
The naive implementation launches one kernel per (i,j) pair — O(log²N) ≈ 378 launches for 100M elements — because each step requires a full GPU synchronization. Round-robin at the kernel level means the CPU and GPU take turns: CPU submits kernel, GPU runs, CPU waits (cudaDeviceSynchronize), repeat. Most cycles are wasted on launch overhead and synchronization gaps. The fix is to batch multiple j steps into a single kernel using shared memory: when elements being compared are within the same block (dist < blockSize), load them all once, do all inner j steps without leaving the kernel, write back once. This collapses 378 kernel launches down to ~50 and eliminates hundreds of synchronization barriers.

**Q5: What is branch divergence and does bitonic sort have it?**
Branch divergence occurs when threads within a warp take different paths at a conditional branch — some execute the if-branch, others the else-branch. The GPU must execute both paths serially with different thread masks, effectively halving throughput for that section. In bitonic sort, the `bool ascending = (idx & (1<<i)) == 0` condition creates divergence if threads in the same warp fall into different groups. For large i, groups are much larger than 32 threads, so all threads in a warp are in the same group — no divergence. But for small i (e.g. i=1, group size=2), threads 0 and 1 go different ways — full divergence. This is unavoidable in bitonic sort's early stages. The optimized index calculation helps by ensuring all active threads do useful work even during potentially divergent steps.

---

## Assignment 3 — Warp Scheduling (GTO, RR, CCWS)

### What Was Built
Extended MacSim, a cycle-accurate GPU simulator written in C++, to implement three warp scheduling policies: Round Robin (RR), Greedy-Then-Oldest (GTO), and Cache-Conscious Warp Scheduling (CCWS). Evaluated all three on six ML workloads.

**Codebase: `src/macsim/core.cpp`** — `run_a_cycle()` and `schedule_warps_*()` functions.

### Simulator Architecture
Each cycle, `run_a_cycle()` does:
1. Decrement LLS scores for all warps (CCWS)
2. Move warps with memory responses from suspended → dispatch queue
3. Push currently running warp back to dispatch queue
4. Call `schedule_warps(policy)` → select next warp
5. Execute one instruction from selected warp's trace buffer
6. If instruction is a memory load/store → `send_mem_req()` → possibly suspend warp

Three queues per core:
- **Dispatch queue** (`c_dispatched_warps`): ready warps
- **Suspended queue** (`c_suspended_warps`): warps waiting on memory responses
- **Running** (`c_running_warp`): currently executing warp (1 at a time)

### Round Robin (RR)
```cpp
bool core_c::schedule_warps_rr() {
    if (!c_dispatched_warps.empty()) {
        c_running_warp = c_dispatched_warps.front();  // take from front
        c_dispatched_warps.erase(c_dispatched_warps.begin());
        return false;  // don't stall
    }
    return true;  // stall — no warps ready
}
```
After execution, warp is pushed to back of queue → pure circular rotation.

**Problem**: Each warp's data gets evicted from L1 by the time it runs again. With 8+ warps, by the time warp 0 runs again all its cache lines are gone. This is **cache thrashing**.

### Greedy-Then-Oldest (GTO)
```cpp
bool core_c::schedule_warps_gto() {
    if (c_dispatched_warps.empty()) {
        prev_scheduled_warp = -1;
        return true;
    }
    // Case 1: previous warp still in dispatch queue → run it again (GREEDY)
    for (auto warp_it = c_dispatched_warps.begin(); ...) {
        if ((*warp_it)->warp_id == prev_scheduled_warp) {
            c_running_warp = *warp_it;
            prev_scheduled_warp = c_running_warp->warp_id;
            return false;
        }
    }
    // Case 2: previous warp stalled → pick warp with smallest timestamp (OLDEST)
    auto oldest = min_element by timestamp;
    c_running_warp = *oldest;
    prev_scheduled_warp = c_running_warp->warp_id;
    return false;
}
```

**Why oldest when stalled?** The oldest warp has been waiting longest → was loaded into L1 earliest → its data is most likely still in cache (fewer other warps have run to evict it). Maximizes chance of L1 hit on warp resumption.

**Why it beats RR on cache hit rate**: GTO gives each warp temporal locality — it runs until it genuinely stalls, reusing its cached data before being evicted. RR switches every cycle regardless.

### CCWS (Cache-Conscious Warp Scheduling)
Throttles warps whose data keeps getting evicted by others.

**VTA (Victim Tag Array)**: per-warp array tracking evicted cache line tags.
- On L1 eviction: evicted tag inserted into the evicted warp's VTA
- On L1 miss: check if missed address is in VTA → VTA hit means "I've been evicted before"

**LLS (Live Liveness Score)**: per-warp score. High = cache-friendly, Low = thrashing victim.
```cpp
// Every cycle: decay all scores
W->ccws_lls_score = max(W->ccws_lls_score - 1, CCWS_LLS_BASE_SCORE);

// On VTA hit: slam score down
llds = max((num_vta_hits * CCWS_LLS_K_THROTTLE * cutoff) / get_insts(), BASE);
c_running_warp->ccws_lls_score = llds;
```

**Scheduling**: each cycle, build schedulable set by sorting warps by LLS descending and taking top warps until cumulative sum exceeds `BASE_SCORE × num_warps`. Run RR only within schedulable set. Low-scoring (thrashing) warps get fewer scheduling opportunities.

### Results (LavaMD benchmark, 8 SMs)

| Policy | Cache Hit Rate | Stall Cycles | Cycles | IPC |
|--------|---------------|-------------|--------|-----|
| RR | 30.49% | 28,827 | 5,561,431 | 7.897 |
| GTO | 79.03% | 71,109 | 5,566,326 | 7.890 |
| CCWS | 30.98% | 29,694 | 5,561,435 | 7.897 |

**Key insight**: GTO achieves 2.6× better cache hit rate but nearly identical total cycles and IPC to RR on LavaMD. Why? LavaMD has enough warps that memory latency is hidden regardless — cache hit rate improved but didn't translate to cycle reduction because the bottleneck was already being hidden. On Crystal_Q12, GTO reduced cycles by 7.4% (512,277 → 474,248).

**Why CCWS ≈ RR here**: CCWS only wins when inter-warp cache thrashing is the actual bottleneck. If the working set fits in L1 naturally or GTO's temporal locality already handles it, CCWS adds overhead without benefit.

### Cross-benchmark Results

| Benchmark | GTO Cache Hit Rate | RR Cache Hit Rate | GTO IPC Gain |
|-----------|-------------------|-------------------|-------------|
| LavaMD_5 | 79.03% | 30.49% | ~0% |
| NN_256K | 40.27% | 38.43% | +10.4% |
| Backprop_8192 | 65.33% | 65.91% | +3.7% |
| Crystal_Q12 | 50.0% | 50.0% | +8.0% |

### 5 Likely Interview Questions

**Q1: Why does GTO improve cache hit rate so dramatically on LavaMD (30% → 79%) but not reduce total execution cycles proportionally?**
GTO's temporal locality means each warp reuses its own cached data before other warps evict it — that's the 2.6× cache hit rate improvement. But on LavaMD, the workload has enough warps that the SM is already hiding memory latency through warp interleaving under RR. Even when RR causes cache misses, the SM switches to another ready warp while the miss is serviced — so those miss cycles are productive cycles for other warps. The total cycle count is nearly the same because the latency hiding is effective regardless of which scheduler is used. GTO would show larger cycle reduction on workloads that are genuinely stall-dominated, where cache misses cause idle cycles that can't be hidden.

**Q2: Walk me through what happens when a warp hits a memory miss in your simulator.**
In `send_mem_req()`, if the L2 cache also misses, the warp is placed in `c_suspended_warps` (keyed by warp_id) and `c_running_warp` is set to NULL. The memory request is dispatched to the RAM model. On subsequent cycles, when the RAM model returns a response, it's placed in `c_memory_responses`. At the start of the next cycle in `run_a_cycle()`, we check the memory response queue — if a suspended warp's ID matches a response, we move it from `c_suspended_warps` back to `c_dispatched_warps`, making it eligible for scheduling again. This models the full latency-hiding mechanism.

**Q3: What is the VTA in CCWS and what problem does it solve?**
The Victim Tag Array is a per-warp data structure that records cache line tags that were evicted from L1 while that warp owned them. When a warp later tries to access the same address and misses in L1, we check its VTA — a VTA hit means "another warp evicted my data and I had to re-fetch it." This is evidence that this warp is a victim of inter-warp cache thrashing. We then lower its LLS score, reducing how frequently the scheduler picks it. This is clever because it's detecting thrashing from the victim's perspective — the warp that keeps losing its data — and gives it fewer scheduling slots so other warps get more time to finish with their data before it's evicted.

**Q4: Why is oldest the right fallback in GTO rather than just taking the front of the queue?**
When GTO's current warp suspends on a memory miss, we need to pick a new warp. The oldest warp (smallest timestamp) has been in the dispatch queue the longest — meaning it was admitted to the SM earliest. Because it's been waiting while other warps ran, its data was loaded into cache earlier and has had less time to be evicted. Round-robin order doesn't give us this — the front of the queue is just whoever happened to be pushed there last. The oldest warp gives us the highest probability that the L1 still holds its working data, preserving the cache-conscious spirit of GTO even during transitions.

**Q5: How does the simulator model warp stall cycles and what do your results show?**
`stall_cycles` is incremented in `run_a_cycle()` when `schedule_warps()` returns true — meaning no warp was available to run (dispatch queue empty, all warps suspended). On LavaMD, RR had 28,827 stall cycles vs GTO's 71,109 — GTO actually had more stalls. This seems counterintuitive, but makes sense: GTO sticks with one warp longer, so when it finally stalls, the other warps may not yet be ready (still in suspended queue). RR distributes execution more evenly, so there's usually a ready warp available. The tradeoff is that RR's ready warps have stale cache — higher stall count for GTO didn't hurt total cycles because its warps hit cache when they do run.

---

## Assignment 4 — Tensor Core & FP16 Mixed Precision

### What Was Built
Extended MacSim (from Assignment 3) to model tensor core instruction execution. Added two components:
1. **Compute Execution Buffer** — tracks in-flight tensor core operations with completion timestamps
2. **Register Dependency Checking** — prevents scheduling warps that need registers still being computed (RAW hazard prevention)

Simulated workloads: gemm, cnn, ffn, gpt2 in both float and half precision variants.

### Key Concepts

**Tensor Cores vs CUDA Cores**
```
CUDA Core:   1 FP32 multiply-add per cycle      → latency = 1 cycle
Tensor Core: 4×4 FP16 matrix multiply per cycle → latency = 32–128 cycles
```
H100 peak: 989 TFLOPS (FP16 Tensor) vs 60 TFLOPS (FP32 CUDA). Same chip, 16× difference.
Tensor cores identified by opcode starting with 'H' (half-precision).

**RAW Hazard (Read After Write)**
```
Cycle 1:   R1 = TensorOp(R2, R3)  ← writing R1, takes 64 cycles
Cycle 2:   R4 = R1 + R5           ← needs R1, but it's not ready!
```
Without dependency checking: reads garbage value from R1 → wrong result. Execution buffer tracks which registers are in-flight and for how long.

**Execution Buffer Entry**
```
{ warp_id, dest_register, completion_cycle }
completion_cycle = current_cycle + get_latency()
get_latency(): opcode[0] == 'H' → tensor_latency (32–128), else → 1
```
Buffer capacity = execution_width. If full → stall warp (can't issue new tensor op).

**Dependency Check**
```cpp
check_dependency(warp_id, next_instruction):
    for each entry in exec_buffer:
        if entry.warp_id == warp_id AND entry.dest_reg ∈ next_inst.src_regs:
            return true  // RAW hazard — don't schedule
    return false  // safe
```

**Scheduling with dependency check (RR variant)**
```
for each warp in dispatch queue:
    if trace_buffer empty OR NOT check_dependency(warp):
        schedule it → return false (no stall)
if all warps blocked: return true (stall)
```

### Execution Width — The Key Parameter
Execution width = number of tensor core operations that can be in-flight simultaneously (buffer slots).

```
Width = 2:  only 2 ops in flight
            → warp 3+ must wait → stall ratio 71% at latency=128

Width = 8:  8 ops in flight simultaneously
            → while Warp 0 waits 128 cycles, Warps 1–7 progress
            → stall ratio drops to 13% (same as no tensor latency)
```

Think of it as ovens in a kitchen: more ovens → chef never waits for one to finish.

### Results

| Metric | Task-1 (scalar, 1-cycle latency) | Task-2 (tensor, 32–128 cycles) |
|--------|----------------------------------|-------------------------------|
| Best Execution Time | 68,713 cycles | 68,713 cycles |
| Worst Execution Time | 69,000 cycles | 93,665 cycles |
| Peak IPC | 3.70 | 3.70 |
| Performance Variation | 0.4% | 36% |
| Stall Ratio (Width=2) | 13% | 71% |
| Stall Ratio (Width=16) | 13% | 13% |

**Width threshold**: Width 2→8 gives 22% improvement. Width 8→16 gives 0.5%. Diminishing returns. Width ≥ 8 captures 95% of peak performance.

**Latency hiding**: At Width≥8, 128-cycle tensor latency causes only 3.5% execution time increase vs 27% at Width=2. Warp interleaving effectively masks even 128-cycle latencies.

### FP16 vs FP32 Simulation Results

| Workload | FP32 Cycles (CC) | FP16 Cycles (CC) | Speedup |
|---------|-----------------|-----------------|---------|
| gemm | 109,220 | 68,512 | 1.59× |
| ffn | 8,256,909 | 1,823,943 | 4.53× |
| gpt2 | 10,002,387 | 5,743,191 | 1.74× |
| cnn | 2,536,533 | 3,380,490 | 0.75× (slower) |

FFN benefits most (4.5×) because it's purely matrix multiplications that map perfectly to tensor cores. GPT-2's 1.74× reflects a mix of tensor-friendly (attention, FFN) and non-tensor operations (layer norm, softmax).

### RR vs GTO for Tensor Cores (opposite conclusion from A3)
- **GTO for memory**: stick with warp → reuse cached data → fewer cache misses (GTO wins)
- **GTO for tensor compute**: stick with warp → warp quickly hits dependency stall → no other warps interleaved → latency not hidden → 15–20% more stalls at narrow widths
- **RR for tensor compute**: rotate through warps → Warp 0 issues TensorOp, blocks → switch to Warp 1, 2, 3... → by time RR returns to Warp 0, op done → perfect latency hiding

**Key learning**: the best scheduling policy depends on the bottleneck. Memory-bound → GTO. Compute-latency-bound → RR.

### Risk of FP16 for Training
FP16 has smaller dynamic range (max ~65,504 vs FP32 ~3.4×10³⁸). Gradients can underflow to zero or overflow to NaN during backprop. Production solution: mixed precision — FP16 forward pass, FP32 gradient accumulation (loss scaling to prevent underflow).

### 5 Likely Interview Questions

**Q1: What is a RAW hazard and how did you prevent it in the tensor core pipeline?**
RAW (Read-After-Write) hazard occurs when an instruction tries to read a register that a prior instruction is still computing. For tensor cores with 32–128 cycle latency, this is critical — if we don't track in-flight operations, a warp could schedule an instruction that reads a register still being computed, getting garbage data. I implemented an execution buffer that stores (warp_id, dest_register, completion_cycle) for every issued tensor op. Before scheduling a warp, `check_dependency()` checks if the warp's next instruction's source registers appear in the buffer for that warp. If there's a match, we skip that warp and try the next one. Only if all warps are blocked do we stall the core.

**Q2: Why does execution width matter so much for tensor core performance?**
Execution width is the number of tensor core operations that can be in-flight simultaneously. At width=2 with 128-cycle latency, only 2 warps can have active tensor ops — once they stall waiting for results, other warps can't issue new ops either (buffer full). Most of the SM sits idle. At width=8, 8 warps can have overlapping tensor ops — while Warp 0 waits 128 cycles, Warps 1–7 are all progressing, hiding the entire latency through interleaving. My results showed: Width=2 gives 71% stall ratio at latency=128, Width=8 drops to 13% — the same as if there were no tensor latency at all. Width 8→16 only gives 0.5% improvement because 8 warps is already enough to fully hide the latency.

**Q3: Why is FP16 4.5× faster than FP32 for FFN layers but only 1.74× for GPT-2?**
FFN (Feed-Forward Network) is almost entirely large matrix multiplications — two linear projections. These map perfectly to tensor cores: H100 Tensor Core hardware is specifically designed for FP16 matmul, giving 989 TFLOPS vs 60 TFLOPS for FP32 CUDA cores. So FFN benefits from the full 16× theoretical advantage (partially realized as 4.5× due to memory bandwidth and overhead). GPT-2 is a full transformer — it includes attention (tensor-friendly), but also layer normalization, softmax, residual additions, and embedding lookups which are element-wise operations that use CUDA cores, not tensor cores. These operations don't benefit from FP16 hardware, diluting the overall speedup to 1.74×.

**Q4: In your simulator, how does `get_latency()` work and why does it check if the opcode starts with 'H'?**
```cpp
int get_latency(opcode) {
    if opcode[0] == 'H': return tensor_latency  // configurable 32–128
    else: return 1
}
```
In NVIDIA's PTX/SASS instruction encoding, half-precision tensor core operations are prefixed with 'H' (e.g., HMMA for half-precision matrix multiply-accumulate). Regular compute operations have latency 1 in our simplified model. The `tensor_latency` parameter is swept from 32–128 cycles in experiments to understand how execution width interacts with latency — this is how we generated the heatmaps showing best performance at (Latency=32–64, Width=8–16).

**Q5: Your results show RR beats GTO for tensor cores but GTO beat RR for memory workloads in A3. Explain the contradiction.**
Both results are actually consistent — they have different bottlenecks. In Assignment 3, the bottleneck was cache thrashing. GTO's greedy approach gives each warp temporal locality — it runs the same warp until memory stall, keeping that warp's data in L1. This reduced cache miss rate from 30% to 79%. In Assignment 4, the bottleneck is compute latency (32–128 cycles for tensor ops). GTO's greedy approach reduces warp interleaving — it keeps running the same warp until it hits a dependency stall, then has no other warps lined up to cover the wait. RR's constant rotation means many warps always have tensor ops in-flight simultaneously, perfectly hiding the latency through overlap. The lesson: GTO wins when you want to exploit temporal locality (memory-bound). RR wins when you want to maximize instruction-level parallelism across warps (compute-latency-bound).

---

## Assignment 5 — SASS / Compiler IR

### What Was Built
A Python static analysis tool that:
1. Parses NVIDIA SASS (Shader ASSembly — actual GPU machine code) from `.sass` files
2. Builds a Control Flow Graph (CFG) with basic blocks
3. Performs type analysis on registers
4. Applies XMAD → IMAD peephole optimization
5. Detects branch divergence via iterative taint analysis

Output: instruction IDs of divergent branches → written to `output_prj5.txt`.

### SASS Basics
SASS is the actual binary instruction set for NVIDIA GPUs (below PTX, which is virtual ISA). Example:
```sass
S2R R0, SR_TID.X          // load thread ID into R0
IMAD R1, R0, 4, RZ        // R1 = R0 * 4 + 0
ISETP.LT.U32 P0, PT, R1, c[0x0][0x164]  // P0 = (R1 < bound)
@P0 BRA target             // branch if P0 — DIVERGENT (depends on thread ID)
```

### Control Flow Graph
- **Basic Block**: maximal sequence of instructions with one entry (no branches into the middle) and one exit (branch/jump at end or fall-through)
- **CFG**: directed graph where nodes = basic blocks, edges = possible control flow transfers (taken branch, fall-through)
- **Predecessors/Successors**: each block tracks `_preds` and `_succs` for traversal

### Taint Analysis (Branch Divergence Detection)

**Goal**: find branch instructions whose outcome depends on thread ID → different threads in a warp take different paths → warp must serialize both paths → 2× performance cost.

**Algorithm**: iterative dataflow analysis (fixed-point).

**Taint sources**:
```python
if inst.opcodes[0] == 'S2R' and inst.operands[1].IsThreadIdx:
    inst_is_tainted = True  # S2R loads threadIdx.x into register
```

**Taint propagation** (3 cases):
```python
# Case 1: instruction's predicate flag is tainted
if inst.pflag is not None and inst.pflag in current_taint:
    inst_is_tainted = True

# Case 2: instruction reads a tainted register
for used_op in inst.GetUses():
    if used_op.IsReg and used_op.Reg in current_taint:
        inst_is_tainted = True

# Case 3: propagate taint to output register
if inst_is_tainted:
    current_taint.add(def_op.Reg)   # ink spreads
else:
    current_taint.discard(def_op.Reg)  # clean write removes taint
```

**Divergence detection**:
```python
if inst_is_tainted and inst.IsBranch():
    branch_divergence_inst_ids.append(inst.id)
```

**Fixed-point iteration** (handles loops):
```python
still_changing = True
while still_changing:
    still_changing = False
    for bb in func.blocks:
        # merge taint from all predecessors
        current_taint = union(taint_at_block_exit[pred] for pred in bb._preds)
        # process instructions...
        if current_taint != taint_at_block_exit[bb]:
            taint_at_block_exit[bb] = current_taint
            still_changing = True  # another pass needed
```

Without iterative approach, a loop where taint from block C flows back to block B (loop back-edge) would not be detected until a second pass. Keep iterating until no block's exit taint changes.

### XMAD → IMAD Optimization

On older architectures (SM52, SM35), no 32-bit integer multiply instruction existed. Compiler emitted 3 XMAD instructions to fake a 32-bit multiply using 16-bit halves:

```sass
# Old (3 instructions) — SM52
XMAD.MRG   T,  A, B.H1, RZ     # multiply low halves, merge high
XMAD       L,  A, B.H1, RZ     # multiply cross terms
XMAD.PSL.CBCC R, A.H1, T, L   # combine with partial sum

# New (1 instruction) — SM75+
IMAD R, A, B, RZ               # 32-bit multiply directly
```

Tool detects the 3-instruction pattern by matching opcodes and operand suffixes (`.H1` = high 16 bits), replaces all 3 with 1 IMAD at the earliest instruction's position. Reduces instruction count → fewer cycles → better IPC.

Two patterns handled:
- **Pattern 1**: XMAD.MRG + XMAD + XMAD.PSL.CBCC (standard 32-bit multiply)
- **Pattern 2**: 4 XMADs + IADD3.RS (wider multiply with accumulation)

### Key Compiler Concepts
- **Def-Use analysis**: `inst.GetDef()` returns register written, `inst.GetUse()` returns registers read. Foundation for dependency analysis and taint tracking
- **Predicate registers (pflag)**: SASS uses predicate registers (P0, P1...) for conditional execution. `@P0 ADD R1, R2, R3` = execute ADD only if P0 is true. Tainted predicate → whole instruction potentially tainted
- **Peephole optimization**: local pattern matching on a small window of instructions, replacing with equivalent but cheaper sequence
- **Dataflow analysis**: propagating facts (here: taint sets) through a CFG until fixed point. Standard compiler technique for dead code elimination, constant propagation, liveness analysis

### 5 Likely Interview Questions

**Q1: What is branch divergence in a GPU context and why is it expensive?**
In a GPU, 32 threads in a warp execute the same instruction simultaneously (SIMT model). Branch divergence occurs when a conditional branch produces different outcomes for different threads — e.g., `if (threadIdx.x < 16)`. When this happens, the GPU must execute both the if-path and the else-path, using a predicate mask to disable the non-participating threads for each path. The cost is serialization: time = time(if-branch) + time(else-branch) instead of max. At worst, with 32 different outcomes you get 32× serialization. Branch divergence on hot loops is one of the biggest GPU performance killers. My tool detects which branches are thread-ID-dependent, allowing developers to restructure code to avoid it.

**Q2: Explain how your taint analysis handles loops — what goes wrong without fixed-point iteration?**
Consider a loop: Block A → Block B (loop body) → Block C (back-edge to B). Without fixed-point iteration, a single forward pass processes B before C — so B doesn't see taint coming from C via the back-edge. If C taints a register that feeds a branch in B, we'd miss that divergence. Fixed-point iteration fixes this: after the first pass, if C's exit taint changed, we mark `still_changing = True` and run another pass. On the second pass, B now sees C's taint in `taint_at_block_exit[C]`, propagates it through B, potentially tainting the branch. We keep iterating until no block's exit taint changes — this is the fixed point, guaranteed to converge since the taint set can only grow (we never remove taint except for clean writes) and is bounded by the number of registers.

**Q3: What is the XMAD → IMAD transformation and why does it matter?**
On NVIDIA architectures before SM75 (Turing), there was no native 32-bit integer multiply instruction. The CUDA compiler decomposed each 32-bit multiply into 3 XMAD instructions operating on 16-bit halves: one for the low-half product, one for cross-terms, one to combine. This means every integer multiply in the original C++ code became 3 instructions in assembly — 3× the instruction count, 3× the scheduling pressure, 3× the decode overhead. My tool detects this 3-instruction pattern by matching opcodes (XMAD.MRG + XMAD + XMAD.PSL.CBCC) and operand suffixes (.H1 for high 16 bits), then replaces all 3 with a single IMAD instruction available on SM75+. This reduces instruction count and improves IPC without changing semantics.

**Q4: How does predicate-based execution in SASS relate to branch divergence?**
SASS uses predicate registers (P0, P1, etc.) for conditional execution instead of jump instructions where possible. `@P0 ADD R1, R2, R3` means "execute ADD only if P0 is true." When a comparison produces P0 based on thread ID (e.g., `ISETP.LT R0, 16 → P0`), P0 is tainted. Any instruction guarded by tainted P0 is also tainted — my Case 2 taint rule handles this. When a `BRA` instruction uses a tainted predicate, it's a divergence point. This predicate model is more fine-grained than C-level branches — multiple predicated instructions can share the same predicate, and my analysis correctly propagates taint through all of them.

**Q5: Why is taint analysis a conservative (over-approximate) analysis?**
Taint analysis may flag branches as divergent that are actually safe — false positives. Example: `R0 = threadIdx.x; R1 = R0 - R0; BRA R1 > 0` — R1 is always 0 regardless of thread ID, so the branch never diverges. But taint analysis marks R0 as tainted, propagates to R1 (R1 uses R0), and flags the BRA. True divergence detection would require knowing runtime values (undecidable in general). Taint analysis is conservative by design — it's better to over-report divergence (miss an optimization opportunity) than under-report (miss a real performance bug). Also, `current_taint.discard(output_reg)` when a non-tainted instruction writes to a register is the one case where we un-taint — this is sound because a clean computation produces a clean output regardless of the register's previous value.

---

## DL Training on GPUs
*(to be filled)*

---

## Distributed Training
*(to be filled)*

---

## Questions & Answers
*(to be filled)*
