#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdint.h>
// ==== DO NOT MODIFY CODE ABOVE THIS LINE ====

#define DTYPE uint16_t

// Add any additional #include headers or helper macros needed
#define LOG_MAX_SHARED 16      //  hyperparameter : we can handle in shared memory (2^13 = 8192 elements) - 2 ^ 14 might be outside H100 GPU's limit 48KB
#define THREADS_PER_BLOCK 1024          // Number of threads per block

// Implement your GPU device kernel(s) here (e.g., the bitonic sort kernel).

// Function to compute the next power of two greater than or equal to size
int nextPowerOf2(int size) {
    //find power using bitwise operations
    int power = 1;
    while (power < size) {
        power <<= 1;
    }
    return power;
}
// GLOBAL MEMORY KERNEL - For large merge steps that don't fit in shared memory
__global__ void bitonic_merge_global(DTYPE *data, int j, int k) {
    // Each thread handles one comparison pair
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;

    //I am finding the two indices to compare: if tid = 5, j = 0, then idx1 = 4, idx2 = 5. This ensures we are always comparing valid pairs. 
    int idx1 = thread_id + (thread_id & ~((1 << j) - 1));
    int idx2 = idx1 + (1 << j);
    
    // Load the two elements to compare
    DTYPE elem1 = data[idx1];
    DTYPE elem2 = data[idx2];
    
    bool ascending = (idx1 & (1 << k)) == 0;
    bool should_swap = (ascending && elem1 > elem2) || (!ascending && elem1 < elem2);
    if(should_swap) {
        DTYPE temp = elem1;
        elem1 = elem2;
        elem2 = temp;
    }
    // Apply the swap
    data[idx1] = elem1;
    data[idx2] = elem2;
}

// SHARED MEMORY MERGE FUNCTION - Performs merge steps within shared memory
__device__ __inline__ void perform_shared_bitonic_merge(DTYPE *shared_data, int k, int j_start, int elements_per_thread) {
    // Perform merge steps from j_start down to 0
    #pragma unroll
    for (int j = j_start; j >= 0; j--) {
        // Each thread processes multiple elements for better memory utilization
        #pragma unroll
        for (int elem_offset = 0; elem_offset < elements_per_thread; elem_offset++) {
            // Calculate global position for this element
            int global_pos = (elements_per_thread * THREADS_PER_BLOCK * blockIdx.x + 
                            threadIdx.x + THREADS_PER_BLOCK * elem_offset);
            
            // Apply the same indexing transformation as global kernel
            int base_idx = global_pos + (global_pos & ~((1 << j) - 1));
            
            // Convert to local shared memory indices
            int local_idx1 = base_idx - 2 * elements_per_thread * THREADS_PER_BLOCK * blockIdx.x;
            int local_idx2 = local_idx1 + (1 << j);
            
            DTYPE elem1 = shared_data[local_idx1];
            DTYPE elem2 = shared_data[local_idx2];
            
            bool ascending = (base_idx & (1 << k)) == 0;
            bool should_swap = (ascending && elem1 > elem2) || (!ascending && elem1 < elem2);
            if(should_swap) {
                DTYPE temp = elem1;
                elem1 = elem2;
                elem2 = temp;
            }
            // Write back to shared memory
            shared_data[local_idx1] = elem1;
            shared_data[local_idx2] = elem2;
        }
        __syncthreads();
    }
}

// SHARED MEMORY KERNEL - Initial sort for small sequences that fit inside shared memory
__global__ void bitonic_sort_shared_initial(DTYPE *data, int elements_per_thread, int total_size) {
    extern __shared__ DTYPE shared_mem[];
    
    // Load elements into shared memory with boundary checks
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        // Global index calculation - threads load 2 elements each with THREADS_PER_BLOCK stride
        int base_idx = threadIdx.x + i * 2 * THREADS_PER_BLOCK + 2 * elements_per_thread * THREADS_PER_BLOCK * blockIdx.x;
        
        if (base_idx >= total_size) {
            // Out of bounds - pad both positions
            shared_mem[threadIdx.x + i * 2 * THREADS_PER_BLOCK] = 65535;
            shared_mem[threadIdx.x + i * 2 * THREADS_PER_BLOCK + THREADS_PER_BLOCK] = 65535;
        } else if (base_idx + THREADS_PER_BLOCK >= total_size) {
            // First element valid, second out of bounds
            shared_mem[threadIdx.x + i * 2 * THREADS_PER_BLOCK] = data[base_idx];
            shared_mem[threadIdx.x + i * 2 * THREADS_PER_BLOCK + THREADS_PER_BLOCK] = 65535;
        } else {
            // Both elements within bounds
            shared_mem[threadIdx.x + i * 2 * THREADS_PER_BLOCK] = data[base_idx];
            shared_mem[threadIdx.x + i * 2 * THREADS_PER_BLOCK + THREADS_PER_BLOCK] = data[base_idx + THREADS_PER_BLOCK];
        }
    }
    __syncthreads();
    
    // Perform bitonic merge operations - limit based on shared memory capacity
    for (int k = 1; k <= min(LOG_MAX_SHARED, 32 - __builtin_clz((unsigned int)total_size)); k++) {
        perform_shared_bitonic_merge(shared_mem, k, k - 1, elements_per_thread);
    }
    
    // Write sorted data back to global memory
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        int base_idx = threadIdx.x + i * 2 * THREADS_PER_BLOCK + 2 * elements_per_thread * THREADS_PER_BLOCK * blockIdx.x;
        
        data[base_idx] = shared_mem[threadIdx.x + i * 2 * THREADS_PER_BLOCK];
        data[base_idx + THREADS_PER_BLOCK] = shared_mem[threadIdx.x + i * 2 * THREADS_PER_BLOCK + THREADS_PER_BLOCK];
    }
}


// SHARED MEMORY KERNEL - Final merge steps after global operations
__global__ void bitonic_sort_shared_final(DTYPE *data, int elements_per_thread, 
                                         int total_size, int k) {
    extern __shared__ DTYPE shared_mem[];
    
    // Load data - same pattern as initial kernel
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        int base_idx = threadIdx.x + i * 2 * THREADS_PER_BLOCK + 2 * elements_per_thread * THREADS_PER_BLOCK * blockIdx.x;
        shared_mem[threadIdx.x + i * 2 * THREADS_PER_BLOCK] = data[base_idx];
        shared_mem[threadIdx.x + i * 2 * THREADS_PER_BLOCK + THREADS_PER_BLOCK] = data[base_idx + THREADS_PER_BLOCK];
    }
    __syncthreads();
    
    // Only perform the remaining merge steps that fit in shared memory
    perform_shared_bitonic_merge(shared_mem, k, LOG_MAX_SHARED - 1, elements_per_thread);
    
    // Write results back
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        int base_idx = threadIdx.x + i * 2 * THREADS_PER_BLOCK + 2 * elements_per_thread * THREADS_PER_BLOCK * blockIdx.x;
        data[base_idx] = shared_mem[threadIdx.x + i * 2 * THREADS_PER_BLOCK];
        data[base_idx + THREADS_PER_BLOCK] = shared_mem[threadIdx.x + i * 2 * THREADS_PER_BLOCK + THREADS_PER_BLOCK];
    }
}

//Perform Bitonic Sort on GPU
void bitonic_sort(int padded_size, int size, DTYPE *d_arr, int log_padded_size) {
    if (padded_size <= 1) return;
    
    int threads_shared = min(THREADS_PER_BLOCK, min(1 << (LOG_MAX_SHARED - 1), padded_size)); // Number of threads per block for shared memory kernels
    int blocks_shared = max((padded_size >> LOG_MAX_SHARED), 1); // Each block sorts 2^LOG_MAX_SHARED elements
    int elements_per_thread = max((1 << (LOG_MAX_SHARED - 1)) / THREADS_PER_BLOCK, 1); //Number of strides, i.e total elements accessed per thread

    // Calculate number of blocks and threads for global memory merges
    int threads_global = min(THREADS_PER_BLOCK, padded_size >> 1);
    if (threads_global <= 0) threads_global = 1;
    int blocks_global = max(((padded_size / threads_global) >> 1), 1);

    // Phase 1: Initial shared memory sort when sequence size <= 2^LOG_MAX_SHARED
    bitonic_sort_shared_initial<<<blocks_shared, threads_shared, (1 << LOG_MAX_SHARED) * sizeof(DTYPE)>>>
                                (d_arr, elements_per_thread, size);

    // Phase 2: Handle larger sequences with hybrid approach
    for (int k = LOG_MAX_SHARED + 1; k <= log_padded_size; k++) {
        // Global memory merge steps
        for (int j = k - 1; j >= LOG_MAX_SHARED; j--) {
            bitonic_merge_global<<<blocks_global, threads_global>>>(d_arr, j, k);
        }
        // Final shared memory optimization
        bitonic_sort_shared_final<<<blocks_shared, threads_shared, (1 << LOG_MAX_SHARED) * sizeof(DTYPE)>>>
                                (d_arr, elements_per_thread, size, k);
    }
}

/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);

    srand(time(NULL));

    DTYPE* arrCpu = (DTYPE*)malloc(size * sizeof(DTYPE));

    for (int i = 0; i < size; i++) {
        arrCpu[i] = rand() % 1000;
    }

    float gpuTime, h2dTime, d2hTime, cpuTime = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
/* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

// arCpu contains the input random array
// arrSortedGpu should contain the sorted array copied from GPU to CPU
DTYPE *arrSortedGpu = (DTYPE*)malloc(size * sizeof(DTYPE));
//Pinning memory for faster access during H2D transfer
cudaHostRegister(arrCpu, size * sizeof(DTYPE), cudaHostRegisterDefault);

// Calculate padded size - must be power of 2
int padded_size = nextPowerOf2(size);
int log_padded_size = (padded_size <= 1) ? 0 : (32 - __builtin_clz((unsigned int)(padded_size - 1)));
// Transfer data (arr_cpu) to device 
DTYPE *d_arr;
cudaMalloc(&d_arr, padded_size * sizeof(DTYPE));
cudaMemset(d_arr, 0, padded_size * sizeof(DTYPE));
cudaMemcpy(d_arr, arrCpu, size * sizeof(DTYPE), cudaMemcpyHostToDevice);
/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2dTime, start, stop);

    cudaEventRecord(start);
    
/* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

// Perform bitonic sort on GPU
bitonic_sort(padded_size, size, d_arr, log_padded_size);

//Pinning memory for faster access during D2H transfer
cudaHostRegister(arrSortedGpu, size * sizeof(DTYPE), cudaHostRegisterDefault);
/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaEventRecord(start);

/* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

// Transfer sorted data back to host (copied to arrSortedGpu)
cudaMemcpy(arrSortedGpu, d_arr, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
cudaFree(d_arr);
/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2hTime, start, stop);

    auto startTime = std::chrono::high_resolution_clock::now();
    
    // CPU sort for performance comparison
    std::sort(arrCpu, arrCpu + size);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    cpuTime = cpuTime / 1000;

    int match = 1;
    for (int i = 0; i < size; i++) {
        if (arrSortedGpu[i] != arrCpu[i]) {
            match = 0;
            break;
        }
    }

    free(arrCpu);
    free(arrSortedGpu);

    if (match)
        printf("\033[1;32mFUNCTIONAL SUCCESS\n\033[0m");
    else {
        printf("\033[1;31mFUNCTIONCAL FAIL\n\033[0m");
        return 0;
    }
    
    printf("\033[1;34mArray size         :\033[0m %d\n", size);
    printf("\033[1;34mCPU Sort Time (ms) :\033[0m %f\n", cpuTime);
    float gpuTotalTime = h2dTime + gpuTime + d2hTime;
    float speedup = (gpuTotalTime > cpuTime) ? (gpuTotalTime/cpuTime) : (cpuTime/gpuTotalTime);
    float meps = size / (gpuTotalTime * 0.001) / 1e6;
    printf("\033[1;34mGPU Sort Time (ms) :\033[0m %f\n", gpuTotalTime);
    printf("\033[1;34mGPU Sort Speed     :\033[0m %f million elements per second\n", meps);
    if (gpuTotalTime < cpuTime) {
        printf("\033[1;32mPERF PASSING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;32m %.2fx \033[1;34mfaster than CPU !!!\033[0m\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
    } else {
        printf("\033[1;31mPERF FAILING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;31m%.2fx \033[1;34mslower than CPU, optimize further!\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
        return 0;
    }

    return 0;
}