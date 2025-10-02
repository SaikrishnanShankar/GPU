#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>
// ==== DO NOT MODIFY CODE ABOVE THIS LINE ====

#define DTYPE int
// Add any additional #include headers or helper macros needed
#define WARP_SIZE 32

// Helper function to find the next power of 2
int nextPowerOf2(int n) {
    if (n <= 0) return 1;
    if ((n & (n - 1)) == 0) return n; // already a power of 2
    int power = 1;
    while (power < n) {
        power <<= 1;
    }
    return power;
}

// Implement your GPU device kernel(s) here (e.g., the bitonic sort kernel).

__global__ void bitonicSortKernel(DTYPE* d_arr, int size, int i, int j) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx >= size) return;

    int dist = 1 << j;
    int b = idx ^ dist;

    if(b >= size || idx >= b) return;

    DTYPE val1 = d_arr[idx];
    DTYPE val2 = d_arr[b];

    int pow2i = 1 << i;
    bool ascending = (idx & pow2i) == 0;

    DTYPE minVal = (val1 < val2) ? val1 : val2;
    DTYPE maxVal = (val1 < val2) ? val2 : val1;

    d_arr[idx] = ascending ? minVal : maxVal;
    d_arr[b] = ascending ? maxVal : minVal;
}

__global__ void bitonicSortShared(DTYPE* d_arr, int size, int stage, int maxJ) {
    extern __shared__ DTYPE shared_data[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int blockSize = blockDim.x;
    int globalIdx = bid * blockSize + tid;

    // Load data into shared memory
    if(globalIdx < size) {
        shared_data[tid] = d_arr[globalIdx];
    } else {
        shared_data[tid] = INT_MAX;
    }
    __syncthreads();

    // Process steps from maxJ down to 0 within shared memory
    for (int j = maxJ; j >= 0; j--) {
        int dist = 1 << j;
        int partner = tid ^ dist;
        
        if (partner < blockSize) {
            int pow2i = 1 << stage;
            bool ascending = (globalIdx & pow2i) == 0;

            if (tid < partner) { // Only lower-index thread does the swap
                DTYPE val1 = shared_data[tid];
                DTYPE val2 = shared_data[partner];

                DTYPE minVal = (val1 < val2) ? val1 : val2;
                DTYPE maxVal = (val1 < val2) ? val2 : val1;

                shared_data[tid] = ascending ? minVal : maxVal;
                shared_data[partner] = ascending ? maxVal : minVal;
            }
        }
        __syncthreads();
    }

    // Write back to global memory
    if (globalIdx < size) {
        d_arr[globalIdx] = shared_data[tid];
    }
}


// Bitonic sort function to be called from main
void bitonicSort(DTYPE* d_arr, int size) {
    int logN = (int)log2(size);
    int blockSize = 512;
    int numBlocks = (size + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(DTYPE);

    for (int i = 1; i <= logN; i++) {
        // Check if we can process some steps in shared memory
        int maxSharedStep = -1;
        for (int j = i - 1; j >= 0; j--) {
            int dist = 1 << j;
            if (dist < blockSize) {
                maxSharedStep = j;
                break;
            }
        }

        if (maxSharedStep >= 0) {
            // First, do the global memory steps
            for (int j = i - 1; j > maxSharedStep; j--) {
                bitonicSortKernel<<<numBlocks, blockSize>>>(d_arr, size, i, j);
                cudaDeviceSynchronize();
            }

            // Then do all remaining steps in shared memory at once
            if (maxSharedStep >= 0) {
                bitonicSortShared<<<numBlocks, blockSize, sharedMemSize>>>(
                    d_arr, size, i, maxSharedStep);
                cudaDeviceSynchronize();
            }
        } else {
            // All steps need global memory (fallback to original method)
            for (int j = i - 1; j >= 0; j--) {
                bitonicSortKernel<<<numBlocks, blockSize>>>(d_arr, size, i, j);
                cudaDeviceSynchronize();
            }
        }
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

// arrCpu contains the input random array
// arrSortedGpu should contain the sorted array copied from GPU to CPU
int *arrSortedGpu = (DTYPE*)malloc(size * sizeof(DTYPE));
// If size is not a power of 2, pad the array to the next power of 2
int newSize = nextPowerOf2(size);
DTYPE *arrCpuPadded = (DTYPE*)malloc(newSize * sizeof(DTYPE));
// Copy original data
memcpy(arrCpuPadded, arrCpu, size * sizeof(DTYPE));
// Pad the rest with max int value
for (int i = size; i < newSize; i++) {
    arrCpuPadded[i] = INT_MAX;
}

// Transfer data (arrCpu) to device
DTYPE *d_arr;
cudaMalloc((void**)&d_arr, newSize * sizeof(DTYPE));
cudaStream_t memStream;
cudaStreamCreate(&memStream);

// Asynchronous host-to-device copy
// cudaMemcpyAsync(d_arr, arrCpuPadded, newSize * sizeof(DTYPE), 
//                 cudaMemcpyHostToDevice, memStream);

// free(arrCpuPadded);
cudaMemcpy(d_arr, arrCpuPadded, newSize * sizeof(DTYPE), cudaMemcpyHostToDevice);
free(arrCpuPadded);
/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2dTime, start, stop);

    cudaEventRecord(start);

/* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

// cudaStreamSynchronize(memStream);
// Perform bitonic sort on GPU
bitonicSort(d_arr, newSize);

/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaEventRecord(start);

/* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

// Transfer sorted data back to host (copied to arrSortedGpu)
// cudaMemcpyAsync(arrSortedGpu, d_arr, size * sizeof(DTYPE), 
//                 cudaMemcpyDeviceToHost, memStream);
// cudaStreamSynchronize(memStream);
// cudaStreamDestroy(memStream);
// cudaFree(d_arr);
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
        printf("\033[1;31mFUNCTIONAL FAIL\n\033[0m");
        return 0;
    }
    
    printf("\033[1;34mArray size         :\033[0m %d\n", size);
    printf("\033[1;34mCPU Sort Time (ms) :\033[0m %f\n", cpuTime);
    float gpuTotalTime = h2dTime + gpuTime + d2hTime;
    int speedup = (gpuTotalTime > cpuTime) ? (gpuTotalTime / cpuTime) : (cpuTime / gpuTotalTime);
    float meps = size / (gpuTotalTime * 0.001) / 1e6;
    printf("\033[1;34mGPU Sort Time (ms) :\033[0m %f\n", gpuTotalTime);
    printf("\033[1;34mGPU Sort Speed     :\033[0m %f million elements per second\n", meps);
    if (gpuTotalTime < cpuTime) {
        printf("\033[1;32mPERF PASSING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;32m %dx \033[1;34mfaster than CPU !!!\033[0m\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
    } else {
        printf("\033[1;31mPERF FAILING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;31m%dx \033[1;34mslower than CPU, optimize further!\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
        return 0;
    }

    return 0;
}


