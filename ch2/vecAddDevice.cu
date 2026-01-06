#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaError_t err = cudaMalloc((void**)&A_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void**)&B_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void**)&C_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // call it
    int threadsPerBlock = 256;
    int numberOfBlocks = ceil(n / 256.0);
    vecAddKernel<<<numberOfBlocks, threadsPerBlock>>>(A_h, B_h, C_h, n);

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);


    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {

}