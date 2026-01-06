#include <stdio.h>

// Square matrices only
__global__
void matmulKernel(float* M, float* N, float* P, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < width) {
        float P_value = 0;

        for (int i = 0; i < width; i++) {
            P_value += M[row*width + i] * N[i*width + col];
        }
        P[row*width + col] = P_value;
    }

}

void printMatrix(float* matrix, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            printf("%.2f ", matrix[row * width + col]);
        }
        printf("\n");
    }
    printf("\n");
}

void matmulCPU(float* M, float* N, float* P, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float value = 0;
            for (int i = 0; i < width; i++) {
                value += M[row * width + i] * N[i * width + col];
            }
            P[row * width + col] = value;
        }
    }
}

bool compareMatrices(float* A, float* B, int width, float epsilon = 1e-5) {
    for (int i = 0; i < width * width; i++) {
        if (fabs(A[i] - B[i]) > epsilon) {
            printf("Mismatch at index %d: CPU=%.6f, GPU=%.6f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}


int main() {
    int BLOCK_WIDTH = 2;
    int matrix_width = 8;

    int size = matrix_width * matrix_width * sizeof(float);

    float* h_M = (float*)malloc(size);
    float* h_N = (float*)malloc(size);
    float* h_P = (float*)malloc(size);

    for (int i = 0; i < matrix_width * matrix_width; i++) {
        h_M[i] = rand() / (float)RAND_MAX;
        h_N[i] = rand() / (float)RAND_MAX;
    }

    float *M, *N, *P;
    cudaMalloc(&M, size);
    cudaMalloc(&N, size);
    cudaMalloc(&P, size);

    cudaMemcpy(M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N, h_N, size, cudaMemcpyHostToDevice);

    dim3 blocksPerGrid(ceil(matrix_width/BLOCK_WIDTH), ceil(matrix_width/BLOCK_WIDTH));
    dim3 threadsPerBlock(BLOCK_WIDTH, BLOCK_WIDTH);

    matmulKernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, P, matrix_width);
    cudaMemcpy(h_P, P, size, cudaMemcpyDeviceToHost);

    float* h_P_cpu = (float*)malloc(size);
    matmulCPU(h_M, h_N, h_P_cpu, matrix_width);
    if (compareMatrices(h_P, h_P_cpu, matrix_width)) {
        printf("Success! GPU and CPU results match.\n");
    } else {
        printf("Error! GPU and CPU results differ.\n");
    }

    printMatrix(h_M, matrix_width);
    printMatrix(h_N, matrix_width);
    printMatrix(h_P, matrix_width);
    cudaFree(M);
    cudaFree(N);
    cudaFree(P);
    free(h_M);
    free(h_N);
    free(h_P);
    free(h_P_cpu);
    return 0;
}