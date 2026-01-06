#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__
void matVecMul(float* B, float* C, float* A, int width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < width) {
        float sum = 0;
        for (int col = 0; col < width; col++) {
            sum += B[row*width + col] * C[col];
        }
        A[row] = sum;
    }
}

void matVecMulCPU(float* B, float* C, float* A, int width) {
    for (int row = 0; row < width; row++) {
        float sum = 0;
        for (int col = 0; col < width; col++) {
            sum += B[row * width + col] * C[col];
        }
        A[row] = sum;
    }
}

bool compareVectors(float* A, float* B, int width, float epsilon = 1e-5) {
    for (int i = 0; i < width; i++) {
        if (fabs(A[i] - B[i]) > epsilon) {
            printf("Mismatch at index %d: CPU=%.6f, GPU=%.6f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

void printVector(float* vec, int width) {
    printf("[");
    for (int i = 0; i < width; i++) {
        printf("%.2f", vec[i]);
        if (i < width - 1) printf(", ");
    }
    printf("]\n");
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

int main() {
    int matrix_width = 8;
    
    int matrix_size = matrix_width * matrix_width * sizeof(float);
    int vector_size = matrix_width * sizeof(float);
    
    float* h_B = (float*)malloc(matrix_size);
    float* h_C = (float*)malloc(vector_size);
    float* h_A = (float*)malloc(vector_size);
    
    for (int i = 0; i < matrix_width * matrix_width; i++) {
        h_B[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < matrix_width; i++) {
        h_C[i] = rand() / (float)RAND_MAX;
    }
    
    float *B_d, *C_d, *A_d;
    cudaMalloc(&B_d, matrix_size);
    cudaMalloc(&C_d, vector_size);
    cudaMalloc(&A_d, vector_size);
    
    cudaMemcpy(B_d, h_B, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, h_C, vector_size, cudaMemcpyHostToDevice);
    
    matVecMul<<<matrix_width, 1>>>(B_d, C_d, A_d, matrix_width);
    
    cudaMemcpy(h_A, A_d, vector_size, cudaMemcpyDeviceToHost);
    
    float* h_A_cpu = (float*)malloc(vector_size);
    matVecMulCPU(h_B, h_C, h_A_cpu, matrix_width);
    
    if (compareVectors(h_A, h_A_cpu, matrix_width)) {
        printf("Success! GPU and CPU results match.\n\n");
    } else {
        printf("Error! GPU and CPU results differ.\n\n");
    }
    
    printf("Matrix B:\n");
    printMatrix(h_B, matrix_width);
    printf("Vector C:\n");
    printVector(h_C, matrix_width);
    printf("Result A:\n");
    printVector(h_A, matrix_width);
    
    cudaFree(B_d);
    cudaFree(C_d);
    cudaFree(A_d);
    free(h_B);
    free(h_C);
    free(h_A);
    free(h_A_cpu);
    
    return 0;
}
