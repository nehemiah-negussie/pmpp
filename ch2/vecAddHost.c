#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    for (int i = 0; i < n; i++) {
        C_h[i] = A_h[i] + B_h[i];
    }
}

int main() {
    int n = 10;
    srand(time(NULL));
    float* A_h = malloc(n*sizeof(float));
    float* B_h = malloc(n*sizeof(float));
    float* C_h = malloc(n*sizeof(float));

    for (int i = 0; i < n; i++){
        A_h[i] = rand() / 1000000;
        B_h[i] = rand() / 1000000;
    }

    vecAdd(A_h, B_h, C_h, n);

    for (int i = 0; i < n; i++){
        printf("A[%d] = %.2f\n", i, A_h[i]);
        printf("B[%d] = %.2f\n", i, B_h[i]);
        printf("C[%d] = %.2f\n", i, C_h[i]);
    }

    return 0;
}