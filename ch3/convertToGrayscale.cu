
__global__
void colorToGrayscaleConversion(unsigned char* Pout, unsigned char* Pin, int width, int height) {
    int x_dimension = blockDim.x * blockIdx.x + threadIdx.x;
    int y_dimension = blockDim.y * blockIdx.y + threadIdx.y;

    if (x_dimension < width && y_dimension < height) {
        int i = y_dimension * width + x_dimension;
        int color_i = i * 3; // 3 channels
        unsigned char r = Pin[color_i];
        unsigned char g = Pin[color_i+1];
        unsigned char b = Pin[color_i+2];

        Pout[i] = 0.21f*r + 0.71f*g + 0.07f*b;

    }


}


int main() {
    int width = 100;
    int height = 50;
    int byte_size =  width * height;
    int color_byte_size = byte_size*3;
    unsigned char* Pin_h = malloc(color_byte_size);

    // fill Pin_h with picture data

    unsigned char *Pin_d, *Pout_d;

    cudaMalloc((void**) &Pin_d, color_byte_size);
    cudaMemcpy((void*) Pin_d, (void*) Pin_h, color_byte_size, cudaMemcpyHostToDevice);

    cudaMalloc((void**) &Pout_d, byte_size);


    dim3 blocksPerGrid(ceil(width/16.0), ceil(height/16.0), 1);
    dim3 threadsPerBlock(16, 16, 1);
    colorToGrayscaleConversion<<<blocksPerGrid, threadsPerBlock>>>>(Pout_d, Pin_d, width, height);

}