
// header file

extern "C" {
    void initData(float *data, int dataSize);
    void computeGPU(float *hostData, int blockSize, int gridSize);
    float sum(float *data, int size);
}

