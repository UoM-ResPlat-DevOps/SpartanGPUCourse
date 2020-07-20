#include <iostream>
#include <stdio.h>

#define checkCudaError(status) { \
	if(status != cudaSuccess) { \
		std::cout << "CUDA Error " << __FILE__ << ", " << __LINE__ \
			<< ": " << cudaGetErrorString(status) << "\n"; \
		exit(-1); \
	} \
}

__global__ void vecAdd(int * a, int * b, int * c, int size) {

	//ADD CODE HERE
	int i = threadIdx.x;
	int j = blockIdx.x*blockDim.x;
	printf("I am in: %d, %d\n", i , j);
	c[i + j] = a[i + j] + b[i + j];
}

int main() {

	//checkCudaError(cudaSetDevice(1));
	int device;
	checkCudaError(cudaGetDevice(&device));
	cudaDeviceProp prop;
	checkCudaError(cudaGetDeviceProperties(&prop, device));
	std::cout << "Device " << device << ": " << prop.name << "\n";
	std::cout << "GPU Cores: " << prop.multiProcessorCount << "\n";
	std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";

	const int GRID_SIZE = 16;
	const int CTA_SIZE = 128;
	const int size = GRID_SIZE * CTA_SIZE;
	int * a, * b, * c;
	int * dev_a, * dev_b, * dev_c;

	a = (int *) malloc (sizeof(int) * size);
	b = (int *) malloc (sizeof(int) * size);
	c = (int *) malloc (sizeof(int) * size);
	if(!a || !b || !c) {
		std::cout << "Error: out of memory\n";
		exit(-1);
	}

	for(int i = 0; i < size; i++) {
		a[i] = i;
		b[i] = i+1;
	}
	memset(c, 0, sizeof(int) * size);

	checkCudaError(cudaMalloc(&dev_a, sizeof(int) * size));
	checkCudaError(cudaMalloc(&dev_b, sizeof(int) * size));	
	checkCudaError(cudaMalloc(&dev_c, sizeof(int) * size));	
	
	checkCudaError(cudaMemcpy(dev_a, a, sizeof(int) * size, cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(dev_b, b, sizeof(int) * size, cudaMemcpyHostToDevice));
	checkCudaError(cudaMemset(dev_c, 0, sizeof(int) * size));

	vecAdd<<<GRID_SIZE, CTA_SIZE>>>(dev_a, dev_b, dev_c, size);

	checkCudaError(cudaDeviceSynchronize());
	checkCudaError(cudaMemcpy(c, dev_c, sizeof(int) * size, cudaMemcpyDeviceToHost));

	for(int i = 0; i < size; i++) {
//		std::cout << i << ": " << c[i] << "\n";
		if(c[i] != i*2+1) {
			std::cout << "Error: c[" << i << "] != " <<
				i*2+1 << "\n";
			exit(-1);
		}
	}
	std::cout << "Pass\n";
}