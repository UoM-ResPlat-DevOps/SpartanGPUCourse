#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)

{
	if (result != cudaSuccess) {
	fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
	assert(result == cudaSuccess);
	}
	return result;
}

int main()
{

/*
* The macro can be wrapped around any function returning
* a value of type `cudaError_t`.
*/

	checkCuda( cudaDeviceSynchronize() )
}
