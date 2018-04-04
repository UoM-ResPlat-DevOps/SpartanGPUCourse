/* kernel routine starts with keyword __global__ */

__global__ void vecadd(float* A, float* B, float* C)
{
  int  i = threadIdx.x;  // threadIdx is a CUDA built-in variable 
  C[i] = A[i] + B[i];
}

int main(int argc, char * argv[])
{
  float *host_A, *host_B, *host_C;
  float *dev_A, *dev_B, *dev_C;
  int n;

  if (argc == 1) n = 1024;
  else n = atoi(argv[1]);
 
  /* 1. allocate host memory */
  host_A = (float*)malloc( n*sizeof(float) );
  host_B = (float*)malloc( n*sizeof(float) );
  host_C = (float*)malloc( n*sizeof(float) );

  /* 2. allocate GPU memory */
  cudaMalloc( &dev_A, n*sizeof(float) );
  cudaMalloc( &dev_B, n*sizeof(float) ); 
  cudaMalloc( &dev_C, n*sizeof(float) ); 

  /* initialize array A and B */
  for(  int i = 0; i < n; ++i ) {
    host_A[i] = (float) i;
    host_B[i] = (float) i;
  }

  /* 3. Copydata (host_A and host_B) to GPU */
  cudaMemcpy( dev_A, host_A, n*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( dev_B, host_B, n*sizeof(float), cudaMemcpyHostToDevice );

  /* 4. call kernel routine to execute on GPU */
  /* launch 1 thread per vector-element, 1024 threads per block */
  vecadd<<<1,n>>>( dev_A, dev_B, dev_C );

  /* transfer results from GPU (dev_C) to CPU (host_C) */
  cudaMemcpy( host_C, dev_C, n*sizeof(float), cudaMemcpyDeviceToHost );
 
  /* free host and GPU memory */
  free(host_A);  
  free(host_B);
  free(host_C);
  cudaFree(dev_A);
  cudaFree(dev_B);
  cudaFree(dev_C);
 
  return( 0 );
}
