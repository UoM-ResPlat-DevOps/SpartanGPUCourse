# Structure of CUDA Code

As with all parallel programming, start with serial code, engage in decomposition, then generate parallel code.

General CPU/GPU code with CUDA will look like:

void CPUFunction()
{
	printf("This function is defined to run on the CPU.\n");
}
__global__ void GPUFunction()
{
	printf("This function is defined to run on the GPU.\n");
}
int main()
{
	CPUFunction();
	GPUFunction<<<1, 1>>>();
	cudaDeviceSynchronize();
}


The __global__ keyword indicates that the following function will run on the GPU, and can be invoked globally, which in this context means either by
the CPU, or, by the GPU.

Often, code executed on the CPU is referred to as host code, and code running on the GPU is referred to as device code.


# Compiling a Sample GPU Job

To run a sample CUDA job start with interactive job.

sinteractive --partition=gpgputest -A hpcadmingpgpu --gres=gpu:p100:4

Load a CUDA module

`module load CUDA/8.0.44-GCC-4.9.2`

To compile 01-hello-gpu-solution.cu, run:

`nvcc 01-hello-gpu-solution.cu -o helloCUDA -gencode arch=compute_60,code=sm_60`

Execute the generated helloCUDA running:

`./helloCUDA`

Or, as an alternative, compile with `-run` at the end of the compilation line which will run the compiled binary right away.

# Examples

All examples with a numerical prefix, 01-, 02- etc are from NVidia.


# Debug with printf

Calling printf from a CUDA kernel function is no different than calling printf on CPU code. In the vector addition example, edit vec_add.cu and insert the following code after line 18:

if(threadIdx.x == 10)
    printf("c[%d] = %dn", id, c[id]);


# Supported Gencode variations for sm and compute

Below are the supported sm variations and sample cards from that generation
Supported on CUDA 7 and later

Fermi (CUDA 3.2 until CUDA 8) (deprecated from CUDA 9):
	SM20 or SM_20, compute_30 – Older cards such as GeForce 400, 500, 600, GT-630

Kepler (CUDA 5 and later):
        SM30 or SM_30, compute_30 – Kepler architecture (generic – Tesla K40/K80, GeForce 700, GT-730)
        Adds support for unified memory programming
        SM35 or SM_35, compute_35 – More specific Tesla K40
        Adds support for dynamic parallelism. Shows no real benefit over SM30 in my experience.
        SM37 or SM_37, compute_37 – More specific Tesla K80
        Adds a few more registers. Shows no real benefit over SM30 in my experience

Maxwell (CUDA 6 and later):
        SM50 or SM_50, compute_50 – Tesla/Quadro M series
        SM52 or SM_52, compute_52 – Quadro M6000 , GeForce 900, GTX-970, GTX-980, GTX Titan X
        SM53 or SM_53, compute_53 – Tegra (Jetson) TX1 / Tegra X1

Pascal (CUDA 8 and later)
        SM60 or SM_60, compute_60 – GP100/Tesla P100 – DGX-1 (Generic Pascal)
        SM61 or SM_61, compute_61 – GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4, Discrete GPU on the NVIDIA Drive PX2
        SM62 or SM_62, compute_62 – Integrated GPU on the NVIDIA Drive PX2, Tegra (Jetson) TX2

Volta (CUDA 9 and later)
        SM70 or SM_70, compute_70 – Tesla V100, GTX 1180 (GV104)
        SM71 or SM_71, compute_71 – probably not implemented
        SM72 or SM_72, compute_72 – currently unknown

Turing (CUDA 10 and later)
        SM80 or SM_80, compute_80 – RTX 2080, Titan RTX, Quadro R8000

(c.f., http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

# CUDA Error Handling

Most CUDA functions return a value of type cudaError_t, which can be used to check for errors when calling a function.

e.g.,

```
cudaError_t err;
err = cudaMallocManaged(&a, N)
// Assume the existence of `a` and `N`.
if (err != cudaSuccess)
// `cudaSuccess` is provided by CUDA.
	{
	printf("Error: %s\n", cudaGetErrorString(err)); // `cudaGetErrorString` is provided by CUDA.
	}
```

