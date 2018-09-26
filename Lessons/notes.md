### Part 1: Definition and History
* The die includes processor cores, render output units, frame buffers, thread scheduler, and texture mapping units
* GPGPU technology is massively parallel compared to CPUs (more cores, more threads), is inexpensive, and programmable (e.g., with CUDA).
* However GPGPU technology is accelerating faster than CPU technology. 

### Part 1: Programming GPGPUs
* GPUs are not suitable for all programming tasks. Investigate other algorithms if possible. Break the problem into discrete sections of work that can be distributed to multiple tasks (decomposition).
* The main challenge to the programmer - keep the GPGPU busy!

### Part 1: OpenACC, OpenMP, CUDA
* The CUDA API extends the C, C++ programming languages. Specific to NVIDIA hardware. 
* Porting CUDA to OpenCL is becoming easier.
* OpenACC Vendor-neutral API is developed by Cray, CAPS, NVidia and PGI for GPU/CPU systems.

### Part 1: OpenCL
* Not taught in this *introductory* course.

### Part 1: GPUs on Spartan
* There are three nodes (only) available for general use (gpu partition), five nodes for physics (two exclusive, three shared), and a much larger set of GPGPU partitions (shortgpgpu, 6 nodes; gpgpu partition 59 nodes, gpgpu-test 6 nodes, deeplearn (engineering purchased) 4 nodes). 
* MDHS inc. StV – 5.22 Nodes, Melbourne Bio – 10.43, Resplat (General Access) – 10.43, MSE – 24.4, MSE Mech Eng – 4.86, Latrobe – 5.22, RMIT – 5.22, Deakin – 5.22. Total Nodes = 71 (excludes 2 nodes for testing)/
* These GPGPU nodes will be presented in 3 subclusters and will be released in stages. 

### Part 1: CUDA and Slurm
* CUDA toolchain is required to make GPUs work as expected. Approximately 250 applications and libraries in total
* For example #SBATCH --partition gpu and #SBATCH --partition gpgpu. For example  For example #SBATCH --gres=gpu:2 will request two GPUs for your job.
* You can specify project at submission time: e.g., sbatch -A projectID script.slurm

### Part 1: Example Slurm Scripts
CPU Example WallClock: 66.138824  CPUTime: 64.785332  Memory: 236.933594 MB
GPU Example WallClock: 18.759842  CPUTime: 17.847618  Memory: 1319.082031 MB

### Part 2: Introduction to OpenACC
GCC > 6.0 supports OpenACC 2.0a - I haven't tested this!

### Part 2: Kernels vs Parallel Directives
See README.md

### Part 3: CUDA Synchronisation

### Part 3: "Hello World", CUDA-style
The kernel is launching with 1 block of threads (the first execution configuration argument) which contains 1 thread (the second configuration argument).

### Part 3: Launching Parallel Kernels
compile and modify `01-basic-parallel.cu` to represent choices.

### Part 3: CUDA Thread Hierarchy Variables
Comile and modify `01-single-block-loop-solution.cu`

### Coordinating Parallel Threads
Compile and modify `02-multi-block-loop.cu`

### Part 3: CPU and GPU Memory Allocation
The program allocates an array, initializes it with integer values on the host, attempts to double each of these values in parallel on the GPU, and then confirms whether or not the doubling operations were successful, on the host.

Currently the program will not work: it is attempting to interact on both the host and the device with an array at pointer a, but has only allocated the array (using malloc) to be accessible on the host. 

Refactor the application to meet the following conditions:
a should be available to both host and device code.
The memory at a should be correctly freed.

Compile and modify `01-double-elements-solution.cu`

### Part 3: Blocks, Threads and Loop Mismatch
The program in 02-mismatched-config-loop.cu allocates memory, using cudaMallocManaged for a 1000 element array of integers, and then seeks to initialize all the values of the array in parallel using a CUDA kernel. 

Assign a value to number_of_blocks that will make sure there are at least as many threads as there are elements in a to work on.

Update the initializeElementsTo kernel to make sure that it does not attempt to work on data elements that are out of range.

Compile and modify `02-mismatched-config-loop-solution.cu`

### Part 3: Data Sets Larger then the Grid
A grid-stride loop in the doubleElements kernel, in order that the grid, which is smaller than N, can reuse threads to cover every element in the array. The program will print whether or not every element in the array has been doubled, currently the program accurately prints FALSE.

### Part 3: CUDA Error Handling
Check README.md Compile and modify 01-add-error-handling-solution.cu 01-add-error-handling.cu
