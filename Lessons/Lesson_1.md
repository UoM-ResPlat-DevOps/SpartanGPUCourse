-- *Slide* --
### Part 0: Goals for today
* Part 1: GPU Technology and UniMelb
* Part 2: OpenACC Pragmas
* Part 3: CUDA Programming
-- *Slide End* --

-- *Slide* --
### Part 0: Slide Repository
* A copy of the slides and same code is available at: https://github.com/UoM-ResPlat-DevOps/SpartanGPUCourse
* Make use of our resources: `man spartan`, `/usr/local/common`, `dashboard.hpc.unimelb.edu.au`, `hpc-support@unimelb.edu.au`
-- *Slide End* --

-- *Slide* --
### Part 1: Definition and History
* The Graphic Processing Unit (GPU) is a processor that was specialised for processing graphics.
* A general purpose GPU recognises that algorithms, not designed for graphics processing, can also make use of the architecture. 
* A CPU has low latency and good throughput (compute a job as quickly as possible), whereas a GPU has high throughput and good latency (compute as many jobs as possible). 
-- *Slide End* --

-- *Slide* --
<img src="https://raw.githubusercontent.com/UoM-ResPlat-DevOps/SpartanGPUCourse/master/Images/gpuimage.png" /><br />
Image from Felipe A. Cruz, GeForce GTX 280
-- *Slide End* --

-- *Slide* --
### Part 1: Programming GPGPUs
* GPU programming is a type of SIMD parallelisation; single instruction, multiple data. 
* Impressive performance gains; 5x, 10x, 20x - and energy efficiency (10x performance, c5x energy per socket) 
* Several APIs; CUDA, OpenACC, OpenMP, and OpenCL.
-- *Slide End* --

-- *Slide* --
<img src="https://raw.githubusercontent.com/UoM-ResPlat-DevOps/SpartanGPUCourse/master/Images/Natoli-CPUvGPU-peak-DP-600x-300x232.png" /> <img src="https://raw.githubusercontent.com/UoM-ResPlat-DevOps/SpartanGPUCourse/master/Images/Natoli-CPUvGPU-peak-mem-bw-600x-300x241.png" /><br />
Images from HPCWire `https://www.hpcwire.com/2016/08/23/2016-important-year-hpc-two-decades/`
-- *Slide End* --

-- *Slide* --
### Part 1: How to Accelerate Code
* Third party libraries and application extensions; minimal effort, excellent performance, not very flexible.
* OpenACC/OpenMP directives; minor effort, lowest performance, portable between GPU/CPU and CPU-only systems.
* Programming extensions (CUDA, OpenCL); significant effort, potentially best performance, very flexible.
-- *Slide End* --

-- *Slide* --
### Part 1: OpenACC, OpenMP
* OpenACC (open accelerators) uses compiler directives to invoke acceleration. Works with C, C++ and Fortran source code. See: http://www.openacc-standard.org
* Since v4.0, OpenMP also supports accelerator directive. See: http://openmp.org
-- *Slide End* --

-- *Slide* --
### Part 1: CUDA
* CUDA was introduced by NVidia in 2006, as a compiler and application toolkit for NVidia GPGPUs. Originally Compute Unified Device Architecture. See: https://developer.nvidia.com/about-cuda
* CUDA does provide a high level of hardware abstraction and automation of thread management. There is also  numerical libraries, such as cuBLAS, and cuFFT.
-- *Slide End* --

-- *Slide* --
### Part 1: OpenCL
* OpenCL will become an industry standard in the future, however is a lower-level specification and therefore harder to program than with CUDA. See: https://www.khronos.org/opencl/
* OpenCL is cross-platform and multiple vendor open standard for C/C++
* Works on a diverse compute resources (e.g., CPU, GPU, DSP, FPGA) and can use same code base on each.
-- *Slide End* --

-- *Slide* --
### Part 1: GPUs on Spartan
* Spartan has a number of GPU partitions on Spartan for projects that are recipients of LE170100200 grant, plus additional departmental purchases. 
* The general gpgpu partition includes four Nvidia P100 GPUs per node; some specialist partitions have V100s (e.g., deeplearn). 
* Check a node with `nvidia-smi`.
-- *Slide End* --

-- *Slide* --
### Part 1: GPGPU Partition on Spartan
* GPGPU partitions are across multiple universities with allocations in proportion to funding. Theoretical maximum performance of 900+ teraflops. 
* If you have a UniMelb project and a GPGPU project you can submit to the CPU partition with `#SBATCH -q normal` and then a CPU partition name e.g., `#SBATCH -p physical`
-- *Slide End* --

-- *Slide* --
### Part 1: CUDA and Slurm
* A number of applications on Spartan have already been compiled with CUDA-specific toolchains
* These are like any other job submission with the following caveats: (1) You will need to specifiy the partition that you are using., (2) You will need to specify the account (projectID) that you are using for the gpgpu partitions., (3) You will need to request a generic resource for your job script. 
-- *Slide End* --

-- *Slide* --
### Part 1: Example Slurm Scripts
* A small number of example GPU-example job submission scripts are available on Spartan.
* For example Tensorflow (`/usr/local/common/Tensorflow`) and NAMD (`/usr/local/common/NAMD`) - the latter has GPU and non-GPU comparisons.
-- *Slide End* --

-- *Slide* --
### Part 2: Introduction to OpenACC
* The general structure of OpenACC is astoundingly simple; see `/usr/local/common/OpenACC`. It combines the process of decomposition on a hetrogenous system with pragmas. However OpenACC only works with a limited range of compilers (on Spartan, PGI compilers only, and GCC 6+). API is for C, C++, and Fortran.
-- *Slide End* --

-- *Slide* --
### Part 2: Portable and Flexible
* OpenACC directives are portable; across operating systems, host CPUs, and accelerators. Can be used  with GPU accelerated libraries, explicit parallel programming languages (e.g., CUDA), MPI, and OpenMP, all in the same program.
* As with OpenMP and OpenMPI directives, the pragmas in OpenACC are treated as comments by compilers that do not use them. 
-- *Slide End* --

-- *Slide* --
### Part 2: Profiling
* Profiling is very important with GPGPUs, and there are good tools with PGI compilers.
* Computation Intensity = Compute Operations / Memory Operations
* Computational Intensity of 1.0 or greater suggests that the loop might run well on a GPU
* Run the example in OpenACC/Profile
-- *Slide End* --

-- *Slide* --
### Part 2: Laplace Equation Decomposition
* Example problem (from the Pawsey Supercomputing Centre); Heating a metal plate, simulated by solving Laplace equation ∇^2 f x,y = 0.
* The steady state response is solved as a 2D grid where the temperature of every ith grid point is an average of its 4 neighbours.
* Compile the example program in `/usr/local/common/OpenACC`, then run the profiler. Notice that loops are identified as intensive regions.
-- *Slide End* --

-- *Slide* --
### Part 2: OpenACC Kernels Construct
* The `kernels` pragma suggests to the compiler to concentrate on loops in the code block.
* The compiler will run in parallel if it can.
* The syntax is `#pragma acc kernels directive [clause]` in C or `!$acc kernels` and `$!acc kernels end` in Fortran.
-- *Slide End* --

-- *Slide* --
### Part 2: Compiler Actions
* The kernels directive descriptive. It tells the compiler that a region can be made parallel. The compiler will analyze the code, identify which data has to be transferred, create a kernel and offload the kernel to the GPU.
* Note that this is different to OpenMP directives, which tell the compiler what to do, i.e., they are prescriptive. There is less control with many OpenMP directives, because it is up to compiler to decide what to do with descriptive directives.
-- *Slide End* --

-- *Slide* --
### Part 2: Compute and Data Bottlenecks
* The introduction of pragmas to the example in `/usr/local/common/OpenACC` actually makes the code slower!
* Simply making a loop available for parallel execution is insufficient. This may remove some compute bottlenecks, but it may also introduce new bottlenecks, such as memory transfer. The CPU (host) and the GPU (device) are separate processors.
-- *Slide End* --

-- *Slide* --
<img src="https://raw.githubusercontent.com/UoM-ResPlat-DevOps/SpartanGPUCourse/master/Images/pci-e_single_dual.png" />><br />
Image from NVidia developer blog
-- *Slide End* --

-- *Slide* --
### Part 2: OpenACC Data Construct
* The `data` construct suggests to the compiler the scoping of data and granularity data movement. It facilitates sharing data between multiple parallel regions.
* As a structure construct it must start and end in the scope of the same function or routine.
* The syntac is `#pragma acc data [clause]` in C or `!$acc data` and `!$acc end data` in Fortran.
-- *Slide End* --

-- *Slide* --
### Part 2: OpenACC Data Construct Clauses
* Several clauses can be used with the kernels, and data constructs (among others), including copy(list), copyin(list), copyout(list), create(list), present(list), presentor\_or\_*(list)
-- *Slide End* --

-- *Slide* --
### Part 2: Data Construct Actions
* The clause copy(list) will read and write data in list from host memory to device memory, copyin(list) will read, copyout(list) will write, create (list) will create and destroy variables on device (temporary buffers), present(list) will declare on device, present_or_*(list) will suggest that the compiler to check if the variable in the list exists or needs an action. Also used as short form e.g. pcopy(list), pcopyin(list), pcopyout(list)
-- *Slide End* --

-- *Slide* --
### Part 2: Array Shaping
* Array shaping for data constructs is often required in C so the compiler can understand the size and shape of the array; Fortran is better at this.
* In C the array shape is described as x[start:count], with another [] for additional dimensions e.g. x[start:count][start:ccount]; 'start' is the start index in the dimension and 'count' is the contiguous memory
address after start. In Fortran, array shape is described as y[start:end].
-- *Slide End* --

-- *Slide* --
### Part 2: Kernels vs Parallel Directives
* The `kernels` directive, used in this course, is a general case statement and is descriptive. The compiler works out what to do,
* The `parallel` directive, is prescriptive and allows further finer-grained control of how the compiler will attempt to structure work on the accelerator. It means the programmer can make errors.
-- *Slide End* --

-- *Slide* --
### Part 2: Parallel Loop Directive
* The `parallel` directive can be combined with a `loop` directive, creating a `parallel loop`, forcing the compiler to process the loop in parallel.
e.g., 
``` #pragma acc parallel loop 
for (int i=0; i<N; i++)
	{ C[i] = A[i] + B[i];
	}
``` 
-- *Slide End* --

-- *Slide* --
### Part 2: Responsibilities
* With the `kernel` directive it is the compiler's responsibility to determine what it safe to parallize. A single directive can cover a large area of code.
* With the `parallel loop` directive, the programmer has the responsibility. May be easier is the programmer is familiar with OpenMP.
-- *Slide End* --

-- *Slide* --
### Part 3: Structure of CUDA Code
* Just like other parallel codes, work with decomposition. General structure of separates CPU functions from CUDA functions. 
* The __global__ keyword indicates that the following function will run on the GPU, and can be invoked globally, which in this context means either by the CPU, or, by the GPU. Functions defined with the __global__ keyword must return type void. The function called to run on a GPU is referred to as a kernel.
-- *Slide End* --

-- *Slide* --
### CUDA Basic C Extensions
* Basic Function modifiers are `__global__` (to be called by the host but executed by the GPU) and `__host__ ` (to be called and executed by the host).
* Basic variable modifiers are `__shared__` (variable in shared memory), and `__syncthreads()` (sync of threads within a block).
* Basic kernel launch paramters are Block size and Grid Size - depends on hardware.
-- *Slide End* --

-- *Slide* --
### Part 3: Execution Configuration
* When a kernel is launched, an execution configuration must be provided, by using the <<< ... >>> syntax just prior to passing the kernel any expected arguments.
* The execution configuration allows programmers to specify the thread hierarchy for a kernel launch, which defines the number of thread groupings (blocks), as well as how many threads to execute in each block.
-- *Slide End* --

-- *Slide* --
### Part 3: CUDA Synchronisation
* Unlike most C/C++, launching CUDA kernels is asynchronous; the CPU code will continue to execute without waiting for the kernel launch to complete.
* A call to `cudaDeviceSynchronize` will cause the host (CPU) code to wait until the device (GPU) code
completes, and only then resume execution on the CPU.
-- *Slide End* --

-- *Slide* --
### Part 3: "Hello World", CUDA-style
* Refactor the "Hello World" example code in `/usr/local/common/CUDA/` to use the GPU.
* Note the compilation process; launch an interactive job, load a CUDA module (e.g., `CUDA/8.0.44-GCC-4.9.2`),  compile with `nvcc vecAdd.cu -o helloWorld -gencode arch=compute_60,code=sm_60`
-- *Slide End* --

-- *Slide* --
### Part 3: CUDA Compilation Flags
* With NVCC the `-arch` flag specifies the name of the NVidia GPU architecture that the CUDA files.
* Architectures can be specified with `sm_XX` and `compute_XX`, for real and virtual architectures respectively.
* Compile for the architecture (both virtual and real), that represents the GPUs you wish to target. i.e., `-gencode arch=compute_XX,code=sm_XX`.
-- *Slide End* --

-- *Slide* --
### Part 3: CUDA Program and Thread Hierarchy
* A CUDA _kernel_ (a C program) is executed by _thread_. Each thread has it own ID, execute same kernel and will access registers and local memory. 
* Threads are grouped into _blocks_, which will access shared memory, and threads in a block can synchronize execution. 
* Blocks are grouped into a _grid_, that can access global memory, and each block must be independent.
-- *Slide End* --

-- *Slide* --
### Part 3: Launching Parallel Kernels
* The execution configuration specificies how many thread blocks, (or blocks) and how many threads they would like each thread block to contain. The general syntax for this is: `<<< NUMBER_OF_BLOCKS, NUMBER_OF_THREADS_PER_BLOCK>>>`
* `someKernel<<<1, 1>>()` Single block, single thread, will run once.
* `someKernel<<<1, 10>>()` Single block, ten thread, will run ten times.
* `someKernel<<<10, 1>>()` Ten blocks, single thread per block, will run ten times.
-- *Slide End* --

-- *Slide* --
### Part 3: CUDA Thread Hierarchy Variables
* Each thread has an index within its thread block, starting at 0. Each block is given an index, starting at 0. CUDA kernels have access to variables identifying both the index of the thread within the block (threadIdx x) that is executing the kernel and the index of the block within the grid (and blockIdx x).
-- *Slide End* --

-- *Slide* --
### Part 3: CUDA Loop Acceleration
* As with other forms of parallelisation, loops in CPU-only applications are promising target for CUDA as each iteration of the loop can be run in parallel in its own thread. 
* To parallelise a loop, two things need to be done; (1) a kernel must be written to do the work of a single iteration of the loop and (2) the execution configuration must ensure the kernel executes the correct number of times.
-- *Slide End* --

-- *Slide* --
### Part 3: Coordinating Parallel Threads
* The number of threads in a block is limited to 1024. Beyond this, coordination is required across multiple thread blocks using the variable `blockDim.x`. When used with `blockIdx.x` and `threadIdx.x`, coordination is achieved across multiple blocks of multiple threads.
* Parallel execution accross multiple blocks of multiple uses the expression threadIdx.x + blockIdx.x * blockDim.x. Modify blockDim.x to increment.
-- *Slide End* --

-- *Slide* --
### Part 3: CPU and GPU Memory Allocation
* The most basic CUDA memory management technique involves a pointer that can be referenced in both host and device code, replace calls to `malloc` and `free` with `cudaMallocManaged` and `cudaFree`.
-- *Slide End* --

-- *Slide* --
<img src="https://raw.githubusercontent.com/UoM-ResPlat-DevOps/SpartanGPUCourse/master/Images/gpumemarch.png" />
-- *Slide End* --

-- *Slide* --
### Part 3: Blocks, Threads and Loop Mismatch
* A common issue is having block/thread sizes mismatched with the iterations of a loop. This can be resolved by (1) writing an execution that creates more threads than necessary and (2) pass an argument, N, into the kernel that represents how many times the kernel should run, and (3) calculating the thread's index within the grid (using tid+bid*bdim), check that this index does not exceed N befor executing.
-- *Slide End* --

-- *Slide* --
### Part 3: Data Sets Larger then the Grid
* The number of threads in a grid may be smaller than the size of a data set (e.g., a grid of 250 threads, an array of 1000 datasets). To resolve this, a grid-stride loop can be used within the kernel.
* Each thread calculates its unique index within the grid using `tid+bid*bdim`, perform its operation on the element at that index within the array, then add to its index the number of threads in the grid until it is out of range of the array.
-- *Slide End* --

-- *Slide* --
### Part 3: The gridDim.x Variable
* CUDA provides a variable giving the number of blocks in a grid, `gridDim.x`. Calculating the total number of threads in a grid then is simply the number of blocks in a grid multiplied by the number of threads in each block, gridDim.x * blockDim.x
-- *Slide End* --

-- *Slide* --
### Part 3: CUDA Error Handling
* Most CUDA functions return a value of type cudaError_t, which can be used to check for errors when calling a function. Launching kernels will not return a value of type `cudaError_t`
* To check for errors occuring at the time of a kernel launch, CUDA provides the `cudaGetLastError` function, which does return a value of type `cudaError_t`.
-- *Slide End* --

-- *Slide* --
### Part 3: CUDA Device Synchronize
* In order to catch errors that occur asynchronously it is necessary to check the error returned by `cudaDeviceSynchronize`, which will return an error if one of the kernel executions it is synchronizing on should fail.
-- *Slide End* --

-- *Slide* --
### References
Accelerate code on GPUs with OpenACC, Pawsey Supercomputing Centre
Accelerating Applications with CUDA C/C++, NVidia
GPU Programming Essentials, Pawsey Supercomputing Centre
Introduction to OpenACC, NVidia
The Graphics Processing Unit (GPU) revolution, Ramu Anandakrishnan, Virginia Polytechnic Institute and State University
Tutorial on GPU computing: With an introduction to CUDA, Felipe A. Cruz, University of Bristol
-- *Slide End* --

-- *Slide* --
<img src="https://raw.githubusercontent.com/UoM-ResPlat-DevOps/SpartanGPUCourse/master/Images/hypnotoad.png" />
-- *Slide End* --

