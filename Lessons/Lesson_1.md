-- *Slide* --
### Part Zero: Goals for today
* Part 1: GPU Technology and UniMelb
* Part 2: OpenACC and OpenMPI Acceleration
* Part 3: CUDA Programming
* Part 4: Debugging, Profiling, and Optimisation
-- *Slide End* --

-- *Slide* --
### Part 1: Definition and History
* The Graphic Processing Unit (GPU) is a processor that was specialised for processing graphics.
* A general purpose GPU recognises that algorithms, not designed for graphics processing, can also make use of the architecture. 
* A CPU has low latency and good throughput (compute a job as quickly as possible), whereas a GPU has high throughput and good latency (compute as many jobs as possible). 
-- *Slide End* --

-- *Slide* --
### Part 1: Programming GPGPUs
* GPU programming is a type of SIMD parallelisation; single instruction, multiple data. 
* Performance improvements come from sending the sequential parts of the code to the CPU, compute intensive SIMD parts to the GPU.
* Impressive performance gains; 5x, 10x, 20x - and energy efficiency (10x performance, c5x energy per socket) 
* Several APIs; CUDA, OpenACC, OpenMP, and OpenCL.
-- *Slide End* --

-- *Slide* --
<img src="https://raw.githubusercontent.com/UoM-ResPlat-DevOps/SpartanGPUCourse/master/Images/Natoli-CPUvGPU-peak-DP-600x-300x232.png" /><img src="https://raw.githubusercontent.com/UoM-ResPlat-DevOps/SpartanGPUCourse/master/Images/Natoli-CPUvGPU-peak-mem-bw-600x-300x241.png" />
Images from HPCWire `https://www.hpcwire.com/2016/08/23/2016-important-year-hpc-two-decades/`
-- *Slide End* --

-- *Slide* --
### Part 1: CUDA, OpenACC, OpenMP
* CUDA was introduced by NVidia in 2006, as a compiler and application toolkit for NVidia GPGPUs. Originally Compute Unified Device Architecture. See: https://developer.nvidia.com/about-cuda
* CUDA does provide a high level of hardware abstraction and automation of thread management. There is also  numerical libraries, such as cuBLAS, and cuFFT.
* OpenACC (open accelerators) uses compiler directives to invoke acceleration. Works with C, C++ and Fortran source code. See: http://www.openacc-standard.org/
* Since v4.0, OpenMP also supports accelerator directive. See: http://openmp.org
-- *Slide End* --

-- *Slide* --
### Part 1: OpenCL
* OpenCL will become an industry standard in the future, however is a lower-level specification and therefore harder to program than with CUDA. See: https://www.khronos.org/opencl/
* OpenCL is cross-platform and multiple vendor open standard for C/C++
* Works on a diverse compute resources (e.g., CPU, GPU, DSP, FPGA) and can use same code base on each.
-- *Slide End* --

-- *Slide* --
### Part 1: GPUs on Spartan
* Spartan has a number of GPU partitions on Spartan; only a few for general use (gpu), many for projects that are recipients of LE170100200 grant (gpgpu partitions), plus additional purchases. 
* The general gpu partition includes four Nvidia K80 GPUs per node, while the newer gpgpu partition includes four Nvidia P100 GPUs per node. 
* GPGPU partitions are across multiple universities with allocations in proportion to funding.
* Theoretical maximum performance of around 900 teraflops. 
-- *Slide End* --

### Part 1: CUDA and Slurm
-- *Slide* --
* A number of applications on Spartan have already been compiled with CUDA-specific toolchains; including CUDA from 7.0.28 to 9.2.88, FFTW, GROMACS, NAMD, OpenMPI, PyTorch, Python, RapidCFD, Tensorflow, Torch, etc.
* These are like any other job submission with the following caveats: (1) You will need to specifiy the partition that you are using., (2) You will need to specify the account (projectID) that you are using for the gpgpu partitions., (3) You will need to request a generic resource for your job script. 
-- *Slide End* --

### Part 1: Example Slurm Scripts
-- *Slide* --
* A small number of example GPU-example job submission scripts are available on Spartan.
* For example Tensorflow (`/usr/local/common/Tensorflow`) and NAMD (`/usr/local/common/NAMD`) - the latter has GPU and non-GPU comparisons.
-- *Slide End* --

-- *Slide* --
### CUDA Program and Memory Hierarchy
* A CUDA _kernel_ (a C program) is executed by _thread_. Each thread has it own ID, and thousands of threads execute same kernel and will access registers and local memory. 
* Threads are grouped into _blocks_, which will access shared memory, and threads in a block can synchronize execution. 
* Blocks are grouped into a _grid_, that can access global memory, and each block must be independent.
-- *Slide End* --

-- *Slide* --
### CUDA Basic C Extensions
* Basic Function modifiers are `__global__` (to be called by the host but executed by the GPU) and `__host__ ` (to be called and executed by the host).
* Basic variable modifiers are `__shared__` (variable in shared memory), and `__syncthreads()` (sync of threads within a block).
* Basic kernel launch paramters are Block size and Grid Size - dependens on hardware.
-- *Slide End* --

-- *Slide* --
### CUDA Simple Example
* Simple example, adds two arrays. Note threadIDx variable, sets threadID.

-- *Slide* --
### GPUs on Spartan
-- *Slide End* --

-- *Slide* --
## Part Two: Application Interfaces
-- *Slide End* --

-- *Slide* --
### Deep Learning Applications
-- *Slide End* --

-- *Slide* --
### CUDA Applications on Spartan
-- *Slide End* --

-- *Slide* --
### OpenACC Compiler directives
-- *Slide End* --

-- *Slide* --
## Part Three: CUDA Programming
-- *Slide End* --

-- *Slide* --
### From Serial to GPU Code

-- *Slide End* --

-- *Slide* --
### Libraries (CUBLAS, CUDART, CUFFT, CURAND etc etc)
-- *Slide End* --

-- *Slide* --
## Part Four: Debugging, Profiling, and Optimisation
-- *Slide End* --

-- *Slide* --
### Profiling
* To enable the cuda profiler to analyse your program, see the example Slurm Script in `/usr/local/common`

-- *Slide End* --

-- *Slide* --
### Debugging
-- *Slide End* --

-- *Slide* --
### Optimisation
-- *Slide End* --


