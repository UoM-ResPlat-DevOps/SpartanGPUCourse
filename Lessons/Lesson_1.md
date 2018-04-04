-- *Slide* --
# Part Zero: Goals for today
* Part 1: Understanding GPU Technology
* Part 2: Understanding Application Interfaces
* Part 3: Understanding CUDA Programming
* Part 4: Understanding Debugging, Profiling, and Optimisation
-- *Slide End* --

-- *Slide* --
## Part One: GPU Technology
-- *Slide End* --

-- *Slide* --
### Definition and History
* The Graphic Processing Unit (GPU) is a processor that was specialised for processing graphics; the die includes processor cores, render output units, frame buffers, thread scheduler, and texture mapping units.
* A general purpose GPU recognises that algorithms, not designed for graphics processing. can also make use of the architecture. GPGPU technology is massively parallel compared to CPUs (more cores, more threads), is inexpensive, and programmable (e.g., with CUDA).
* A CPU has low latency and good throughput (compute a job as quickly as possible), whereas a GPU has high throughput and good latency (compute as many jobs as possible). However GPGPU technology is accelerating faster than CPU technology. 
-- *Slide End* --

-- *Slide* --
### Programming GPGPUs
* CUDA was introduced by NVidia in 2006, as a compilerr and application toolkit for NVidia GPGPUs. The CUDA API extends the C, C++ programming languages. CUDA is, however, vendor specific. 
* OpenCL will become an industry standard in the future, however is a lower-level specification and therefore harder to program than with CUDA. Porting CUDA to OpenCL is becoming easier.
* CUDA does provide a high level of hardware abstraction and automation of thread management. The main challenge to the programmer - keep the GPGPU busy!
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
### Early Usage
-- *Slide End* --

-- *Slide* --
### Stream processing and general purpose GPUs
-- *Slide End* --

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


