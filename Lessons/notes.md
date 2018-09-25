### Part 1: Definition and History
* The die includes processor cores, render output units, frame buffers, thread scheduler, and texture mapping units
* GPGPU technology is massively parallel compared to CPUs (more cores, more threads), is inexpensive, and programmable (e.g., with CUDA).
* However GPGPU technology is accelerating faster than CPU technology. 

### Part 1: Programming GPGPUs
* GPUs are not suitable for all programming tasks. Investigate other algorithms if possible. Break the problem into discrete sections of work that can be distributed to multiple tasks (decomposition).
* The main challenge to the programmer - keep the GPGPU busy!

### Part 1: CUDA, OpenACC, OpenMP
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




### CUDA Program and Memory Hierarchy

### CUDA Basic C Extensions
