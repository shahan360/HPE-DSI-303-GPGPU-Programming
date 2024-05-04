1. Parallelizing the Compute Function:
The compute function, responsible for calculating forces and energies, is parallelized using OpenACC.
#pragma acc parallel loop directive is used to parallelize the outer loop over atoms, allowing each atom's computation to be executed concurrently on the GPU.
Inside the outer loop, #pragma acc loop vector independent directive is used to parallelize the inner loop over atom pairs, ensuring independent iterations can be executed concurrently on GPU threads.

2. Parallelizing the Update Function:
The update function, responsible for updating atom positions and velocities, is parallelized using OpenACC.
#pragma acc parallel loop directive is used to parallelize the loop over atoms, allowing each atom's update computation to be executed concurrently on the GPU.

These parallelizations leverage the parallel computing capabilities of GPUs to accelerate the simulation of molecular dynamics.

Compilation Instructions:
gcc -o md_parallelized md.c -lm

Execution Instructions:
./md_parallelized

