/**
 * @file curandTdgl.cu
 * @brief Defines the CUDA kernel for initializing CURAND random number generator states.
 *
 * This file implements the global CUDA function declared in `curandTdgl.cuh`.
 * Each thread initializes its own `curandState` instance using a global seed
 * and its unique thread index. These RNG states are later used for stochastic
 * components of simulations such as the TDGL or LLG equations.
 */

#include "curandTdgl.cuh"  // Header declaration

 // ============================================================================
 // Global Function Definition
 // ============================================================================

 /**
  * @brief Initialize CURAND states on the GPU.
  *
  * Each thread computes its unique index and initializes one CURAND state
  * with a user-provided seed. This ensures independent random number sequences
  * across threads.
  *
  * @param states   Pointer to device memory containing CURAND states.
  * @param seed     Base random seed for initialization.
  * @param gridSize Total number of states (corresponding to simulation sites or threads).
  *
  * @details
  * - The `curand_init()` function initializes each RNG state with:
  *   - `seed`      : A common random seed for reproducibility.
  *   - `sequence`  : Unique per-thread identifier (`idx`).
  *   - `offset`    : Set to `0` to start each sequence from its beginning.
  *
  * @note
  * This function should be called once at the start of the simulation, before
  * generating any random numbers on the GPU.
  *
  * **Example usage:**
  * @code
  * int gridSize = 1024;
  * dim3 threads(256);
  * dim3 blocks((gridSize + threads.x - 1) / threads.x);
  * init_curand_states<<<blocks, threads>>>(d_states, 1234UL, gridSize);
  * @endcode
  */
__global__ void init_curand_states(curandState* states, unsigned long seed, int gridSize) {
    // Compute unique thread index across the grid.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we stay within bounds of the state array.
    if (idx < gridSize) {
        // Initialize a CURAND RNG state for this thread.
        curand_init(seed, idx, 0, &states[idx]);
    }
}
