#pragma once
/**
 * @file initCurandStates.cuh
 * @brief Declares the CUDA kernel for initializing CURAND random number generator (RNG) states.
 *
 * This header provides a global kernel declaration to initialize an array of
 * `curandState` objects on the GPU. Each thread in the CUDA grid initializes
 * its own RNG state using the provided seed and unique thread index.
 *
 * This kernel should be launched before any stochastic simulation that
 * requires GPU-based random number generation (e.g., thermal noise in TDGL or LLG models).
 *
 * Dependencies:
 *  - `<curand_kernel.h>` : CUDA CURAND RNG API
 *  - `<device_launch_parameters.h>` : CUDA thread/block launch utilities
 *  - `<thrust/execution_policy.h>` : For Thrust execution compatibility
 */

#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <thrust/execution_policy.h>

 // ============================================================================
 // Global Kernel Declaration
 // ============================================================================

 /**
  * @brief Initializes CURAND states for all threads in a given grid.
  *
  * Each thread will initialize one `curandState` entry using the provided seed
  * and its unique global index.
  *
  * @param states   Pointer to device array of `curandState` objects (size ? gridSize)
  * @param seed     Random seed used for state initialization
  * @param gridSize Total number of RNG states (typically equals number of simulation sites or threads)
  *
  * @note Call this function once before any stochastic computations.
  *       Example launch:
  *       @code
  *       int N = gridSize;
  *       dim3 blocks((N + 255) / 256);
  *       dim3 threads(256);
  *       init_curand_states<<<blocks, threads>>>(d_states, 1234UL, N);
  *       @endcode
  */
__global__ void init_curand_states(curandState* states, unsigned long seed, int gridSize);
