/**
 * @file randomTdgl.cu
 * @brief Implements the TDGL random vector generator functor.
 *
 * This source file provides the device implementation of the
 * `tdgl_random<T>::operator()` function. Each thread generates a stochastic
 * 3D vector (`Vec3<T>`) using a CUDA random number generator (`curandState`),
 * modulated by a spatially dependent variance.
 *
 * @details
 * The function supports three variance configurations:
 * - **Single value:** uniform noise amplitude across the grid.
 * - **Layer-wise values:** variance specified per z-slice.
 * - **Fully spatial values:** variance specified per grid cell.
 *
 * The resulting random vectors are used as thermal noise sources in
 * Time-Dependent Ginzburg–Landau (TDGL) simulations.
 */

#pragma once

 //------------------------------------------------------------------------------
 // Thrust and project includes
 //------------------------------------------------------------------------------
#include <thrust/random.h>
#include "randomTdgl.cuh"

//==============================================================================
//! @brief Device operator that generates a random 3D vector for stochastic TDGL noise.
//!
//! @tparam T Numeric precision type (e.g., float or double).
//! @param idx Global cell index in the simulation grid.
//! @return Randomized 3D vector (`Vec3<T>`) scaled by local variance.
//!
//! @details
//! - Uses `curand_normal_double()` for sampling Gaussian-distributed components.
//! - Each thread operates independently using its own RNG state.
//! - The variance can be uniform, layer-wise, or per-cell depending on
//!   the length of the `varianceVec` array.
//!
//! @note The function updates the `curandState` array in-place to ensure
//!       reproducible, continuous random sequences across iterations.
//==============================================================================
template <typename T>
__device__ Vec3<T> tdgl_random<T>::operator()(int idx) const
{
    T variance;

    //----------------------------------------------------------------------
    // Select variance based on configuration
    //----------------------------------------------------------------------
    if (varianceVecSize == 1)
    {
        // Single variance value (uniform across the entire grid)
        variance = varianceVec[0];
    }
    else if (varianceVecSize == Ncz)
    {
        // Layer-wise variance: one variance value per z-slice
        int z = idx / (gridSize / Ncz);
        variance = varianceVec[z];
    }
    else
    {
        // Per-cell variance: one value per grid point
        variance = varianceVec[idx];
    }

    //----------------------------------------------------------------------
    // Load current RNG state
    //----------------------------------------------------------------------
    curandState currentState = states[idx];

    //----------------------------------------------------------------------
    // Generate Gaussian random components
    //----------------------------------------------------------------------
    T x = curand_normal_double(&currentState);
    T y = curand_normal_double(&currentState);
    T z = curand_normal_double(&currentState);

    //----------------------------------------------------------------------
    // Save updated RNG state
    //----------------------------------------------------------------------
    states[idx] = currentState;

    //----------------------------------------------------------------------
    // Apply variance scaling
    //----------------------------------------------------------------------
    x *= variance;
    y *= variance;
    z *= variance;

    //----------------------------------------------------------------------
    // Return stochastic vector
    //----------------------------------------------------------------------
    return Vec3<T>(x, y, z);
}
