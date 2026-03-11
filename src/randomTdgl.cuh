/**
 * @file randomTdgl.cuh
 * @brief Defines random number generation functor for TDGL simulations.
 *
 * This header provides a templated functor used to generate random
 * three-component vectors (`Vec3<T>`) for stochastic noise terms in
 * Time-Dependent Ginzburg–Landau (TDGL) equations.
 *
 * @details
 * The functor interfaces with `curandState` for per-thread random number
 * generation on the GPU. Each thread produces a random `Vec3<T>` based on
 * variance parameters provided in a device vector. This allows spatially
 * varying stochastic contributions to polarization or magnetization dynamics.
 */

#pragma once
#ifndef _RANDOMTDGL_CUH_
#define _RANDOMTDGL_CUH_

//------------------------------------------------------------------------------
// Standard and CUDA includes
//------------------------------------------------------------------------------
#include "Vec3.h"
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


//------------------------------------------------------------------------------
// Project includes
//------------------------------------------------------------------------------


//==============================================================================
//! @brief Functor for generating random 3D vectors for TDGL stochastic terms.
//!
//! @tparam T Numeric precision type (e.g., float or double).
//!
//! @details
//! This struct provides an overloaded device `operator()` that produces
//! random 3D vectors following a specified variance profile. Each element
//! of the output field corresponds to a spatial grid cell used in TDGL
//! simulations.
//!
//! The generated vectors are used to introduce stochastic fluctuations
//! in polarization or magnetization dynamics.
//==============================================================================
template <typename T>
struct tdgl_random
{
    //--------------------------------------------------------------------------
    // Member data
    //--------------------------------------------------------------------------

    curandState* states;    //!< Pointer to the array of random number generator (RNG) states.
    T* varianceVec;         //!< Pointer to the variance vector (defining stochastic strength per cell).
    int varianceVecSize;    //!< Number of entries in the variance vector.
    int Ncz;                //!< Number of cells along the z-axis (depth dimension).
    int gridSize;           //!< Total number of spatial grid points in the simulation.

    //--------------------------------------------------------------------------
    //! @brief Constructor: initializes RNG state and simulation parameters.
    //!
    //! @param states_          Pointer to initialized `curandState` array.
    //! @param varianceVec_     Pointer to variance vector in device memory.
    //! @param varianceVecSize_ Number of variance values available.
    //! @param gridSize_        Total number of grid points in the simulation.
    //! @param Ncz_             Number of cells in the z-direction.
    //--------------------------------------------------------------------------
    __host__ __device__
    tdgl_random(
        curandState* states_,
        T* varianceVec_,
        int varianceVecSize_,
        int gridSize_,
        int Ncz_
    )
        : states(states_),
          varianceVec(varianceVec_),
          varianceVecSize(varianceVecSize_),
          gridSize(gridSize_),
          Ncz(Ncz_)
    {
    }

    //--------------------------------------------------------------------------
    //! @brief Device operator that generates a random 3D vector for a given index.
    //!
    //! @param idx Global thread or cell index.
    //! @return Randomized vector (`Vec3<T>`) with variance scaling.
    //!
    //! @details
    //! Each component of the resulting vector is drawn from a uniform or
    //! Gaussian distribution (depending on implementation in `randomTdgl.cu`).
    //! The variance vector provides spatial modulation of noise amplitude.
    //--------------------------------------------------------------------------
    __device__ inline Vec3<T> operator()(int idx) const;
};

//------------------------------------------------------------------------------
// Include device implementation
//------------------------------------------------------------------------------
#include "randomTdgl.cu"

#endif // _RANDOMTDGL_CUH_
