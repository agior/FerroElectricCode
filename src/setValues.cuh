/**
 * @file setValues.cuh
 * @brief Declares the parameter initialization routine and includes its CUDA implementation.
 *
 * This header provides the declaration for the templated function `set_values<T>()`, which
 * initializes or normalizes simulation parameters (e.g., coefficients, scaling factors, or
 * material constants). The actual function definition is included from `setValues.cu`.
 *
 * Dependencies:
 * - `Vec3.h` : Defines the 3D vector class used for spatial or field quantities.
 * - `coefficientVector.cuh` : Contains coefficient structures or device vectors.
 * - `curandTdgl.cuh` : Provides random number generation utilities (CURAND-based) for TDGL simulations.
 *
 * @note The `set_values` function is templated to support multiple precision types (`float`, `double`, etc.).
 */

#pragma once

 //------------------------------------------------------------------------------
 // Include dependencies
 //------------------------------------------------------------------------------
#include <iostream>
#include <limits>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include "Vec3.h"                 ///< 3D vector utilities
#include "coefficientVector.cuh"  ///< Coefficient definitions and related device data
#include "curandTdgl.cuh"         ///< Random number generation utilities

//------------------------------------------------------------------------------
// Function Declarations
//------------------------------------------------------------------------------

/**
 * @brief Sets or normalizes material and simulation parameters.
 *
 * This templated function configures problem-specific constants or scaling factors
 * before the main time integration begins.
 *
 * @tparam T Numeric precision type (e.g., float or double).
 */
template <typename T>
void set_values();

//------------------------------------------------------------------------------
// Include the corresponding CUDA implementation
//------------------------------------------------------------------------------
#include "setValues.cu"
