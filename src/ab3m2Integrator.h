#pragma once
#ifndef AB3M2INTEGRATOR_H
#define AB3M2INTEGRATOR_H

/**
 * @file ab3m2Integrator.h
 * @brief Declares the Adams–Bashforth–Moulton 3rd–2nd order (AB3M2) integrator
 *        for ferroelectric polarization dynamics.
 *
 * This header defines the template function interface for the AB3M2 time integration
 * scheme. The AB3M2 method combines a 3rd-order explicit Adams–Bashforth predictor
 * with a 2nd-order implicit Adams–Moulton corrector to achieve stability and accuracy
 * in the numerical evolution of nonlinear TDGL equations.
 *
 * Dependencies:
 * - Thrust (device/host vectors, execution policies, functional operators)
 * - rk5Integrator.h      → For optional high-accuracy reference integration
 * - externalField.h      → Handles external and AC field contributions
 * - ab3m2Steps.h         → Contains per-step update kernels for AB3M2
 * - rk5Constant.h        → Constants shared across integrators
 * - errorSteps.h         → Numerical error estimation and correction
 * - tdgl.cuh             → Time-dependent Ginzburg–Landau (TDGL) equation kernels
 * - output.hpp           → Output handling and diagnostics
 */

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include <cmath>

 // Core numerical physics modules
#include "Vec3.h"              ///< Custom 3D vector class (templated)
#include "rk5Integrator.h"     ///< Runge–Kutta 5th-order integrator (for comparison)
#include "externalField.h"     ///< External and AC field contribution handling
#include "ab3m2Steps.h"        ///< Adams–Bashforth–Moulton step function definitions
#include "rk5Constant.h"       ///< Common numerical constants (e.g., dt, tolerances)
#include "errorSteps.h"        ///< Local error estimation utilities
#include "tdgl.cuh"            ///< Time-dependent Ginzburg–Landau solver
#include "output.hpp"          ///< File and stream output utilities

// ----------------------------------------------------------------------------
// Function Declaration
// ----------------------------------------------------------------------------

/**
 * @brief Performs one or multiple AB3M2 integration cycles on the polarization field.
 *
 * @tparam T                 Floating-point type (typically `float` or `double`)
 * @param polarization       [in/out] Device vector containing polarization components
 * @param hFerro             [in/out] Device vector of effective field components
 * @param noiseVector        [in]     Device vector of stochastic noise components
 * @param gridSize           [in]     Total number of grid points
 * @param Ncz                [in]     Number of z-slices in the computational domain
 * @param totalTime          [in]     Total simulation time
 */
template <typename T>
void ab3m2(
    thrust::device_vector<Vec3<T>>& polarization,
    thrust::device_vector<Vec3<T>>& hFerro,
    thrust::device_vector<Vec3<T>>& noiseVector,
    int gridSize,
    int Ncz,
    T totalTime
);

// ----------------------------------------------------------------------------
// Implementation Include
// ----------------------------------------------------------------------------

#include "ab3m2Integrator.cuh"

#endif // AB3M2INTEGRATOR_H
