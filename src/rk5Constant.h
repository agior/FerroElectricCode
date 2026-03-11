/**
 * @file rk5Constant.h
 * @brief Declares the fixed-step fifth-order Runge–Kutta (RK5) integration routine.
 *
 * This header defines the interface for the `rk5FixedStep()` function, which performs
 * time integration of the polarization field using a fifth-order Runge–Kutta method.
 *
 * @details
 * The RK5 method is used for solving the Time-Dependent Ginzburg–Landau (TDGL)
 * equations with high accuracy. It accounts for effective fields, thermal noise,
 * and optional external driving fields. The function operates on device vectors
 * managed by the Thrust library to leverage GPU acceleration.
 *
 * The corresponding implementation is provided in **rk5Constant.cuh**.
 */

#pragma once
#ifndef RK5CONSTANT_H
#define RK5CONSTANT_H

 //------------------------------------------------------------------------------
 // Thrust includes
 //------------------------------------------------------------------------------
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

//------------------------------------------------------------------------------
// Standard includes
//------------------------------------------------------------------------------
#include <cmath>

//------------------------------------------------------------------------------
// Project includes
//------------------------------------------------------------------------------
#include "Vec3.h"
#include "coefficientVector.cuh"
#include "externalField.h"
#include "rk5Steps.h"
#include "errorSteps.h"
#include "tdgl.cuh"

//==============================================================================
//! @brief Fixed-step RK5 integration scheme for polarization dynamics.
//!
//! @tparam T Numeric precision type (e.g., float or double).
//!
//! @param polarization Device vector holding current polarization field values (`Vec3<T>`).
//! @param hFerro       Device vector containing effective field components.
//! @param noiseVector  Device vector storing stochastic noise contributions.
//! @param gridSize     Total number of spatial elements in the simulation grid.
//! @param Ncz          Number of cells along the z-axis (depth).
//! @param currentTime  Current normalized simulation time.
//! @param stepCounting Current integration step index (used for noise and output control).
//!
//! @details
//! Performs one or multiple fixed-time-step updates using the fifth-order Runge–Kutta
//! method. The function computes field derivatives via `rk5Steps.h` routines and applies
//! optional noise and external field effects. The method is stable and suitable for
//! moderately stiff TDGL problems.
//==============================================================================
template <typename T>
void rk5FixedStep(
    thrust::device_vector<Vec3<T>>& polarization,  //!< Result polarization vector
    thrust::device_vector<Vec3<T>>& hFerro,        //!< Input effective field vector
    thrust::device_vector<Vec3<T>>& noiseVector,   //!< Random noise vector
    int gridSize,                                  //!< Total grid size
    int Ncz,                                       //!< Number of cells in z-direction
    T currentTime,                                 //!< Current simulation time
    int stepCounting                               //!< Integration step counter
);

//------------------------------------------------------------------------------
// Include the implementation of RK5 integrator
//------------------------------------------------------------------------------
#include "rk5Constant.cuh"

#endif // RK5CONSTANT_H
