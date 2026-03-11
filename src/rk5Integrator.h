/**
 * @file rk5Integrator.h
 * @brief Declares the full fifth-order Runge–Kutta–Fehlberg (RK45) integrator for TDGL dynamics.
 *
 * This header defines the interface for the `rk5()` function, implementing an
 * adaptive Runge–Kutta–Fehlberg (RK45) scheme used to integrate the
 * Time-Dependent Ginzburg–Landau (TDGL) equations on GPU.
 *
 * @details
 * The RK45 integrator combines a 5th-order accurate solution with an embedded
 * 4th-order estimate to automatically control integration step size based on
 * local error. It accounts for deterministic field dynamics, stochastic
 * thermal noise, and optional external fields.
 *
 * The CUDA `__constant__` coefficients define the Butcher tableau constants
 * for the RK45 scheme (Fehlberg method). The implementation is located in
 * **rk5Integrator.cuh**.
 */

#pragma once
#ifndef RK5INTEGRATOR_H
#define RK5INTEGRATOR_H

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
#include "output.hpp"
#include "output_handle.h"

//==============================================================================
//! @name RK45 Butcher tableau constants (Fehlberg coefficients)
//! @brief Device constants defining time-step weights for the RK45 method.
//==============================================================================
// clang-format off
__constant__ Type_var aa1 = 0.0;
__constant__ Type_var aa2 = 1.0 / 5.0;
__constant__ Type_var aa3 = 3.0 / 10.0;
__constant__ Type_var aa4 = 3.0 / 5.0;
__constant__ Type_var aa5 = 1.0;
__constant__ Type_var aa6 = 7.0 / 8.0;
// clang-format on

//==============================================================================
//! @brief Adaptive Runge–Kutta–Fehlberg (RK45) integrator for TDGL equations.
//!
//! @tparam T Numeric precision type (e.g., float or double).
//!
//! @param polarization Device vector holding the polarization (order parameter) field.
//! @param hFerro       Device vector containing effective field components.
//! @param noiseVector  Device vector containing stochastic noise contributions.
//! @param gridSize     Total number of grid cells in the simulation.
//! @param Ncz          Number of cells along the z-axis (depth).
//! @param totalTime    Total integration time for the simulation.
//!
//! @details
//! This function executes the full RK45 integration loop, performing adaptive
//! step-size control using embedded 4th/5th order error estimates. It updates
//! polarization vectors based on deterministic and stochastic field dynamics,
//! applies boundary conditions, and handles optional output via `output_handle.h`.
//!
//! The method is particularly suited for problems where stability and precision
//! are required under thermal noise and strong nonlinear couplings.
//==============================================================================
template <typename T>
void rk5(
    thrust::device_vector<Vec3<T>>& polarization,  //!< Polarization field (output)
    thrust::device_vector<Vec3<T>>& hFerro,        //!< Effective field input
    thrust::device_vector<Vec3<T>>& noiseVector,   //!< Random noise field
    int gridSize,                                  //!< Total number of grid cells
    int Ncz,                                       //!< Number of cells along z-axis
    T totalTime                                    //!< Total simulation time
);

//------------------------------------------------------------------------------
// Include implementation
//------------------------------------------------------------------------------
#include "rk5Integrator.cuh"

#endif // RK5INTEGRATOR_H
