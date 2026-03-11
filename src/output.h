/**
 * @file output.h
 * @brief Declares output utility functions for CUDA-based polarization simulations.
 *
 * This header defines a set of templated functions responsible for writing
 * simulation data (polarization, constants, shapes, alpha vectors, and gradient vectors)
 * from device memory to host files. These utilities support performance tracking
 * and data export for post-processing and visualization.
 *
 * @details
 * Functions here primarily work with `thrust::device_vector` containers and operate
 * in both host and device contexts, depending on the operation. The implementation
 * details are included in the corresponding `output.cuh` file.
 */

#pragma once
#ifndef OUTPUT_H
#define OUTPUT_H

 //------------------------------------------------------------------------------
 // Standard and Thrust includes
 //------------------------------------------------------------------------------
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <chrono>
#include <fstream>
#include <iomanip>

#include "Vec3.h"

//==============================================================================
//! @brief Outputs polarization data to a file and records execution timing.
//!
//! @tparam T Numeric precision type (e.g., float or double).
//! @param d_vec Device vector containing polarization field values (`Vec3<T>`).
//! @param Ncx Number of cells in the x-direction.
//! @param Ncy Number of cells in the y-direction.
//! @param Ncz Number of cells in the z-direction.
//! @param N_d  Total number of spatial grid points (flattened index size).
//! @param time Simulation time corresponding to the output frame.
//!
//! @details
//! Writes polarization data from the device vector to an output file for visualization
//! and diagnostics. Also logs timing information for performance analysis.
//==============================================================================
template <typename T>
void getOutput(
    thrust::device_vector<Vec3<T>>& d_vec,
    int& Ncx,
    int& Ncy,
    int& Ncz,
    int& N_d,
    T& time
);

//==============================================================================
//! @brief Outputs constant parameter values stored on the device.
//!
//! @tparam T Numeric precision type (e.g., float or double).
//! @param d_vec Device vector containing constant parameter values.
//!
//! @details
//! Exports constant simulation parameters (such as field coefficients or materials
//! constants) from device memory to a file for inspection or reproducibility checks.
//==============================================================================
template <typename T>
void getConstant(thrust::device_vector<T>& d_vec);

//==============================================================================
//! @brief Outputs geometric shape data stored on the device.
//!
//! @tparam T Numeric precision type (e.g., float or double).
//! @param shapeVector Device vector defining geometry or masking shapes.
//!
//! @details
//! Typically used to record device-side shape or geometry arrays (e.g., boundary masks,
//! domain configurations) into host-accessible output files.
//==============================================================================
template <typename T>
void getShape(thrust::device_vector<T>& shapeVector);

//==============================================================================
//! @brief Saves normalized alpha vectors to an output file.
//!
//! @tparam T Numeric precision type (e.g., float or double).
//!
//! @details
//! Exports precomputed or dynamically updated normalized alpha field values
//! for diagnostics or further computation.
//==============================================================================
template <typename T>
void saveAlphaVectorsToFile();

//==============================================================================
//! @brief Saves normalized gradient vectors to an output file.
//!
//! @tparam T Numeric precision type (e.g., float or double).
//!
//! @details
//! Writes gradient field data (e.g., polarization gradient coefficients)
//! to a file for post-processing or verification.
//==============================================================================
template <typename T>
void saveGradientVectorsToFile();

//------------------------------------------------------------------------------
// Include implementation (CUDA-related definitions)
//------------------------------------------------------------------------------
#include "output.cuh"

#endif // OUTPUT_H
