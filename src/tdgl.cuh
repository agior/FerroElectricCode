/**
 * @file tdgl.cuh
 * @brief Declares the main TDGL (Time-Dependent Ginzburg–Landau) equation evaluator and includes its implementation.
 *
 * This header defines the function interface for computing the total effective field acting
 * on the polarization (or order parameter) according to the TDGL formalism. The implementation
 * in `tdgl.cu` combines all relevant physical field contributions (Landau, Ginzburg, electrostatic,
 * elastic, and external fields) along with stochastic noise.
 *
 * Dependencies:
 * - `Vec3.h`               : Defines 3D vector class for polarization and fields.
 * - `tdglFunction.cuh`     : Provides helper functions for TDGL calculations.
 * - `landauFreeField.cuh`  : Computes Landau energy contribution.
 * - `ginzburgField.cuh`    : Computes gradient (Ginzburg) term.
 * - `electrostaticField.cuh`: Handles electric potential / depolarization field contributions.
 * - `elastic_field.cuh`    : Elastic coupling contributions.
 * - `randomTdgl.cuh`       : Handles random noise generation for stochastic TDGL.
 * - `externalField.h`      : Provides time-varying external field inputs.
 * - `coefficientVector.cuh`: Holds material coefficients.
 * - `output.h`             : Output and logging utilities.
 */

#pragma once

 //------------------------------------------------------------------------------
 // Include dependencies
 //------------------------------------------------------------------------------
#include <fstream>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>

#include "Vec3.h"
#include "tdglFunction.cuh"
#include "landauFreeField.cuh"
#include "ginzburgField.cuh"
#include "electrostaticField.cuh"
#include "elastic_field.cuh"
#include "externalField.h"
#include "randomTdgl.cuh"
#include "coefficientVector.cuh"
#include "output.h"

//------------------------------------------------------------------------------
// Function Declaration
//------------------------------------------------------------------------------

/**
 * @brief Computes the total effective TDGL field acting on the polarization.
 *
 * This function aggregates all relevant physical field contributions (Landau, Ginzburg,
 * electrostatic, elastic, and external) and optionally includes stochastic noise.
 * It operates on device vectors for GPU parallelization using Thrust.
 *
 * @tparam T Numeric precision type (e.g., float or double).
 * @param polarization Device vector holding current polarization values (Vec3<T>).
 * @param hFerro Device vector for effective field values (Vec3<T>).
 * @param noiseVector Device vector storing random noise contributions.
 * @param gridSize Total number of grid elements in the simulation.
 * @param Ncz Number of cells along the z-axis (depth).
 * @param currentTime Current simulation time.
 * @param stepCounting Current step counter (used for controlling noise inclusion).
 * @return Device vector of computed TDGL effective fields (Vec3<T>).
 */
template <typename T>
thrust::device_vector<Vec3<T>> calculate_tdgl(
	thrust::device_vector<Vec3<T>>& polarization,    ///< Polarization vector
	thrust::device_vector<Vec3<T>>& hFerro,          ///< Effective field accumulator
	thrust::device_vector<Vec3<T>>& noiseVector,     ///< RNG state or stochastic field
	int gridSize,                                    ///< Total grid size
	int Ncz,                                         ///< Number of cells along z-axis
	T currentTime,                                   ///< Current simulation time
	int stepCounting                                 ///< Step counter for noise timing
);

//------------------------------------------------------------------------------
// Include the CUDA implementation
//------------------------------------------------------------------------------
#include "tdgl.cu"
