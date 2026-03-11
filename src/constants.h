/**
 * @file constants.h
 * @brief Declarations for all configuration, initialization, and setup routines
 *        used throughout the CUDA-based TDGL simulation framework.
 *
 * This header provides function prototypes to:
 *   - Load geometry, material, and field configuration data.
 *   - Initialize electrostatic tensors and gradient terms.
 *   - Configure output and simulation parameters.
 *   - Support both scalar (T) and vector (Vec3<T>) data loading using Thrust device vectors.
 *
 * @note Function implementations are provided in "constants.cuh".
 */

#pragma once
#ifndef CONSTANTS_H
#define CONSTANTS_H

 //------------------------------------------------------------------------------
 // Standard and Thrust includes
 //------------------------------------------------------------------------------
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>

//------------------------------------------------------------------------------
// Project-specific includes
//------------------------------------------------------------------------------
#include "Vec3.h"
#include "parameters.h"
#include "electrostaticTensor.cuh"

//==============================================================================
//! @name Data Input Functions
//! These functions handle reading scalar and vector data into GPU memory.
//==============================================================================

/**
 * @brief Loads scalar values from a text or binary file into a Thrust device vector.
 * @tparam T Numeric type (float, double, etc.)
 * @param filename  Path to input file containing scalar values.
 * @param vec       Thrust device vector to be filled with loaded data.
 */
template <typename T>
void inputValues(std::string& filename, thrust::device_vector<T>& vec);

/**
 * @brief Loads 3-component vector values (Vec3<T>) from a file into a Thrust device vector.
 * @tparam T Numeric type (float, double, etc.)
 * @param filename  Path to input file.
 * @param vec       Thrust device vector to be filled with Vec3<T> entries.
 */
template <typename T>
void inputValuesVec3(const std::string& filename, thrust::device_vector<Vec3<T>>& vec);

/**
 * @brief Loads shape configuration (spatial geometry of the simulation domain).
 * @tparam T Numeric type (float, double, etc.)
 * @param shapeVector Thrust device vector representing shape data.
 */
template <typename T>
void shapeIn(thrust::device_vector<T>& shapeVector);

//==============================================================================
//! @name Polarization Initialization
//! These functions configure the initial polarization field for the simulation.
//==============================================================================

/**
 * @brief Loads the initial polarization vector from an input stream.
 * @param initial_pol Struct containing polarization parameters.
 * @param filei       Input stream for reading data.
 * @param N           Total number of grid points.
 */
void load_initial_polarization(initial_polarization& initial_pol, std::ifstream& filei, int N);

/**
 * @brief Loads initial polarization configuration from predefined input file.
 * @param initial_pol Struct containing polarization parameters.
 * @param N           Total number of grid points.
 */
void load_initialP_configuration(initial_polarization& initial_pol, int N);

//==============================================================================
//! @name Geometry and Simulation Domain
//! Functions for reading and configuring geometry information.
//==============================================================================

/**
 * @brief Loads geometry information (domain size, spacing, time parameters) from an input stream.
 * @param FE_geom Struct holding finite element geometry parameters.
 * @param filei   Input stream.
 */
void load_geometry(FE_geometry& FE_geom, std::ifstream& filei);

/**
 * @brief Loads geometry configuration from a predefined input file.
 * @param FE_geom Struct holding finite element geometry parameters.
 */
void load_geometry_configuration(FE_geometry& FE_geom);

//==============================================================================
//! @name Landau Free Energy Parameters
//! Functions for loading and configuring Landau energy coefficients.
//==============================================================================

/**
 * @brief Loads Landau free energy parameters from an input stream.
 * @param landau_free_param Struct holding Landau coefficients.
 * @param filei             Input stream.
 * @param N                 Number of coefficients or grid points.
 */
void load_Landau_free(landau_free& landau_free_param, std::ifstream& filei, int N);


/**
 * @brief Loads Landau free energy configuration from predefined input file.
 * @param landau_free_param Struct holding Landau coefficients.
 * @param N                 Number of coefficients or grid points.
 */
void load_Landau_configuration(landau_free& landau_free_param, int N);

/**
 * @brief Loads Elastic energy parameters from an input stream.
 * @param elastic_field_param Struct holding Landau coefficients.
 * @param filei             Input stream.
 * @param N                 Number of coefficients or grid points.
 */
void load_elastic(elastic& elastic_field_param, std::ifstream& filei, int N);

/**
 * @brief Loads Elastic energy configuration from predefined input file.
 * @param elastic_field_param Struct holding Landau coefficients.
 * @param N                 Number of coefficients or grid points.
 */
void load_elastic_configuration(elastic& elastic_field_param, int N);

//==============================================================================
//! @name External Field Configuration
//! Handles electric/magnetic field setup for external influences.
//==============================================================================

/**
 * @brief Loads external electric field configuration from an input stream.
 * @param external_field Struct holding external field parameters.
 * @param filei          Input stream.
 * @param N              Number of entries or time steps.
 */
void load_external(field_external& external_field, std::ifstream& filei, int N);

/**
 * @brief Loads external field configuration from predefined input file.
 * @param external_field Struct holding external field parameters.
 * @param N              Number of entries or time steps.
 */
void load_external_configuration(field_external& external_field, int N);

//==============================================================================
//! @name Gradient Energy Terms
//! Functions related to gradient field coefficients and tensor fields.
//==============================================================================

/**
 * @brief Loads gradient parameters (used in gradient energy terms) from an input stream.
 * @param filei Input stream.
 * @param N     Number of gradient coefficients.
 * @return Struct of type `gradient` containing loaded parameters.
 */
gradient load_gradient_field(std::ifstream& filei, int N);

/**
 * @brief Loads gradient configuration from a file.
 * @param filename Path to configuration file.
 * @param N        Number of gradient coefficients.
 * @return Struct of type `gradient` containing loaded parameters.
 */
gradient load_Gradient_configuration(const char* filename, int N);

//==============================================================================
//! @name Electrostatic Tensor and Demagnetization Setup
//! Functions for setting up tensor fields for electrostatic calculations.
//==============================================================================

/**
 * @brief Loads tensor configuration for electrostatics.
 * @param filename Path to tensor configuration file.
 * @return Struct of type `set_tensor` describing tensor setup.
 */
set_tensor conf_tensor(const char* filename);

/**
 * @brief Number of threads per CUDA block (externally defined elsewhere).
 */
extern int threadsPerBlock;

/**
 * @brief Configures demagnetization (electrostatic) tensors for simulation.
 * @param Mcx, Mcy, Mcz Dimensions of the computational domain.
 * @param delta          Spatial discretization step.
 * @param conf_tens      Tensor configuration parameters.
 * @return Struct `set_DEMAG` describing demagnetization setup.
 */
set_DEMAG set_demagnetization(int Mcx, int Mcy, int Mcz, Type_var delta, set_tensor conf_tens);

/**
 * @brief Sets overall simulation parameters based on geometry and tensor setup for the electrostatic field.
 * @param FE_geom Finite element geometry parameters.
 * @param conf_tens Tensor configuration parameters.
 * @return Struct `SET_parameters` with combined configuration.
 */
SET_parameters set_configuration(FE_geometry FE_geom, set_tensor conf_tens);

//==============================================================================
//! @name Output Configuration
//! Functions for loading and managing simulation output parameters.
//==============================================================================

/**
 * @brief Loads output configuration from an input stream.
 * @param set_out Struct holding output control parameters.
 * @param filei   Input stream.
 */
void load_output(set_output& set_out, std::ifstream& filei);

/**
 * @brief Loads output configuration from predefined input file.
 * @param set_out Struct holding output control parameters.
 */
void load_output_configuration(set_output& set_out);

//==============================================================================
//! @name Implementation Include
//! Includes all function definitions declared above.
//==============================================================================
#include "constants.cuh"

#endif // CONSTANTS_H
