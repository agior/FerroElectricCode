/**
 * @file output_utils.h
 * @brief Declares output and logging utilities for polarization and field data.
 *
 * This header defines file I/O routines for exporting simulation results,
 * including magnetization, polarization, energy, and field data. It also
 * provides error reporting and OOMMF-compatible file output functions.
 *
 * @details
 * These routines handle structured output for simulations based on the
 * Landau–Ginzburg–Devonshire (LGD) or Time-Dependent Ginzburg–Landau (TDGL)
 * frameworks. Output includes both global (averaged) and local field data,
 * enabling detailed post-processing and visualization.
 */

#pragma once

 //------------------------------------------------------------------------------
 // Standard library includes
 //------------------------------------------------------------------------------
#include <string>
#include <fstream>

//------------------------------------------------------------------------------
// Project includes
//------------------------------------------------------------------------------
#include "Vec3.cuh"
#include "parameters.h"

//------------------------------------------------------------------------------
// Thrust includes
//------------------------------------------------------------------------------
#include <thrust/host_vector.h>

//==============================================================================
//! @brief Resets and prepares output directories for a new simulation run.
//!
//! @details
//! Removes existing data folders and recreates a clean directory structure
//! for current simulation outputs (polarization, field, and energy data).
//==============================================================================
void resetOutputFolders();

//==============================================================================
//! @brief Reports a fatal settings error and safely terminates the simulation.
//!
//! @param message Descriptive error message to display or log.
//!
//! @details
//! Logs the error message to both console and file output streams. Used to
//! halt execution when invalid parameter combinations or configuration issues
//! are detected during initialization.
//==============================================================================
void reportFatalSettingsError(const std::string& message);

//==============================================================================
//! @brief Writes magnetization data to an OOMMF-compatible `.omf` file.
//!
//! @param magnetization Pointer to the array of magnetization vectors (`Vec3<Type_var>`).
//! @param sim_id        Simulation identifier (used in file naming).
//! @param local_counter Snapshot index or output counter.
//!
//! @details
//! Generates structured output suitable for visualization in OOMMF (Object
//! Oriented MicroMagnetic Framework). File naming follows internal conventions
//! for organized multi-run management.
//==============================================================================
void writeOommfFile(const Vec3<Type_var>* magnetization, int sim_id, int local_counter);

//==============================================================================
//! @brief Generates a standardized OOMMF output filename.
//!
//! @param sim_id        Simulation identifier.
//! @param local_counter Output index.
//! @return Constructed filename string.
//!
//! @details
//! Produces filenames based on simulation IDs and snapshot counters to
//! maintain consistent versioned output organization.
//==============================================================================
std::string generateOommfFileName(int sim_id, int local_counter);

//==============================================================================
//! @brief Saves magnetization data snapshot to a file.
//!
//! @param magnetization Pointer to device/host magnetization data array.
//! @param sim_id        Simulation identifier for file tagging.
//==============================================================================
void save_m(const Vec3<Type_var>* magnetization, int sim_id);

//==============================================================================
//! @brief Writes general simulation output, including magnetization and snapshots.
//!
//! @param counterMagnetization Counter for magnetization outputs.
//! @param counterSnapshot      Counter for snapshot files.
//! @param time                 Current simulation time.
//! @param magnetization        Pointer to magnetization field data.
//!
//! @details
//! Invoked periodically during the time integration loop to record
//! magnetization states and metadata for analysis or restart.
//==============================================================================
void writeOutput(int& counterMagnetization, int& counterSnapshot, Type_var time, const Vec3<Type_var>* magnetization);

//==============================================================================
//! @brief Writes per-cell energy data to an output file.
//!
//! @param energy     Pointer to the computed energy array.
//! @param size       Number of elements in the energy array.
//! @param stepIndex  Current simulation step index.
//==============================================================================
void writeEnergyToFile(const Type_var* energy, int size, int stepIndex);

//==============================================================================
//! @brief Writes averaged polarization vector to file.
//!
//! @param time          Simulation time at which average is computed.
//! @param magnetization Pointer to polarization (or magnetization) data.
//==============================================================================
void writeAvgPolarizationToFile(Type_var time, const Vec3<Type_var>* magnetization);

//==============================================================================
//! @brief Writes averaged total energy to file.
//!
//! @param time   Current simulation time.
//! @param energy Pointer to total energy value(s).
//==============================================================================
void writeAvgEnergyToFile(Type_var time, const Type_var* energy);

//==============================================================================
//! @brief Writes averaged field values to file.
//!
//! @param time  Current simulation time.
//! @param field Pointer to averaged field data (e.g., effective field).
//==============================================================================
void writeAvgFieldToFile(Type_var time, const Vec3<Type_var>* field);

//==============================================================================
//! @brief Writes local polarization values (spatially resolved) to file.
//!
//! @param polarization Pointer to polarization vector field array.
//! @param stepIndex    Current simulation step index.
//==============================================================================
void writeLocalPolarizationToFile(const Vec3<Type_var>* polarization, int stepIndex);

//==============================================================================
//! @brief Writes local field data to file.
//!
//! @param localField Pointer to local effective field array.
//! @param stepIndex  Simulation step index for output naming.
//==============================================================================
void writeLocalFieldToFile(const Vec3<Type_var>* localField, int stepIndex);
