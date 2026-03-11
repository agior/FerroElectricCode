/**
 * @file handleAllOutputs.h
 * @brief Centralized output handler for simulation data streams.
 *
 * This header defines the templated `handleAllOutputs()` function,
 * which coordinates all data-writing operations for a given simulation step.
 * It determines which outputs to produce based on user-defined flags and
 * intervals specified in the global `set_output` structure.
 *
 * @details
 * The function manages multiple categories of outputs:
 * - Polarization (spin) snapshots
 * - Local and averaged polarization
 * - Local and averaged energy
 * - Local and averaged effective field
 *
 * All writing operations are delegated to specialized output functions
 * declared in `output.hpp` (e.g., `writeOutput()`, `writeAvgFieldToFile()`, etc.).
 * This ensures modular, maintainable, and efficient data management
 * in large-scale CUDA simulations.
 */

#pragma once

 //------------------------------------------------------------------------------
 // Standard and Thrust includes
 //------------------------------------------------------------------------------
#include <thrust/host_vector.h>
#include <iostream>

//------------------------------------------------------------------------------
// Project includes
//------------------------------------------------------------------------------
#include "Vec3.h"
#include "output.hpp"

//------------------------------------------------------------------------------
// External global variables
//------------------------------------------------------------------------------
extern set_output set_out;

//==============================================================================
//! @brief Handles all enabled simulation outputs based on current configuration.
//!
//! @tparam T Numeric precision type (e.g., float or double).
//!
//! @param counterMagnetization Counter controlling magnetization output intervals.
//! @param counterSnapshot      Counter controlling snapshot output intervals.
//! @param stepCounting         Current simulation step index.
//! @param currentTime          Current simulation time.
//! @param h_polarization       Host vector holding polarization (or magnetization) data.
//! @param h_field              Host vector holding effective field data.
//! @param total_energy         Host vector holding per-cell or averaged energy values.
//!
//! @details
//! This function checks various flags and output intervals defined in the
//! `set_output` configuration structure. When a condition is met, it triggers
//! the corresponding output writing routine. This function ensures that only
//! the requested outputs are produced at the correct simulation steps, reducing
//! I/O overhead during long simulations.
//!
//! @note This routine operates on host-side data, typically copied from device
//! memory prior to being passed into this function.
//==============================================================================
template <typename T>
void handleAllOutputs(
    int& counterMagnetization,
    int& counterSnapshot,
    int stepCounting,
    T currentTime,
    thrust::host_vector<Vec3<T>>& h_polarization,
    thrust::host_vector<Vec3<T>>& h_field,
    thrust::host_vector<T>& total_energy
)
{
    //--------------------------------------------------------------------------
    // Polarization (Spin) Output
    //--------------------------------------------------------------------------
    if (set_out.FLAG_SPIN == 1)
    {
        if (counterMagnetization == set_out.interval_output ||
            counterSnapshot == set_out.interval_polarization)
        {
            writeOutput(
                counterMagnetization,
                counterSnapshot,
                currentTime,
                thrust::raw_pointer_cast(h_polarization.data())
            );
        }
    }

    //--------------------------------------------------------------------------
    // Local Polarization Output
    //--------------------------------------------------------------------------
    if (set_out.FLAG_LOCAL_POLARIZATION == 1 &&
        (stepCounting % set_out.interval_local_polarization == 0))
    {
        writeLocalPolarizationToFile(
            thrust::raw_pointer_cast(h_polarization.data()),
            stepCounting
        );
    }

    //--------------------------------------------------------------------------
    // Average Polarization Output
    //--------------------------------------------------------------------------
    if (set_out.FLAG_AVERAGE_POLARIZATION == 1 &&
        (stepCounting % set_out.interval_average_polarization == 0))
    {
        writeAvgPolarizationToFile(
            currentTime,
            thrust::raw_pointer_cast(h_polarization.data())
        );
    }

    //--------------------------------------------------------------------------
    // Local Energy Output
    //--------------------------------------------------------------------------
    if (set_out.FLAG_LOCAL_ENERGY == 1 &&
        (stepCounting % set_out.interval_local_energy == 0))
    {
        const int total_size = FE_geom.Ncx * FE_geom.Ncy * FE_geom.Ncz * FE_geom.N_d;
        writeEnergyToFile(
            total_energy.data(),
            total_size,
            stepCounting
        );
    }

    //--------------------------------------------------------------------------
    // Average Energy Output
    //--------------------------------------------------------------------------
    if (set_out.FLAG_AVERAGE_ENERGY == 1 &&
        (stepCounting % set_out.interval_average_energy == 0))
    {
        writeAvgEnergyToFile(
            currentTime,
            thrust::raw_pointer_cast(total_energy.data())
        );
    }

    //--------------------------------------------------------------------------
    // Local Field Output
    //--------------------------------------------------------------------------
    if (set_out.FLAG_LOCAL_FIELD == 1 &&
        (stepCounting % set_out.interval_local_field == 0))
    {
        writeLocalFieldToFile(
            thrust::raw_pointer_cast(h_field.data()),
            stepCounting
        );
    }

    //--------------------------------------------------------------------------
    // Average Field Output
    //--------------------------------------------------------------------------
    if (set_out.FLAG_AVERAGE_FIELD == 1 &&
        (stepCounting % set_out.interval_average_field == 0))
    {
        writeAvgFieldToFile(
            currentTime,
            thrust::raw_pointer_cast(h_field.data())
        );
    }
}
