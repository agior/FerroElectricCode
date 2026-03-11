/**
 * @file landau_energy.cuh
 * @brief Computes total free energy contributions (Landau + gradient + electrostatic + external fields).
 *
 * This routine evaluates the spatial energy density contributions for the
 * time-dependent Ginzburg–Landau (TDGL) ferroelectric model. The returned
 * device vector contains the per-cell free energy density.
 *
 * The following energy terms are included conditionally:
 *  - Landau polynomial free energy (FLAG_LANDAU)
 *  - Gradient energy (FLAG_GRADIENT)
 *  - Electrostatic energy (FLAG_TENSOR)
 *  - External field coupling (FLAG_FIELD, FLAG_FIELD_AC, FLAG_FIELD_PULSE)
 */

#pragma once

#include "landauFreeEnergy.cuh"
#include "coefficientVector.cuh"
#include "ginzburgEnergy.cuh"
#include "tdgl.cu"
#include "output.h"

 //------------------------------------------------------------------------------
 // External global structures shared across the simulation modules
 //------------------------------------------------------------------------------
extern landau_free               landau_free_param;
extern FE_geometry               FE_geom;
extern initial_polarization      initial_pol;
extern field_external            external_field;
extern field_external_normalized external_field_vectors;
extern gradient                  gradient_field_param;
extern elastic                   elastic_field_param;
extern landau_free_final         landau_vectors;
extern gradient_final            gradient_vectors;
extern SET_parameters            conf;
extern set_tensor                conf_tens;

//==============================================================================
//! @brief Compute the total local energy density per grid cell.
//!
//! Evaluates all relevant energetic contributions to the system:
//!  - Landau free energy (local polynomial expansion)
//!  - Gradient (Ginzburg) energy (spatial derivatives of polarization)
//!  - Electrostatic coupling (P · E)
//!  - External field coupling (P · H_ext)
//!
//! @tparam T Floating-point precision type (float or double).
//! @param polarization Polarization field (device vector of Vec3<T>).
//! @return Device vector of scalar energy densities (T) per grid point.
//==============================================================================
template <typename T>
thrust::device_vector<T> landau_energy(
    thrust::device_vector<Vec3<T>> polarization //!< Polarization field (device)
)
{
    //----------------------------------------------------------------------
    // 1. Initialize energy vector and grid parameters
    //----------------------------------------------------------------------
    int gridSize = FE_geom.Ncx * FE_geom.Ncy * FE_geom.Ncz * FE_geom.N_d;

    thrust::device_vector<T> energyVector(gridSize);
    thrust::fill(energyVector.begin(), energyVector.end(), static_cast<T>(0));

    auto index = thrust::make_counting_iterator(0);

    //----------------------------------------------------------------------
    // 2. Apply spontaneous polarization scaling (normalization step)
    //----------------------------------------------------------------------
    thrust::host_vector<Vec3<T>> spontaneous_polarization =
        landau_free_param.d_spontaneousp_ref;

    T spontaneous_value = spontaneous_polarization[0].z;

    thrust::device_vector<Vec3<T>> polarization_copy(gridSize);
    thrust::transform(
        index,
        index + gridSize,
        polarization_copy.begin(),
        scalarMultiplication<T>(
            spontaneous_value,
            thrust::raw_pointer_cast(polarization.data())
        )
    );

    //----------------------------------------------------------------------
    // 3. Landau free energy contribution
    //----------------------------------------------------------------------
    if (landau_free_param.FLAG_LANDAU == 1)
    {
        thrust::transform(
            index,
            index + gridSize,
            energyVector.begin(),
            landauFreeEnergy<Type_var>(  // NOTE: Type_var matches simulation typedef
                thrust::raw_pointer_cast(polarization.data()),
                thrust::raw_pointer_cast(landau_vectors.vector_alpha1.data()),
                thrust::raw_pointer_cast(landau_vectors.vector_alpha2.data()),
                thrust::raw_pointer_cast(landau_vectors.vector_alpha3.data()),
                thrust::raw_pointer_cast(landau_vectors.vector_alpha4.data()),
                thrust::raw_pointer_cast(landau_vectors.vector_alpha5.data()),
                thrust::raw_pointer_cast(landau_vectors.vector_alpha6.data()),
                thrust::raw_pointer_cast(landau_free_param.uni_anistropy_z.data()),
                landau_vectors.vector_alpha1.size(),
                landau_vectors.vector_alpha2.size(),
                landau_vectors.vector_alpha3.size(),
                landau_vectors.vector_alpha4.size(),
                landau_vectors.vector_alpha5.size(),
                landau_vectors.vector_alpha6.size(),
                landau_free_param.uni_anistropy_z.size(),
                gridSize,
                FE_geom.Ncz,
                FE_geom.N_d,
                FE_geom.FLAG_STUDY
            )
        );
    }

    //----------------------------------------------------------------------
    // 4. Gradient (Ginzburg) energy contribution
    //----------------------------------------------------------------------
    if (gradient_field_param.FLAG_GRADIENT == 1)
    {
        thrust::transform(
            index,
            index + gridSize,
            energyVector.begin(),
            gradientEnergy<T>(
                thrust::raw_pointer_cast(FE_geom.shape.data()),
                gradient_field_param.FLAG_BC,
                FE_geom.Ncx,
                FE_geom.Ncy,
                FE_geom.Ncz,
                FE_geom.delta_x,
                FE_geom.delta_y,
                FE_geom.delta_z,
                thrust::raw_pointer_cast(polarization.data()),
                thrust::raw_pointer_cast(energyVector.data()),
                1, 1,
                thrust::raw_pointer_cast(gradient_vectors.vector_G1.data()),
                thrust::raw_pointer_cast(gradient_vectors.vector_G2.data()),
                thrust::raw_pointer_cast(gradient_vectors.vector_G3.data()),
                thrust::raw_pointer_cast(gradient_vectors.vector_G4.data()),
                gradient_field_param.FLAG_G1,
                gradient_field_param.FLAG_G2,
                gradient_field_param.FLAG_G3,
                gradient_field_param.FLAG_G4
            )
        );
    }

    //----------------------------------------------------------------------
    // 5. Electrostatic energy contribution (if enabled)
    //----------------------------------------------------------------------
    thrust::device_vector<Vec3<T>> electrostatic = FE_geom.electrostatic_field;
    thrust::device_vector<T> electro_energy(gridSize);

    if (conf_tens.FLAG_TENSOR == 2)
    {
        thrust::transform(
            index,
            index + gridSize,
            electro_energy.begin(),
            multiplyVectors<T>(
                thrust::raw_pointer_cast(electrostatic.data()),
                thrust::raw_pointer_cast(polarization.data())
            )
        );

        // Add electrostatic contribution to total energy
        thrust::transform(
            electro_energy.begin(),
            electro_energy.end(),
            energyVector.begin(),
            energyVector.begin(),
            thrust::plus<T>()
        );
    }

    //----------------------------------------------------------------------
    // 6. External field energy contribution (if field flags are active)
    //----------------------------------------------------------------------
    if (external_field.FLAG_FIELD == 1 ||
        external_field.FLAG_FIELD_AC == 1 ||
        external_field.FLAG_FIELD_PULSE == 1)
    {
        thrust::device_vector<Vec3<T>> externalFieldVec =
            external_field.external_field_storage;

        thrust::device_vector<T> external_energy(gridSize);

        // Compute energy from P · H_ext
        thrust::transform(
            index,
            index + gridSize,
            external_energy.begin(),
            multiplyVectors<T>(
                thrust::raw_pointer_cast(externalFieldVec.data()),
                thrust::raw_pointer_cast(polarization.data())
            )
        );

        // Accumulate into total energy
        thrust::transform(
            external_energy.begin(),
            external_energy.end(),
            energyVector.begin(),
            energyVector.begin(),
            thrust::plus<T>()
        );
    }

    //----------------------------------------------------------------------
    // 7. Return final per-cell energy density
    //----------------------------------------------------------------------
    return energyVector;
}
