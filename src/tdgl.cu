/**
 * @file calculate_tdgl.cuh
 * @brief Computes the effective TDGL field contributions and polarization time derivative.
 *
 * This module assembles all physical field components contributing to the
 * time-dependent Ginzburg–Landau (TDGL) equation of polarization dynamics.
 * It evaluates Landau, gradient, electrostatic, elastic, and stochastic noise
 * terms, returning the resulting time derivative of polarization (dP/dt).
 *
 * The function operates entirely on CUDA device memory via Thrust transforms.
 */

#pragma once

#include "tdgl.cuh"
#include <thrust/execution_policy.h>
#include <fstream>

 //------------------------------------------------------------------------------
 // External simulation parameter structures
 //------------------------------------------------------------------------------
extern landau_free              landau_free_param;
extern FE_geometry              FE_geom;
extern initial_polarization     initial_pol;
extern field_external           external_field;
extern field_external_normalized external_field_vectors;
extern gradient                 gradient_field_param;
extern elastic                  elastic_field_param;
extern landau_free_final        landau_vectors;
extern gradient_final           gradient_vectors;
extern SET_parameters           conf;
extern set_tensor               conf_tens;

//==============================================================================
//! @brief Compute the time derivative of polarization (dP/dt) from TDGL equation.
//!
//! This function assembles all effective fields acting on the polarization:
//! Landau free energy, gradient, external, electrostatic, elastic, and noise.
//!
//! @tparam T Floating-point precision type (float or double).
//! @param polarization Current polarization field (device vector).
//! @param hFerro Effective field (device vector, modified in place).
//! @param noiseVector Device vector containing random noise (updated if enabled).
//! @param gridSize Total number of grid elements.
//! @param Ncz Number of cells along the z-axis.
//! @param currentTime Current simulation time.
//! @param stepCounting Step counter used for stochastic noise timing.
//!
//! @return Device vector of Vec3<T> representing dP/dt for each grid element.
//==============================================================================
template <typename T>
thrust::device_vector<Vec3<T>> calculate_tdgl(
    thrust::device_vector<Vec3<T>>& polarization,   //!< Polarization vector
    thrust::device_vector<Vec3<T>>& hFerro,         //!< Effective field vector
    thrust::device_vector<Vec3<T>>& noiseVector,    //!< Random noise vector
    int gridSize,                                   //!< Total grid size
    int Ncz,                                        //!< Cells along z-axis
    T currentTime,                                  //!< Current simulation time
    int stepCounting                                //!< Step counter
)
{
    //----------------------------------------------------------------------
    // Allocate device vector for polarization rate of change (dP/dt)
    //----------------------------------------------------------------------
    thrust::device_vector<Vec3<T>> dp_dt(gridSize);
    auto index = thrust::make_counting_iterator(0);

    //----------------------------------------------------------------------
    // 1. Landau free-energy contribution
    //----------------------------------------------------------------------
    if (landau_free_param.FLAG_LANDAU) {
        thrust::transform(
            index,
            index + gridSize,
            hFerro.begin(),
            getLandauField<T>(
                thrust::raw_pointer_cast(polarization.data()),
                thrust::raw_pointer_cast(hFerro.data()),
                thrust::raw_pointer_cast(FE_geom.shape.data()),
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
                Ncz,
                FE_geom.N_d,
                FE_geom.FLAG_STUDY
            )
        );
    }

    //----------------------------------------------------------------------
    // 2. Gradient field contribution
    //----------------------------------------------------------------------
    if (gradient_field_param.FLAG_GRADIENT) {
        thrust::transform(
            index,
            index + gridSize,
            hFerro.begin(),
            gradientField<T>(
                thrust::raw_pointer_cast(FE_geom.shape.data()),
                gradient_field_param.FLAG_BC,
                FE_geom.Ncx,
                FE_geom.Ncy,
                FE_geom.Ncz,
                FE_geom.delta_x,
                FE_geom.delta_y,
                FE_geom.delta_z,
                thrust::raw_pointer_cast(polarization.data()),
                thrust::raw_pointer_cast(hFerro.data()),
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
    // 3. External field contribution (DC, AC, and pulse fields)
    //----------------------------------------------------------------------
    external_field.external_field_storage.resize(gridSize);
    thrust::fill(
        external_field.external_field_storage.begin(),
        external_field.external_field_storage.end(),
        Vec3<T>()
    );

    if (external_field.FLAG_FIELD != 0 ||
        external_field.FLAG_FIELD_AC != 0 ||
        external_field.FLAG_FIELD_PULSE != 0)
    {
        thrust::transform(
            index,
            index + gridSize,
            external_field.external_field_storage.begin(),
            getExternal<T>(
                external_field.FLAG_FIELD,
                external_field.FLAG_FIELD_AC,
                external_field.FLAG_FIELD_PULSE,
                external_field_vectors.vector_field.size(),
                external_field_vectors.vector_Ac_field.size(),
                external_field.d_AC_frequency.size(),
                external_field_vectors.vector_AC_phase.size(),
                external_field.d_pulse_field.size(),
                thrust::raw_pointer_cast(polarization.data()),
                thrust::raw_pointer_cast(landau_free_param.d_alpha1_ref.data()),
                thrust::raw_pointer_cast(external_field_vectors.MagSp.data()),
                thrust::raw_pointer_cast(landau_free_param.d_relaxation_ref.data()),
                thrust::raw_pointer_cast(external_field.external_field_storage.data()),
                thrust::raw_pointer_cast(external_field_vectors.vector_field.data()),
                thrust::raw_pointer_cast(external_field_vectors.vector_Ac_field.data()),
                thrust::raw_pointer_cast(external_field_vectors.vector_AC_phase.data()),
                thrust::raw_pointer_cast(external_field.d_AC_frequency.data()),
                thrust::raw_pointer_cast(FE_geom.shape.data()),
                currentTime,
                thrust::raw_pointer_cast(external_field.d_pulse_time.data()),
                thrust::raw_pointer_cast(external_field.d_pulse_field.data()),
                thrust::raw_pointer_cast(FE_geom.shape.data()),
                external_field.H_theta_pulse,
                external_field.H_phi_pulse,
                gridSize,
                FE_geom.Ncz,
                FE_geom.N_d,
                FE_geom.FLAG_STUDY
            )
        );
    }

    // Add external field to total effective field
    thrust::transform(
        index,
        index + gridSize,
        hFerro.begin(),
        addVectors<T>(
            thrust::raw_pointer_cast(hFerro.data()),
            thrust::raw_pointer_cast(external_field.external_field_storage.data())
        )
    );

    //----------------------------------------------------------------------
    // 4. Electrostatic field contribution
    //----------------------------------------------------------------------
    FE_geom.electrostatic_field.resize(gridSize);
    thrust::fill(
        FE_geom.electrostatic_field.begin(),
        FE_geom.electrostatic_field.end(),
        Vec3<T>()
    );

    if (conf_tens.FLAG_TENSOR) {
        electrostaticField(
            thrust::raw_pointer_cast(polarization.data()),
            thrust::raw_pointer_cast(FE_geom.electrostatic_field.data()),
            conf
        );
    }

    // Add electrostatic field to total effective field
    thrust::transform(
        index,
        index + gridSize,
        hFerro.begin(),
        addVectors<T>(
            thrust::raw_pointer_cast(hFerro.data()),
            thrust::raw_pointer_cast(FE_geom.electrostatic_field.data())
        )
    );

    //----------------------------------------------------------------------
    // 5. Random thermal noise (if enabled)
    //----------------------------------------------------------------------
    if (landau_free_param.FLAG_NOISE == 1 &&
        stepCounting < landau_free_param.steps_noise)
    {
        thrust::transform(
            index,
            index + gridSize,
            noiseVector.begin(),
            tdgl_random<T>(
                landau_free_param.states,
                thrust::raw_pointer_cast(landau_vectors.vector_variance.data()),
                landau_vectors.vector_variance.size(),
                gridSize,
                Ncz
            )
        );
    }
    else if (landau_free_param.FLAG_NOISE == 2) {
        thrust::transform(
            index,
            index + gridSize,
            noiseVector.begin(),
            tdgl_random<T>(
                landau_free_param.states,
                thrust::raw_pointer_cast(landau_vectors.vector_variance.data()),
                landau_vectors.vector_variance.size(),
                gridSize,
                Ncz
            )
        );
    }

    //-----------------------------------------------------------------------
    // 6. Elastic field contribution (optional)
    //-----------------------------------------------------------------------
   
    if (elastic_field_param.FLAG_ELASTIC == 1) {

        int FLAG_OUTPUT_FILES = 0;
        // Call Elastic Field
        compute_elastic_field(
            polarization, hFerro, elastic_field_param.elastic_energy,
            FE_geom.Ncx, FE_geom.Ncy, FE_geom.Ncz,
            FE_geom.delta_x, FE_geom.delta_y, FE_geom.delta_z,
            elastic_field_param.Q11[0], elastic_field_param.Q12[0], elastic_field_param.Q44[0],
            elastic_field_param.C11[0], elastic_field_param.C12[0], elastic_field_param.C44[0],
            FLAG_OUTPUT_FILES,
            elastic_field_param.sigma_xx_ext[0], elastic_field_param.sigma_yy_ext[0], elastic_field_param.sigma_zz_ext[0],
            elastic_field_param.sigma_xy_ext[0], elastic_field_param.sigma_xz_ext[0], elastic_field_param.sigma_yz_ext[0],
            elastic_field_param.FLAG_DISPLACEMENT_FIELD,
            elastic_field_param.FLAG_POLARIZATION_FIELD
        );
    }
    
    //----------------------------------------------------------------------
    // 7. Final TDGL derivative evaluation (dP/dt)
    //----------------------------------------------------------------------
    thrust::transform(
        index,
        index + gridSize,
        dp_dt.begin(),
        tdgl_functor<T>(
            thrust::raw_pointer_cast(hFerro.data()),
            thrust::raw_pointer_cast(noiseVector.data()),
            thrust::raw_pointer_cast(landau_free_param.d_relaxation.data()),
            static_cast<int>(landau_free_param.d_relaxation.size()),
            gridSize,
            Ncz
        )
    );

    //----------------------------------------------------------------------
    // Return computed polarization derivative
    //----------------------------------------------------------------------
    return dp_dt;
}
