/**
 * @file externalField.cuh
 * @brief Defines the functor `getExternal` for computing the external effective field
 *        contribution in the TDGL simulation.
 *
 * This file provides the CUDA device and host implementation of the external field
 * operator, which combines static, alternating (AC), and pulse field contributions
 * to compute the total effective field at each grid site.
 */

#ifndef EXTERNALFIELD_CUH
#define EXTERNALFIELD_CUH

#define _USE_MATH_DEFINES
#include <cmath>
#include "externalField.h"

 //==============================================================================
 //! @brief Functor operator to compute the effective external field at a grid point.
 //!
 //! This operator combines multiple contributions:
 //! - External static fields (`externalF_flag`)
 //! - Alternating current (AC) fields (`AcField_flag`)
 //! - Pulse fields (`pulseField_flag`)
 //!
 //! @tparam T Numeric type (e.g., float or double).
 //! @param idx Index of the grid point.
 //! @return Updated effective field vector (Vec3<T>) for the given grid point.
 //==============================================================================
template <typename T>
__host__ __device__
Vec3<T> getExternal<T>::operator()(int idx) const
{
    //--------------------------------------------------------------------------
    // Initialize effective field from the ferroelectric field array
    //--------------------------------------------------------------------------
    Vec3<T> heff = hFerro[idx];
    int sim_id = idx / (gridSize / N_d);  // Simulation domain ID

    //--------------------------------------------------------------------------
    // Check if current index belongs to an active region (shapeVector == 1)
    //--------------------------------------------------------------------------
    if (shapeVector[idx] == 1) {

        //----------------------------------------------------------------------
        // (1) Static external field contribution
        //----------------------------------------------------------------------
        if (externalF_flag != 0) {
            Vec3<T> externalField;

            if (externalFVectorSize == 1) {
                externalField = externalF[0];
            }
            else if (externalFVectorSize == Ncz) {
                int z = idx / (gridSize / Ncz);
                externalField = externalF[z];
            }
            else if (externalFVectorSize == gridSize) {
                externalField = externalF[idx];
            }
            else if (externalFVectorSize == N_d && externalF_flag == 4 && flag_study == 1) {
                externalField = externalF[sim_id];
            }
            else {
                externalField = externalF[0];
            }

            // Subtract external field from total effective field
            heff -= externalField;
        }

        //----------------------------------------------------------------------
        // (2) Alternating current (AC) field contribution
        //----------------------------------------------------------------------
        if (AcField_flag != 0) {
            Vec3<T> AcField;   // Amplitude of AC field
            Vec3<T> fi;        // Phase vector
            T shape = shapeField[idx];
            T freq;             // Frequency value

            //-------------------------------
            // Select AC field amplitude
            //-------------------------------
            if (hAcVectorSize == 1) {
                AcField = h_Ac[0];
            }
            else if (hAcVectorSize == Ncz) {
                int z = idx / (gridSize / Ncz);
                AcField = h_Ac[z];
            }
            else if (hAcVectorSize == gridSize) {
                AcField = h_Ac[idx];
            }
            else if (hAcVectorSize == N_d && flag_study == 1) {
                AcField = h_Ac[sim_id];
            }
            else {
                AcField = h_Ac[0];
            }

            //-------------------------------
            // Select phase vector
            //-------------------------------
            if (phaseVectorSize == 1) {
                fi = phase_fi[0];
            }
            else if (phaseVectorSize == Ncz) {
                int z = idx / (gridSize / Ncz);
                fi = phase_fi[z];
            }
            else if (phaseVectorSize == gridSize) {
                fi = phase_fi[idx];
            }
            else if (phaseVectorSize == N_d && flag_study == 1) {
                fi = phase_fi[sim_id];
            }
            else {
                fi = phase_fi[0];
            }

            //-------------------------------
            // Select frequency
            //-------------------------------
            if (fVectorSize == 1) {
                freq = frequency[0];
            }
            else if (fVectorSize == Ncz) {
                int z = idx / (gridSize / Ncz);
                freq = frequency[z];
            }
            else if (fVectorSize == gridSize) {
                freq = frequency[idx];
            }
            else if (fVectorSize == N_d && flag_study == 1) {
                freq = frequency[sim_id];
            }
            else {
                freq = frequency[0];
            }

            //-------------------------------
            // Apply sinusoidal AC field
            //-------------------------------
            T two_pi_f = 2.0 * PI * freq;
            heff.x -= AcField.x * sin(two_pi_f * time + fi.x);
            heff.y -= AcField.y * sin(two_pi_f * time + fi.y);
            heff.z -= AcField.z * sin(two_pi_f * time + fi.z);
        }

        //----------------------------------------------------------------------
        // (3) Pulse field contribution
        //----------------------------------------------------------------------
        if (pulseField_flag != 0) {
            T alpha1_val = alpha1_ref[0];
            T spontaneousP_val = spontaneousP[0];
            T relaxation_val = relaxation_ref[0];

            // Convert polar and azimuthal angles from degrees to radians
            T theta_pulse_val = theta_pulse * (PI / 180.0);
            T phi_pulse_val = phi_pulse * (PI / 180.0);

            // Loop through pulse intervals and interpolate the field
            for (int i = 0; i < pulseFieldVectorSize; i++) {
                if (time > pulse_time[i] && time <= pulse_time[i + 1]) {
                    T field1 = pulse_field[i];
                    T field2 = pulse_field[i + 1];
                    T fieldNormalizer = abs(alpha1_val) * spontaneousP_val;
                    field1 /= fieldNormalizer;
                    field2 /= fieldNormalizer;

                    T time1 = pulse_time[i];
                    T time2 = pulse_time[i + 1];
                    T timeNormalizer = abs(alpha1_val) * relaxation_val;
                    time1 *= timeNormalizer;
                    time2 *= timeNormalizer;

                    // Linear interpolation between pulse values
                    T h_pulse = field1 + ((field2 - field1) / (time2 - time1)) * (time - time1);

                    // Apply directional components of the pulse
                    heff.x -= h_pulse * sin(theta_pulse_val) * cos(phi_pulse_val);
                    heff.y -= h_pulse * sin(theta_pulse_val) * sin(phi_pulse_val);
                    heff.z -= h_pulse * cos(theta_pulse_val);
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // Return final effective field for this grid element
    //--------------------------------------------------------------------------
    return heff;
}

#endif // EXTERNALFIELD_CUH
