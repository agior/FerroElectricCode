/**
 * @file externalField.h
 * @brief Declares CUDA-compatible functor for computing external field contributions in TDGL simulations.
 *
 * This header defines the template functor `getExternal<T>`, which calculates the total
 * external field acting on the polarization or magnetization vectors.
 * It incorporates static, AC, and pulse field components, each of which can be
 * selectively enabled through control flags.
 *
 * The corresponding implementation is provided in **externalField.cuh**.
 */

#ifndef EXTERNALFIELD_H
#define EXTERNALFIELD_H

#include "cuda_runtime.h"
#include "parameters.h"
#include "Vec3.h"
#include <thrust/execution_policy.h>

 //------------------------------------------------------------------------------
 // Struct: getExternal
 //------------------------------------------------------------------------------
 //! @brief CUDA-compatible functor to compute external field contributions.
 //!
 //! @tparam T Numeric type (e.g., float, double).
 //!
 //! The functor aggregates contributions from different external sources:
 //! - Static external field
 //! - AC (time-varying sinusoidal) field
 //! - Pulse field (transient excitation)
 //!
 //! Each contribution can be individually toggled through its corresponding flag.
 //! The operator() computes the net external field vector at a specific grid index.
template <typename T>
struct getExternal
{
    //--------------------------------------------------------------------------
    // Field control flags
    //--------------------------------------------------------------------------
    int externalF_flag;    //!< Enables/disables static external field
    int AcField_flag;      //!< Enables/disables AC field contribution
    int pulseField_flag;   //!< Enables/disables pulse field contribution

    //--------------------------------------------------------------------------
    // Vector sizes for field arrays
    //--------------------------------------------------------------------------
    int externalFVectorSize;   //!< Size of external field vector array
    int hAcVectorSize;         //!< Size of AC field vector array
    int fVectorSize;           //!< Size of frequency vector
    int phaseVectorSize;       //!< Size of phase vector
    int pulseFieldVectorSize;  //!< Size of pulse field vector

    //--------------------------------------------------------------------------
    // Pointers to simulation field data
    //--------------------------------------------------------------------------
    Vec3<T>* polarization;     //!< Polarization or magnetization array
    T* alpha1_ref;             //!< Material parameter (Landau coefficient)
    T* spontaneousP;           //!< Spontaneous polarization
    T* relaxation_ref;         //!< Relaxation coefficient
    Vec3<T>* hFerro;           //!< Effective field array
    Vec3<T>* externalF;        //!< External static field array
    Vec3<T>* h_Ac;             //!< AC field vector array
    Vec3<T>* phase_fi;         //!< Phase vector (for AC fields)

    //--------------------------------------------------------------------------
    // Scalars and arrays for field parameters
    //--------------------------------------------------------------------------
    T* frequency;              //!< Frequency array for AC fields
    int* shapeField;           //!< Shape configuration for AC/pulse field
    T time;                    //!< Current simulation time
    T* pulse_time;             //!< Pulse timing array
    T* pulse_field;            //!< Pulse field magnitude array
    int* shapeVector;          //!< Shape vector for pulse orientation

    //--------------------------------------------------------------------------
    // Angular parameters (pulse field orientation)
    //--------------------------------------------------------------------------
    T theta_pulse;             //!< Polar angle of pulse field
    T phi_pulse;               //!< Azimuthal angle of pulse field

    //--------------------------------------------------------------------------
    // Grid and configuration parameters
    //--------------------------------------------------------------------------
    int gridSize;              //!< Total grid size
    int Ncz;                   //!< Number of cells in z direction
    int N_d;                   //!< Auxiliary parameter (domain count or similar)
    int flag_study;            //!< Flag for study configuration (runtime mode control)

    //--------------------------------------------------------------------------
    // Constructor
    //--------------------------------------------------------------------------
    //! @brief Initializes the getExternal functor with simulation parameters and pointers.
    //!
    //! @param externalF_flag_ Flag to enable/disable static external field.
    //! @param AcField_flag_ Flag to enable/disable AC field.
    //! @param pulseField_flag_ Flag to enable/disable pulse field.
    //! @param externalFVectorSize_ Size of external field vector.
    //! @param hAcVectorSize_ Size of AC field vector.
    //! @param fVectorSize_ Size of frequency array.
    //! @param phaseVectorSize_ Size of phase vector.
    //! @param pulseFieldVectorSize_ Size of pulse field array.
    //! @param polarization_ Pointer to polarization/magnetization data.
    //! @param alpha1_ref_ Pointer to Landau coefficient array.
    //! @param spontaneousP_ Pointer to spontaneous polarization array.
    //! @param relaxation_ref_ Pointer to relaxation coefficient array.
    //! @param hFerro_ Pointer to effective field array.
    //! @param externalF_ Pointer to external field array.
    //! @param h_Ac_ Pointer to AC field vector array.
    //! @param phase_fi_ Pointer to phase vector array.
    //! @param frequency_ Pointer to frequency array.
    //! @param shapeField_ Pointer to field shape array.
    //! @param time_ Current simulation time.
    //! @param pulse_time_ Pointer to pulse timing array.
    //! @param pulse_field_ Pointer to pulse field array.
    //! @param shapeVector_ Pointer to shape orientation array.
    //! @param theta_pulse_ Polar angle of pulse field.
    //! @param phi_pulse_ Azimuthal angle of pulse field.
    //! @param gridSize_ Total grid size.
    //! @param Ncz_ Number of cells along z.
    //! @param N_d_ Auxiliary parameter (domain count).
    //! @param flag_study_ Configuration flag for study mode.
    getExternal(
        int  externalF_flag_,
        int  AcField_flag_,
        int  pulseField_flag_,
        int  externalFVectorSize_,
        int  hAcVectorSize_,
        int  fVectorSize_,
        int  phaseVectorSize_,
        int  pulseFieldVectorSize_,
        Vec3<T>* polarization_,
        T* alpha1_ref_,
        T* spontaneousP_,
        T* relaxation_ref_,
        Vec3<T>* hFerro_,
        Vec3<T>* externalF_,
        Vec3<T>* h_Ac_,
        Vec3<T>* phase_fi_,
        T* frequency_,
        int* shapeField_,
        T   time_,
        T* pulse_time_,
        T* pulse_field_,
        int* shapeVector_,
        T   theta_pulse_,
        T   phi_pulse_,
        int gridSize_,
        int Ncz_,
        int N_d_,
        int flag_study_
    )
        : externalF_flag(externalF_flag_),
        AcField_flag(AcField_flag_),
        pulseField_flag(pulseField_flag_),
        externalFVectorSize(externalFVectorSize_),
        hAcVectorSize(hAcVectorSize_),
        fVectorSize(fVectorSize_),
        phaseVectorSize(phaseVectorSize_),
        pulseFieldVectorSize(pulseFieldVectorSize_),
        polarization(polarization_),
        alpha1_ref(alpha1_ref_),
        spontaneousP(spontaneousP_),
        relaxation_ref(relaxation_ref_),
        hFerro(hFerro_),
        externalF(externalF_),
        h_Ac(h_Ac_),
        phase_fi(phase_fi_),
        frequency(frequency_),
        shapeField(shapeField_),
        time(time_),
        pulse_time(pulse_time_),
        pulse_field(pulse_field_),
        shapeVector(shapeVector_),
        theta_pulse(theta_pulse_),
        phi_pulse(phi_pulse_),
        gridSize(gridSize_),
        Ncz(Ncz_),
        N_d(N_d_),
        flag_study(flag_study_)
    {
    }

    //--------------------------------------------------------------------------
    // Operator
    //--------------------------------------------------------------------------
    //! @brief Compute the external field contribution at a specific grid index.
    //!
    //! @param idx Grid index to evaluate.
    //! @return Net external field vector (Vec3<T>) at the given position.
    __host__ __device__
        Vec3<T> operator()(int idx) const;
};

//------------------------------------------------------------------------------
// Implementation include
//------------------------------------------------------------------------------
#include "externalField.cuh"

#endif // EXTERNALFIELD_H
