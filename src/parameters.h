/**
 * @file parameters.h
 * @brief Defines core simulation parameters, constants, and data structures for
 *        the TDGL polarization dynamics framework.
 *
 * This header centralizes physical constants, GPU configuration parameters,
 * structure definitions for Landau free energy, external fields, gradient terms,
 * and numerical integration control.
 */

#ifndef _PARAMETERS_H
#define _PARAMETERS_H

 //==============================================================================
 // Includes
 //==============================================================================
#include "Vec3.h"

// Optional workaround for IntelliSense, if needed
#ifdef __INTELLISENSE___
    // Add IntelliSense-specific macros or stubs here if required
#endif

// CUDA and standard headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <algorithm>
#include <cufft.h>
#include <cuda.h>
#include <cuComplex.h>
#include <curand.h>                   
#include <curand_kernel.h>            // Random number generation on GPU

//==============================================================================
// General configuration
//==============================================================================
#define N_BUFFER 1000                  //!< Buffer size used when reading files
const int MAX_GPU_COUNT = 32;          //!< Maximum number of GPUs supported

//==============================================================================
// Type definitions
//==============================================================================
typedef double Type_var;               //!< Base floating-point precision type
typedef double3 Type_var3;             //!< 3-component vector (double precision)
typedef cuDoubleComplex cu_Tens_Type;  //!< CUDA double-precision complex tensor
typedef long double DT_double;         //!< Extended precision floating-point
typedef cufftDoubleComplex HD_Complex_Type; //!< FFT complex data type

//==============================================================================
// FFT configuration flags
//==============================================================================
#define SPLITTED_FFT3D 0     //!< 1 = Perform 1D FFT, 0 = Perform full 3D FFT
#define MEMORY 1             //!< 1 = Print memory usage, 0 = Disable memory logs
#define FLAG_AMP 0           //!< 1 = Load amplitude from file, 0 = Disable

//==============================================================================
// Constant condition handling macros
//==============================================================================
#define ZERO 0
#define CONST_CONST 1
#define VETT_CONST 2
#define CONST_VETT 3
#define VETT_VETT 4

#define CONST_CONST_CONST 1
#define CONST_CONST_VETT 2
#define CONST_VETT_CONST 3
#define CONST_VETT_VETT 4
#define VETT_CONST_CONST 5
#define VETT_CONST_VETT 6
#define VETT_VETT_CONST 7
#define VETT_VETT_VETT 8

#define ADP 0 // Placeholder macro, likely related to adaptive integration

//==============================================================================
// Physical constants
//==============================================================================
#define LANDE_G 2
#define PLANCK_RED 1.054571817e-34      //!< Reduced Planck constant (J·s)
#define BOHRM 9.274e-24                 //!< Bohr magneton (J/T)
#define carica 1.6022e-19               //!< Elementary charge (C)
#define GAMMA0 2.21e5                   //!< Gyromagnetic ratio
#define GAMMA1 2.5320e5                 //!< Alternative gyromagnetic constant
#define PI 3.1415926535897932           //!< Mathematical constant π
#define MU 1.256637061435917e-06        //!< Magnetic permeability of free space

//==============================================================================
// CUDA error checking utilities
//==============================================================================
#define check(cudacall) { int err=cudacall; if (err!=cudaSuccess) std::cout<<"CUDA ERROR "<<err<<" at line "<<__LINE__<<"'s "<<#cudacall<<"\n"; }
#define checkAllocation(ptr) { if(ptr == nullptr) std::cout << "Allocation failed at line " << __LINE__ << "\n"; }

//==============================================================================
// Landau free energy parameter structures
//==============================================================================

//! @brief Stores raw (unprocessed) Landau free energy parameters and GPU vectors.
typedef struct {
    int FLAG_LANDAU;
    int FLAG_ALPHA0;
    int FLAG_ALPHA2;
    int FLAG_ALPHA3;
    int FLAG_ALPHA4;
    int FLAG_ALPHA5;
    int FLAG_ALPHA6;
    int FLAG_TEMPERATURE;
    int FLAG_TEMPERATURE_NOISE;
    int FLAG_TRANSITION_TEMP;
    int FLAG_RELAXATION;
    int FLAG_NOISE;
    int steps_noise;
    int FLAG_SUSCIPTIBILITY_WEIGHTS;
    int FLAG_UNI_ANIS_Z;

    thrust::device_vector<Type_var> d_alpha0;
    thrust::device_vector<Type_var> d_alpha1_ref;
    thrust::device_vector<Type_var> d_alpha1;
    thrust::device_vector<Type_var> d_alpha2;
    thrust::device_vector<Type_var> d_alpha3;
    thrust::device_vector<Type_var> d_alpha4;
    thrust::device_vector<Type_var> d_alpha5;
    thrust::device_vector<Type_var> d_alpha6;
    thrust::device_vector<Vec3<Type_var>> d_spontaneousp_ref;
    thrust::device_vector<Type_var> d_temperature;
    thrust::device_vector<Type_var> d_temperature_noise;
    thrust::device_vector<Type_var> d_transition_temp;
    thrust::device_vector<Type_var> d_relaxation;
    thrust::device_vector<Type_var> d_relaxation_ref;
    thrust::device_vector<Type_var> d_susciptibilityWeights;
    thrust::device_vector<Type_var> uni_anistropy_z;

    curandState* states; //!< GPU RNG states for thermal noise
} landau_free;

//! @brief Stores final processed (normalized) Landau parameter vectors.
typedef struct {
    thrust::device_vector<Type_var> vector_alpha1;
    thrust::device_vector<Type_var> vector_alpha2;
    thrust::device_vector<Type_var> vector_alpha3;
    thrust::device_vector<Type_var> vector_alpha4;
    thrust::device_vector<Type_var> vector_alpha5;
    thrust::device_vector<Type_var> vector_alpha6;
    thrust::device_vector<Vec3<Type_var>> vector_spontaneousp;
    thrust::device_vector<Type_var> vector_temperature;
    thrust::device_vector<Type_var> vector_transition_temp;
    thrust::device_vector<Type_var> vector_relaxation;
    thrust::device_vector<Type_var> vector_variance;
} landau_free_final;

//==============================================================================
// External electric field parameter structures
//==============================================================================

//! @brief Stores AC, DC, and pulse external field configurations.
typedef struct {
    int FLAG_FIELD;
    int FLAG_INPUT_TYPE;
    int FLAG_FIELD_AC;
    int FLAG_FREQUENCY;
    int FLAG_PHASE;
    int FLAG_FIELD_PULSE;
    int nfield;

    thrust::device_vector<Vec3<Type_var>> d_field;
    thrust::device_vector<Vec3<Type_var>> d_AC_field;
    thrust::device_vector<Type_var> d_AC_frequency;
    thrust::device_vector<Vec3<Type_var>> d_AC_phase;
    thrust::device_vector<Type_var> d_pulse_field;
    thrust::device_vector<Type_var> d_pulse_time;
    thrust::device_vector<Vec3<Type_var>> external_field_storage;

    Type_var H_theta_pulse;
    Type_var H_phi_pulse;
} field_external;

//! @brief Stores normalized field vectors for GPU use.
typedef struct {
    thrust::device_vector<Vec3<Type_var>> vector_field;
    thrust::device_vector<Vec3<Type_var>> vector_Ac_field;
    thrust::device_vector<Vec3<Type_var>> vector_AC_phase;
    thrust::device_vector<Type_var> MagSp;
} field_external_normalized;

//==============================================================================
// Gradient energy terms
//==============================================================================

//! @brief Gradient coefficients and flags for Ginzburg-Landau energy terms.
typedef struct {
    int FLAG_GRADIENT;
    int FLAG_G1;
    int FLAG_G2;
    int FLAG_G3;
    int FLAG_G4;
    int FLAG_BC;

    thrust::device_vector<Type_var> d_G0;
    thrust::device_vector<Type_var> d_G1;
    thrust::device_vector<Type_var> d_G2;
    thrust::device_vector<Type_var> d_G3;
    thrust::device_vector<Type_var> d_G4;
} gradient;

//! @brief Final normalized gradient energy vectors.
typedef struct {
    thrust::device_vector<Type_var> vector_G1;
    thrust::device_vector<Type_var> vector_G2;
    thrust::device_vector<Type_var> vector_G3;
    thrust::device_vector<Type_var> vector_G4;
} gradient_final;


//! @brief Contains spatial discretization and simulation time control parameters.
struct elastic {
    int FLAG_ELASTIC;
    int FLAG_DISPLACEMENT_FIELD;
    int FLAG_POLARIZATION_FIELD;
    int FLAG_STRESS;

    int FLAG_Q11;
    int FLAG_Q12;
    int FLAG_Q44;

    int FLAG_C11;
    int FLAG_C12;
    int FLAG_C44;

    int FLAG_SIGMA_XX;
    int FLAG_SIGMA_YY;
    int FLAG_SIGMA_ZZ;
    int FLAG_SIGMA_XY;
    int FLAG_SIGMA_XZ;
    int FLAG_SIGMA_YZ;

    thrust::device_vector<Type_var> Q11;
    thrust::device_vector<Type_var> Q12;
    thrust::device_vector<Type_var> Q44;

    thrust::device_vector<Type_var> C11;
    thrust::device_vector<Type_var> C12;
    thrust::device_vector<Type_var> C44;

    thrust::device_vector<Type_var> sigma_xx_ext;
    thrust::device_vector<Type_var> sigma_yy_ext;
    thrust::device_vector<Type_var> sigma_zz_ext;
    thrust::device_vector<Type_var> sigma_xy_ext;
    thrust::device_vector<Type_var> sigma_xz_ext;
    thrust::device_vector<Type_var> sigma_yz_ext;

    thrust::device_vector<Vec3<Type_var>> displacement;
    thrust::device_vector<Type_var> elastic_energy;
};

//==============================================================================
// Initial polarization field configuration
//==============================================================================

//! @brief Initial polarization setup parameters.
typedef struct {
    int FLAG_INICIALP;
    thrust::device_vector<Vec3<Type_var>> initial_pol_vector;
} initial_polarization;

//==============================================================================
// Finite difference geometry and simulation parameters
//==============================================================================

//! @brief Contains spatial discretization and simulation time control parameters.
struct FE_geometry {
    int FLAG_STUDY;
    int N_d;
    int Ncx;
    int Ncy;
    int Ncz;
    int FLAG_SHAPE;

    thrust::device_vector<int> shape;
    thrust::device_vector<Vec3<Type_var>> electrostatic_field;
    thrust::device_vector<Type_var> electrostatic_energy;
    thrust::device_vector<Type_var> ferro_energy;

    Type_var delta_x;
    Type_var delta_y;
    Type_var delta_z;
    Type_var h;

    Type_var time_sim;
    Type_var dtime;
    Type_var stop_sim;
    Type_var dtime_min;
    Type_var dtime_max;

    int FLAG_INTEGRATOR;
    int FLAG_ENERGY;
    int interval_energy;

    Type_var error_tolerance;
    Type_var max_dt_times;
    Type_var min_dt_times;

    int start_adapting;
    int n_elements;
};

//==============================================================================
// FFT-based electrostatic field storage and configuration
//==============================================================================

//! @brief Holds GPU buffers for FFT-based demagnetization tensor components.
typedef struct {
    int FLAG_CALC;
    HD_Complex_Type* cuSDxx;
    HD_Complex_Type* cuSDyy;
    HD_Complex_Type* cuSDzz;
    HD_Complex_Type* cuSDxy;
    HD_Complex_Type* cuSDxz;
    HD_Complex_Type* cuSDyz;
} set_DEMAG;

//! @brief Defines FFT grid sizes and DEMAG tensor parameters.
typedef struct {
    set_DEMAG demag_param;
    int Mcx;
    int Mcy;
    int Mcz;
} SET_parameters;

//==============================================================================
// Tensor and padding configuration
//==============================================================================

//! @brief Stores dielectric tensor configuration and grid padding information.
typedef struct {
    int FLAG_TENSOR;
    Type_var Dx;
    Type_var Dy;
    Type_var Dz;
    int FLAG_PADDING;
    int PAD_cx;
    int PAD_cy;
    int PAD_cz;
} set_tensor;

//==============================================================================
// Output control parameters
//==============================================================================

//! @brief Controls simulation output intervals for polarization, field, and energy.
typedef struct {
    int num_iteration;
    int interval_output;
    int interval_polarization;
    int FLAG_SPIN;
    int FLAG_LOCAL_POLARIZATION;
    int interval_local_polarization;
    int FLAG_AVERAGE_POLARIZATION;
    int interval_average_polarization;
    int FLAG_LOCAL_ENERGY;
    int interval_local_energy;
    int FLAG_AVERAGE_ENERGY;
    int interval_average_energy;
    int FLAG_LOCAL_FIELD;
    int interval_local_field;
    int FLAG_AVERAGE_FIELD;
    int interval_average_field;
} set_output;

#endif // _PARAMETERS_H
