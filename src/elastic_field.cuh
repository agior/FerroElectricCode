/**
 * @file elasticField.h
 * @brief CUDA C++ header defining kernels, functors, and helper routines for elastic field computation.
 *
 * This module provides kernel and functor declarations for computing strain,
 * stress, and elastic contributions in phase-field or ferroelectric simulations.
 * It integrates cuFFT-based spectral methods and Thrust device operations.
 */

#pragma once

 //------------------------------------------------------------------------------
 // Includes
 //------------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include <cufft.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Vec3.h"  ///< Requires Vec3.h and Vec3.cuh to be present in include path.

/**
 * @file elasticField.h
 * @brief CUDA C++ header defining kernels, functors, and helper routines for elastic field computation.
 *
 * This module provides kernel and functor declarations for computing strain,
 * stress, and elastic contributions in phase-field or ferroelectric simulations.
 * It integrates cuFFT-based spectral methods and Thrust device operations.
 */

 //------------------------------------------------------------------------------
 // Constants and Macros
 //------------------------------------------------------------------------------
#ifndef M_PI
#define M_PI 3.14159265358979323846  ///< Mathematical constant π.
#endif

//------------------------------------------------------------------------------
// CUDA Error Checking Macros
//------------------------------------------------------------------------------

/**
 * @brief Checks CUDA runtime calls and exits on error.
 */
#define CHECK_CUDA(call)                                                       \
do {                                                                          \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                             \
                __FILE__, __LINE__, cudaGetErrorString(err));                 \
        exit(1);                                                              \
    }                                                                         \
} while (0)

 /**
  * @brief Checks cuFFT library calls and exits on error.
  */
#define CHECK_CUFFT(call)                                                      \
do {                                                                          \
    cufftResult err = call;                                                   \
    if (err != CUFFT_SUCCESS) {                                               \
        fprintf(stderr, "cuFFT error %s:%d: %d\n",                            \
                __FILE__, __LINE__, (int)err);                                \
        exit(1);                                                              \
    }                                                                         \
} while (0)

  //------------------------------------------------------------------------------
  // Type Aliases
  //------------------------------------------------------------------------------
using cuC = cufftDoubleComplex;  ///< Alias for double-precision cuFFT complex type.

//------------------------------------------------------------------------------
// Helper Function Declarations
//------------------------------------------------------------------------------

/**
 * @brief Compute the 1D flattened index from 3D coordinates.
 */
inline int idx3(int i, int j, int k, int Nx, int Ny, int Nz);

/**
 * @brief Save a 2D slice from a 3D array to a file.
 */
void save_slice(const std::string& fname,
    const thrust::host_vector<double>& arr,
    int Nx, int Ny, int Nz, int zslice);

//------------------------------------------------------------------------------
// Kernel Declarations
//------------------------------------------------------------------------------

__global__ void solve_kernel(
    const cuC* exx, const cuC* eyy, const cuC* ezz,
    const cuC* exy, const cuC* exz, const cuC* eyz,
    cuC* ux_hat, cuC* uy_hat, cuC* uz_hat,
    int Nx, int Ny, int Nz,
    double Lx, double Ly, double Lz,
    double C11, double C12, double C44
);

__global__ void deriv_kernel(
    const cuC* u_hat,
    cuC* out_x, cuC* out_y, cuC* out_z,
    int Nx, int Ny, int Nz,
    double Lx, double Ly, double Lz
);

//------------------------------------------------------------------------------
// Functor Declarations
//------------------------------------------------------------------------------

struct ConvertFromCuCToVec3Functor {
    // Input arrays
    cuC* ux_hat; cuC* uy_hat; cuC* uz_hat;
    cuC* ux_x_hat; cuC* ux_y_hat; cuC* ux_z_hat;
    cuC* uy_x_hat; cuC* uy_y_hat; cuC* uy_z_hat;
    cuC* uz_x_hat; cuC* uz_y_hat; cuC* uz_z_hat;

    // Output arrays
    Vec3<double>* U_out;
    Vec3<double>* Ux_deriv_out;
    Vec3<double>* Uy_deriv_out;
    Vec3<double>* Uz_deriv_out;

    double invN;  ///< Normalization factor (1/N).
    int N;        ///< Total number of grid points.

    __host__ __device__
        ConvertFromCuCToVec3Functor(
            cuC* ux_hat_, cuC* uy_hat_, cuC* uz_hat_,
            cuC* ux_x_hat_, cuC* ux_y_hat_, cuC* ux_z_hat_,
            cuC* uy_x_hat_, cuC* uy_y_hat_, cuC* uy_z_hat_,
            cuC* uz_x_hat_, cuC* uz_y_hat_, cuC* uz_z_hat_,
            Vec3<double>* U_out_, Vec3<double>* Ux_deriv_out_,
            Vec3<double>* Uy_deriv_out_, Vec3<double>* Uz_deriv_out_,
            double invN_, int N_);

    __host__ __device__
        void operator()(const int& idx) const;
};

struct FullPhysicsFunctor {
    // Input arrays
    Vec3<double>* P;
    Vec3<double>* U;
    Vec3<double>* Ux_deriv;
    Vec3<double>* Uy_deriv;
    Vec3<double>* Uz_deriv;

    // Output arrays
    Vec3<double>* eps_norm_out;
    Vec3<double>* eps_shear_out;
    double* fE1_out;
    double* fE2_out;
    double* fE3_out;
    Vec3<double>* field_P_out;
    Vec3<double>* heff;
    double* elastic_energy;
    double* field_mag_out;

    // Material coefficients
    double Q11, Q12, Q44;
    double C11, C12, C44;
    double q11, q12, q44;
    double b11, b12;

    // externally-induced homogeneous strain components (Voigt)
    double eps_ext_xx, eps_ext_yy, eps_ext_zz;
    double eps_ext_xy, eps_ext_xz, eps_ext_yz;

    int FLAG_USE_DISPLACEMENT_FIELD; ///< if 0, skip displacement-dependent part (use eps_ext only)

    int FLAG_USE_POLARIZATION_FIELD;

    int N;  ///< Total number of grid points.

    __host__ __device__
        FullPhysicsFunctor(
            Vec3<double>* P_, Vec3<double>* U_, Vec3<double>* Ux_deriv_,
            Vec3<double>* Uy_deriv_, Vec3<double>* Uz_deriv_,
            Vec3<double>* eps_norm_out_, Vec3<double>* eps_shear_out_,
            double* fE1_out_, double* fE2_out_, double* fE3_out_,
            Vec3<double>* field_P_out_, double* field_mag_out_, Vec3<double>* heff_, double* elastic_energy_,
            double Q11_, double Q12_, double Q44_,
            double C11_, double C12_, double C44_,
            double q11_, double q12_, double q44_,
            double b11_, double b12_, int N_,
            double eps_ext_xx_, double eps_ext_yy_, double eps_ext_zz_,
            double eps_ext_xy_, double eps_ext_xz_, double eps_ext_yz_,
            int FLAG_USE_DISPLACEMENT_FIELD_, int FLAG_USE_POLARIZATION_FIELD_);

    __host__ __device__
        void operator()(const int& idx) const;
};

//------------------------------------------------------------------------------
// Main Host Function Declaration
//------------------------------------------------------------------------------

// NOTE: added 6 external stress inputs (Voigt) and a toggle to skip displacement field.
// sigma_* are stress components (units same as C_). If you don't want an external stress, pass zeros.
// FLAG_USE_DISPLACEMENT_FIELD: 1 => full (FFT displacement) ; 0 => skip displacement-dependent solve (faster)
void compute_elastic_field(
    thrust::device_vector<Vec3<double>>& d_P,
    thrust::device_vector<Vec3<double>>& d_heff,
    thrust::device_vector<double>& d_elastic_energy,
    int Nx, int Ny, int Nz,
    double Lx, double Ly, double Lz,
    double Q11, double Q12, double Q44,
    double C11, double C12, double C44,
    int FLAG_OUTPUT_FILES,
    double sigma_xx_ext,
    double sigma_yy_ext,
    double sigma_zz_ext,
    double sigma_xy_ext,
    double sigma_xz_ext,
    double sigma_yz_ext,
    int FLAG_USE_DISPLACEMENT_FIELD,
    int FLAG_POLARIZATION_FIELD
);
