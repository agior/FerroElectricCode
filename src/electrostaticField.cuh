#ifndef _HDEMAG_H
#define _HDEMAG_H

#include "parameters.h"
#include "Vec3.h"

extern int threadsPerBlock;

// ================= Utility Functions =================
int nextPowerOf2(int n);

// ================= FFT Operations =================
int splitted_3dfft_calc(
    HD_Complex_Type* d_dataX,
    HD_Complex_Type* d_dataY,
    HD_Complex_Type* d_dataZ,
    int Mcx, int Mcy, int Mcz,
    int Ncx, int Ncy, int Ncz,
    int fft_direction
);

// ================= Electrostatic Field =================
/*
void electrostaticField(
    Type_var*, Type_var*, Type_var*,
    HD_Complex_Type*, HD_Complex_Type*, HD_Complex_Type*,
    HD_Complex_Type*, HD_Complex_Type*, HD_Complex_Type*,
    Type_var*, Type_var*, Type_var*,
    int, int, int, int, int, int
);
*/

void electrostaticField(
    Vec3<Type_var>* P,
    Vec3<Type_var>* Hdmg,
    SET_parameters& params
);

// ================= CUDA Kernels =================

// With Ms weighting
__global__ void kernel3_copyPolarization_Ms(
    Vec3<Type_var>* cu_m,
    HD_Complex_Type* cuTempx,
    HD_Complex_Type* cuTempy,
    HD_Complex_Type* cuTempz,
    Type_var* cu_weights,
    int Ncx, int Ncy, int Ncz,
    int N_d,
    int Mcx, int Mcy, int Mcz,
    int weightVecSize
);

// Standard copy
__global__ void kernel_copyPolarization(
    Vec3<Type_var>* cu_m,
    HD_Complex_Type* cuTempx,
    HD_Complex_Type* cuTempy,
    HD_Complex_Type* cuTempz,
    int Ncx, int Ncy, int Ncz,
    int N_d,
    int Mcx, int Mcy, int Mcz
);

// Scalar version
__global__ void kernel1_copyPolarization(
    Type_var* cu_m,
    HD_Complex_Type* cuTemp,
    int Ncx, int Ncy, int Ncz,
    int N_d,
    int Mcx, int Mcy, int Mcz
);

// Demagnetizing field calculation
__global__ void kernel_H_calc(
    HD_Complex_Type* dTempx,
    HD_Complex_Type* dTempy,
    HD_Complex_Type* dTempz,
    cu_Tens_Type* dSDxx,
    cu_Tens_Type* dSDyy,
    cu_Tens_Type* dSDzz,
    cu_Tens_Type* dSDxy,
    cu_Tens_Type* dSDxz,
    cu_Tens_Type* dSDyz,
    int Mcx, int Mcy, int Mcz,
    int N_d
);

// Electrostatic field kernels
__global__ void kernel_electrostaticField_calc(
    HD_Complex_Type* cuTempx,
    HD_Complex_Type* cuTempy,
    HD_Complex_Type* cuTempz,
    Vec3<Type_var>* cuHdmg,
    int Ncx, int Ncy, int Ncz,
    int N_d,
    int Mcx, int Mcy, int Mcz
);

__global__ void kernel1_electrostaticField_calc(
    HD_Complex_Type* cuTemp,
    Type_var* cuHdmg,
    int Ncx, int Ncy, int Ncz,
    int Mcx, int Mcy, int Mcz
);

#endif  // _HDEMAG_H
