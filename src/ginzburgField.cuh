#pragma once

#include "Vec3.h"
#include <thrust/execution_policy.h>

// ===============================================================
// gradientField<T>
// ---------------------------------------------------------------
// CUDA-compatible functor for calculating the Ginzburg-Landau
// gradient field contribution.
// ===============================================================

template<typename T>
struct gradientField
{
    // ===========================================================
    // Constructor
    // ===========================================================
    gradientField
    (
        int* shape,                 // Device shape
        int flag_BC,                // Boundary condition flag
        int dim1,                   // Number of cells along X
        int dim2,                   // Number of cells along Y
        int dim3,                   // Number of cells along Z
        T stride1,                  // Cell size along X-axis
        T stride2,                  // Cell size along Y-axis
        T stride3,                  // Cell size along Z-axis
        Vec3<T>* p,                 // Polarization vector
        Vec3<T>* fieldFerro,        // Effective field vector 
        int calcFlag,               // Flag to calculate gradient field
        int ginzburgFlag,           // Ginzburg flag
        T* G1_,
        T* G2_,
        T* G3_,
        T* G4_,
        int FLAG_G1_,
        int FLAG_G2_,
        int FLAG_G3_,
        int FLAG_G4_
    );

    // ===========================================================
    // Operator (Functor)
    // ===========================================================
    __host__ __device__ Vec3<T> operator()(const int index) const;

private:
    // ===========================================================
    // Simulation and geometry parameters
    // ===========================================================
    const int boundaryConditionsFlag;
    const int* device;
    const int dimX;
    const int dimY;
    const int dimXY;
    const int dimZ;
    const T invStrideX;
    const T invStrideY;
    const T invStrideZ;

    // ===========================================================
    // Physical fields
    // ===========================================================
    Vec3<T>* polarization;
    Vec3<T>* hFerro;

    // ===========================================================
    // Ginzburg-Landau parameters
    // ===========================================================
    const int FLAG_GINZBURG;
    const int nears;
    const T* G1;
    const T* G2;
    const T* G3;
    const T* G4;
    const int FLAG_G1;
    const int FLAG_G2;
    const int FLAG_G3;
    const int FLAG_G4;

    // ===========================================================
    // First- and second-order derivative helpers
    // ===========================================================
    __host__ __device__ Vec3<T> Dv1x(const int index, const int i) const;
    __host__ __device__ Vec3<T> Dv1y(const int index, const int j) const;
    __host__ __device__ Vec3<T> Dv1z(const int index, const int k) const;

    __host__ __device__ Vec3<T> Dv2x(const int index, const int i) const;
    __host__ __device__ Vec3<T> Dv2y(const int index, const int j) const;
    __host__ __device__ Vec3<T> Dv2z(const int index, const int k) const;

    // ===========================================================
    // Product derivatives
    // ===========================================================
    __host__ __device__ Vec3<T> p_Dv2x(const int index, const int i, const int baseIdx) const;
    __host__ __device__ Vec3<T> p_Dv2y(const int index, const int j, const int baseIdx) const;
    __host__ __device__ Vec3<T> p_Dv2z(const int index, const int k, const int baseIdx) const;

    __host__ __device__ Vec3<T> p_Dvxy(const int index, const int i, const int j, const int baseIdx) const;
    __host__ __device__ Vec3<T> p_Dvyz(const int index, const int j, const int k, const int baseIdx) const;
    __host__ __device__ Vec3<T> p_Dvxz(const int index, const int i, const int k, const int baseIdx) const;

    // ===========================================================
    // Coefficient accessors
    // ===========================================================
    __host__ __device__ T GetG1(const int index, const int flagG1, const T* G1) const;
    __host__ __device__ T GetG2(const int index, const int flagG2, const T* G2) const;
    __host__ __device__ T GetG3(const int index, const int flagG3, const T* G3) const;
    __host__ __device__ T GetG4(const int index, const int flagG4, const T* G4) const;

    // ===========================================================
    // Derivative operators - Simple Copy BC
    // ===========================================================
    __host__ __device__ Vec3<T> Dv2_2x_simpleCopy(const int index, const int i, const int baseIdx) const;
    __host__ __device__ Vec3<T> Dv2_2y_simpleCopy(const int index, const int j, const int baseIdx) const;
    __host__ __device__ Vec3<T> Dv2_2z_simpleCopy(const int index, const int k, const int baseIdx) const;

    __host__ __device__ Vec3<T> Dv2_xy_simpleCopy(const int index, const int i, const int j, const int baseIdx) const;
    __host__ __device__ Vec3<T> Dv2_yz_simpleCopy(const int index, const int j, const int k, const int baseIdx) const;
    __host__ __device__ Vec3<T> Dv2_xz_simpleCopy(const int index, const int i, const int k, const int baseIdx) const;

    __host__ __device__ Vec3<T> Dv1_2x_simpleCopy(const int index, const int i) const;
    __host__ __device__ Vec3<T> Dv1_2y_simpleCopy(const int index, const int j) const;
    __host__ __device__ Vec3<T> Dv1_2z_simpleCopy(const int index, const int k) const;

    // ===========================================================
    // Derivative operators - Open BC
    // ===========================================================
    __host__ __device__ Vec3<T> Dv1_2x_open(const int index, const int i) const;
    __host__ __device__ Vec3<T> Dv1_2y_open(const int index, const int j) const;
    __host__ __device__ Vec3<T> Dv1_2z_open(const int index, const int k) const;

    __host__ __device__ Vec3<T> Dv2_2x_open(const int index, const int i, const int baseIdx) const;
    __host__ __device__ Vec3<T> Dv2_2y_open(const int index, const int j, const int baseIdx) const;
    __host__ __device__ Vec3<T> Dv2_2z_open(const int index, const int k, const int baseIdx) const;

    __host__ __device__ Vec3<T> Dv2_xy_open(const int index, const int i, const int j, const int baseIdx) const;
    __host__ __device__ Vec3<T> Dv2_yz_open(const int index, const int j, const int k, const int baseIdx) const;
    __host__ __device__ Vec3<T> Dv2_xz_open(const int index, const int i, const int k, const int baseIdx) const;

    __host__ __device__ Vec3<T> Dv2_4x_open(const int index, const int i) const;
    __host__ __device__ Vec3<T> Dv2_4y_open(const int index, const int j) const;
    __host__ __device__ Vec3<T> Dv2_4z_open(const int index, const int k) const;

    // ===========================================================
    // Derivative operators - Periodic BC
    // ===========================================================
    __host__ __device__ Vec3<T> Dv1_2x_periodic(const int index, const int i) const;
    __host__ __device__ Vec3<T> Dv1_2y_periodic(const int index, const int j) const;
    __host__ __device__ Vec3<T> Dv1_2z_periodic(const int index, const int k) const;

    __host__ __device__ Vec3<T> Dv2_2x_periodic(const int index, const int i, const int baseIdx) const;
    __host__ __device__ Vec3<T> Dv2_2y_periodic(const int index, const int j, const int baseIdx) const;
    __host__ __device__ Vec3<T> Dv2_2z_periodic(const int index, const int k, const int baseIdx) const;

    __host__ __device__ Vec3<T> Dv2_xy_periodic(const int index, const int i, const int j, const int baseIdx) const;
    __host__ __device__ Vec3<T> Dv2_yz_periodic(const int index, const int j, const int k, const int baseIdx) const;
    __host__ __device__ Vec3<T> Dv2_xz_periodic(const int index, const int i, const int k, const int baseIdx) const;
};

// ===============================================================
// Implementation
// ===============================================================
#include "ginzburgField.cu"
