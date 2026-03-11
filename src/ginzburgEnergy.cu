/**
 * @file ginzburgEnergy.cu
 * @brief Implementation of CUDA functor `gradientEnergy<T>` for Ginzburg–Landau gradient energy computation.
 *
 * Provides definitions for all device/host functions used by the `gradientEnergy` struct,
 * including boundary condition–aware spatial derivatives and anisotropic Ginzburg coefficient retrieval.
 */

#pragma once

 //------------------------------------------------------------------------------
 // Standard C / C++ Libraries
 //------------------------------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <chrono>

//------------------------------------------------------------------------------
// Thrust and CUDA Libraries
//------------------------------------------------------------------------------
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>

//------------------------------------------------------------------------------
// Project Headers
//------------------------------------------------------------------------------
#include "ginzburgEnergy.cuh"

//------------------------------------------------------------------------------
// Constructor: gradientEnergy<T>
//------------------------------------------------------------------------------
template <typename T>
gradientEnergy<T>::gradientEnergy(
    int* shape, int flag_BC,
    int dim1, int dim2, int dim3,
    T stride1, T stride2, T stride3,
    Vec3<T>* p, T* energyFerro,
    int calcFlag, int ginzburgFlag,
    T* G1_, T* G2_, T* G3_, T* G4_,
    int FLAG_G1_, int FLAG_G2_, int FLAG_G3_, int FLAG_G4_
)
    : device(shape),
    boundaryConditionsFlag(flag_BC),
    dimX(dim1),
    dimY(dim2),
    dimXY(dim1* dim2),
    dimZ(dim3),
    invStrideX(1.0 / stride1),
    invStrideY(1.0 / stride2),
    invStrideZ(1.0 / stride3),
    polarization(p),
    eFerro(energyFerro),
    nears(calcFlag),
    FLAG_GINZBURG(ginzburgFlag),
    G1(G1_),
    G2(G2_),
    G3(G3_),
    G4(G4_),
    FLAG_G1(FLAG_G1_),
    FLAG_G2(FLAG_G2_),
    FLAG_G3(FLAG_G3_),
    FLAG_G4(FLAG_G4_)
{
}

//------------------------------------------------------------------------------
// Derivative Computations (x, y, z directions)
//------------------------------------------------------------------------------
template <typename T>
__host__ __device__
Vec3<T> gradientEnergy<T>::Dv1x_energy(const int index, const int i, const int baseIdx) const
{
    switch (boundaryConditionsFlag)
    {
    case 0: // Simple copy BC
        switch (nears)
        {
        case 1: return Dv1_2x_simpleCopy_energy(index, i, baseIdx);
        default: return {};
        }

    case 1: // Open BC
        switch (nears)
        {
        case 1: return Dv1_2x_open_energy(index, i, baseIdx);
        default: return {};
        }

    case 2: // Periodic BC
        switch (nears)
        {
        case 1: return Dv1_2x_periodic_energy(index, i, baseIdx);
        default: return {};
        }

    default:
        return {};
    }
}

template <typename T>
__host__ __device__
Vec3<T> gradientEnergy<T>::Dv1y_energy(const int index, const int j, const int baseIdx) const
{
    switch (boundaryConditionsFlag)
    {
    case 0: // Simple copy BC
        switch (nears)
        {
        case 1: return Dv1_2y_simpleCopy_energy(index, j, baseIdx);
        default: return {};
        }

    case 1: // Open BC
        switch (nears)
        {
        case 1: return Dv1_2y_open_energy(index, j, baseIdx);
        default: return {};
        }

    case 2: // Periodic BC
        switch (nears)
        {
        case 1: return Dv1_2y_periodic_energy(index, j, baseIdx);
        default: return {};
        }

    default:
        return {};
    }
}

template <typename T>
__host__ __device__
Vec3<T> gradientEnergy<T>::Dv1z_energy(const int index, const int k, const int baseIdx) const
{
    switch (boundaryConditionsFlag)
    {
    case 0: // Simple copy BC
        switch (nears)
        {
        case 1: return Dv1_2z_simpleCopy_energy(index, k, baseIdx);
        default: return {};
        }

    case 1: // Open BC
        switch (nears)
        {
        case 1: return Dv1_2z_open_energy(index, k, baseIdx);
        default: return {};
        }

    case 2: // Periodic BC
        switch (nears)
        {
        case 1: return Dv1_2z_periodic_energy(index, k, baseIdx);
        default: return {};
        }

    default:
        return {};
    }
}

//------------------------------------------------------------------------------
// Ginzburg Coefficient Retrieval
//------------------------------------------------------------------------------
template <typename T>
__host__ __device__
T gradientEnergy<T>::GetG1_energy(const int index, const int flagG1, const T* G1) const
{
    switch (flagG1)
    {
    case 1: return G1[0];
    case 2: return G1[index / dimXY];
    case 3: return G1[index];
    default: return static_cast<T>(0.0);
    }
}

template <typename T>
__host__ __device__
T gradientEnergy<T>::GetG2_energy(const int index, const int flagG2, const T* G2) const
{
    switch (flagG2)
    {
    case 1: return G2[0];
    case 2: return G2[index / dimXY];
    case 3: return G2[index];
    default: return static_cast<T>(0.0);
    }
}

template <typename T>
__host__ __device__
T gradientEnergy<T>::GetG3_energy(const int index, const int flagG3, const T* G3) const
{
    switch (flagG3)
    {
    case 1: return G3[0];
    case 2: return G3[index / dimXY];
    case 3: return G3[index];
    default: return static_cast<T>(0.0);
    }
}

template <typename T>
__host__ __device__
T gradientEnergy<T>::GetG4_energy(const int index, const int flagG4, const T* G4) const
{
    switch (flagG4)
    {
    case 1: return G4[0];
    case 2: return G4[index / dimXY];
    case 3: return G4[index];
    default: return static_cast<T>(0.0);
    }
}
// ===============================================================
// gradientEnergy<T> Implementation
// ===============================================================

template<typename T>
__host__ __device__ T gradientEnergy<T>::operator()(const int index) const
{
    int simSize = dimX * dimY * dimZ;
    int simId = index / simSize;
    int localIdx = index % simSize;
    int baseIdx = simId * simSize;

    int k = localIdx / dimXY;
    int j = (localIdx - k * dimXY) / dimX;
    int i = localIdx - k * dimXY - j * dimX;

    T h = eFerro[index];

    if (device[index])
    {
        // Compute polarization gradients along x, y, z
        Vec3<T> p_x = Dv1x_energy(localIdx, i, baseIdx);
        Vec3<T> p_y = Dv1y_energy(localIdx, j, baseIdx);
        Vec3<T> p_z = Dv1z_energy(localIdx, k, baseIdx);

        // Retrieve Ginzburg-Landau constants
        T G1_val = GetG1_energy(localIdx, FLAG_G1, G1);
        T G2_val = GetG2_energy(localIdx, FLAG_G2, G2);
        T G3_val = GetG3_energy(localIdx, FLAG_G3, G3);
        T G4_val = GetG4_energy(localIdx, FLAG_G4, G4);

        // Gradient energy contribution
        h += 0.5 * G1_val * ((p_x.x * p_x.x) + (p_y.y * p_y.y) + (p_z.z * p_z.z)) +
            G2_val * ((p_x.x * p_y.y) + (p_y.y * p_z.z) + (p_z.z * p_x.x)) +
            0.5 * G3_val * (((p_y.x + p_x.y) * (p_y.x + p_x.y)) +
                ((p_z.y + p_y.z) * (p_z.y + p_y.z)) +
                ((p_x.z + p_z.x) * (p_x.z + p_z.x))) +
            0.5 * G4_val * (((p_y.x - p_x.y) * (p_y.x - p_x.y)) -
                ((p_z.y - p_y.z) * (p_z.y - p_y.z)) +
                ((p_x.z - p_z.x) * (p_x.z - p_z.x)));
    }

    return h;
}


// ===============================================================
// Nearest neighbours first-order derivative - Simple Copy BC
// ===============================================================

template<typename T>
__host__ __device__ Vec3<T> gradientEnergy<T>::Dv1_2x_simpleCopy_energy(
    const int index, const int i, const int baseIdx) const
{
    if (dimX == 1) return {};

    return (polarization[baseIdx + (index + (i == (dimX - 1) ? 0 : 1))] -
        polarization[baseIdx + (index + (i == 0 ? 0 : -1))]) * (0.5 * invStrideX);
}

template<typename T>
__host__ __device__ Vec3<T> gradientEnergy<T>::Dv1_2y_simpleCopy_energy(
    const int index, const int j, const int baseIdx) const
{
    if (dimY == 1) return {};

    return (polarization[baseIdx + (index + (j == (dimY - 1) ? 0 : dimX))] -
        polarization[baseIdx + (index + (j == 0 ? 0 : -dimX))]) * (0.5 * invStrideY);
}

template<typename T>
__host__ __device__ Vec3<T> gradientEnergy<T>::Dv1_2z_simpleCopy_energy(
    const int index, const int k, const int baseIdx) const
{
    if (dimZ == 1) return {};

    return (polarization[baseIdx + (index + (k == (dimZ - 1) ? 0 : dimXY))] -
        polarization[baseIdx + (index + (k == 0 ? 0 : -dimXY))]) * (0.5 * invStrideZ);
}


// ===============================================================
// Nearest neighbours first-order derivative - Open BC
// ===============================================================

template<typename T>
__host__ __device__ Vec3<T> gradientEnergy<T>::Dv1_2x_open_energy(
    const int index, const int i, const int baseIdx) const
{
    if (dimX == 1) return {};

    return ((i == (dimX - 1) ? Vec3<T>() : polarization[baseIdx + (index + 1)]) -
        (i == 0 ? Vec3<T>() : polarization[baseIdx + (index - 1)])) * (0.5 * invStrideX);
}

template<typename T>
__host__ __device__ Vec3<T> gradientEnergy<T>::Dv1_2y_open_energy(
    const int index, const int j, const int baseIdx) const
{
    if (dimY == 1) return {};

    return ((j == (dimY - 1) ? Vec3<T>() : polarization[baseIdx + (index + dimX)]) -
        (j == 0 ? Vec3<T>() : polarization[baseIdx + (index - dimX)])) * (0.5 * invStrideY);
}

template<typename T>
__host__ __device__ Vec3<T> gradientEnergy<T>::Dv1_2z_open_energy(
    const int index, const int k, const int baseIdx) const
{
    if (dimZ == 1) return {};

    return ((k == (dimZ - 1) ? Vec3<T>() : polarization[baseIdx + (index + dimXY)]) -
        (k == 0 ? Vec3<T>() : polarization[baseIdx + (index - dimXY)])) * (0.5 * invStrideZ);
}


// ===============================================================
// Nearest neighbours first-order derivative - Periodic BC
// ===============================================================

template<typename T>
__host__ __device__ Vec3<T> gradientEnergy<T>::Dv1_2x_periodic_energy(
    const int index, const int i, const int baseIdx) const
{
    if (dimX == 1) return {};

    return (polarization[baseIdx + (index + (i == (dimX - 1) ? -i : 1))] -
        polarization[baseIdx + (index + (i == 0 ? (dimX - 1) : -1))]) * (0.5 * invStrideX);
}

template<typename T>
__host__ __device__ Vec3<T> gradientEnergy<T>::Dv1_2y_periodic_energy(
    const int index, const int j, const int baseIdx) const
{
    if (dimY == 1) return {};

    return (polarization[baseIdx + (index + (j == (dimY - 1) ? -j * dimX : dimX))] -
        polarization[baseIdx + (index + (j == 0 ? (dimY - 1) * dimX : -dimX))]) * (0.5 * invStrideY);
}

template<typename T>
__host__ __device__ Vec3<T> gradientEnergy<T>::Dv1_2z_periodic_energy(
    const int index, const int k, const int baseIdx) const
{
    if (dimZ == 1) return {};

    return (polarization[baseIdx + (index + (k == (dimZ - 1) ? -k * dimXY : dimXY))] -
        polarization[baseIdx + (index + (k == 0 ? (dimZ - 1) * dimXY : -dimXY))]) * (0.5 * invStrideZ);
}
