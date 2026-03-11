/**
 * @file ginzburgEnergy.h
 * @brief Declares CUDA-compatible functor for computing Ginzburg–Landau gradient energy contributions.
 *
 * This functor, `gradientEnergy<T>`, computes the gradient (Ginzburg) energy term
 * for each grid point based on polarization field derivatives. It supports various
 * boundary conditions (simple copy, open, periodic) and handles anisotropic
 * Ginzburg coefficients (G1–G4).
 *
 * The implementation is provided in **ginzburgEnergy.cu**.
 */

#pragma once

#include "Vec3.h"
#include <thrust/execution_policy.h>

 //------------------------------------------------------------------------------
 // Struct: gradientEnergy
 //------------------------------------------------------------------------------
 //! @brief CUDA functor for calculating the gradient energy term in TDGL models.
 //!
 //! @tparam T Numeric type (e.g., float or double).
 //!
 //! The gradient energy term contributes to the free energy functional in
 //! time-dependent Ginzburg–Landau (TDGL) simulations. This functor evaluates
 //! the energy density based on spatial derivatives of polarization vectors
 //! under different boundary condition schemes.
template <typename T>
struct gradientEnergy
{
    //--------------------------------------------------------------------------
    // Constructor
    //--------------------------------------------------------------------------
    //! @brief Initializes the gradientEnergy functor with geometric and field parameters.
    //!
    //! @param shape Device array representing simulation shape or mask.
    //! @param flag_BC Boundary condition flag (e.g., 0=simple, 1=open, 2=periodic).
    //! @param dim1 Number of cells along x direction.
    //! @param dim2 Number of cells along y direction.
    //! @param dim3 Number of cells along z direction.
    //! @param stride1 Cell spacing along x-axis.
    //! @param stride2 Cell spacing along y-axis.
    //! @param stride3 Cell spacing along z-axis.
    //! @param p Pointer to polarization vector array.
    //! @param energyFerro Pointer to output energy array.
    //! @param calcFlag Flag controlling gradient field calculation.
    //! @param ginzburgFlag Ginzburg computation flag (enables/disables contribution).
    //! @param G1_–G4_ Ginzburg coefficients (may be anisotropic).
    //! @param FLAG_G1_–FLAG_G4_ Flags to enable each corresponding G coefficient.
    gradientEnergy(
        int* shape, int flag_BC,
        int dim1, int dim2, int dim3,
        T stride1, T stride2, T stride3,
        Vec3<T>* p, T* energyFerro,
        int calcFlag, int ginzburgFlag,
        T* G1_, T* G2_, T* G3_, T* G4_,
        int FLAG_G1_, int FLAG_G2_, int FLAG_G3_, int FLAG_G4_
    );

    //--------------------------------------------------------------------------
    // Operator
    //--------------------------------------------------------------------------
    //! @brief Compute gradient energy contribution at a given grid index.
    //! @param index Linear index of the grid cell.
    //! @return Gradient energy (scalar).
    __host__ __device__
        T operator()(const int index) const;

private:
    //--------------------------------------------------------------------------
    // Boundary and device configuration
    //--------------------------------------------------------------------------
    const int boundaryConditionsFlag; //!< Boundary condition mode
    const int* device;                //!< Device shape/mask pointer
    const int dimX;                   //!< Number of cells along x
    const int dimY;                   //!< Number of cells along y
    const int dimXY;                  //!< Precomputed dimX * dimY
    const int dimZ;                   //!< Number of cells along z
    const T invStrideX;               //!< Inverse of cell spacing (x)
    const T invStrideY;               //!< Inverse of cell spacing (y)
    const T invStrideZ;               //!< Inverse of cell spacing (z)

    //--------------------------------------------------------------------------
    // Polarization and energy arrays
    //--------------------------------------------------------------------------
    Vec3<T>* polarization;            //!< Polarization vector field
    T* eFerro;                        //!< Gradient energy array (output)

    //--------------------------------------------------------------------------
    // Ginzburg energy parameters
    //--------------------------------------------------------------------------
    const int FLAG_GINZBURG;          //!< Enables/disables Ginzburg term
    const int nears;                  //!< Number of nearest neighbors (stencil size)
    const T* G1;                      //!< Ginzburg coefficient G1
    const T* G2;                      //!< Ginzburg coefficient G2
    const T* G3;                      //!< Ginzburg coefficient G3
    const T* G4;                      //!< Ginzburg coefficient G4
    const int FLAG_G1;                //!< Enables/disables G1 term
    const int FLAG_G2;                //!< Enables/disables G2 term
    const int FLAG_G3;                //!< Enables/disables G3 term
    const int FLAG_G4;                //!< Enables/disables G4 term

    //--------------------------------------------------------------------------
    // Private helper functions
    //--------------------------------------------------------------------------

    //--- First-order derivative functions for DMI or gradient calculations ---
    __host__ __device__ Vec3<T> Dv1x_energy(const int index, const int i, const int baseIdx) const;
    __host__ __device__ Vec3<T> Dv1y_energy(const int index, const int j, const int baseIdx) const;
    __host__ __device__ Vec3<T> Dv1z_energy(const int index, const int k, const int baseIdx) const;

    //--- Accessors for anisotropic Ginzburg coefficients ---
    __host__ __device__ T GetG1_energy(const int index, const int flagG2, const T* G1) const;
    __host__ __device__ T GetG2_energy(const int index, const int flagG3, const T* G2) const;
    __host__ __device__ T GetG3_energy(const int index, const int flagG4, const T* G3) const;
    __host__ __device__ T GetG4_energy(const int index, const int flagG5, const T* G4) const;

    //--- Derivative operators under different boundary conditions ---

    // Simple copy boundary conditions
    __host__ __device__ Vec3<T> Dv1_2x_simpleCopy_energy(const int index, const int i, const int baseIdx) const;
    __host__ __device__ Vec3<T> Dv1_2y_simpleCopy_energy(const int index, const int j, const int baseIdx) const;
    __host__ __device__ Vec3<T> Dv1_2z_simpleCopy_energy(const int index, const int k, const int baseIdx) const;

    // Open boundary conditions
    __host__ __device__ Vec3<T> Dv1_2x_open_energy(const int index, const int i, const int baseIdx) const;
    __host__ __device__ Vec3<T> Dv1_2y_open_energy(const int index, const int j, const int baseIdx) const;
    __host__ __device__ Vec3<T> Dv1_2z_open_energy(const int index, const int k, const int baseIdx) const;

    // Periodic boundary conditions
    __host__ __device__ Vec3<T> Dv1_2x_periodic_energy(const int index, const int i, const int baseIdx) const;
    __host__ __device__ Vec3<T> Dv1_2y_periodic_energy(const int index, const int j, const int baseIdx) const;
    __host__ __device__ Vec3<T> Dv1_2z_periodic_energy(const int index, const int k, const int baseIdx) const;
};

//------------------------------------------------------------------------------
// Implementation include
//------------------------------------------------------------------------------
#include "ginzburgEnergy.cu"
