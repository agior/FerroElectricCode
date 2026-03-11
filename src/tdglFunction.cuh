/**
 * @file tdglFunction.cuh
 * @brief Defines the TDGL functor for computing time-dependent Ginzburg–Landau field updates.
 *
 * This header declares the `tdgl_functor` structure, which provides the device-side
 * operator used to evaluate the TDGL equation for each grid element. It combines
 * contributions from deterministic effective fields and stochastic noise, scaled by
 * the local relaxational coefficients.
 *
 * Dependencies:
 * - `Vec3.h`          : Defines the 3D vector class template for field and polarization data.
 * - `randomTdgl.cuh`  : Provides random number generation utilities for stochastic TDGL noise.
 * - `<thrust/execution_policy.h>` : Used for controlling Thrust execution on host or device.
 *
 * The CUDA implementation of the operator is included from `tdglFunction.cu`.
 */

#pragma once

 //------------------------------------------------------------------------------
 // Include dependencies
 //------------------------------------------------------------------------------
#include <thrust/execution_policy.h>

#include "Vec3.h"           ///< 3D vector structure for field representation
#include "randomTdgl.cuh"   ///< RNG utilities for stochastic contributions

//------------------------------------------------------------------------------
// TDGL Functor Definition
//------------------------------------------------------------------------------

/**
 * @brief Functor implementing the TDGL equation for polarization dynamics.
 *
 * The `tdgl_functor` structure defines the callable operator used by Thrust algorithms
 * to compute the rate of change of polarization (`dP/dt`) at each grid index.
 * It combines deterministic effective field components and stochastic noise, scaled
 * by the relaxation coefficients (`L_val`).
 *
 * @tparam T Numeric precision type (e.g., float or double).
 */
template <typename T>
struct tdgl_functor {
	//--------------------------------------------------------------------------
	// Member variables
	//--------------------------------------------------------------------------
	Vec3<T>* h;               ///< Pointer to device vector containing effective field values
	Vec3<T>* noiseVec;        ///< Pointer to device vector containing noise values
	T* L_val;                 ///< Pointer to device array of relaxation coefficients
	int LVectorSize;          ///< Size of the relaxation coefficient array
	int gridSize;             ///< Total grid size (number of elements)
	int Ncz;                  ///< Number of cells along the z-axis

	//--------------------------------------------------------------------------
	// Constructor
	//--------------------------------------------------------------------------
	/**
	 * @brief Constructs the TDGL functor with required device pointers and parameters.
	 *
	 * @param h_ Pointer to effective field vector.
	 * @param noiseVec_ Pointer to noise vector.
	 * @param L_val_ Pointer to relaxation coefficient values.
	 * @param LVectorSize_ Size of the relaxation coefficient array.
	 * @param gridSize_ Total number of grid points.
	 * @param Ncz_ Number of cells along z-axis.
	 */
	tdgl_functor(Vec3<T>* h_, Vec3<T>* noiseVec_, T* L_val_, int LVectorSize_, int gridSize_, int Ncz_)
		: h(h_), noiseVec(noiseVec_), L_val(L_val_),
		LVectorSize(LVectorSize_), gridSize(gridSize_), Ncz(Ncz_) {
	}

	//--------------------------------------------------------------------------
	// Operator Overload
	//--------------------------------------------------------------------------
	/**
	 * @brief Computes the TDGL derivative for a given grid index.
	 *
	 * This operator is executed on both host and device (`__host__ __device__`)
	 * and performs the evaluation of the TDGL function:
	 * \f[
	 *   \frac{dP}{dt} = L \cdot (h + \text{noise})
	 * \f]
	 * where `h` represents the deterministic field and `noise` the stochastic term.
	 *
	 * @param idx Grid index for which to compute the update.
	 * @return Computed `Vec3<T>` representing the TDGL update at index `idx`.
	 */
	__host__ __device__ Vec3<T> operator()(int idx) const;
};

//------------------------------------------------------------------------------
// Include CUDA Implementation
//------------------------------------------------------------------------------
#include "tdglFunction.cu"
