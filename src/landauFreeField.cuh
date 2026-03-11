#pragma once
#ifndef _LANDAUFREEFIELD_CUH_
#define _LANDAUFREEFIELD_CUH_

#include "Vec3.h"
#include <thrust/execution_policy.h>

/**
 * @brief Landau effective field functor for ferroelectric simulations.
 *
 * This functor computes the local effective field contribution derived from
 * the Landau-Devonshire free energy expansion, using the polarization vector field
 * and Landau coefficients (α₁ … α₆). It supports uniform, z-dependent, or fully
 * spatially varying coefficients and can be executed on host or device.
 *
 * @tparam T Numeric type (e.g., float or double).
 */
template <typename T>
struct getLandauField
{
	// ------------------------------------------------------------------------
	// Member variables
	// ------------------------------------------------------------------------

	/// Pointer to polarization vector field at each grid point.
	Vec3<T>* polarization;

	/// Pointer to array storing the computed effective field.
	Vec3<T>* hFerro;

	/// Pointer to shape mask (e.g., to distinguish active/inactive regions).
	int* shape;

	/// Landau expansion coefficients α₁, α₁₁, α₁₂, α₁₁₁, α₁₁₂, α₁₂₃.
	T* alphaOne;
	T* alphaTwo;
	T* alphaThree;
	T* alphaFour;
	T* alphaFive;
	T* alphaSix;
	T* uni_anisotropy;

	/// Sizes of the coefficient arrays (can be 1, Ncz, or gridSize).
	int alpha1VecSizes;
	int alpha2VecSizes;
	int alpha3VecSizes;
	int alpha4VecSizes;
	int alpha5VecSizes;
	int alpha6VecSizes;
	int anisVecSizes;

	/// Total number of grid points.
	int gridSize;

	/// Number of grid cells along z direction (used for z-dependent coefficients).
	int Ncz;

	/// Number of domains or active cells (e.g., for masked operations).
	int N_d;

	/// Optional flag controlling study or mode selection.
	int flag_study;

	// ------------------------------------------------------------------------
	// Constructor
	// ------------------------------------------------------------------------

	/**
	 * @brief Construct a new getLandauField functor.
	 *
	 * @param polarization_ Pointer to polarization vector field.
	 * @param hFerro_ Pointer to output effective field array.
	 * @param shape_ Pointer to shape mask.
	 * @param alphaOne_ Pointer to α₁ array.
	 * @param alphaTwo_ Pointer to α₁₁ array.
	 * @param alphaThree_ Pointer to α₁₂ array.
	 * @param alphaFour_ Pointer to α₁₁₁ array.
	 * @param alphaFive_ Pointer to α₁₁₂ array.
	 * @param alphaSix_ Pointer to α₁₂₃ array.
	 * @param alpha1VecSizes_ Size of α₁ array.
	 * @param alpha2VecSizes_ Size of α₁₁ array.
	 * @param alpha3VecSizes_ Size of α₁₂ array.
	 * @param alpha4VecSizes_ Size of α₁₁₁ array.
	 * @param alpha5VecSizes_ Size of α₁₁₂ array.
	 * @param alpha6VecSizes_ Size of α₁₂₃ array.
	 * @param gridSize_ Total number of grid points.
	 * @param Ncz_ Number of grid points along z direction.
	 * @param N_d_ Number of active domain points.
	 * @param flag_study_ Optional flag to control mode or study.
	 */
	__host__ __device__
		getLandauField(Vec3<T>* polarization_,
			Vec3<T>* hFerro_,
			int* shape_,
			T* alphaOne_,
			T* alphaTwo_,
			T* alphaThree_,
			T* alphaFour_,
			T* alphaFive_,
			T* alphaSix_,
			T* uni_anisotropy_,
			int alpha1VecSizes_,
			int alpha2VecSizes_,
			int alpha3VecSizes_,
			int alpha4VecSizes_,
			int alpha5VecSizes_,
			int alpha6VecSizes_,
			int anisVecSizes_,
			int gridSize_,
			int Ncz_,
			int N_d_,
			int flag_study_)
		: polarization(polarization_),
		hFerro(hFerro_),
		shape(shape_),
		alphaOne(alphaOne_),
		alphaTwo(alphaTwo_),
		alphaThree(alphaThree_),
		alphaFour(alphaFour_),
		alphaFive(alphaFive_),
		alphaSix(alphaSix_),
		uni_anisotropy(uni_anisotropy_),
		alpha1VecSizes(alpha1VecSizes_),
		alpha2VecSizes(alpha2VecSizes_),
		alpha3VecSizes(alpha3VecSizes_),
		alpha4VecSizes(alpha4VecSizes_),
		alpha5VecSizes(alpha5VecSizes_),
		alpha6VecSizes(alpha6VecSizes_),
		anisVecSizes(anisVecSizes_),
		gridSize(gridSize_),
		Ncz(Ncz_),
		N_d(N_d_),
		flag_study(flag_study_)
	{
	}

	// ------------------------------------------------------------------------
	// Functor call operator
	// ------------------------------------------------------------------------

	/**
	 * @brief Compute the Landau effective field at the given grid index.
	 *
	 * This evaluates the derivative of the Landau free energy with respect
	 * to polarization at the specified grid point.
	 *
	 * @param idx Index in the polarization field array.
	 * @return Effective field contribution at the given point.
	 */
	__host__ __device__
		Vec3<T> operator()(int idx) const;
};

// Implementation of operator() is in the corresponding .cu file
#include "landaufreeField.cu"

#endif // _LANDAUFREEFIELD_CUH_
