#pragma once

#include "Vec3.h"
#include <thrust/execution_policy.h>

/**
 * @brief Landau free energy functor for evaluating energy density at each grid point.
 *
 * This struct stores pointers to polarization vectors and Landau expansion coefficients
 * (α₁ through α₆) and provides an `operator()` to evaluate the Landau-Devonshire free energy
 * functional at a given grid index. It is designed to run on both host and device.
 *
 * @tparam T Numeric type (e.g., float or double).
 */
template <typename T>
struct landauFreeEnergy
{
	// ------------------------------------------------------------------------
	// Member variables
	// ------------------------------------------------------------------------

	/// Pointer to array of polarization vectors at each grid point.
	Vec3<T>* polarization;

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

	/// Number of grid cells along the z direction (used for z-dependent coefficients).
	int Ncz;

	/// Number of domains or active cells (e.g., for masked operations).
	int N_d;

	/// Optional flag controlling study or mode selection.
	int flag_study;

	// ------------------------------------------------------------------------
	// Constructor
	// ------------------------------------------------------------------------

	/**
	 * @brief Construct a new landauFreeEnergy functor.
	 *
	 * @param polarization_ Pointer to polarization vector field.
	 * @param alphaOne_ Pointer to α₁ coefficients array.
	 * @param alphaTwo_ Pointer to α₁₁ coefficients array.
	 * @param alphaThree_ Pointer to α₁₂ coefficients array.
	 * @param alphaFour_ Pointer to α₁₁₁ coefficients array.
	 * @param alphaFive_ Pointer to α₁₁₂ coefficients array.
	 * @param alphaSix_ Pointer to α₁₂₃ coefficients array.
	 * @param alpha1VecSizes_ Size of α₁ array.
	 * @param alpha2VecSizes_ Size of α₁₁ array.
	 * @param alpha3VecSizes_ Size of α₁₂ array.
	 * @param alpha4VecSizes_ Size of α₁₁₁ array.
	 * @param alpha5VecSizes_ Size of α₁₁₂ array.
	 * @param alpha6VecSizes_ Size of α₁₂₃ array.
	 * @param gridSize_ Total number of grid points.
	 * @param Ncz_ Number of grid points along z.
	 */
	__host__ __device__
		landauFreeEnergy(Vec3<T>* polarization_,
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
	 * @brief Evaluate the Landau free energy at a given grid index.
	 *
	 * @param idx Index in the polarization field array.
	 * @return Landau free energy density at the given grid point.
	 */
	__host__ __device__
		T operator()(int idx) const;
};

// Implementation of operator() is in the corresponding .cu file
#include "landauFreeEnergy.cu"
