#pragma once
#include "landauFreeEnergy.cuh"

/**
 * @brief Computes the Landau free energy at a given grid index.
 *
 * This operator evaluates the Landau-Devonshire free energy functional
 * based on the local polarization vector and the temperature-dependent
 * Landau coefficients (α₁ ... α₆). It supports both uniform and spatially
 * varying coefficients along the z-direction.
 *
 * @tparam T Numeric type (e.g., float or double).
 * @param idx Linear index within the polarization grid.
 * @return Energy contribution at the specified index.
 */
template <typename T>
__host__ __device__
T landauFreeEnergy<T>::operator()(int idx) const
{
	// ------------------------------------------------------------------------
	// Retrieve Landau coefficients α1 ... α6 for this grid index
	// ------------------------------------------------------------------------
	T alphaOne_val, alphaTwo_val, alphaThree_val;
	T alphaFour_val, alphaFive_val, alphaSix_val, uni_anis_val;

	Vec3<T> p = polarization[idx];

    //--------------------------------------------------------------------------
    // Determine current simulation ID for multi-domain systems
    //--------------------------------------------------------------------------
    int simId = idx / (gridSize / N_d);

    //--------------------------------------------------------------------------
    // Retrieve α₁ parameter based on its defined vector size
    //--------------------------------------------------------------------------
    if (alpha1VecSizes == 1)                      alphaOne_val = alphaOne[0];
    else if (alpha1VecSizes == Ncz)               alphaOne_val = alphaOne[idx / (gridSize / Ncz)];
    else if (alpha1VecSizes == gridSize)          alphaOne_val = alphaOne[idx];
    else if (alpha1VecSizes == N_d && flag_study) alphaOne_val = alphaOne[simId];
    else                                          alphaOne_val = alphaOne[0];

    //--------------------------------------------------------------------------
    // Retrieve α₁₁ parameter
    //--------------------------------------------------------------------------
    if (alpha2VecSizes == 1)                      alphaTwo_val = alphaTwo[0];
    else if (alpha2VecSizes == Ncz)               alphaTwo_val = alphaTwo[idx / (gridSize / Ncz)];
    else if (alpha2VecSizes == gridSize)          alphaTwo_val = alphaTwo[idx];
    else if (alpha2VecSizes == N_d && flag_study) alphaTwo_val = alphaTwo[simId];
    else                                          alphaTwo_val = alphaTwo[0];

    //--------------------------------------------------------------------------
    // Retrieve α₁₂ parameter
    //--------------------------------------------------------------------------
    if (alpha3VecSizes == 1)                      alphaThree_val = alphaThree[0];
    else if (alpha3VecSizes == Ncz)               alphaThree_val = alphaThree[idx / (gridSize / Ncz)];
    else if (alpha3VecSizes == gridSize)          alphaThree_val = alphaThree[idx];
    else if (alpha3VecSizes == N_d && flag_study) alphaThree_val = alphaThree[simId];
    else                                          alphaThree_val = alphaThree[0];

    //--------------------------------------------------------------------------
    // Retrieve α₁₁₁ parameter
    //--------------------------------------------------------------------------
    if (alpha4VecSizes == 1)                      alphaFour_val = alphaFour[0];
    else if (alpha4VecSizes == Ncz)               alphaFour_val = alphaFour[idx / (gridSize / Ncz)];
    else if (alpha4VecSizes == gridSize)          alphaFour_val = alphaFour[idx];
    else if (alpha4VecSizes == N_d && flag_study) alphaFour_val = alphaFour[simId];
    else                                          alphaFour_val = alphaFour[0];

    //--------------------------------------------------------------------------
    // Retrieve α₁₁₂ parameter
    //--------------------------------------------------------------------------
    if (alpha5VecSizes == 1)                      alphaFive_val = alphaFive[0];
    else if (alpha5VecSizes == Ncz)               alphaFive_val = alphaFive[idx / (gridSize / Ncz)];
    else if (alpha5VecSizes == gridSize)          alphaFive_val = alphaFive[idx];
    else if (alpha5VecSizes == N_d && flag_study) alphaFive_val = alphaFive[simId];
    else                                          alphaFive_val = alphaFive[0];

    //--------------------------------------------------------------------------
    // Retrieve α₁₂₃ parameter
    //--------------------------------------------------------------------------
    if (alpha6VecSizes == 1)                      alphaSix_val = alphaSix[0];
    else if (alpha6VecSizes == Ncz)               alphaSix_val = alphaSix[idx / (gridSize / Ncz)];
    else if (alpha6VecSizes == gridSize)          alphaSix_val = alphaSix[idx];
    else if (alpha6VecSizes == N_d && flag_study) alphaSix_val = alphaSix[simId];
    else                                          alphaSix_val = alphaSix[0];

    if (anisVecSizes == 1)                      uni_anis_val = uni_anisotropy[0];
    else if (anisVecSizes == Ncz)               uni_anis_val = uni_anisotropy[idx / (gridSize / Ncz)];
    else if (anisVecSizes == gridSize)          uni_anis_val = uni_anisotropy[idx];
    else if (anisVecSizes == N_d && flag_study) uni_anis_val = uni_anisotropy[simId];
    else                                        uni_anis_val = uni_anisotropy[0];

	// ------------------------------------------------------------------------
	// Precompute frequently used polarization terms
	// ------------------------------------------------------------------------
	T p_x2 = p.x * p.x;
	T p_y2 = p.y * p.y;
	T p_z2 = p.z * p.z;

	T sumSquares = p_x2 + p_y2 + p_z2;
	T sumsquare2 = sumSquares * sumSquares;

	T p_x3 = p_x2 * p.x;
	T p_y3 = p_y2 * p.y;
	T p_z3 = p_z2 * p.z;

	T p_x4 = p_x3 * p.x;
	T p_y4 = p_y3 * p.y;
	T p_z4 = p_z3 * p.z;

	T px2_sum_py2 = p_x2 + p_y2;
	T py2_sum_pz2 = p_y2 + p_z2;
	T px2_sum_pz2 = p_x2 + p_z2;

	// ------------------------------------------------------------------------
	// Compute Landau free energy density
	// ------------------------------------------------------------------------
	T energy =
		alphaOne_val * (p_x2 + p_y2 + p_z2 * uni_anis_val) +
		alphaTwo_val * sumsquare2 +
		(alphaThree_val - 2 * alphaTwo_val) *
		(p_x2 * p_y2 + p_y2 * p_z2 + p_z2 * p_x2) +
		alphaFour_val * (sumSquares * sumsquare2) +
		(alphaFive_val - 3 * alphaFour_val) *
		(p_x4 * py2_sum_pz2 + p_y4 * px2_sum_pz2 + p_z4 * px2_sum_py2) +
		(alphaSix_val - 6 * alphaFour_val) * p_x2 * p_y2 * p_z2;

	return energy;
}
