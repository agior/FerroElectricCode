#pragma once

// ===============================================================
// Includes
// ===============================================================
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <chrono>

#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include "ginzburgField.cuh"

// ===============================================================
// gradientField<T> Implementation
// ===============================================================

template<typename T>
gradientField<T>::gradientField
(
	int* shape, int flag_BC,
	int dim1, int dim2, int dim3,
	T stride1, T stride2, T stride3,
	Vec3<T>* p, Vec3<T>* fieldFerro,
	int calcFlag, int ginzburgFlag,
	T* G1_, T* G2_, T* G3_, T* G4_,
	int FLAG_G1_, int FLAG_G2_, int FLAG_G3_, int FLAG_G4_
)
	:
	device(shape),
	boundaryConditionsFlag(flag_BC),
	dimX(dim1), dimY(dim2), dimXY(dim1* dim2), dimZ(dim3),
	invStrideX(1.0 / stride1), invStrideY(1.0 / stride2), invStrideZ(1.0 / stride3),
	polarization(p), hFerro(fieldFerro),
	nears(calcFlag), FLAG_GINZBURG(ginzburgFlag),
	G1(G1_), G2(G2_), G3(G3_), G4(G4_),
	FLAG_G1(FLAG_G1_), FLAG_G2(FLAG_G2_), FLAG_G3(FLAG_G3_), FLAG_G4(FLAG_G4_)
{
}

// ===============================================================
// Dv1x()
// ---------------------------------------------------------------
// Computes first-order derivative along X depending on BC type
// ===============================================================
template<typename T>
__host__ __device__
Vec3<T> gradientField<T>::Dv1x(const int index, const int i) const
{
	switch (boundaryConditionsFlag)
	{
	case 0:
	case 3:
		switch (nears)
		{
		case 1: return Dv1_2x_simpleCopy(index, i);
		default: return {};
		}

	case 1:
		switch (nears)
		{
		case 1: return Dv1_2x_copy_iDMI(index, i);
		default: return {};
		}

	case 2:
		switch (nears)
		{
		case 1: return Dv1_2x_copy_bDMI(index, i);
		default: return {};
		}

	case 4:
		switch (nears)
		{
		case 1: return Dv1_2x_open(index, i);
		default: return {};
		}

	case 5:
	case 6:
	case 7:
		switch (nears)
		{
		case 1: return Dv1_2x_periodic(index, i);
		default: return {};
		}

	default:
		return {};
	}
}

// ===============================================================
// Dv1y()
// ---------------------------------------------------------------
// Computes first-order derivative along Y depending on BC type
// ===============================================================
template<typename T>
__host__ __device__
Vec3<T> gradientField<T>::Dv1y(const int index, const int j) const
{
	switch (boundaryConditionsFlag)
	{
	case 0:
	case 3:
	case 7:
		switch (nears)
		{
		case 1: return Dv1_2y_simpleCopy(index, j);
		default: return {};
		}

	case 1:
		switch (nears)
		{
		case 1: return Dv1_2y_copy_iDMI(index, j);
		default: return {};
		}

	case 2:
		switch (nears)
		{
		case 1: return Dv1_2y_copy_bDMI(index, j);
		default: return {};
		}

	case 4:
		switch (nears)
		{
		case 1: return Dv1_2y_open(index, j);
		default: return {};
		}

	case 5:
	case 6:
		switch (nears)
		{
		case 1: return Dv1_2y_periodic(index, j);
		default: return {};
		}

	default:
		return {};
	}
}

// ===============================================================
// Dv1z()
// ---------------------------------------------------------------
// Computes first-order derivative along Z depending on BC type
// ===============================================================
template<typename T>
__host__ __device__
Vec3<T> gradientField<T>::Dv1z(const int index, const int k) const
{
	switch (boundaryConditionsFlag)
	{
	case 0:
	case 3:
	case 6:
	case 7:
		switch (nears)
		{
		case 1: return Dv1_2z_simpleCopy(index, k);
		default: return {};
		}

	case 1:
		switch (nears)
		{
			// Placeholder for iDMI derivative (if implemented later)
			// case 1: return Dv1_2z_copy_iDMI(index, k);
		default: return {};
		}

	case 2:
		switch (nears)
		{
			// Placeholder for bDMI derivative (if implemented later)
			// case 1: return Dv1_2z_copy_bDMI(index, k);
		default: return {};
		}

	case 4:
		switch (nears)
		{
		case 1: return Dv1_2z_open(index, k);
		default: return {};
		}

	case 5:
		switch (nears)
		{
		case 1: return Dv1_2z_periodic(index, k);
		default: return {};
		}

	default:
		return {};
	}
}

// ===============================================================
// Dv2x()
// ---------------------------------------------------------------
// Computes second-order derivative along X depending on BC type
// ===============================================================
template<typename T>
__host__ __device__
Vec3<T> gradientField<T>::Dv2x(const int index, const int i) const
{
	switch (boundaryConditionsFlag)
	{
	case 0:
	case 3:
		switch (nears)
		{
		case 1: return Dv2_2x_simpleCopy(index, i);
		default: return {};
		}

	case 1:
		switch (nears)
		{
			// Placeholder for iDMI second-order derivative
		default: return {};
		}

	case 2:
		switch (nears)
		{
			// Placeholder for bDMI second-order derivative
		default: return {};
		}

	case 4:
		switch (nears)
		{
		case 1: return Dv2_2x_open(index, i);
		default: return {};
		}

	case 5:
	case 6:
	case 7:
		switch (nears)
		{
		case 1: return Dv2_2x_periodic(index, i);
		default: return {};
		}

	default:
		return {};
	}
}
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv2y(const int index, const int j) const
{
	switch (boundaryConditionsFlag)
	{
	case 0:
	case 3:
	case 7:
		switch (nears)
		{
			// case 0: return ((j == 0) || j == (dimY - 1)) ? Dv2_2y_simpleCopy(index, j) : Dv2_4y_???(index, j);
		case 1: return Dv2_2y_simpleCopy(index, j);
			// case 2: return Dv2_4y_???(index, j);
		default: return {};
		}

	case 1:
		switch (nears)
		{
			// case 0: return ((j == 0) || j == (dimY - 1)) ? Dv2_2y_copy_iDMI(index, j) : Dv2_4y_copy_iDMI(index, j);
			// case 1: return Dv2_2y_copy_iDMI(index, j);
			// case 2: return Dv2_4y_copy_iDMI(index, j);
		default: return {};
		}

	case 2:
		switch (nears)
		{
			// case 0: return ((j == 0) || j == (dimY - 1)) ? Dv2_2y_copy_bDMI(index, j) : Dv2_4y_copy_bDMI(index, j);
			// case 1: return Dv2_2y_copy_bDMI(index, j);
			// case 2: return Dv2_4y_copy_bDMI(index, j);
		default: return {};
		}

	case 4:
		switch (nears)
		{
			// case 0: return ((j == 0) || j == (dimY - 1)) ? Dv2_2y_open(index, j) : Dv2_4y_open(index, j);
		case 1: return Dv2_2y_open(index, j);
			// case 2: return Dv2_4y_open(index, j);
		default: return {};
		}

	case 5:
	case 6:
		switch (nears)
		{
			// case 0: return ((j == 0) || j == (dimY - 1)) ? Dv2_2y_periodic(index, j) : Dv2_4y_periodic(index, j);
		case 1: return Dv2_2y_periodic(index, j);
			// case 2: return Dv2_4y_periodic(index, j);
		default: return {};
		}
	}
}

template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv2z(const int index, const int k) const
{
	switch (boundaryConditionsFlag)
	{
	case 0:
	case 3:
	case 6:
	case 7:
		switch (nears)
		{
			// case 0: return ((k == 0) || k == (dimZ - 1)) ? Dv2_2z_simpleCopy(index, k) : Dv2_4z_???(index, k);
		case 1: return Dv2_2z_simpleCopy(index, k);
			// case 2: return Dv2_4z_???(index, k);
		default: return {};
		}

	case 1:
		switch (nears)
		{
			// case 0: return ((k == 0) || k == (dimZ - 1)) ? Dv2_2z_copy_iDMI(index, k) : Dv2_4z_copy_iDMI(index, k);
			// case 1: return Dv2_2z_copy_iDMI(index, k);
			// case 2: return Dv2_4z_copy_iDMI(index, k);
		default: return {};
		}

	case 2:
		switch (nears)
		{
			// case 0: return ((k == 0) || k == (dimZ - 1)) ? Dv2_2z_copy_bDMI(index, k) : Dv2_4z_copy_bDMI(index, k);
			// case 1: return Dv2_2z_copy_bDMI(index, k);
			// case 2: return Dv2_4z_copy_bDMI(index, k);
		default: return {};
		}

	case 4:
		switch (nears)
		{
			// case 0: return ((k == 0) || k == (dimZ - 1)) ? Dv2_2z_open(index, k) : Dv2_4z_open(index, k);
		case 1: return Dv2_2z_open(index, k);
			// case 2: return Dv2_4z_open(index, k);
		default: return {};
		}

	case 5:
		switch (nears)
		{
			// case 0: return ((k == 0) || k == (dimZ - 1)) ? Dv2_2z_periodic(index, k) : Dv2_4z_periodic(index, k);
		case 1: return Dv2_2z_periodic(index, k);
			// case 2: return Dv2_4z_periodic(index, k);
		default: return {};
		}
	}
}

/**
 * @brief Partial gradient computation along X with base index for derived updates.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::p_Dv2x(const int index, const int i, const int baseIdx) const
{
	switch (boundaryConditionsFlag)
	{
	case 0:
		switch (nears)
		{
			// case 0: return ((i == 0) || i == (dimX - 1)) ? Dv2_2x_simpleCopy(index, i) : Dv2_4x_???(index, i);
		case 1: return Dv2_2x_simpleCopy(index, i, baseIdx);
			// case 2: return Dv2_4x_???(index, i);
		default: return {};
		}

	case 1:
		switch (nears)
		{
			// case 0: return ((i == 0) || i == (dimX - 1)) ? Dv2_2x_open(index, i) : Dv2_4x_open(index, i);
		case 1: return Dv2_2x_open(index, i, baseIdx);
			// case 2: return Dv2_4x_open(index, i);
		default: return {};
		}

	case 2:
		switch (nears)
		{
			// case 0: return ((i == 0) || i == (dimX - 1)) ? Dv2_2x_periodic(index, i) : Dv2_4x_periodic(index, i);
		case 1: return Dv2_2x_periodic(index, i, baseIdx);
			// case 2: return Dv2_4x_periodic(index, i);
		default: return {};
		}
	}
}

/**
 * @brief Partial gradient computation along Y with base index for derived updates.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::p_Dv2y(const int index, const int j, const int baseIdx) const
{
	switch (boundaryConditionsFlag)
	{
	case 0:
		switch (nears)
		{
			// case 0: return ((j == 0) || j == (dimY - 1)) ? Dv2_2y_simpleCopy(index, j) : Dv2_4y_???(index, j);
		case 1: return Dv2_2y_simpleCopy(index, j, baseIdx);
			// case 2: return Dv2_4y_???(index, j);
		default: return {};
		}

	case 1:
		switch (nears)
		{
			// case 0: return ((j == 0) || j == (dimY - 1)) ? Dv2_2y_open(index, j) : Dv2_4y_open(index, j);
		case 1: return Dv2_2y_open(index, j, baseIdx);
			// case 2: return Dv2_4y_open(index, j);
		default: return {};
		}

	case 2:
		switch (nears)
		{
			// case 0: return ((j == 0) || j == (dimY - 1)) ? Dv2_2y_periodic(index, j) : Dv2_4y_periodic(index, j);
		case 1: return Dv2_2y_periodic(index, j, baseIdx);
			// case 2: return Dv2_4y_periodic(index, j);
		default: return {};
		}
	}
}

/**
 * @brief Partial gradient computation along Z with base index for derived updates.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::p_Dv2z(const int index, const int k, const int baseIdx) const
{
	switch (boundaryConditionsFlag)
	{
	case 0:
		switch (nears)
		{
			// case 0: return ((k == 0) || k == (dimZ - 1)) ? Dv2_2z_simpleCopy(index, k) : Dv2_4z_???(index, k);
		case 1: return Dv2_2z_simpleCopy(index, k, baseIdx);
			// case 2: return Dv2_4z_???(index, k);
		default: return {};
		}

	case 1:
		switch (nears)
		{
			// case 0: return ((k == 0) || k == (dimZ - 1)) ? Dv2_2z_open(index, k) : Dv2_4z_open(index, k);
		case 1: return Dv2_2z_open(index, k, baseIdx);
			// case 2: return Dv2_4z_open(index, k);
		default: return {};
		}

	case 2:
		switch (nears)
		{
			// case 0: return ((k == 0) || k == (dimZ - 1)) ? Dv2_2z_periodic(index, k) : Dv2_4z_periodic(index, k);
		case 1: return Dv2_2z_periodic(index, k, baseIdx);
			// case 2: return Dv2_4z_periodic(index, k);
		default: return {};
		}
	}
}template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::p_Dvxy(const int index, const int i, const int j, const int baseIdx) const
{
	switch (boundaryConditionsFlag)
	{
	case 0:
		switch (nears)
		{
			// case 0: return ((i == 0) || i == (dimX - 1)) ? Dv2_2x_simpleCopy(index, i) : Dv2_4x_???(index, i);
		case 1: return Dv2_xy_simpleCopy(index, i, j, baseIdx);
			// case 2: return Dv2_4x_???(index, i);
		default: return {};
		}

	case 1:
		switch (nears)
		{
			// case 0: return ((i == 0) || i == (dimX - 1)) ? Dv2_2x_open(index, i) : Dv2_4x_open(index, i);
		case 1: return Dv2_xy_open(index, i, j, baseIdx);
			// case 2: return Dv2_4x_open(index, i);
		default: return {};
		}

	case 2:
		switch (nears)
		{
			// case 0: return ((i == 0) || i == (dimX - 1)) ? Dv2_2x_periodic(index, i) : Dv2_4x_periodic(index, i);
		case 1: return Dv2_xy_periodic(index, i, j, baseIdx);
			// case 2: return Dv2_4x_periodic(index, i);
		default: return {};
		}
	}
}

/**
 * @brief Computes the gradient in YZ direction at a given index based on boundary conditions.
 *
 * @tparam T Numeric type.
 * @param index   Linear index in the grid.
 * @param j       Y-coordinate index.
 * @param k       Z-coordinate index.
 * @param baseIdx Base index offset.
 * @return Vec3<T> Gradient vector at (j, k).
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::p_Dvyz(const int index, const int j, const int k, const int baseIdx) const
{
	switch (boundaryConditionsFlag)
	{
	case 0:
		switch (nears)
		{
			// case 0: return ((i == 0) || i == (dimX - 1)) ? Dv2_2x_simpleCopy(index, i) : Dv2_4x_???(index, i);
		case 1: return Dv2_yz_simpleCopy(index, j, k, baseIdx);
			// case 2: return Dv2_4x_???(index, i);
		default: return {};
		}

	case 1:
		switch (nears)
		{
			// case 0: return ((i == 0) || i == (dimX - 1)) ? Dv2_2x_open(index, i) : Dv2_4x_open(index, i);
		case 1: return Dv2_yz_open(index, j, k, baseIdx);
			// case 2: return Dv2_4x_open(index, i);
		default: return {};
		}

	case 2:
		switch (nears)
		{
			// case 0: return ((i == 0) || i == (dimX - 1)) ? Dv2_2x_periodic(index, i) : Dv2_4x_periodic(index, i);
		case 1: return Dv2_yz_periodic(index, j, k, baseIdx);
			// case 2: return Dv2_4x_periodic(index, i);
		default: return {};
		}
	}
}

/**
 * @brief Computes the gradient in XZ direction at a given index based on boundary conditions.
 *
 * @tparam T Numeric type.
 * @param index   Linear index in the grid.
 * @param i       X-coordinate index.
 * @param k       Z-coordinate index.
 * @param baseIdx Base index offset.
 * @return Vec3<T> Gradient vector at (i, k).
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::p_Dvxz(const int index, const int i, const int k, const int baseIdx) const
{
	switch (boundaryConditionsFlag)
	{
	case 0:
		switch (nears)
		{
			// case 0: return ((i == 0) || i == (dimX - 1)) ? Dv2_2x_simpleCopy(index, i) : Dv2_4x_???(index, i);
		case 1: return Dv2_xz_simpleCopy(index, i, k, baseIdx);
			// case 2: return Dv2_4x_???(index, i);
		default: return {};
		}

	case 1:
		switch (nears)
		{
			// case 0: return ((i == 0) || i == (dimX - 1)) ? Dv2_2x_open(index, i) : Dv2_4x_open(index, i);
		case 1: return Dv2_xz_open(index, i, k, baseIdx);
			// case 2: return Dv2_4x_open(index, i);
		default: return {};
		}

	case 2:
		switch (nears)
		{
			// case 0: return ((i == 0) || i == (dimX - 1)) ? Dv2_2x_periodic(index, i) : Dv2_4x_periodic(index, i);
		case 1: return Dv2_xz_periodic(index, i, k, baseIdx);
			// case 2: return Dv2_4x_periodic(index, i);
		default: return {};
		}
	}
}

/**
 * @brief Retrieves the G1 value based on its flag.
 *
 * @tparam T Numeric type.
 * @param index  Linear index.
 * @param flagG1 Control flag for access mode.
 * @param G1     Pointer to G1 array.
 * @return T Value from G1 based on flagG1.
 */
template<typename T>
__host__ __device__ T gradientField<T>::GetG1(const int index, const int flagG1, const T* G1) const
{
	// if (constants == nullptr) return 0.0;
	switch (flagG1)
	{
	case 1: return G1[0];

	case 2:
	{
		int z = index / dimXY;
		return G1[z];
	}

	case 3: return G1[index];

	default: return 0.0;
	}
}

/**
 * @brief Retrieves the G2 value based on its flag.
 *
 * @tparam T Numeric type.
 * @param index  Linear index.
 * @param flagG2 Control flag for access mode.
 * @param G2     Pointer to G2 array.
 * @return T Value from G2 based on flagG2.
 */
template<typename T>
__host__ __device__ T gradientField<T>::GetG2(const int index, const int flagG2, const T* G2) const
{
	// if (constants == nullptr) return 0.0;
	switch (flagG2)
	{
	case 1: return G2[0];

	case 2:
	{
		int z = index / dimXY;
		return G2[z];
	}

	case 3: return G2[index];

	default: return 0.0;
	}
}

/**
 * @brief Retrieves the G3 value based on its flag.
 *
 * @tparam T Numeric type.
 * @param index  Linear index.
 * @param flagG3 Control flag for access mode.
 * @param G3     Pointer to G3 array.
 * @return T Value from G3 based on flagG3.
 */
template<typename T>
__host__ __device__ T gradientField<T>::GetG3(const int index, const int flagG3, const T* G3) const
{
	// if (constants == nullptr) return 0.0;
	switch (flagG3)
	{
	case 1: return G3[0];

	case 2:
	{
		int z = index / dimXY;
		return G3[z];
	}

	case 3: return G3[index];

	default: return 0.0;
	}
}

/**
 * @brief Retrieves the G4 value based on its flag.
 *
 * @tparam T Numeric type.
 * @param index  Linear index.
 * @param flagG4 Control flag for access mode.
 * @param G4     Pointer to G4 array.
 * @return T Value from G4 based on flagG4.
 */
template<typename T>
__host__ __device__ T gradientField<T>::GetG4(const int index, const int flagG4, const T* G4) const
{
	// if (constants == nullptr) return 0.0;
	switch (flagG4)
	{
	case 1: return G4[0];

	case 2:
	{
		int z = index / dimXY;
		return G4[z];
	}

	case 3: return G4[index];

	default: return 0.0;
	}
}/**
 * @brief Computes the gradient field at a given linear index by applying
 *        second-order derivatives and G coefficients.
 *
 * @tparam T Numeric type.
 * @param index Linear index within the simulation domain.
 * @return Vec3<T> Gradient vector after applying differential operators.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::operator()(const int index) const
{
	int simSize = dimX * dimY * dimZ;
	int simId = index / simSize;
	int localIdx = index % simSize;
	int baseIdx = simId * simSize;

	int k = localIdx / dimXY;
	int j = (localIdx - k * dimXY) / dimX;
	int i = localIdx - k * dimXY - j * dimX;

	Vec3<T> h = hFerro[index];

	if (device[index])
	{
		// Second-order derivatives along principal and mixed directions
		Vec3<T> p_xx = p_Dv2x(localIdx, i, baseIdx);
		Vec3<T> p_yy = p_Dv2y(localIdx, j, baseIdx);
		Vec3<T> p_zz = p_Dv2z(localIdx, k, baseIdx);
		Vec3<T> p_xy = p_Dvxy(localIdx, i, j, baseIdx);
		Vec3<T> p_yz = p_Dvyz(localIdx, j, k, baseIdx);
		Vec3<T> p_xz = p_Dvxz(localIdx, i, k, baseIdx);

		// Coefficients for gradient terms
		T G1_val = GetG1(localIdx, FLAG_G1, G1);
		T G2_val = GetG2(localIdx, FLAG_G2, G2);
		T G3_val = GetG3(localIdx, FLAG_G3, G3);
		T G4_val = GetG4(localIdx, FLAG_G4, G4);

		// Accumulate contributions to h from each derivative term
		h.x += -G1_val * p_xx.x
			- G2_val * (p_xy.y + p_xz.z)
			- G3_val * (p_yy.x + p_xy.y + p_zz.x + p_xz.z)
			- G4_val * (p_yy.x - p_xy.y + p_zz.x - p_xz.z);

		h.y += -G1_val * p_yy.y
			- G2_val * (p_xy.x + p_yz.z)
			- G3_val * (p_xx.y + p_xy.x + p_zz.y + p_yz.z)
			- G4_val * (p_xx.y - p_xy.x + p_zz.y - p_yz.z);

		h.z += -G1_val * p_zz.z
			- G2_val * (p_xz.x + p_yz.y)
			- G3_val * (p_xx.z + p_xz.x + p_yy.z + p_yz.y)
			- G4_val * (p_xx.z - p_xz.x + p_yy.z - p_yz.y);
	}

	return h;
}

/**
 * @brief First-order derivative along X with simple copy boundary condition.
 *
 * @tparam T Numeric type.
 * @param index Linear index.
 * @param i     X-coordinate index.
 * @return Vec3<T> Gradient in X direction.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv1_2x_simpleCopy(const int index, const int i) const
{
	if (dimX == 1) return {};

	return (polarization[index + (i == (dimX - 1) ? 0 : 1)] -
		polarization[index + (i == 0 ? 0 : -1)]) * (0.5 * invStrideX);
}

/**
 * @brief First-order derivative along Y with simple copy boundary condition.
 *
 * @tparam T Numeric type.
 * @param index Linear index.
 * @param j     Y-coordinate index.
 * @return Vec3<T> Gradient in Y direction.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv1_2y_simpleCopy(const int index, const int j) const
{
	if (dimY == 1) return {};

	return (polarization[index + (j == (dimY - 1) ? 0 : dimX)] -
		polarization[index + (j == 0 ? 0 : -dimX)]) * (0.5 * invStrideY);
}

/**
 * @brief First-order derivative along Z with simple copy boundary condition.
 *
 * @tparam T Numeric type.
 * @param index Linear index.
 * @param k     Z-coordinate index.
 * @return Vec3<T> Gradient in Z direction.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv1_2z_simpleCopy(const int index, const int k) const
{
	if (dimZ == 1) return {};

	return (polarization[index + (k == (dimZ - 1) ? 0 : dimXY)] -
		polarization[index + (k == 0 ? 0 : -dimXY)]) * (0.5 * invStrideZ);
}

/**
 * @brief Second-order derivative along X with simple copy boundary condition.
 *
 * @tparam T Numeric type.
 * @param index   Linear index.
 * @param i       X-coordinate index.
 * @param baseIdx Base index offset.
 * @return Vec3<T> Second derivative in X direction.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv2_2x_simpleCopy(const int index, const int i, const int baseIdx) const
{
	if (dimX == 1) return {};

	return (polarization[baseIdx + (index + (i == 0 ? 0 : -1))] -
		polarization[baseIdx + index] -
		polarization[baseIdx + index] +
		polarization[baseIdx + (index + (i == (dimX - 1) ? 0 : 1))]) *
		invStrideX * invStrideX;
}

/**
 * @brief Second-order derivative along Y with simple copy boundary condition.
 *
 * @tparam T Numeric type.
 * @param index   Linear index.
 * @param j       Y-coordinate index.
 * @param baseIdx Base index offset.
 * @return Vec3<T> Second derivative in Y direction.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv2_2y_simpleCopy(const int index, const int j, const int baseIdx) const
{
	if (dimY == 1) return {};

	return (polarization[baseIdx + (index + (j == 0 ? 0 : -dimX))] -
		polarization[baseIdx + index] -
		polarization[baseIdx + index] +
		polarization[baseIdx + (index + (j == (dimY - 1) ? 0 : dimX))]) *
		invStrideY * invStrideY;
}

/**
 * @brief Second-order derivative along Z with simple copy boundary condition.
 *
 * @tparam T Numeric type.
 * @param index   Linear index.
 * @param k       Z-coordinate index.
 * @param baseIdx Base index offset.
 * @return Vec3<T> Second derivative in Z direction.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv2_2z_simpleCopy(const int index, const int k, const int baseIdx) const
{
	if (dimZ == 1) return {};

	return (polarization[baseIdx + (index + (k == 0 ? 0 : -dimXY))] -
		polarization[baseIdx + index] -
		polarization[baseIdx + index] +
		polarization[baseIdx + (index + (k == (dimZ - 1) ? 0 : dimXY))]) *
		invStrideZ * invStrideZ;
}

/**
 * @brief Second-order mixed derivative in XY with simple copy boundary condition.
 *
 * @tparam T Numeric type.
 * @param index   Linear index.
 * @param i       X-coordinate index.
 * @param j       Y-coordinate index.
 * @param baseIdx Base index offset.
 * @return Vec3<T> Mixed derivative in XY plane.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv2_xy_simpleCopy(
	const int index, const int i, const int j, const int baseIdx) const
{
	if (dimX == 1 || dimY == 1) return {};

	Vec3<T> p_xy =
		(polarization[baseIdx + (index + (i == (dimX - 1) ? 0 : 1) + (j == (dimY - 1) ? 0 : dimX))] -
			polarization[baseIdx + (index + (i == (dimX - 1) ? 0 : 1) + (j == 0 ? 0 : -dimX))] -
			polarization[baseIdx + (index + (i == 0 ? 0 : -1) + (j == (dimY - 1) ? 0 : dimX))] +
			polarization[baseIdx + (index + (i == 0 ? 0 : -1) + (j == 0 ? 0 : -dimX))]) *
		0.25 * invStrideX * invStrideY;

	return p_xy;
}

/**
 * @brief Second-order mixed derivative in YZ with simple copy boundary condition.
 *
 * @tparam T Numeric type.
 * @param index   Linear index.
 * @param j       Y-coordinate index.
 * @param k       Z-coordinate index.
 * @param baseIdx Base index offset.
 * @return Vec3<T> Mixed derivative in YZ plane.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv2_yz_simpleCopy(
	const int index, const int j, const int k, const int baseIdx) const
{
	if (dimY == 1 || dimZ == 1) return {};

	Vec3<T> p_yz =
		(polarization[baseIdx + (index + (j == (dimY - 1) ? 0 : dimX) + (k == (dimZ - 1) ? 0 : dimXY))] -
			polarization[baseIdx + (index + (j == (dimY - 1) ? 0 : dimX) + (k == 0 ? 0 : -dimXY))] -
			polarization[baseIdx + (index + (j == 0 ? 0 : -dimX) + (k == (dimZ - 1) ? 0 : dimXY))] +
			polarization[baseIdx + (index + (j == 0 ? 0 : -dimX) + (k == 0 ? 0 : -dimXY))]) *
		0.25 * invStrideY * invStrideZ;

	return p_yz;
}

/**
 * @brief Second-order mixed derivative in XZ with simple copy boundary condition.
 *
 * @tparam T Numeric type.
 * @param index   Linear index.
 * @param i       X-coordinate index.
 * @param k       Z-coordinate index.
 * @param baseIdx Base index offset.
 * @return Vec3<T> Mixed derivative in XZ plane.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv2_xz_simpleCopy(
	const int index, const int i, const int k, const int baseIdx) const
{
	if (dimX == 1 || dimZ == 1) return {};

	Vec3<T> p_xz =
		(polarization[baseIdx + (index + (i == (dimX - 1) ? 0 : 1) + (k == (dimZ - 1) ? 0 : dimXY))] -
			polarization[baseIdx + (index + (i == (dimX - 1) ? 0 : 1) + (k == 0 ? 0 : -dimXY))] -
			polarization[baseIdx + (index + (i == 0 ? 0 : -1) + (k == (dimZ - 1) ? 0 : dimXY))] +
			polarization[baseIdx + (index + (i == 0 ? 0 : -1) + (k == 0 ? 0 : -dimXY))]) *
		0.25 * invStrideX * invStrideZ;

	return p_xz;
}
/**
 * @brief Compute first-order derivative in x-direction with open boundary conditions.
 *
 * @tparam T Numeric type.
 * @param index Linear index in the array.
 * @param i     X-coordinate index.
 * @return First-order derivative vector at the given point.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv1_2x_open(const int index, const int i) const
{
	if (dimX == 1) return {};

	return ((i == (dimX - 1) ? Vec3<T>() : polarization[index + 1])
		- (i == 0 ? Vec3<T>() : polarization[index - 1])) * (0.5 * invStrideX);
}

/**
 * @brief Compute first-order derivative in y-direction with open boundary conditions.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv1_2y_open(const int index, const int j) const
{
	if (dimY == 1) return {};

	return ((j == (dimY - 1) ? Vec3<T>() : polarization[index + dimX])
		- (j == 0 ? Vec3<T>() : polarization[index - dimX])) * (0.5 * invStrideY);
}

/**
 * @brief Compute first-order derivative in z-direction with open boundary conditions.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv1_2z_open(const int index, const int k) const
{
	if (dimZ == 1) return {};

	return ((k == (dimZ - 1) ? Vec3<T>() : polarization[index + dimXY])
		- (k == 0 ? Vec3<T>() : polarization[index - dimXY])) * (0.5 * invStrideZ);
}

/**
 * @brief Compute second-order derivative in x-direction with open boundary conditions.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv2_2x_open(const int index, const int i, const int baseIdx) const
{
	if (dimX == 1) return {};

	return ((i == 0 ? Vec3<T>() : polarization[baseIdx + (index - 1)]) - polarization[baseIdx + index]
		- polarization[baseIdx + index] +
		(i == (dimX - 1) ? Vec3<T>() : polarization[baseIdx + (index + 1)])) * invStrideX * invStrideX;
}

/**
 * @brief Compute second-order derivative in y-direction with open boundary conditions.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv2_2y_open(const int index, const int j, const int baseIdx) const
{
	if (dimY == 1) return {};

	return ((j == 0 ? Vec3<T>() : polarization[baseIdx + (index - dimX)]) - polarization[baseIdx + index]
		- polarization[baseIdx + index] +
		(j == (dimY - 1) ? Vec3<T>() : polarization[baseIdx + (index + dimX)])) * invStrideY * invStrideY;
}

/**
 * @brief Compute second-order derivative in z-direction with open boundary conditions.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv2_2z_open(const int index, const int k, const int baseIdx) const
{
	if (dimZ == 1) return {};

	return ((k == 0 ? Vec3<T>() : polarization[baseIdx + (index - dimXY)]) - polarization[baseIdx + index]
		- polarization[baseIdx + index] +
		(k == (dimZ - 1) ? Vec3<T>() : polarization[baseIdx + (index + dimXY)])) * invStrideZ * invStrideZ;
}

/**
 * @brief Compute mixed second-order derivative in xy-plane with open boundary conditions.
 */
template <typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv2_xy_open(const int index, const int i, const int j, const int baseIdx) const
{
	if (dimX == 1 || dimY == 1) return {};

	Vec3<T> p_xy =
		((i == (dimX - 1) || j == (dimY - 1) ? Vec3<T>() : polarization[baseIdx + (index + 1 + dimX)]) -
			(i == (dimX - 1) || j == 0 ? Vec3<T>() : polarization[baseIdx + (index + 1 - dimX)]) -
			(i == 0 || j == (dimY - 1) ? Vec3<T>() : polarization[baseIdx + (index - 1 + dimX)]) +
			(i == 0 || j == 0 ? Vec3<T>() : polarization[baseIdx + (index - 1 - dimX)]))
		* 0.25 * invStrideX * invStrideY;

	return p_xy;
}

/**
 * @brief Compute mixed second-order derivative in yz-plane with open boundary conditions.
 */
template <typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv2_yz_open(const int index, const int j, const int k, const int baseIdx) const
{
	if (dimY == 1 || dimZ == 1) return {};

	Vec3<T> p_yz =
		((j == (dimY - 1) || k == (dimZ - 1) ? Vec3<T>() : polarization[baseIdx + (index + dimX + dimXY)]) -
			(j == (dimY - 1) || k == 0 ? Vec3<T>() : polarization[baseIdx + (index + dimX - dimXY)]) -
			(j == 0 || k == (dimZ - 1) ? Vec3<T>() : polarization[baseIdx + (index - dimX + dimXY)]) +
			(j == 0 || k == 0 ? Vec3<T>() : polarization[baseIdx + (index - dimX - dimXY)]))
		* 0.25 * invStrideY * invStrideZ;

	return p_yz;
}

/**
 * @brief Compute mixed second-order derivative in xz-plane with open boundary conditions.
 */
template <typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv2_xz_open(const int index, const int i, const int k, const int baseIdx) const
{
	if (dimX == 1 || dimZ == 1) return {};

	Vec3<T> p_xz =
		((i == (dimX - 1) || k == (dimZ - 1) ? Vec3<T>() : polarization[baseIdx + (index + 1 + dimXY)]) -
			(i == (dimX - 1) || k == 0 ? Vec3<T>() : polarization[baseIdx + (index + 1 - dimXY)]) -
			(i == 0 || k == (dimZ - 1) ? Vec3<T>() : polarization[baseIdx + (index - 1 + dimXY)]) +
			(i == 0 || k == 0 ? Vec3<T>() : polarization[baseIdx + (index - 1 - dimXY)]))
		* 0.25 * invStrideX * invStrideZ;

	return p_xz;
}

/**
 * @brief Compute next-nearest neighbour second-order derivative in x-direction with open boundary conditions.
 *        Uses fourth-order accurate central difference stencil.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv2_4x_open(const int index, const int i) const
{
	if (dimX == 1) return {};

	return
		((i < 2 ? Vec3<T>() : -(1.0 / 12.0) * polarization[index - 2]) +
			(i == 0 ? Vec3<T>() : +(4.0 / 3.0) * polarization[index - 1]) +
			(i == (dimX - 1) ? Vec3<T>() : +(4.0 / 3.0) * polarization[index + 1]) +
			(i > (dimX - 3) ? Vec3<T>() : -(1.0 / 12.0) * polarization[index + 2]))
		* invStrideX * invStrideX;
}

/**
 * @brief Compute next-nearest neighbour second-order derivative in y-direction with open boundary conditions.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv2_4y_open(const int index, const int j) const
{
	if (dimY == 1) return {};

	return
		((j < 2 ? Vec3<T>() : -(1.0 / 12.0) * polarization[index - 2 * dimX]) +
			(j == (dimY - 1) ? Vec3<T>() : +(4.0 / 3.0) * polarization[index + dimX]) +
			(j == 0 ? Vec3<T>() : +(4.0 / 3.0) * polarization[index - dimX]) +
			(j > (dimY - 3) ? Vec3<T>() : -(1.0 / 12.0) * polarization[index + 2 * dimX]))
		* invStrideY * invStrideY;
}

/**
 * @brief Compute next-nearest neighbour second-order derivative in z-direction with open boundary conditions.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv2_4z_open(const int index, const int k) const
{
	if (dimZ == 1) return {};

	return
		((k < 2 ? Vec3<T>() : -(1.0 / 12.0) * polarization[index - 2 * dimXY]) +
			(k == 0 ? Vec3<T>() : +(4.0 / 3.0) * polarization[index - dimXY]) +
			(k == (dimZ - 1) ? Vec3<T>() : +(4.0 / 3.0) * polarization[index + dimXY]) +
			(k > (dimZ - 3) ? Vec3<T>() : -(1.0 / 12.0) * polarization[index + 2 * dimXY]))
		* invStrideZ * invStrideZ;
}

/**
 * @brief Compute first-order derivative in x-direction with periodic boundary conditions.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv1_2x_periodic(const int index, const int i) const
{
	if (dimX == 1) return {};

	return (polarization[index + (i == (dimX - 1) ? -i : 1)]
		- polarization[index + (i == 0 ? (dimX - 1) : -1)]) * (0.5 * invStrideX);
}

/**
 * @brief Compute first-order derivative in y-direction with periodic boundary conditions.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv1_2y_periodic(const int index, const int j) const
{
	if (dimY == 1) return {};

	return (polarization[index + (j == (dimY - 1) ? -j * dimX : dimX)]
		- polarization[index + (j == 0 ? (dimY - 1) * dimX : -dimX)]) * (0.5 * invStrideY);
}

/**
 * @brief Compute first-order derivative in z-direction with periodic boundary conditions.
 */
template<typename T>
__host__ __device__ Vec3<T> gradientField<T>::Dv1_2z_periodic(const int index, const int k) const
{
	if (dimZ == 1) return {};

	return (polarization[index + (k == (dimZ - 1) ? -k * dimXY : dimXY)]
		- polarization[index + (k == 0 ? (dimZ - 1) * dimXY : -dimXY)]) * (0.5 * invStrideZ);
}
// ============================================================================
// Nearest neighbours second order derivative — periodic boundary conditions
// ============================================================================

/**
 * @brief Computes the second-order derivative in the X-direction with periodic boundary conditions.
 * @tparam T Data type of the vector field.
 * @param index Linear index in the polarization array.
 * @param i X-coordinate index.
 * @param baseIdx Base index offset for the polarization field.
 * @return Second-order derivative vector along X.
 */
template<typename T>
__host__ __device__
Vec3<T> gradientField<T>::Dv2_2x_periodic(const int index, const int i, const int baseIdx) const
{
	if (dimX == 1) return {};

	return (
		polarization[baseIdx + (index + (i == 0 ? (dimX - 1) : -1))] -
		polarization[baseIdx + index] -
		polarization[baseIdx + index] +
		polarization[baseIdx + (index + (i == (dimX - 1) ? -i : 1))]
		) * invStrideX * invStrideX;
}

/**
 * @brief Computes the second-order derivative in the Y-direction with periodic boundary conditions.
 * @tparam T Data type of the vector field.
 * @param index Linear index in the polarization array.
 * @param j Y-coordinate index.
 * @param baseIdx Base index offset for the polarization field.
 * @return Second-order derivative vector along Y.
 */
template<typename T>
__host__ __device__
Vec3<T> gradientField<T>::Dv2_2y_periodic(const int index, const int j, const int baseIdx) const
{
	if (dimY == 1) return {};

	return (
		polarization[baseIdx + (index + (j == 0 ? (dimY - 1) * dimX : -dimX))] -
		polarization[baseIdx + index] -
		polarization[baseIdx + index] +
		polarization[baseIdx + (index + (j == (dimY - 1) ? -j * dimX : dimX))]
		) * invStrideY * invStrideY;
}

/**
 * @brief Computes the second-order derivative in the Z-direction with periodic boundary conditions.
 * @tparam T Data type of the vector field.
 * @param index Linear index in the polarization array.
 * @param k Z-coordinate index.
 * @param baseIdx Base index offset for the polarization field.
 * @return Second-order derivative vector along Z.
 */
template<typename T>
__host__ __device__
Vec3<T> gradientField<T>::Dv2_2z_periodic(const int index, const int k, const int baseIdx) const
{
	if (dimZ == 1) return {};

	return (
		polarization[baseIdx + (index + (k == 0 ? (dimZ - 1) * dimXY : -dimXY))] -
		polarization[baseIdx + index] -
		polarization[baseIdx + index] +
		polarization[baseIdx + (index + (k == (dimZ - 1) ? -k * dimXY : dimXY))]
		) * invStrideZ * invStrideZ;
}

// ============================================================================
// Mixed derivatives — periodic boundary conditions
// ============================================================================

/**
 * @brief Computes the mixed second-order derivative in the XY-plane with periodic boundary conditions.
 * @tparam T Data type of the vector field.
 * @param index Linear index in the polarization array.
 * @param i X-coordinate index.
 * @param j Y-coordinate index.
 * @param baseIdx Base index offset for the polarization field.
 * @return Mixed derivative vector ∂²/∂x∂y.
 */
template <typename T>
__host__ __device__
Vec3<T> gradientField<T>::Dv2_xy_periodic(const int index, const int i, const int j, const int baseIdx) const
{
	if (dimX == 1 || dimY == 1) return {};

	Vec3<T> p_xy = (
		polarization[baseIdx + (index + (i == (dimX - 1) ? -i : 1) + (j == (dimY - 1) ? -j * dimX : dimX))] -
		polarization[baseIdx + (index + (i == (dimX - 1) ? -i : 1) + (j == 0 ? (dimY - 1) * dimX : -dimX))] -
		polarization[baseIdx + (index + (i == 0 ? (dimX - 1) : -1) + (j == (dimY - 1) ? -j * dimX : dimX))] +
		polarization[baseIdx + (index + (i == 0 ? (dimX - 1) : -1) + (j == 0 ? (dimY - 1) * dimX : -dimX))]
		) * 0.25 * invStrideX * invStrideY;

	return p_xy;
}

/**
 * @brief Computes the mixed second-order derivative in the YZ-plane with periodic boundary conditions.
 * @tparam T Data type of the vector field.
 * @param index Linear index in the polarization array.
 * @param j Y-coordinate index.
 * @param k Z-coordinate index.
 * @param baseIdx Base index offset for the polarization field.
 * @return Mixed derivative vector ∂²/∂y∂z.
 */
template<typename T>
__host__ __device__
Vec3<T> gradientField<T>::Dv2_yz_periodic(const int index, const int j, const int k, const int baseIdx) const
{
	if (dimY == 1 || dimZ == 1) return {};

	Vec3<T> p_yz = (
		polarization[baseIdx + (index + (j == (dimY - 1) ? -j * dimX : dimX) + (k == (dimZ - 1) ? -k * dimXY : dimXY))] -
		polarization[baseIdx + (index + (j == (dimY - 1) ? -j * dimX : dimX) + (k == 0 ? (dimZ - 1) * dimXY : -dimXY))] -
		polarization[baseIdx + (index + (j == 0 ? (dimY - 1) * dimX : -dimX) + (k == (dimZ - 1) ? -k * dimXY : dimXY))] +
		polarization[baseIdx + (index + (j == 0 ? (dimY - 1) * dimX : -dimX) + (k == 0 ? (dimZ - 1) * dimXY : -dimXY))]
		) * 0.25 * invStrideY * invStrideZ;

	return p_yz;
}

/**
 * @brief Computes the mixed second-order derivative in the XZ-plane with periodic boundary conditions.
 * @tparam T Data type of the vector field.
 * @param index Linear index in the polarization array.
 * @param i X-coordinate index.
 * @param k Z-coordinate index.
 * @param baseIdx Base index offset for the polarization field.
 * @return Mixed derivative vector ∂²/∂x∂z.
 */
template <typename T>
__host__ __device__
Vec3<T> gradientField<T>::Dv2_xz_periodic(const int index, const int i, const int k, const int baseIdx) const
{
	if (dimX == 1 || dimZ == 1) return {};

	Vec3<T> p_xz = (
		polarization[baseIdx + (index + (i == (dimX - 1) ? -i : 1) + (k == (dimZ - 1) ? -k * dimXY : dimXY))] -
		polarization[baseIdx + (index + (i == (dimX - 1) ? -i : 1) + (k == 0 ? (dimZ - 1) * dimXY : -dimXY))] -
		polarization[baseIdx + (index + (i == 0 ? (dimX - 1) : -1) + (k == (dimZ - 1) ? -k * dimXY : dimXY))] +
		polarization[baseIdx + (index + (i == 0 ? (dimX - 1) : -1) + (k == 0 ? (dimZ - 1) * dimXY : -dimXY))]
		) * 0.25 * invStrideX * invStrideZ;

	return p_xz;
}
