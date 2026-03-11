#pragma once
#include "tdglFunction.cuh"
template<typename T>
__host__ __device__ Vec3<T> tdgl_functor<T>::operator()(int idx) const {
	// Result vector
	Vec3<T> dp_dt;
	T variance;
	T L;

	if (LVectorSize == 1) {
		// Use single L value
		L = L_val[0];
	}
	else if (LVectorSize == Ncz) {
		// Compute z index
		int z = idx / (gridSize / Ncz);
		L = L_val[z];
	}
	else {
		// Use L per index
		L = L_val[idx];
	}

	// Compute dp_dt
	 dp_dt = (-L * h[idx]) + noiseVec[idx];

	return dp_dt;
}