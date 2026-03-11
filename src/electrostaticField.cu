#define _USE_MATH_DEFINES
#include <cmath>
#include <fstream>
#include <iostream>
#include <chrono>
#include <iomanip>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cufftXt.h>

#include "electrostaticField.cuh"
#include "parameters.h"
#include "Vec3.h"

// --------------------------------------------------
// Global Variables
// --------------------------------------------------
extern FE_geometry FE_geom;
extern landau_free landau_free_param;

std::ofstream file_out;

// --------------------------------------------------
// Utility Function
// --------------------------------------------------
int nextPowerOf2(int n) {
	if (n == 0) return 1;
	if ((n & (n - 1)) == 0) return n << 1;

	n--;
	for (unsigned int i = 1; i < sizeof(n) * CHAR_BIT; i *= 2) {
		n |= n >> i;
	}
	return n + 1;
}

// --------------------------------------------------
// Main Electrostatic Field Routine
// --------------------------------------------------
void electrostaticField(
	Vec3<Type_var>* cu_m,                  // Input polarization (x, y, z)
	Vec3<Type_var>* electrostatic_field,   // Output field (x, y, z)
	SET_parameters& conf                   // Simulation parameters
) {
	// --------------------------------------------------
	// Grid & Simulation Setup
	// --------------------------------------------------
	int Mcx = conf.Mcx;
	int Mcy = conf.Mcy;
	int Mcz = conf.Mcz;
	int N_sim = FE_geom.N_d;

	// FFT parameters
	const int rank = 3;
	int n[3] = { Mcz, Mcy, Mcx };
	int inembed[3] = { Mcz, Mcy, Mcx };
	int onembed[3] = { Mcz, Mcy, Mcx };
	int istride = 1, ostride = 1;
	int idist = Mcx * Mcy * Mcz;
	int odist = idist;

	// Demag tensors
	HD_Complex_Type* cuSDxx = conf.demag_param.cuSDxx;
	HD_Complex_Type* cuSDyy = conf.demag_param.cuSDyy;
	HD_Complex_Type* cuSDzz = conf.demag_param.cuSDzz;
	HD_Complex_Type* cuSDxy = conf.demag_param.cuSDxy;
	HD_Complex_Type* cuSDxz = conf.demag_param.cuSDxz;
	HD_Complex_Type* cuSDyz = conf.demag_param.cuSDyz;

	// --------------------------------------------------
	// Allocate Temporary Buffers
	// --------------------------------------------------
	HD_Complex_Type* cuTempx, * cuTempy, * cuTempz;
	size_t totalSize = sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz * N_sim;

	check(cudaMalloc(&cuTempx, totalSize));
	check(cudaMalloc(&cuTempy, totalSize));
	check(cudaMalloc(&cuTempz, totalSize));

	check(cudaMemset(cuTempx, 0, totalSize));
	check(cudaMemset(cuTempy, 0, totalSize));
	check(cudaMemset(cuTempz, 0, totalSize));

	// --------------------------------------------------
	// Kernel: Copy Polarization
	// --------------------------------------------------
	cudaError c_ret;
	int blockSize, minGridSize, gridSize;

	if (landau_free_param.FLAG_SUSCIPTIBILITY_WEIGHTS == 1) {
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel_copyPolarization, 0, 0);
		gridSize = (FE_geom.Ncx * FE_geom.Ncy * FE_geom.Ncz * FE_geom.N_d + blockSize - 1) / blockSize;

		kernel_copyPolarization << <gridSize, blockSize >> > (
			cu_m, cuTempx, cuTempy, cuTempz,
			FE_geom.Ncx, FE_geom.Ncy, FE_geom.Ncz,
			FE_geom.N_d, Mcx, Mcy, Mcz
			);
		cudaDeviceSynchronize();
	}
	else if (landau_free_param.FLAG_SUSCIPTIBILITY_WEIGHTS == 2 ||
		landau_free_param.FLAG_SUSCIPTIBILITY_WEIGHTS == 3) {
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel3_copyPolarization_Ms, 0, 0);
		gridSize = (FE_geom.Ncx * FE_geom.Ncy * FE_geom.Ncz * FE_geom.N_d + blockSize - 1) / blockSize;

		kernel3_copyPolarization_Ms << <gridSize, blockSize >> > (
			cu_m, cuTempx, cuTempy, cuTempz,
			thrust::raw_pointer_cast(landau_free_param.d_susciptibilityWeights.data()),
			FE_geom.Ncx, FE_geom.Ncy, FE_geom.Ncz,
			FE_geom.N_d, Mcx, Mcy, Mcz,
			landau_free_param.d_susciptibilityWeights.size()
			);
		cudaDeviceSynchronize();
	}

	// Error check
	c_ret = cudaGetLastError();
	if (c_ret != cudaSuccess) {
		printf("CUDA Error in kernel_copyPolarization: %s\n", cudaGetErrorString(c_ret));
		exit(EXIT_FAILURE);
	}

	// --------------------------------------------------
	// Forward FFT
	// --------------------------------------------------
	cufftHandle plan;
	if (SPLITTED_FFT3D == 1) {
		int fft_res = splitted_3dfft_calc(cuTempx, cuTempy, cuTempz,
			Mcx, Mcy, Mcz,
			FE_geom.Ncx, FE_geom.Ncy, FE_geom.Ncz,
			CUFFT_FORWARD);
		if (fft_res != 0) {
			printf("Error during 3D FFT computation (forward)\n");
		}
	}
	else {
		check(cufftPlanMany(&plan, rank, n, inembed, istride, idist,
			onembed, ostride, odist, CUFFT_Z2Z, N_sim));

		check(cufftExecZ2Z(plan, cuTempx, cuTempx, CUFFT_FORWARD));
		check(cufftExecZ2Z(plan, cuTempy, cuTempy, CUFFT_FORWARD));
		check(cufftExecZ2Z(plan, cuTempz, cuTempz, CUFFT_FORWARD));

		cufftDestroy(plan);
	}

	// --------------------------------------------------
	// Kernel: Field Calculation in k-space
	// --------------------------------------------------
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel_H_calc, 0, 0);
	gridSize = (Mcx * Mcy * Mcz * FE_geom.N_d + blockSize - 1) / blockSize;

	kernel_H_calc << <gridSize, blockSize >> > (
		cuTempx, cuTempy, cuTempz,
		cuSDxx, cuSDyy, cuSDzz,
		cuSDxy, cuSDxz, cuSDyz,
		Mcx, Mcy, Mcz, FE_geom.N_d
		);
	cudaDeviceSynchronize();

	c_ret = cudaGetLastError();
	if (c_ret != cudaSuccess) {
		printf("CUDA Error in kernel_H_calc: %s\n", cudaGetErrorString(c_ret));
		exit(EXIT_FAILURE);
	}

	// --------------------------------------------------
	// Inverse FFT
	// --------------------------------------------------
	if (SPLITTED_FFT3D == 1) {
		int fft_res = splitted_3dfft_calc(cuTempx, cuTempy, cuTempz,
			Mcx, Mcy, Mcz,
			FE_geom.Ncx, FE_geom.Ncy, FE_geom.Ncz,
			CUFFT_INVERSE);
		if (fft_res != 0) {
			printf("Error during inverse 3D FFT computation\n");
		}
	}
	else {
		cufftHandle plan_inv;
		check(cufftPlanMany(&plan_inv, rank, n, inembed, istride, idist,
			onembed, ostride, odist, CUFFT_Z2Z, N_sim));

		check(cufftExecZ2Z(plan_inv, cuTempx, cuTempx, CUFFT_INVERSE));
		check(cufftExecZ2Z(plan_inv, cuTempy, cuTempy, CUFFT_INVERSE));
		check(cufftExecZ2Z(plan_inv, cuTempz, cuTempz, CUFFT_INVERSE));

		cufftDestroy(plan_inv);
	}

	// --------------------------------------------------
	// Kernel: Convert to Real Space Field
	// --------------------------------------------------
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel_electrostaticField_calc, 0, 0);
	gridSize = (FE_geom.Ncx * FE_geom.Ncy * FE_geom.Ncz * FE_geom.N_d + blockSize - 1) / blockSize;

	kernel_electrostaticField_calc << <gridSize, blockSize >> > (
		cuTempx, cuTempy, cuTempz, electrostatic_field,
		FE_geom.Ncx, FE_geom.Ncy, FE_geom.Ncz, FE_geom.N_d,
		Mcx, Mcy, Mcz
		);

	c_ret = cudaGetLastError();
	if (c_ret != cudaSuccess) {
		printf("CUDA Error in kernel_electrostaticField_calc: %s\n", cudaGetErrorString(c_ret));
		exit(EXIT_FAILURE);
	}

	// --------------------------------------------------
	// Cleanup
	// --------------------------------------------------
	cudaFree(cuTempx);
	cudaFree(cuTempy);
	cudaFree(cuTempz);
}
#include <cufft.h>
#include <cufftXt.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

#include "Vec3.h"

// ============================================================================
// 1D Split 3D FFT (Sequential Dumb Implementation)
// ============================================================================
int splitted_3dfft_calc(
	HD_Complex_Type* d_dataX,
	HD_Complex_Type* d_dataY,
	HD_Complex_Type* d_dataZ,
	int Mcx, int Mcy, int Mcz,
	int Ncx, int Ncy, int Ncz,
	int fft_direction
) {
	cufftResult res;
	cufftHandle planx, plany, planz;

	int rank = 1;  // 1D FFT
	int n[1];
	int istride, ostride;
	int idist, odist;
	int my_inembed[] = { 0 };
	int my_onembed[] = { 0 };
	int batch;

	// ------------------------------------------------------------------------
	// FFT along X-axis
	// ------------------------------------------------------------------------
	n[0] = Mcx;
	istride = ostride = 1;
	idist = odist = Mcx;
	batch = Mcy * Mcz;

	res = cufftPlanMany(&planx, rank, n,
		my_inembed, istride, idist,
		my_onembed, ostride, odist,
		CUFFT_Z2Z, batch);
	if (res != CUFFT_SUCCESS) {
		printf("ERROR: cufftPlanMany(planx) failed\n");
		return 1;
	}

	// ------------------------------------------------------------------------
	// FFT along Y-axis (split across Mcz planes)
	// ------------------------------------------------------------------------
	n[0] = Mcy;
	istride = ostride = Mcx;
	idist = odist = 1;
	batch = Mcx;

	res = cufftPlanMany(&plany, rank, n,
		my_inembed, istride, idist,
		my_onembed, ostride, odist,
		CUFFT_Z2Z, batch);
	if (res != CUFFT_SUCCESS) {
		printf("ERROR: cufftPlanMany(plany) failed\n");
		return 1;
	}

	// ------------------------------------------------------------------------
	// FFT along Z-axis
	// ------------------------------------------------------------------------
	n[0] = Mcz;
	istride = ostride = Mcx * Mcy;
	idist = odist = 1;
	batch = Mcx * Mcy;

	res = cufftPlanMany(&planz, rank, n,
		my_inembed, istride, idist,
		my_onembed, ostride, odist,
		CUFFT_Z2Z, batch);
	if (res != CUFFT_SUCCESS) {
		printf("ERROR: cufftPlanMany(planz) failed\n");
		return 1;
	}

	// ------------------------------------------------------------------------
	// Execute FFT along X
	// ------------------------------------------------------------------------
	if (cufftExecZ2Z(planx, d_dataX, d_dataX, fft_direction) != CUFFT_SUCCESS ||
		cufftExecZ2Z(planx, d_dataY, d_dataY, fft_direction) != CUFFT_SUCCESS ||
		cufftExecZ2Z(planx, d_dataZ, d_dataZ, fft_direction) != CUFFT_SUCCESS) {
		printf("ERROR: FFT along X failed\n");
		return 1;
	}

	// ------------------------------------------------------------------------
	// Execute FFT along Y
	// ------------------------------------------------------------------------
	for (int k = 0; k < Mcz; ++k) {
		size_t offset = static_cast<size_t>(k) * Mcx * Mcy;

		if (cufftExecZ2Z(plany, d_dataX + offset, d_dataX + offset, fft_direction) != CUFFT_SUCCESS ||
			cufftExecZ2Z(plany, d_dataY + offset, d_dataY + offset, fft_direction) != CUFFT_SUCCESS ||
			cufftExecZ2Z(plany, d_dataZ + offset, d_dataZ + offset, fft_direction) != CUFFT_SUCCESS) {
			printf("ERROR: FFT along Y failed (plane %d)\n", k);
			return 1;
		}
	}

	// ------------------------------------------------------------------------
	// Execute FFT along Z
	// ------------------------------------------------------------------------
	if (cufftExecZ2Z(planz, d_dataX, d_dataX, fft_direction) != CUFFT_SUCCESS ||
		cufftExecZ2Z(planz, d_dataY, d_dataY, fft_direction) != CUFFT_SUCCESS ||
		cufftExecZ2Z(planz, d_dataZ, d_dataZ, fft_direction) != CUFFT_SUCCESS) {
		printf("ERROR: FFT along Z failed\n");
		return 1;
	}

	// ------------------------------------------------------------------------
	// Cleanup
	// ------------------------------------------------------------------------
	cufftDestroy(planx);
	cufftDestroy(plany);
	cufftDestroy(planz);

	return 0;
}

// ============================================================================
// Kernel: Electrostatic Field Calculation in Fourier Space
// ============================================================================
__global__ void kernel_H_calc(
	HD_Complex_Type* dTempx,
	HD_Complex_Type* dTempy,
	HD_Complex_Type* dTempz,
	cu_Tens_Type* dSDxx, cu_Tens_Type* dSDyy, cu_Tens_Type* dSDzz,
	cu_Tens_Type* dSDxy, cu_Tens_Type* dSDxz, cu_Tens_Type* dSDyz,
	int Mcx, int Mcy, int Mcz, int N_d
) {
	unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int totalVoxels = Mcx * Mcy * Mcz;

	if (globalIndex >= totalVoxels * N_d) return;

	int simId = globalIndex / totalVoxels;
	int localIndex = globalIndex % totalVoxels;
	int index = simId * totalVoxels + localIndex;

	// Load into registers (faster access)
	cuDoubleComplex mx = dTempx[index];
	cuDoubleComplex my = dTempy[index];
	cuDoubleComplex mz = dTempz[index];

	cuDoubleComplex sdxx = dSDxx[localIndex];
	cuDoubleComplex sdyy = dSDyy[localIndex];
	cuDoubleComplex sdzz = dSDzz[localIndex];
	cuDoubleComplex sdxy = dSDxy[localIndex];
	cuDoubleComplex sdxz = dSDxz[localIndex];
	cuDoubleComplex sdyz = dSDyz[localIndex];

	// Magnetic field components (complex domain)
	dTempx[index] = cuCadd(cuCmul(mx, sdxx),
		cuCadd(cuCmul(my, sdxy), cuCmul(mz, sdxz)));
	dTempy[index] = cuCadd(cuCmul(mx, sdxy),
		cuCadd(cuCmul(my, sdyy), cuCmul(mz, sdyz)));
	dTempz[index] = cuCadd(cuCmul(mx, sdxz),
		cuCadd(cuCmul(my, sdyz), cuCmul(mz, sdzz)));
}

// ============================================================================
// Kernel: Copy Polarization with weighting (multi-simulation support)
// ============================================================================
__global__ void kernel3_copyPolarization_Ms(
	Vec3<Type_var>* cu_m,
	HD_Complex_Type* cuTempx,
	HD_Complex_Type* cuTempy,
	HD_Complex_Type* cuTempz,
	Type_var* cu_weights,
	int Ncx, int Ncy, int Ncz, int N_d,
	int Mcx, int Mcy, int Mcz,
	int weightVecSize
) {
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int totalCells = Ncx * Ncy * Ncz * N_d;

	if (index >= totalCells) return;

	int localGridSize = Ncx * Ncy * Ncz;
	int simId = index / localGridSize;
	int localIndex = index % localGridSize;

	int k = localIndex / (Ncx * Ncy);
	int j = (localIndex - k * Ncx * Ncy) / Ncx;
	int i = localIndex - k * Ncx * Ncy - j * Ncx;

	unsigned int Mindex = i + j * Mcx + k * Mcx * Mcy + simId * Mcx * Mcy * Mcz;

	// Select weighting factor
	Type_var weight = 1.0;
	if (weightVecSize == Ncz) {
		weight = cu_weights[k];
	}
	else if (weightVecSize == Ncx * Ncy * Ncz) {
		weight = cu_weights[localIndex];
	}

	Vec3<Type_var> m = cu_m[localIndex + simId * localGridSize];
	m *= weight;

	// Store real components for FFT input
	cuTempx[Mindex].x = m.x;
	cuTempy[Mindex].x = m.y;
	cuTempz[Mindex].x = m.z;
}
//------------------------------------------------------------------------------
// CUDA Kernels: Polarization Copy and Electrostatic Field Calculation
//------------------------------------------------------------------------------

__global__ void kernel_copyPolarization(
	Vec3<Type_var>* cu_m,
	HD_Complex_Type* cuTempx,
	HD_Complex_Type* cuTempy,
	HD_Complex_Type* cuTempz,
	int Ncx, int Ncy, int Ncz, int N_d,
	int Mcx, int Mcy, int Mcz
) {
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int i, j, k, Mindex;

	// CHANGE START (multiplying N_d and handling index)
	if (index < Ncx * Ncy * Ncz * N_d) {
		int localGridSize = Ncx * Ncy * Ncz;
		int simId = index / localGridSize;
		int localIndex = index % localGridSize;

		// Map local 3D coordinates (i, j, k)
		k = localIndex / (Ncx * Ncy);
		j = (localIndex - k * Ncx * Ncy) / Ncx;
		i = localIndex - k * Ncx * Ncy - j * Ncx;

		// Obtain the padded array position (Nsizes → Ksizes)
		Mindex = i + j * Mcx + k * Mcx * Mcy + simId * Mcx * Mcy * Mcz;

		Vec3<Type_var> m = cu_m[localIndex + simId * localGridSize];
		// CHANGE END

		// Set the real components (cufftComplex is typedef of float2)
		cuTempx[Mindex].x = m.x;
		cuTempy[Mindex].x = m.y;
		cuTempz[Mindex].x = m.z;
	}
}

//------------------------------------------------------------------------------
// Split kernel (1 of 3): Copy polarization data (scalar version)
// Used with CUDA streams for concurrent execution
//------------------------------------------------------------------------------
__global__ void kernel1_copyPolarization(
	Type_var* cu_m,
	HD_Complex_Type* cuTemp,
	int Ncx, int Ncy, int Ncz, int N_d,
	int Mcx, int Mcy, int Mcz
) {
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int i, j, k, Mindex;

	// CHANGE START (multiplying N_d and handling index)
	if (index < Ncx * Ncy * Ncz * N_d) {
		int localGridSize = Ncx * Ncy * Ncz;
		int simId = index / localGridSize;
		int localIndex = index % localGridSize;

		// Map local 3D coordinates (i, j, k)
		k = localIndex / (Ncx * Ncy);
		j = (localIndex - k * Ncx * Ncy) / Ncx;
		i = localIndex - k * Ncx * Ncy - j * Ncx;

		// Obtain the padded array position (Nsizes → Ksizes)
		Mindex = i + j * Mcx + k * Mcx * Mcy + simId * Mcx * Mcy * Mcz;

		// Set the real component (cufftComplex is typedef of float2)
		cuTemp[Mindex].x = cu_m[localIndex + simId * localGridSize];
	}
}

//------------------------------------------------------------------------------
// Kernel: Electrostatic Field Calculation (vector version)
//------------------------------------------------------------------------------
__global__ void kernel_electrostaticField_calc(
	HD_Complex_Type* cuTempx,
	HD_Complex_Type* cuTempy,
	HD_Complex_Type* cuTempz,
	Vec3<Type_var>* electrostatic_field,
	int Ncx, int Ncy, int Ncz, int N_d,
	int Mcx, int Mcy, int Mcz
) {
	Type_var factor = 1.0 / (4.0 * M_PI * 8.8541878188e-12 * 45.0 * Mcx * Mcy * Mcz);

	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int i, j, k, Hindex;

	if (index < Ncx * Ncy * Ncz * N_d) {
		// CHANGE START (handling index)
		int localGridSize = Ncx * Ncy * Ncz;
		int simId = index / localGridSize;
		int localIndex = index % localGridSize;

		// Map local 3D coordinates (i, j, k)
		k = localIndex / (Ncx * Ncy);
		j = (localIndex - k * Ncx * Ncy) / Ncx;
		i = localIndex - k * Ncx * Ncy - j * Ncx;

		Hindex = i + j * Mcx + k * Mcx * Mcy + simId * Mcx * Mcy * Mcz;
		// CHANGE END

		// Compute electrostatic field contribution
		Vec3<Type_var> temp = {
			cuCreal(cuTempx[Hindex]),
			cuCreal(cuTempy[Hindex]),
			cuCreal(cuTempz[Hindex])
		};

		 temp *= factor;
		// Accumulate result
		electrostatic_field[index] -= temp;
	}
}

//------------------------------------------------------------------------------
// Split kernel (1 of 3): Electrostatic Field Calculation (scalar version)
// Used with CUDA streams for concurrency
//------------------------------------------------------------------------------
__global__ void kernel1_electrostaticField_calc(
	HD_Complex_Type* cuTemp,
	Type_var* electrostatic_field,
	int Ncx, int Ncy, int Ncz,
	int Mcx, int Mcy, int Mcz
) {
	Type_var factor = 1.0 / (4.0 * M_PI * Mcx * Mcy * Mcz);
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int i, j, k, Hindex;

	if (index < Ncx * Ncy * Ncz) {
		// Compute 3D coordinates
		k = index / (Ncx * Ncy);
		j = (index - k * Ncx * Ncy) / Ncx;
		i = index - k * Ncx * Ncy - j * Ncx;

		Hindex = i + j * Mcx + k * Mcx * Mcy;

		// Apply electrostatic correction
		electrostatic_field[index] -= cuCreal(cuTemp[Hindex]) * factor;
	}
}
