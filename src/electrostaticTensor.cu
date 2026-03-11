#include "electroStaticTensor.cuh"
#include "electrostaticField.cuh"
#include "parameters.h"
#include <time.h>

#include <fstream>
#include <iostream>

//namespace std
;

#include <complex>



#include "cufft.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
//#include <cuda_runtime.h> //necessario per cudamemcopy
//#include <cuda_runtime_api.h> 
#include  "device_launch_parameters.h" //necessatrio per riconoscere blockDim



void electroStaticTensor(HD_Complex_Type* cuSDxx, HD_Complex_Type* cuSDyy, HD_Complex_Type* cuSDzz,
	HD_Complex_Type* cuSDxy, HD_Complex_Type* cuSDxz, HD_Complex_Type* cuSDyz, Type_var h, int Mcx, int Mcy, int Mcz, set_tensor conf_tens) {

	// qui si troverà il calcolo dei tensori e il calcolo dell' FFT diviso per (Mcx*Mcy*Mcz)


	int kz2 = (int)Mcz / 2.0;
	int ky2 = (int)Mcy / 2.0;
	int kx2 = (int)Mcx / 2.0;


	// aggiungere altri if

	if (conf_tens.FLAG_TENSOR == 1)
	{

		// definition and allocation of tensor CPU
		cu_Tens_Type* SDxx = new cu_Tens_Type[Mcx * Mcy * Mcz];
		cu_Tens_Type* SDyy = new cu_Tens_Type[Mcx * Mcy * Mcz];
		cu_Tens_Type* SDzz = new cu_Tens_Type[Mcx * Mcy * Mcz];
		cu_Tens_Type* SDxy = new cu_Tens_Type[Mcx * Mcy * Mcz];
		cu_Tens_Type* SDxz = new cu_Tens_Type[Mcx * Mcy * Mcz];
		cu_Tens_Type* SDyz = new cu_Tens_Type[Mcx * Mcy * Mcz];


		memset(SDxx, 0, sizeof(cu_Tens_Type) * Mcx * Mcy * Mcz);
		memset(SDyy, 0, sizeof(cu_Tens_Type) * Mcx * Mcy * Mcz);
		memset(SDzz, 0, sizeof(cu_Tens_Type) * Mcx * Mcy * Mcz);
		memset(SDxy, 0, sizeof(cu_Tens_Type) * Mcx * Mcy * Mcz);
		memset(SDxz, 0, sizeof(cu_Tens_Type) * Mcx * Mcy * Mcz);
		memset(SDyz, 0, sizeof(cu_Tens_Type) * Mcx * Mcy * Mcz);


		// definition and allocation of tensor CPU
		DT_double* NDxx = new DT_double[Mcx * Mcy * Mcz];
		DT_double* NDyy = new DT_double[Mcx * Mcy * Mcz];
		DT_double* NDzz = new DT_double[Mcx * Mcy * Mcz];
		DT_double* NDxy = new DT_double[Mcx * Mcy * Mcz];
		DT_double* NDxz = new DT_double[Mcx * Mcy * Mcz];
		DT_double* NDyz = new DT_double[Mcx * Mcy * Mcz];


		int n1, n2, n3, nn1, nn2, nn3;
		DT_double xn1, xn2, xn3; //aggiunto con la versione 3Marzo
		int index;


		time_t start_t, end_t;
		double diff_t;
		time(&start_t);

		for (n3 = 0, nn3 = (-kz2 + 1); n3 < Mcz && nn3 <= kz2; n3++, nn3++) {
			for (n2 = 0, nn2 = (-ky2 + 1); n2 < Mcy && nn2 <= ky2; n2++, nn2++) {
				printf("electroStaticTensor completion = %d % \r", int(100 * (n3 * Mcy + n2) / (Mcz * Mcy)));
				fflush(stdout);
				for (n1 = 0, nn1 = (-kx2 + 1); n1 < Mcx && nn1 <= kx2; n1++, nn1++) {

					xn1 = nn1 * 1.0e0;
					xn2 = nn2 * 1.0e0;
					xn3 = nn3 * h;

					index = n1 + n2 * Mcx + n3 * Mcx * Mcy;

					if (nn1 == 0 && nn2 == 0 && nn3 == 0) {

						NDxx[index] = host_Dz(h, 1.0e0, 1.0e0);
						NDyy[index] = host_Dz(1.0e0, h, 1.0e0);
						NDzz[index] = host_Dz(1.0e0, 1.0e0, h);
						NDxy[index] = 0.0e0;
						NDxz[index] = 0.0e0;
						NDyz[index] = 0.0e0;

					}
					else {

						NDxx[index] = host_fnxx(xn1, xn2, xn3, 1.e0, 1.0e0, h);
						NDyy[index] = host_fnxx(xn2, xn1, xn3, 1.e0, 1.0e0, h);
						NDzz[index] = host_fnxx(xn3, xn2, xn1, h, 1.0e0, 1.0e0);
						NDxy[index] = host_fnxy(xn1, xn2, xn3, 1.e0, 1.0e0, h);
						NDxz[index] = host_fnxy(xn1, xn3, xn2, 1.e0, h, 1.0e0);
						NDyz[index] = host_fnxy(xn2, xn3, xn1, 1.e0, h, 1.0e0);
					}
				}
			}
		}

		//printf("\n 1 NDxx: %.15Lf: \n", NDxx[100]);

		time(&end_t);
		diff_t = difftime(end_t, start_t);
		printf("Execution time = %7.0f s (%5.1f min) (%3.1f h)\n", diff_t, diff_t / 60, diff_t / 3600);

		int index1 = 0, index2 = 0;
		for (n3 = 0; n3 <= (Mcz - 1); n3++) {
			for (n2 = 0; n2 <= (Mcy - 1); n2++) {
				for (n1 = 0; n1 <= (Mcx - 1); n1++) {

					nn1 = n1 + (kx2 - 1);
					nn2 = n2 + (ky2 - 1);
					nn3 = n3 + (kz2 - 1);

					if (n3 > (kz2)) {
						nn3 = n3 - (kz2 + 1);
					}
					if (n2 > (ky2)) {
						nn2 = n2 - (ky2 + 1);
					}
					if (n1 > (kx2)) {
						nn1 = n1 - (kx2 + 1);
					}

					index1 = n1 + n2 * Mcx + n3 * Mcx * Mcy;
					index2 = nn1 + nn2 * Mcx + nn3 * Mcx * Mcy;

					// make_cuDobleComplex, returns a cuDoubleComplex from its
					// real and imaginary part


					SDxx[index1] = make_cuDoubleComplex(NDxx[index2], 0.0e0);
					SDyy[index1] = make_cuDoubleComplex(NDyy[index2], 0.0e0);
					SDzz[index1] = make_cuDoubleComplex(NDzz[index2], 0.0e0);
					SDxy[index1] = make_cuDoubleComplex(NDxy[index2], 0.0e0);
					SDxz[index1] = make_cuDoubleComplex(NDxz[index2], 0.0e0);
					SDyz[index1] = make_cuDoubleComplex(NDyz[index2], 0.0e0);

				}
			}
		}
		delete[] NDxx;
		delete[] NDyy;
		delete[] NDzz;
		delete[] NDxy;
		delete[] NDxz;
		delete[] NDyz;

		/*
				cudaStream_t streamA, streamB, streamC, streamD, streamE, streamF;
				cudaStreamCreate(&streamA);
				cudaStreamCreate(&streamB);
				cudaStreamCreate(&streamC);
				cudaStreamCreate(&streamD);
				cudaStreamCreate(&streamE);
				cudaStreamCreate(&streamF);

				/*
				cudaMemcpyAsync(cuSDxx, SDxx, sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz, cudaMemcpyHostToDevice, streamA);
				cudaMemcpyAsync(cuSDyy, SDyy, sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz, cudaMemcpyHostToDevice, streamB);
				cudaMemcpyAsync(cuSDzz, SDzz, sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz, cudaMemcpyHostToDevice, streamC);
				cudaMemcpyAsync(cuSDxy, SDxy, sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz, cudaMemcpyHostToDevice, streamD);
				cudaMemcpyAsync(cuSDxz, SDxz, sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz, cudaMemcpyHostToDevice, streamE);
				cudaMemcpyAsync(cuSDyz, SDyz, sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz, cudaMemcpyHostToDevice, streamF);

				cudaDeviceSynchronize();

				//printf("\n SDxx: %.15Lf: \n", SDxx[100].x);

				cudaStreamDestroy(streamA);
				cudaStreamDestroy(streamB);
				cudaStreamDestroy(streamC);
				cudaStreamDestroy(streamD);
				cudaStreamDestroy(streamE);
				cudaStreamDestroy(streamF); */

		check(cudaMemcpy(cuSDxx, SDxx, sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz, cudaMemcpyHostToDevice)); //controllo tensori ok
		check(cudaMemcpy(cuSDyy, SDyy, sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz, cudaMemcpyHostToDevice));//controllo tensori ok
		check(cudaMemcpy(cuSDzz, SDzz, sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz, cudaMemcpyHostToDevice));//controllo tensori ok

		check(cudaMemcpy(cuSDxy, SDxy, sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz, cudaMemcpyHostToDevice));//controllo tensori ok
		check(cudaMemcpy(cuSDxz, SDxz, sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz, cudaMemcpyHostToDevice));//controllo tensori ok
		check(cudaMemcpy(cuSDyz, SDyz, sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz, cudaMemcpyHostToDevice));//controllo tensori ok
		/*
		HD_Complex_Type* prova1 = new HD_Complex_Type[Mcx * Mcy * Mcz];
		HD_Complex_Type* prova2 = new HD_Complex_Type[Mcx * Mcy * Mcz];
		HD_Complex_Type* prova3 = new HD_Complex_Type[Mcx * Mcy * Mcz];
		HD_Complex_Type* prova4 = new HD_Complex_Type[Mcx * Mcy * Mcz];
		HD_Complex_Type* prova5 = new HD_Complex_Type[Mcx * Mcy * Mcz];
		HD_Complex_Type* prova6 = new HD_Complex_Type[Mcx * Mcy * Mcz];

		check(cudaMemcpy(prova1, cuSDxx, sizeof(HD_Complex_Type) * Mcx* Mcy* Mcz, cudaMemcpyDeviceToHost));
		check(cudaMemcpy(prova2, cuSDyy, sizeof(HD_Complex_Type) * Mcx* Mcy* Mcz, cudaMemcpyDeviceToHost));
		check(cudaMemcpy(prova3, cuSDzz, sizeof(HD_Complex_Type) * Mcx* Mcy* Mcz, cudaMemcpyDeviceToHost));
		check(cudaMemcpy(prova4, cuSDxy, sizeof(HD_Complex_Type) * Mcx* Mcy* Mcz, cudaMemcpyDeviceToHost));
		check(cudaMemcpy(prova5, cuSDxz, sizeof(HD_Complex_Type) * Mcx* Mcy* Mcz, cudaMemcpyDeviceToHost));
		check(cudaMemcpy(prova6, cuSDyz, sizeof(HD_Complex_Type) * Mcx* Mcy* Mcz, cudaMemcpyDeviceToHost));

		FILE* fout;
		fout = fopen("tensoriHdemag.dat", "w+");

		for (int i = 0; i < Mcx * Mcy * Mcz; i++) {
			//printf("\n % d: real %e - Img %e \n", i, prova1[i].x, prova1[i].y);
			fprintf(fout, "%.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f\n", prova1[i].x, prova1[i].y, prova2[i].x, prova2[i].y, prova3[i].x, prova3[i].y, prova4[i].x, prova4[i].y, prova5[i].x, prova5[i].y, prova6[i].x, prova6[i].y, );
		}
		fclose(fout);
		*/
		delete[]SDxx;
		delete[]SDyy;
		delete[]SDzz;
		delete[]SDxy;
		delete[]SDxz;
		delete[]SDyz;
	}


	else if (conf_tens.FLAG_TENSOR == 2) {

		cudaError c_ret = cudaGetLastError();


		time_t start_t, end_t;
		double diff_t;
		time(&start_t);
		Type_var* ND1, * ND2, * ND3, * ND4, * ND5, * ND6;

		//allocate variable in GPU
		cudaMalloc((void**)&(ND1), sizeof(Type_var) * (Mcx * Mcy * Mcz));
		cudaMalloc((void**)&(ND2), sizeof(Type_var) * (Mcx * Mcy * Mcz));
		cudaMalloc((void**)&(ND3), sizeof(Type_var) * (Mcx * Mcy * Mcz));
		cudaMalloc((void**)&(ND4), sizeof(Type_var) * (Mcx * Mcy * Mcz));
		cudaMalloc((void**)&(ND5), sizeof(Type_var) * (Mcx * Mcy * Mcz));
		cudaMalloc((void**)&(ND6), sizeof(Type_var) * (Mcx * Mcy * Mcz));

		int threadsPerBlock2 = 256;
		dim3 threads2(threadsPerBlock2);
		//int M = (Mcx) * (Mcy) * (Mcz);
		dim3 blocksPerGrid2(ceil(Type_var(Mcx * Mcy * Mcz) / threads2.x));
		//unsigned int blocksPerGrid2 = (M + threadsPerBlock2 - 1) / threadsPerBlock2;

		kernel_tensor1partea << <blocksPerGrid2, threads2 >> > (ND1, h, Mcx, Mcy, Mcz);
		c_ret = cudaGetLastError();

		if (c_ret != cudaSuccess) {
			printf("CUDA Error in kernel_tensor1partea: %s\n", cudaGetErrorString(c_ret));
			exit(-1);
		}
		kernel_tensor2parte << <blocksPerGrid2, threads2 >> > (cuSDxx, ND1, h, Mcx, Mcy, Mcz);
		c_ret = cudaGetLastError();

		if (c_ret != cudaSuccess) {
			printf("CUDA Error in kernel_tensor2partea: %s\n", cudaGetErrorString(c_ret));
			exit(-1);
		}
		kernel_tensor1parteb << <blocksPerGrid2, threads2 >> > (ND2, h, Mcx, Mcy, Mcz);
		c_ret = cudaGetLastError();

		if (c_ret != cudaSuccess) {
			printf("CUDA Error in kernel_tensor1parteb: %s\n", cudaGetErrorString(c_ret));
			exit(-1);
		}
		kernel_tensor2parte << <blocksPerGrid2, threads2 >> > (cuSDyy, ND2, h, Mcx, Mcy, Mcz);
		c_ret = cudaGetLastError();

		if (c_ret != cudaSuccess) {
			printf("CUDA Error in kernel_tensor2parte: %s\n", cudaGetErrorString(c_ret));
			exit(-1);
		}
		kernel_tensor1partec << <blocksPerGrid2, threads2 >> > (ND3, h, Mcx, Mcy, Mcz);
		c_ret = cudaGetLastError();

		if (c_ret != cudaSuccess) {
			printf("CUDA Error in kernel_tensor1partec: %s\n", cudaGetErrorString(c_ret));
			exit(-1);
		}
		kernel_tensor2parte << <blocksPerGrid2, threads2 >> > (cuSDzz, ND3, h, Mcx, Mcy, Mcz);
		c_ret = cudaGetLastError();

		if (c_ret != cudaSuccess) {
			printf("CUDA Error in kernel_tensor2parte: %s\n", cudaGetErrorString(c_ret));
			exit(-1);
		}
		kernel_tensor1parted << <blocksPerGrid2, threads2 >> > (ND4, h, Mcx, Mcy, Mcz);
		c_ret = cudaGetLastError();

		if (c_ret != cudaSuccess) {
			printf("CUDA Error in kernel_tensor1parted: %s\n", cudaGetErrorString(c_ret));
			exit(-1);
		}
		kernel_tensor2parte << <blocksPerGrid2, threads2 >> > (cuSDxy, ND4, h, Mcx, Mcy, Mcz);
		c_ret = cudaGetLastError();

		if (c_ret != cudaSuccess) {
			printf("CUDA Error in kernel_tensor2parte: %s\n", cudaGetErrorString(c_ret));
			exit(-1);
		}
		kernel_tensor1partee << <blocksPerGrid2, threads2 >> > (ND5, h, Mcx, Mcy, Mcz);
		c_ret = cudaGetLastError();

		if (c_ret != cudaSuccess) {
			printf("CUDA Error in kernel_tensor1partee: %s\n", cudaGetErrorString(c_ret));
			exit(-1);
		}
		kernel_tensor2parte << <blocksPerGrid2, threads2 >> > (cuSDxz, ND5, h, Mcx, Mcy, Mcz);
		c_ret = cudaGetLastError();

		if (c_ret != cudaSuccess) {
			printf("CUDA Error in kernel_tensor2parte: %s\n", cudaGetErrorString(c_ret));
			exit(-1);
		}
		kernel_tensor1partef << <blocksPerGrid2, threads2 >> > (ND6, h, Mcx, Mcy, Mcz);
		c_ret = cudaGetLastError();

		if (c_ret != cudaSuccess) {
			printf("CUDA Error in kernel_tensor1partef: %s\n", cudaGetErrorString(c_ret));
			exit(-1);
		}
		kernel_tensor2parte << <blocksPerGrid2, threads2 >> > (cuSDyz, ND6, h, Mcx, Mcy, Mcz);
		c_ret = cudaGetLastError();

		if (c_ret != cudaSuccess) {
			printf("CUDA Error in kernel_tensor2parte: %s\n", cudaGetErrorString(c_ret));
			exit(-1);
		}
		cudaDeviceSynchronize();
/*

		HD_Complex_Type* prova1 = new HD_Complex_Type[Mcx * Mcy * Mcz];
		HD_Complex_Type* prova2 = new HD_Complex_Type[Mcx * Mcy * Mcz];
		HD_Complex_Type* prova3 = new HD_Complex_Type[Mcx * Mcy * Mcz];
		HD_Complex_Type* prova4 = new HD_Complex_Type[Mcx * Mcy * Mcz];
		HD_Complex_Type* prova5 = new HD_Complex_Type[Mcx * Mcy * Mcz];
		HD_Complex_Type* prova6 = new HD_Complex_Type[Mcx * Mcy * Mcz];

		check(cudaMemcpy(prova1, cuSDxx, sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz, cudaMemcpyDeviceToHost));
		check(cudaMemcpy(prova2, cuSDyy, sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz, cudaMemcpyDeviceToHost));
		check(cudaMemcpy(prova3, cuSDzz, sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz, cudaMemcpyDeviceToHost));
		check(cudaMemcpy(prova4, cuSDxy, sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz, cudaMemcpyDeviceToHost));
		check(cudaMemcpy(prova5, cuSDxz, sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz, cudaMemcpyDeviceToHost));
		check(cudaMemcpy(prova6, cuSDyz, sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz, cudaMemcpyDeviceToHost));

		FILE* fout;
		fout = fopen("tensoriHdemag.dat", "w+");

		for (int i = 0; i < Mcx * Mcy * Mcz; i++) {
			//printf("\n % d: real %e - Img %e \n", i, prova1[i].x, prova1[i].y);
			fprintf(fout, "%.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f\n", prova1[i].x, prova1[i].y, prova2[i].x, prova2[i].y, prova3[i].x, prova3[i].y, prova4[i].x, prova4[i].y, prova5[i].x, prova5[i].y, prova6[i].x, prova6[i].y, );
		}
		fclose(fout);
		*/

		time(&end_t);
		diff_t = difftime(end_t, start_t);
		printf("Execution time = %7.0f s (%5.1f min) (%3.1f h)\n", diff_t, diff_t / 60, diff_t / 3600);


		cudaFree(ND1);
		cudaFree(ND2);
		cudaFree(ND3);
		cudaFree(ND4);
		cudaFree(ND5);
		cudaFree(ND6);



	}


	int fft_res;

	//-- 1st 3D FFT COMPUTATION (ON TENSOR DATA) --// (rimosso dalla fine di "demag_tensor_calc" e inserito qui)
	if (SPLITTED_FFT3D == 1) {
		fft_res = splitted_3dfft_calc(cuSDxx, cuSDyy, cuSDzz, Mcx, Mcy, Mcz, kx2, ky2, kz2, CUFFT_FORWARD);
		if (fft_res != 0)
			printf("Error during the 1st 3D-FFT computation (part a)\n\n");
		fft_res = splitted_3dfft_calc(cuSDxy, cuSDxz, cuSDyz, Mcx, Mcy, Mcz, kx2, ky2, kz2, CUFFT_FORWARD);
		if (fft_res != 0)
			printf("Error during the 1st 3D-FFT computation (part b)\n\n");
	}
	else if (SPLITTED_FFT3D == 0) {

		cufftHandle plan;

		cufftPlan3d(&plan, Mcz, Mcy, Mcx, CUFFT_Z2Z);

		check(cufftExecZ2Z(plan, cuSDxx, cuSDxx, CUFFT_FORWARD));
		check(cufftExecZ2Z(plan, cuSDyy, cuSDyy, CUFFT_FORWARD));
		check(cufftExecZ2Z(plan, cuSDzz, cuSDzz, CUFFT_FORWARD));
		check(cufftExecZ2Z(plan, cuSDxy, cuSDxy, CUFFT_FORWARD));
		check(cufftExecZ2Z(plan, cuSDxz, cuSDxz, CUFFT_FORWARD));
		check(cufftExecZ2Z(plan, cuSDyz, cuSDyz, CUFFT_FORWARD));


		cufftDestroy(plan);
	}

	/*
	HD_Complex_Type* prova1 = new HD_Complex_Type[Mcx * Mcy * Mcz];
	HD_Complex_Type* prova2 = new HD_Complex_Type[Mcx * Mcy * Mcz];
	HD_Complex_Type* prova3 = new HD_Complex_Type[Mcx * Mcy * Mcz];
	HD_Complex_Type* prova4 = new HD_Complex_Type[Mcx * Mcy * Mcz];
	HD_Complex_Type* prova5 = new HD_Complex_Type[Mcx * Mcy * Mcz];
	HD_Complex_Type* prova6 = new HD_Complex_Type[Mcx * Mcy * Mcz];

	check(cudaMemcpy(prova1, cuSDxx, sizeof(HD_Complex_Type)* Mcx* Mcy* Mcz, cudaMemcpyDeviceToHost));
	check(cudaMemcpy(prova2, cuSDyy, sizeof(HD_Complex_Type)* Mcx* Mcy* Mcz, cudaMemcpyDeviceToHost));
	check(cudaMemcpy(prova3, cuSDzz, sizeof(HD_Complex_Type)* Mcx* Mcy* Mcz, cudaMemcpyDeviceToHost));
	check(cudaMemcpy(prova4, cuSDxy, sizeof(HD_Complex_Type)* Mcx* Mcy* Mcz, cudaMemcpyDeviceToHost));
	check(cudaMemcpy(prova5, cuSDxz, sizeof(HD_Complex_Type)* Mcx* Mcy* Mcz, cudaMemcpyDeviceToHost));
	check(cudaMemcpy(prova6, cuSDyz, sizeof(HD_Complex_Type)* Mcx* Mcy* Mcz, cudaMemcpyDeviceToHost));

	FILE* fout;
	fout = fopen("cuSDzz_fft.dat", "w+");

	for (int i = 0; i < Mcx * Mcy * Mcz; i++) {
		//printf("\n % d: real %.15f - Img %.15f \n", i, prova1[i].x, prova1[i].y);
		fprintf(fout, "%e %e \n", prova3[i].x, prova3[i].y);
	}
	fclose(fout);*/

}




DT_double host_fi(DT_double x)
{
	DT_double fi;

	fi = log(x + sqrt(1 + x * x));

	return fi;
}


DT_double host_f(DT_double x, DT_double y, DT_double z)
{
	DT_double f, x2, y2, z2, R;

	x2 = x * x;
	y2 = y * y;
	z2 = z * z;
	R = sqrtl(x2 + y2 + z2);
	f = .0e0;

	if (z2 > .0e0) {
		f = (1.0e0 / 6.0e0) * (2 * x2 - y2 - z2) * R;
		if (x2 > .0e0 && y2 > .0e0) {
			f = f - x * y * z * atanl(y * z / (x * R));
		}
		if (y2 > .0e0) {
			f = f + 0.5 * y * (z2 - x2) * host_fi(y / sqrtl(x2 + z2));
		}
		if (x2 > .0e0 || y2 > .0e0) {
			f = f + 0.5 * z * (y2 - x2) * host_fi(z / sqrtl(x2 + y2));
		}
	}
	else {
		f = (1.0e0 / 6.0e0) * (2 * x2 - y2) * R;
		if (x2 > .0e0 && y2 > .0e0) {
			f = f - 0.5 * y * x2 * host_fi(y / fabsl(x));
		}
	}

	return f;
}


DT_double host_g(DT_double x, DT_double y, DT_double z)
{
	DT_double g, x2, y2, z2, R;

	x2 = x * x;
	y2 = y * y;
	z2 = z * z;
	R = sqrtl(x2 + y2 + z2);
	g = -x * y * R / 3.0;

	if (z2 > .0e0) {
		g = g - (z * z2 / 6.0) * atanl(x * y / (z * R))
			+ (y / 6.0) * (3 * z2 - y2) * host_fi(x / sqrtl(y2 + z2))
			+ (x / 6.0) * (3 * z2 - x2) * host_fi(y / sqrtl(x2 + z2));

		if (y2 > .0e0) {
			g = g - (z * y2 / 2.0) * atanl(x * z / (y * R));
		}
		if (x2 > .0e0) {
			g = g - (z * x2 / 2.0) * atanl(y * z / (x * R));
		}
		if (x2 > .0e0 && y2 > .0e0) {
			g = g + (x * y * z) * host_fi(z / sqrtl(x2 + y2));
		}
	}
	else {
		if (y2 > .0e0) {
			g = g - (y * y2 / 6.0) * host_fi(x / fabsl(y));
		}
		if (x2 > .0e0) {
			g = g - (x * x2 / 6.0) * host_fi(y / fabsl(x));
		}
	}

	return g;
}

DT_double host_f2(DT_double x, DT_double y, DT_double z)
{
	DT_double f2;

	f2 = host_f(x, y, z) - host_f(x, .0e0, z) - host_f(x, y, .0e0) + host_f(x, .0e0, .0e0);

	return f2;
}


DT_double host_g2(DT_double x, DT_double y, DT_double z)
{
	DT_double g2;

	g2 = host_g(x, y, z) - host_g(x, y, .0e0);

	return g2;
}

DT_double host_fnxx(DT_double x, DT_double y, DT_double z, DT_double dx, DT_double dy, DT_double dz) {

	DT_double fnxx;

	fnxx = -powl((dx * dy * dz), (-1.0)) * (
		8.0 * host_f2(x, y, z)
		- 4.0 * (host_f2(x, y, z + dz) + host_f2(x, y, z - dz) + host_f2(x, y + dy, z)
			+ host_f2(x, y - dy, z) + host_f2(x + dx, y, z) + host_f2(x - dx, y, z))
		+ 2.0 * (host_f2(x, y + dy, z + dz) + host_f2(x, y + dy, z - dz)
			+ host_f2(x, y - dy, z + dz) + host_f2(x, y - dy, z - dz)
			+ host_f2(x + dx, y + dy, z) + host_f2(x + dx, y - dy, z)
			+ host_f2(x - dx, y + dy, z) + host_f2(x - dx, y - dy, z)
			+ host_f2(x + dx, y, z + dz) + host_f2(x + dx, y, z - dz)
			+ host_f2(x - dx, y, z + dz) + host_f2(x - dx, y, z - dz))
		- 1.0 * (host_f2(x + dx, y + dy, z + dz) + host_f2(x + dx, y + dy, z - dz)
			+ host_f2(x + dx, y - dy, z + dz) + host_f2(x + dx, y - dy, z - dz)
			+ host_f2(x - dx, y + dy, z + dz) + host_f2(x - dx, y + dy, z - dz)
			+ host_f2(x - dx, y - dy, z + dz) + host_f2(x - dx, y - dy, z - dz)));

	return fnxx;

}

DT_double host_fnxy(DT_double x, DT_double y, DT_double z, DT_double dx, DT_double dy, DT_double dz)
{
	DT_double fnxy;

	fnxy = -powl((dx * dy * dz), (-1.0)) * (
		8.0 * host_g2(x, y, z)
		- 4.0 * (host_g2(x, y, z + dz) + host_g2(x, y, z - dz) + host_g2(x, y + dy, z)
			+ host_g2(x, y - dy, z) + host_g2(x + dx, y, z) + host_g2(x - dx, y, z))
		+ 2.0 * (host_g2(x, y + dy, z + dz) + host_g2(x, y + dy, z - dz)
			+ host_g2(x, y - dy, z + dz) + host_g2(x, y - dy, z - dz)
			+ host_g2(x + dx, y + dy, z) + host_g2(x + dx, y - dy, z)
			+ host_g2(x - dx, y + dy, z) + host_g2(x - dx, y - dy, z)
			+ host_g2(x + dx, y, z + dz) + host_g2(x + dx, y, z - dz)
			+ host_g2(x - dx, y, z + dz) + host_g2(x - dx, y, z - dz))
		- 1.0 * (host_g2(x + dx, y + dy, z + dz) + host_g2(x + dx, y + dy, z - dz)
			+ host_g2(x + dx, y - dy, z + dz) + host_g2(x + dx, y - dy, z - dz)
			+ host_g2(x - dx, y + dy, z + dz) + host_g2(x - dx, y + dy, z - dz)
			+ host_g2(x - dx, y - dy, z + dz) + host_g2(x - dx, y - dy, z - dz)));

	return fnxy;
}

DT_double host_Dz(DT_double a, DT_double b, DT_double c)
{
	DT_double a2, b2, c2, Dz;
	DT_double sqrta2b2, sqrta2c2, sqrtb2c2, sqrta2b2c2;

	a2 = a * a;
	b2 = b * b;
	c2 = c * c;

	sqrta2b2 = sqrtl(a2 + b2);
	sqrta2c2 = sqrtl(a2 + c2);
	sqrtb2c2 = sqrtl(b2 + c2);
	sqrta2b2c2 = sqrtl(a2 + b2 + c2);

	Dz = -(((b2 - c2) / (2.0 * b * c)) * logl((sqrta2b2c2 - a) / (sqrta2b2c2 + a))
		+ ((a2 - c2) / (2.0 * a * c)) * logl((sqrta2b2c2 - b) / (sqrta2b2c2 + b))
		+ (b / (2.0 * c)) * logl((sqrta2b2 + a) / (sqrta2b2 - a))
		+ (a / (2.0 * c)) * logl((sqrta2b2 + b) / (sqrta2b2 - b))
		+ (c / (2.0 * a)) * logl((sqrtb2c2 - b) / (sqrtb2c2 + b))
		+ (c / (2.0 * b)) * logl((sqrta2c2 - a) / (sqrta2c2 + a))
		+ 2.0 * atanl((a * b) / (c * sqrta2b2c2))
		+ (powl(a, 3.0) + powl(b, 3.0) - 2.0 * powl(c, 3.0)) / (3.0 * a * b * c)
		+ (powl(a, 2.0) + powl(b, 2.0) - 2.0 * powl(c, 2.0)) / (3.0 * a * b * c) * sqrta2b2c2
		+ c / (a * b) * (sqrta2c2 + sqrtb2c2)
		- (powl(sqrta2b2, 3.0) + powl(sqrtb2c2, 3.0) + powl(sqrta2c2, 3.0)) / (3.0 * a * b * c)
		) * 4.0;

	return Dz;
}



__device__ double fi(double x)
{
	double fi;

	fi = log(x + sqrt(1 + x * x));

	//return fi;
	return fi;
}

__device__  double f(double x, double y, double z)
{
	double f, x2, y2, z2, R;

	x2 = x * x;
	y2 = y * y;
	z2 = z * z;
	R = sqrt(x2 + y2 + z2);
	f = .0e0;

	if (z2 > .0e0) {
		f = (1.0e0 / 6.0e0) * (2 * x2 - y2 - z2) * R;
		if (x2 > .0e0 && y2 > .0e0) {
			f = f - x * y * z * atan(y * z / (x * R));
		}
		if (y2 > .0e0) {
			f = f + 0.5 * y * (z2 - x2) * fi(y / sqrt(x2 + z2));
		}
		if (x2 > .0e0 || y2 > .0e0) {
			f = f + 0.5 * z * (y2 - x2) * fi(z / sqrt(x2 + y2));
		}
	}
	else {
		f = (1.0e0 / 6.0e0) * (2 * x2 - y2) * R;
		if (x2 > .0e0 && y2 > .0e0) {
			f = f - 0.5 * y * x2 * fi(y / fabs(x));
		}
	}

	//return f;
	return f;
}


__device__  double g(double x, double y, double z)
{
	double g, x2, y2, z2, R;

	x2 = x * x;
	y2 = y * y;
	z2 = z * z;
	R = sqrt(x2 + y2 + z2);
	g = -x * y * R / 3.0;

	if (z2 > .0e0) {
		g = g - (z * z2 / 6.0) * atan(x * y / (z * R))
			+ (y / 6.0) * (3 * z2 - y2) * fi(x / sqrt(y2 + z2))
			+ (x / 6.0) * (3 * z2 - x2) * fi(y / sqrt(x2 + z2));

		if (y2 > .0e0) {
			g = g - (z * y2 / 2.0) * atan(x * z / (y * R));
		}
		if (x2 > .0e0) {
			g = g - (z * x2 / 2.0) * atan(y * z / (x * R));
		}
		if (x2 > .0e0 && y2 > .0e0) {
			g = g + (x * y * z) * fi(z / sqrt(x2 + y2));
		}
	}
	else {
		if (y2 > .0e0) {
			g = g - (y * y2 / 6.0) * fi(x / fabs(y));
		}
		if (x2 > .0e0) {
			g = g - (x * x2 / 6.0) * fi(y / fabs(x));
		}
	}

	//return g;
	return  g;
}

__device__ double f2(double x, double y, double z)
{
	double f2;

	f2 = f(x, y, z) - f(x, .0e0, z) - f(x, y, .0e0) + f(x, .0e0, .0e0);

	//return f2;
	return f2;
}


__device__  double g2(double x, double y, double z)
{
	double g2;

	g2 = g(x, y, z) - g(x, y, .0e0);

	return  g2;
	//return g2;
}
/*
static __device__  double fnxx(double x, double y, double z, double dx, double dy, double dz) {

	double fnxx;

	fnxx = -pow((dx * dy * dz), (-1.0)) * (
		8.0 * f2(x, y, z)
		- 4.0 * (f2(x, y, z + dz) + f2(x, y, z - dz) + f2(x, y + dy, z)
			+ f2(x, y - dy, z) + f2(x + dx, y, z) + f2(x - dx, y, z))
		+ 2.0 * (f2(x, y + dy, z + dz) + f2(x, y + dy, z - dz)
			+ f2(x, y - dy, z + dz) + f2(x, y - dy, z - dz)
			+ f2(x + dx, y + dy, z) + f2(x + dx, y - dy, z)
			+ f2(x - dx, y + dy, z) + f2(x - dx, y - dy, z)
			+ f2(x + dx, y, z + dz) + f2(x + dx, y, z - dz)
			+ f2(x - dx, y, z + dz) + f2(x - dx, y, z - dz))
		- 1.0 * (f2(x + dx, y + dy, z + dz) + f2(x + dx, y + dy, z - dz)
			+ f2(x + dx, y - dy, z + dz) + f2(x + dx, y - dy, z - dz)
			+ f2(x - dx, y + dy, z + dz) + f2(x - dx, y + dy, z - dz)
			+ f2(x - dx, y - dy, z + dz) + f2(x - dx, y - dy, z - dz)));

	return fnxx;

}
*/
/*
static __device__  double fnxx2(double x, double y, double z, double dx, double dy, double dz) {

	double fnxx;

	fnxx = -pow((dx * dy * dz), (-1.0)) * (
		8.0 * f2(x, y, z)
		- 4.0 * (f2(x, y, z + dz) + f2(x, y, z - dz) + f2(x, y + dy, z)
			+ f2(x, y - dy, z) + f2(x + dx, y, z) + f2(x - dx, y, z))
		+ 2.0 * (f2(x, y + dy, z + dz) + f2(x, y + dy, z - dz)
			+ f2(x, y - dy, z + dz) + f2(x, y - dy, z - dz)
			+ f2(x + dx, y + dy, z) + f2(x + dx, y - dy, z)
			+ f2(x - dx, y + dy, z) + f2(x - dx, y - dy, z)
			+ f2(x + dx, y, z + dz) + f2(x + dx, y, z - dz)
			+ f2(x - dx, y, z + dz) + f2(x - dx, y, z - dz))
		- 1.0 * (f2(x + dx, y + dy, z + dz) + f2(x + dx, y + dy, z - dz)
			+ f2(x + dx, y - dy, z + dz) + f2(x + dx, y - dy, z - dz)
			+ f2(x - dx, y + dy, z + dz) + f2(x - dx, y + dy, z - dz)
			+ f2(x - dx, y - dy, z + dz) + f2(x - dx, y - dy, z - dz)));

	return fnxx;

}*/
/*
static __device__  double fnxx3(double x, double y, double z, double dx, double dy, double dz) {

	double fnxx;

	fnxx = -pow((dx * dy * dz), (-1.0)) * (
		8.0 * f2(x, y, z)
		- 4.0 * (f2(x, y, z + dz) + f2(x, y, z - dz) + f2(x, y + dy, z)
			+ f2(x, y - dy, z) + f2(x + dx, y, z) + f2(x - dx, y, z))
		+ 2.0 * (f2(x, y + dy, z + dz) + f2(x, y + dy, z - dz)
			+ f2(x, y - dy, z + dz) + f2(x, y - dy, z - dz)
			+ f2(x + dx, y + dy, z) + f2(x + dx, y - dy, z)
			+ f2(x - dx, y + dy, z) + f2(x - dx, y - dy, z)
			+ f2(x + dx, y, z + dz) + f2(x + dx, y, z - dz)
			+ f2(x - dx, y, z + dz) + f2(x - dx, y, z - dz))
		- 1.0 * (f2(x + dx, y + dy, z + dz) + f2(x + dx, y + dy, z - dz)
			+ f2(x + dx, y - dy, z + dz) + f2(x + dx, y - dy, z - dz)
			+ f2(x - dx, y + dy, z + dz) + f2(x - dx, y + dy, z - dz)
			+ f2(x - dx, y - dy, z + dz) + f2(x - dx, y - dy, z - dz)));

	return fnxx;

}*/

/*
static __device__  double fnxy(double x, double y, double z, double dx, double dy, double dz)
{
	double fnxy;

	fnxy = -pow((dx * dy * dz), (-1.0)) * (
		8.0 * g2(x, y, z)
		- 4.0 * (g2(x, y, z + dz) + g2(x, y, z - dz) + g2(x, y + dy, z)
			+ g2(x, y - dy, z) + g2(x + dx, y, z) + g2(x - dx, y, z))
		+ 2.0 * (g2(x, y + dy, z + dz) + g2(x, y + dy, z - dz)
			+ g2(x, y - dy, z + dz) + g2(x, y - dy, z - dz)
			+ g2(x + dx, y + dy, z) + g2(x + dx, y - dy, z)
			+ g2(x - dx, y + dy, z) + g2(x - dx, y - dy, z)
			+ g2(x + dx, y, z + dz) + g2(x + dx, y, z - dz)
			+ g2(x - dx, y, z + dz) + g2(x - dx, y, z - dz))
		- 1.0 * (g2(x + dx, y + dy, z + dz) + g2(x + dx, y + dy, z - dz)
			+ g2(x + dx, y - dy, z + dz) + g2(x + dx, y - dy, z - dz)
			+ g2(x - dx, y + dy, z + dz) + g2(x - dx, y + dy, z - dz)
			+ g2(x - dx, y - dy, z + dz) + g2(x - dx, y - dy, z - dz)));

	return fnxy;
}*/
/*
static __device__  double fnxy2(double x, double y, double z, double dx, double dy, double dz)
{
	double fnxy;

	fnxy = -pow((dx * dy * dz), (-1.0)) * (
		8.0 * g2(x, y, z)
		- 4.0 * (g2(x, y, z + dz) + g2(x, y, z - dz) + g2(x, y + dy, z)
			+ g2(x, y - dy, z) + g2(x + dx, y, z) + g2(x - dx, y, z))
		+ 2.0 * (g2(x, y + dy, z + dz) + g2(x, y + dy, z - dz)
			+ g2(x, y - dy, z + dz) + g2(x, y - dy, z - dz)
			+ g2(x + dx, y + dy, z) + g2(x + dx, y - dy, z)
			+ g2(x - dx, y + dy, z) + g2(x - dx, y - dy, z)
			+ g2(x + dx, y, z + dz) + g2(x + dx, y, z - dz)
			+ g2(x - dx, y, z + dz) + g2(x - dx, y, z - dz))
		- 1.0 * (g2(x + dx, y + dy, z + dz) + g2(x + dx, y + dy, z - dz)
			+ g2(x + dx, y - dy, z + dz) + g2(x + dx, y - dy, z - dz)
			+ g2(x - dx, y + dy, z + dz) + g2(x - dx, y + dy, z - dz)
			+ g2(x - dx, y - dy, z + dz) + g2(x - dx, y - dy, z - dz)));

	return fnxy;
}*/
/*
static __device__  double fnxy3(double x, double y, double z, double dx, double dy, double dz)
{
	double fnxy;

	fnxy = -pow((dx * dy * dz), (-1.0)) * (
		8.0 * g2(x, y, z)
		- 4.0 * (g2(x, y, z + dz) + g2(x, y, z - dz) + g2(x, y + dy, z)
			+ g2(x, y - dy, z) + g2(x + dx, y, z) + g2(x - dx, y, z))
		+ 2.0 * (g2(x, y + dy, z + dz) + g2(x, y + dy, z - dz)
			+ g2(x, y - dy, z + dz) + g2(x, y - dy, z - dz)
			+ g2(x + dx, y + dy, z) + g2(x + dx, y - dy, z)
			+ g2(x - dx, y + dy, z) + g2(x - dx, y - dy, z)
			+ g2(x + dx, y, z + dz) + g2(x + dx, y, z - dz)
			+ g2(x - dx, y, z + dz) + g2(x - dx, y, z - dz))
		- 1.0 * (g2(x + dx, y + dy, z + dz) + g2(x + dx, y + dy, z - dz)
			+ g2(x + dx, y - dy, z + dz) + g2(x + dx, y - dy, z - dz)
			+ g2(x - dx, y + dy, z + dz) + g2(x - dx, y + dy, z - dz)
			+ g2(x - dx, y - dy, z + dz) + g2(x - dx, y - dy, z - dz)));

	return fnxy;
}*/
/*
static __device__  double Dz(double a, double b, double c)
{
	double a2, b2, c2, Dz;
	double sqrta2b2, sqrta2c2, sqrtb2c2, sqrta2b2c2;

	a2 = a * a;
	b2 = b * b;
	c2 = c * c;

	sqrta2b2 = sqrt(a2 + b2);
	sqrta2c2 = sqrt(a2 + c2);
	sqrtb2c2 = sqrt(b2 + c2);
	sqrta2b2c2 = sqrt(a2 + b2 + c2);

	Dz = -(((b2 - c2) / (2.0 * b * c)) * log((sqrta2b2c2 - a) / (sqrta2b2c2 + a))
		+ ((a2 - c2) / (2.0 * a * c)) * log((sqrta2b2c2 - b) / (sqrta2b2c2 + b))
		+ (b / (2.0 * c)) * log((sqrta2b2 + a) / (sqrta2b2 - a))
		+ (a / (2.0 * c)) * log((sqrta2b2 + b) / (sqrta2b2 - b))
		+ (c / (2.0 * a)) * log((sqrtb2c2 - b) / (sqrtb2c2 + b))
		+ (c / (2.0 * b)) * log((sqrta2c2 - a) / (sqrta2c2 + a))
		+ 2.0 * atan((a * b) / (c * sqrta2b2c2))
		+ (pow(a, 3.0) + pow(b, 3.0) - 2.0 * pow(c, 3.0)) / (3.0 * a * b * c)
		+ (pow(a, 2.0) + pow(b, 2.0) - 2.0 * pow(c, 2.0)) / (3.0 * a * b * c) * sqrta2b2c2
		+ c / (a * b) * (sqrta2c2 + sqrtb2c2)
		- (pow(sqrta2b2, 3.0) + pow(sqrtb2c2, 3.0) + pow(sqrta2c2, 3.0)) / (3.0 * a * b * c)
		) * 4.0;

	return Dz;
}*/

/*
static __device__  double Dz2(double a, double b, double c)
{
	double a2, b2, c2, Dz;
	double sqrta2b2, sqrta2c2, sqrtb2c2, sqrta2b2c2;

	a2 = a * a;
	b2 = b * b;
	c2 = c * c;

	sqrta2b2 = sqrt(a2 + b2);
	sqrta2c2 = sqrt(a2 + c2);
	sqrtb2c2 = sqrt(b2 + c2);
	sqrta2b2c2 = sqrt(a2 + b2 + c2);

	Dz = -(((b2 - c2) / (2.0 * b * c)) * log((sqrta2b2c2 - a) / (sqrta2b2c2 + a))
		+ ((a2 - c2) / (2.0 * a * c)) * log((sqrta2b2c2 - b) / (sqrta2b2c2 + b))
		+ (b / (2.0 * c)) * log((sqrta2b2 + a) / (sqrta2b2 - a))
		+ (a / (2.0 * c)) * log((sqrta2b2 + b) / (sqrta2b2 - b))
		+ (c / (2.0 * a)) * log((sqrtb2c2 - b) / (sqrtb2c2 + b))
		+ (c / (2.0 * b)) * log((sqrta2c2 - a) / (sqrta2c2 + a))
		+ 2.0 * atan((a * b) / (c * sqrta2b2c2))
		+ (pow(a, 3) + pow(b, 3) - 2.0 * pow(c, 3)) / (3.0 * a * b * c)
		+ (pow(a, 2) + pow(b, 2) - 2.0 * pow(c, 2)) / (3.0 * a * b * c) * sqrta2b2c2
		+ c / (a * b) * (sqrta2c2 + sqrtb2c2)
		- (pow(sqrta2b2, 3) + pow(sqrtb2c2, 3) + pow(sqrta2c2, 3)) / (3.0 * a * b * c)
		) * 4.0;

	return Dz;
}
*/

/*
static __device__  double Dz3(double a, double b, double c)
{
	double a2, b2, c2, Dz;
	double sqrta2b2, sqrta2c2, sqrtb2c2, sqrta2b2c2;

	a2 = a * a;
	b2 = b * b;
	c2 = c * c;

	sqrta2b2 = sqrt(a2 + b2);
	sqrta2c2 = sqrt(a2 + c2);
	sqrtb2c2 = sqrt(b2 + c2);
	sqrta2b2c2 = sqrt(a2 + b2 + c2);

	Dz = -(((b2 - c2) / (2.0 * b * c)) * log((sqrta2b2c2 - a) / (sqrta2b2c2 + a))
		+ ((a2 - c2) / (2.0 * a * c)) * log((sqrta2b2c2 - b) / (sqrta2b2c2 + b))
		+ (b / (2.0 * c)) * log((sqrta2b2 + a) / (sqrta2b2 - a))
		+ (a / (2.0 * c)) * log((sqrta2b2 + b) / (sqrta2b2 - b))
		+ (c / (2.0 * a)) * log((sqrtb2c2 - b) / (sqrtb2c2 + b))
		+ (c / (2.0 * b)) * log((sqrta2c2 - a) / (sqrta2c2 + a))
		+ 2.0 * atan((a * b) / (c * sqrta2b2c2))
		+ (pow(a, 3) + pow(b, 3) - 2.0 * pow(c, 3)) / (3.0 * a * b * c)
		+ (pow(a, 2) + pow(b, 2) - 2.0 * pow(c, 2)) / (3.0 * a * b * c) * sqrta2b2c2
		+ c / (a * b) * (sqrta2c2 + sqrtb2c2)
		- (pow(sqrta2b2, 3) + pow(sqrtb2c2, 3) + pow(sqrta2c2, 3)) / (3.0 * a * b * c)
		) * 4.0;

	return Dz;
}
*/


__global__ void kernel_tensor1partea(Type_var* NDxx, Type_var h, int Mcx, int Mcy, int Mcz) {


	int kz2 = (int)Mcz / 2.0;
	int ky2 = (int)Mcy / 2.0;
	int kx2 = (int)Mcx / 2.0;

	unsigned int ind = blockDim.x * blockIdx.x + threadIdx.x;

	int n3, n2, n1;


	if (ind < Mcx * Mcy * Mcy) {
		//printf("\n Valore di ind %d: \n", ind);
		n3 = ind / (Mcx) / (Mcy);
		n2 = (ind - n3 * (Mcx) * (Mcy)) / (Mcx);
		n1 = ind - n3 * (Mcx) * (Mcy)-n2 * (Mcx);

		if ((-kx2 + 1 + n1) == 0 && (-ky2 + 1 + n2) == 0 && (-kz2 + 1 + n3) == 0) {

			double a = h;
			double b = 1.0e0, c = 1.0e0;

			double a2, b2, c2, Dz;
			double sqrta2b2, sqrta2c2, sqrtb2c2, sqrta2b2c2;

			a2 = a * a;
			b2 = b * b;
			c2 = c * c;

			sqrta2b2 = sqrt(a2 + b2);
			sqrta2c2 = sqrt(a2 + c2);
			sqrtb2c2 = sqrt(b2 + c2);
			sqrta2b2c2 = sqrt(a2 + b2 + c2);

			Dz = -(((b2 - c2) / (2.0 * b * c)) * log((sqrta2b2c2 - a) / (sqrta2b2c2 + a))
				+ ((a2 - c2) / (2.0 * a * c)) * log((sqrta2b2c2 - b) / (sqrta2b2c2 + b))
				+ (b / (2.0 * c)) * log((sqrta2b2 + a) / (sqrta2b2 - a))
				+ (a / (2.0 * c)) * log((sqrta2b2 + b) / (sqrta2b2 - b))
				+ (c / (2.0 * a)) * log((sqrtb2c2 - b) / (sqrtb2c2 + b))
				+ (c / (2.0 * b)) * log((sqrta2c2 - a) / (sqrta2c2 + a))
				+ 2.0 * atan((a * b) / (c * sqrta2b2c2))
				+ (pow(a, 3.0) + pow(b, 3.0) - 2.0 * pow(c, 3.0)) / (3.0 * a * b * c)
				+ (pow(a, 2.0) + pow(b, 2.0) - 2.0 * pow(c, 2.0)) / (3.0 * a * b * c) * sqrta2b2c2
				+ c / (a * b) * (sqrta2c2 + sqrtb2c2)
				- (pow(sqrta2b2, 3.0) + pow(sqrtb2c2, 3.0) + pow(sqrta2c2, 3.0)) / (3.0 * a * b * c)
				) * 4.0;

			NDxx[n1 + n2 * Mcx + n3 * Mcx * Mcy] = Dz;
			//NDxx[n1 + n2 * Mcx + n3 * Mcx * Mcy] = Dz(h, 1.0e0, 1.0e0);

		}
		else {
			//printf("\n Valore di ind %d: \n", ind);
			double x = (-kx2 + 1 + n1) * 1.0;
			double y = (-ky2 + 1 + n2) * 1.0;
			double z = (-kz2 + 1 + n3) * h;
			double dx = 1.e0;  double dy = 1.0e0;  double dz = h;

			double fnxx = -pow((dx * dy * dz), (-1.0)) * (
				8.0 * f2(x, y, z)
				- 4.0 * (f2(x, y, z + dz) + f2(x, y, z - dz) + f2(x, y + dy, z)
					+ f2(x, y - dy, z) + f2(x + dx, y, z) + f2(x - dx, y, z))
				+ 2.0 * (f2(x, y + dy, z + dz) + f2(x, y + dy, z - dz)
					+ f2(x, y - dy, z + dz) + f2(x, y - dy, z - dz)
					+ f2(x + dx, y + dy, z) + f2(x + dx, y - dy, z)
					+ f2(x - dx, y + dy, z) + f2(x - dx, y - dy, z)
					+ f2(x + dx, y, z + dz) + f2(x + dx, y, z - dz)
					+ f2(x - dx, y, z + dz) + f2(x - dx, y, z - dz))
				- 1.0 * (f2(x + dx, y + dy, z + dz) + f2(x + dx, y + dy, z - dz)
					+ f2(x + dx, y - dy, z + dz) + f2(x + dx, y - dy, z - dz)
					+ f2(x - dx, y + dy, z + dz) + f2(x - dx, y + dy, z - dz)
					+ f2(x - dx, y - dy, z + dz) + f2(x - dx, y - dy, z - dz)));

			//NDxx[n1 + n2 * Mcx + n3 * Mcx * Mcy] = fnxx((-kx2 + 1 + n1) * 1.0, (-ky2 + 1 + n2) * 1.0, (-kz2 + 1 + n3) * h, 1.e0, 1.0e0, h);
			NDxx[n1 + n2 * Mcx + n3 * Mcx * Mcy] = fnxx;

		}
	}


}


__global__
void kernel_tensor1parteb(Type_var* NDyy, Type_var h, int Mcx, int Mcy, int Mcz) {


	int kz2 = (int)Mcz / 2.0;
	int ky2 = (int)Mcy / 2.0;
	int kx2 = (int)Mcx / 2.0;

	unsigned int ind = blockDim.x * blockIdx.x + threadIdx.x;

	int n3, n2, n1;

	if (ind < Mcx * Mcy * Mcy) {

		n3 = ind / (Mcx) / (Mcy);
		n2 = (ind - n3 * (Mcx) * (Mcy)) / (Mcx);
		n1 = ind - n3 * (Mcx) * (Mcy)-n2 * (Mcx);

		if ((-kx2 + 1 + n1) == 0 && (-ky2 + 1 + n2) == 0 && (-kz2 + 1 + n3) == 0) {

			double a = 1.0e0; double b = h; double c = 1.0e0;
			double a2, b2, c2;
			double sqrta2b2, sqrta2c2, sqrtb2c2, sqrta2b2c2;

			a2 = a * a;
			b2 = b * b;
			c2 = c * c;

			sqrta2b2 = sqrt(a2 + b2);
			sqrta2c2 = sqrt(a2 + c2);
			sqrtb2c2 = sqrt(b2 + c2);
			sqrta2b2c2 = sqrt(a2 + b2 + c2);

			double Dz2 = -(((b2 - c2) / (2.0 * b * c)) * log((sqrta2b2c2 - a) / (sqrta2b2c2 + a))
				+ ((a2 - c2) / (2.0 * a * c)) * log((sqrta2b2c2 - b) / (sqrta2b2c2 + b))
				+ (b / (2.0 * c)) * log((sqrta2b2 + a) / (sqrta2b2 - a))
				+ (a / (2.0 * c)) * log((sqrta2b2 + b) / (sqrta2b2 - b))
				+ (c / (2.0 * a)) * log((sqrtb2c2 - b) / (sqrtb2c2 + b))
				+ (c / (2.0 * b)) * log((sqrta2c2 - a) / (sqrta2c2 + a))
				+ 2.0 * atan((a * b) / (c * sqrta2b2c2))
				+ (pow(a, 3.0) + pow(b, 3.0) - 2.0 * pow(c, 3.0)) / (3.0 * a * b * c)
				+ (pow(a, 2.0) + pow(b, 2.0) - 2.0 * pow(c, 2.0)) / (3.0 * a * b * c) * sqrta2b2c2
				+ c / (a * b) * (sqrta2c2 + sqrtb2c2)
				- (pow(sqrta2b2, 3.0) + pow(sqrtb2c2, 3.0) + pow(sqrta2c2, 3.0)) / (3.0 * a * b * c)
				) * 4.0;

			NDyy[n1 + n2 * Mcx + n3 * Mcx * Mcy] = Dz2;
			//NDyy[n1 + n2 * Mcx + n3 * Mcx * Mcy] = Dz2(1.0e0, h, 1.0e0);

		}
		else {

			double x = (-ky2 + 1 + n2) * 1.0;
			double y = (-kx2 + 1 + n1) * 1.0;
			double z = (-kz2 + 1 + n3) * h;
			double dx = 1.e0; double dy = 1.e0; double dz = h;
			double fnxx2;

			fnxx2 = -pow((dx * dy * dz), (-1.0)) * (
				8.0 * f2(x, y, z)
				- 4.0 * (f2(x, y, z + dz) + f2(x, y, z - dz) + f2(x, y + dy, z)
					+ f2(x, y - dy, z) + f2(x + dx, y, z) + f2(x - dx, y, z))
				+ 2.0 * (f2(x, y + dy, z + dz) + f2(x, y + dy, z - dz)
					+ f2(x, y - dy, z + dz) + f2(x, y - dy, z - dz)
					+ f2(x + dx, y + dy, z) + f2(x + dx, y - dy, z)
					+ f2(x - dx, y + dy, z) + f2(x - dx, y - dy, z)
					+ f2(x + dx, y, z + dz) + f2(x + dx, y, z - dz)
					+ f2(x - dx, y, z + dz) + f2(x - dx, y, z - dz))
				- 1.0 * (f2(x + dx, y + dy, z + dz) + f2(x + dx, y + dy, z - dz)
					+ f2(x + dx, y - dy, z + dz) + f2(x + dx, y - dy, z - dz)
					+ f2(x - dx, y + dy, z + dz) + f2(x - dx, y + dy, z - dz)
					+ f2(x - dx, y - dy, z + dz) + f2(x - dx, y - dy, z - dz)));

			NDyy[n1 + n2 * Mcx + n3 * Mcx * Mcy] = fnxx2;
			//NDyy[n1 + n2 * Mcx + n3 * Mcx * Mcy] = fnxx2((-ky2 + 1 + n2) * 1.0, (-kx2 + 1 + n1) * 1.0, (-kz2 + 1 + n3) * h, 1.e0, 1.0e0, h);	

		}
	}


}


__global__
void kernel_tensor1partec(Type_var* NDzz, Type_var h, int Mcx, int Mcy, int Mcz) {

	double a = 1.0e0; double b = 1.0e0; double c = h;

	int kz2 = (int)Mcz / 2.0;
	int ky2 = (int)Mcy / 2.0;
	int kx2 = (int)Mcx / 2.0;

	unsigned int ind = blockDim.x * blockIdx.x + threadIdx.x;

	int n3, n2, n1;

	if (ind < Mcx * Mcy * Mcy) {

		n3 = ind / (Mcx) / (Mcy);
		n2 = (ind - n3 * (Mcx) * (Mcy)) / (Mcx);
		n1 = ind - n3 * (Mcx) * (Mcy)-n2 * (Mcx);

		if ((-kx2 + 1 + n1) == 0 && (-ky2 + 1 + n2) == 0 && (-kz2 + 1 + n3) == 0) {

			double a2, b2, c2, Dz3;
			double sqrta2b2, sqrta2c2, sqrtb2c2, sqrta2b2c2;

			a2 = a * a;
			b2 = b * b;
			c2 = c * c;

			sqrta2b2 = sqrt(a2 + b2);
			sqrta2c2 = sqrt(a2 + c2);
			sqrtb2c2 = sqrt(b2 + c2);
			sqrta2b2c2 = sqrt(a2 + b2 + c2);

			Dz3 = -(((b2 - c2) / (2.0 * b * c)) * log((sqrta2b2c2 - a) / (sqrta2b2c2 + a))
				+ ((a2 - c2) / (2.0 * a * c)) * log((sqrta2b2c2 - b) / (sqrta2b2c2 + b))
				+ (b / (2.0 * c)) * log((sqrta2b2 + a) / (sqrta2b2 - a))
				+ (a / (2.0 * c)) * log((sqrta2b2 + b) / (sqrta2b2 - b))
				+ (c / (2.0 * a)) * log((sqrtb2c2 - b) / (sqrtb2c2 + b))
				+ (c / (2.0 * b)) * log((sqrta2c2 - a) / (sqrta2c2 + a))
				+ 2.0 * atan((a * b) / (c * sqrta2b2c2))
				+ (pow(a, 3.0) + pow(b, 3.0) - 2.0 * pow(c, 3.0)) / (3.0 * a * b * c)
				+ (pow(a, 2.0) + pow(b, 2.0) - 2.0 * pow(c, 2.0)) / (3.0 * a * b * c) * sqrta2b2c2
				+ c / (a * b) * (sqrta2c2 + sqrtb2c2)
				- (pow(sqrta2b2, 3.0) + pow(sqrtb2c2, 3.0) + pow(sqrta2c2, 3.0)) / (3.0 * a * b * c)
				) * 4.0;
			NDzz[n1 + n2 * Mcx + n3 * Mcx * Mcy] = Dz3;
			//NDzz[n1 + n2 * Mcx + n3 * Mcx * Mcy] = Dz3(1.0e0, 1.0e0, h);
		}
		else {

			double x = (-kz2 + 1 + n3) * h;
			double y = (-ky2 + 1 + n2) * 1.0;
			double z = (-kx2 + 1 + n1) * 1.0;
			double dx = h;
			double dy = 1.0e0; double dz = 1.0e0;
			double fnxx3;

			fnxx3 = -pow((dx * dy * dz), (-1.0)) * (
				8.0 * f2(x, y, z)
				- 4.0 * (f2(x, y, z + dz) + f2(x, y, z - dz) + f2(x, y + dy, z)
					+ f2(x, y - dy, z) + f2(x + dx, y, z) + f2(x - dx, y, z))
				+ 2.0 * (f2(x, y + dy, z + dz) + f2(x, y + dy, z - dz)
					+ f2(x, y - dy, z + dz) + f2(x, y - dy, z - dz)
					+ f2(x + dx, y + dy, z) + f2(x + dx, y - dy, z)
					+ f2(x - dx, y + dy, z) + f2(x - dx, y - dy, z)
					+ f2(x + dx, y, z + dz) + f2(x + dx, y, z - dz)
					+ f2(x - dx, y, z + dz) + f2(x - dx, y, z - dz))
				- 1.0 * (f2(x + dx, y + dy, z + dz) + f2(x + dx, y + dy, z - dz)
					+ f2(x + dx, y - dy, z + dz) + f2(x + dx, y - dy, z - dz)
					+ f2(x - dx, y + dy, z + dz) + f2(x - dx, y + dy, z - dz)
					+ f2(x - dx, y - dy, z + dz) + f2(x - dx, y - dy, z - dz)));

			NDzz[n1 + n2 * Mcx + n3 * Mcx * Mcy] = fnxx3;
			//NDzz[n1 + n2 * Mcx + n3 * Mcx * Mcy] = fnxx3((-kz2 + 1 + n3) * h, (-ky2 + 1 + n2) * 1.0, (-kx2 + 1 + n1) * 1.0, h, 1.0e0, 1.0e0);

		}
	}


}

__global__
void kernel_tensor1parted(Type_var* NDxy, Type_var h, int Mcx, int Mcy, int Mcz) {


	int kz2 = (int)Mcz / 2.0;
	int ky2 = (int)Mcy / 2.0;
	int kx2 = (int)Mcx / 2.0;

	unsigned int ind = blockDim.x * blockIdx.x + threadIdx.x;

	int n3, n2, n1;

	if (ind < Mcx * Mcy * Mcy) {

		n3 = ind / (Mcx) / (Mcy);
		n2 = (ind - n3 * (Mcx) * (Mcy)) / (Mcx);
		n1 = ind - n3 * (Mcx) * (Mcy)-n2 * (Mcx);

		if ((-kx2 + 1 + n1) == 0 && (-ky2 + 1 + n2) == 0 && (-kz2 + 1 + n3) == 0) {

			NDxy[n1 + n2 * Mcx + n3 * Mcx * Mcy] = 0.0e0;

		}
		else {

			double x = (-kx2 + 1 + n1) * 1.0;
			double y = (-ky2 + 1 + n2) * 1.0;
			double z = (-kz2 + 1 + n3) * h;
			double dx = 1.e0; double dy = 1.e0; double dz = h;
			double fnxy;

			fnxy = -pow((dx * dy * dz), (-1.0)) * (
				8.0 * g2(x, y, z)
				- 4.0 * (g2(x, y, z + dz) + g2(x, y, z - dz) + g2(x, y + dy, z)
					+ g2(x, y - dy, z) + g2(x + dx, y, z) + g2(x - dx, y, z))
				+ 2.0 * (g2(x, y + dy, z + dz) + g2(x, y + dy, z - dz)
					+ g2(x, y - dy, z + dz) + g2(x, y - dy, z - dz)
					+ g2(x + dx, y + dy, z) + g2(x + dx, y - dy, z)
					+ g2(x - dx, y + dy, z) + g2(x - dx, y - dy, z)
					+ g2(x + dx, y, z + dz) + g2(x + dx, y, z - dz)
					+ g2(x - dx, y, z + dz) + g2(x - dx, y, z - dz))
				- 1.0 * (g2(x + dx, y + dy, z + dz) + g2(x + dx, y + dy, z - dz)
					+ g2(x + dx, y - dy, z + dz) + g2(x + dx, y - dy, z - dz)
					+ g2(x - dx, y + dy, z + dz) + g2(x - dx, y + dy, z - dz)
					+ g2(x - dx, y - dy, z + dz) + g2(x - dx, y - dy, z - dz)));

			NDxy[n1 + n2 * Mcx + n3 * Mcx * Mcy] = fnxy;
			//NDxy[n1 + n2 * Mcx + n3 * Mcx * Mcy] = fnxy((-kx2 + 1 + n1) * 1.0, (-ky2 + 1 + n2) * 1.0, (-kz2 + 1 + n3) * h, 1.e0, 1.0e0, h);

		}
	}


}


__global__
void kernel_tensor1partee(Type_var* NDxz, Type_var h, int Mcx, int Mcy, int Mcz) {


	int kz2 = (int)Mcz / 2.0;
	int ky2 = (int)Mcy / 2.0;
	int kx2 = (int)Mcx / 2.0;

	unsigned int ind = blockDim.x * blockIdx.x + threadIdx.x;

	int n3, n2, n1;

	if (ind < Mcx * Mcy * Mcy) {

		n3 = ind / (Mcx) / (Mcy);
		n2 = (ind - n3 * (Mcx) * (Mcy)) / (Mcx);
		n1 = ind - n3 * (Mcx) * (Mcy)-n2 * (Mcx);

		if ((-kx2 + 1 + n1) == 0 && (-ky2 + 1 + n2) == 0 && (-kz2 + 1 + n3) == 0) {

			NDxz[n1 + n2 * Mcx + n3 * Mcx * Mcy] = 0.0e0;

		}
		else {
			double x = (-kx2 + 1 + n1) * 1.0;
			double y = (-kz2 + 1 + n3) * h;
			double z = (-ky2 + 1 + n2) * 1.0;
			double dx = 1.e0; double dy = h; double dz = 1.e0;
			double fnxy2 = -pow((dx * dy * dz), (-1.0)) * (
				8.0 * g2(x, y, z)
				- 4.0 * (g2(x, y, z + dz) + g2(x, y, z - dz) + g2(x, y + dy, z)
					+ g2(x, y - dy, z) + g2(x + dx, y, z) + g2(x - dx, y, z))
				+ 2.0 * (g2(x, y + dy, z + dz) + g2(x, y + dy, z - dz)
					+ g2(x, y - dy, z + dz) + g2(x, y - dy, z - dz)
					+ g2(x + dx, y + dy, z) + g2(x + dx, y - dy, z)
					+ g2(x - dx, y + dy, z) + g2(x - dx, y - dy, z)
					+ g2(x + dx, y, z + dz) + g2(x + dx, y, z - dz)
					+ g2(x - dx, y, z + dz) + g2(x - dx, y, z - dz))
				- 1.0 * (g2(x + dx, y + dy, z + dz) + g2(x + dx, y + dy, z - dz)
					+ g2(x + dx, y - dy, z + dz) + g2(x + dx, y - dy, z - dz)
					+ g2(x - dx, y + dy, z + dz) + g2(x - dx, y + dy, z - dz)
					+ g2(x - dx, y - dy, z + dz) + g2(x - dx, y - dy, z - dz)));

			NDxz[n1 + n2 * Mcx + n3 * Mcx * Mcy] = fnxy2;
			//NDxz[n1 + n2 * Mcx + n3 * Mcx * Mcy] = fnxy2((-kx2 + 1 + n1) * 1.0, (-kz2 + 1 + n3) * h, (-ky2 + 1 + n2) * 1.0, 1.e0, h, 1.0e0);

		}
	}


}

__global__
void kernel_tensor1partef(Type_var* NDyz, Type_var h, int Mcx, int Mcy, int Mcz) {


	int kz2 = (int)Mcz / 2.0;
	int ky2 = (int)Mcy / 2.0;
	int kx2 = (int)Mcx / 2.0;

	unsigned int ind = blockDim.x * blockIdx.x + threadIdx.x;

	int n3, n2, n1;

	if (ind < Mcx * Mcy * Mcy) {

		n3 = ind / (Mcx) / (Mcy);
		n2 = (ind - n3 * (Mcx) * (Mcy)) / (Mcx);
		n1 = ind - n3 * (Mcx) * (Mcy)-n2 * (Mcx);

		if ((-kx2 + 1 + n1) == 0 && (-ky2 + 1 + n2) == 0 && (-kz2 + 1 + n3) == 0) {

			NDyz[n1 + n2 * Mcx + n3 * Mcx * Mcy] = 0.0e0;

		}
		else {
			double x = (-ky2 + 1 + n2) * 1.0;
			double y = (-kz2 + 1 + n3) * h;
			double z = (-kx2 + 1 + n1) * 1.0;
			double dx = 1.e0; double dy = h; double dz = 1.e0;

			double fnxy3 = -pow((dx * dy * dz), (-1.0)) * (
				8.0 * g2(x, y, z)
				- 4.0 * (g2(x, y, z + dz) + g2(x, y, z - dz) + g2(x, y + dy, z)
					+ g2(x, y - dy, z) + g2(x + dx, y, z) + g2(x - dx, y, z))
				+ 2.0 * (g2(x, y + dy, z + dz) + g2(x, y + dy, z - dz)
					+ g2(x, y - dy, z + dz) + g2(x, y - dy, z - dz)
					+ g2(x + dx, y + dy, z) + g2(x + dx, y - dy, z)
					+ g2(x - dx, y + dy, z) + g2(x - dx, y - dy, z)
					+ g2(x + dx, y, z + dz) + g2(x + dx, y, z - dz)
					+ g2(x - dx, y, z + dz) + g2(x - dx, y, z - dz))
				- 1.0 * (g2(x + dx, y + dy, z + dz) + g2(x + dx, y + dy, z - dz)
					+ g2(x + dx, y - dy, z + dz) + g2(x + dx, y - dy, z - dz)
					+ g2(x - dx, y + dy, z + dz) + g2(x - dx, y + dy, z - dz)
					+ g2(x - dx, y - dy, z + dz) + g2(x - dx, y - dy, z - dz)));

			NDyz[n1 + n2 * Mcx + n3 * Mcx * Mcy] = fnxy3;
			//NDyz[n1 + n2 * Mcx + n3 * Mcx * Mcy] = fnxy3((-ky2 + 1 + n2) * 1.0, (-kz2 + 1 + n3) * h, (-kx2 + 1 + n1) * 1.0, 1.e0, h, 1.0e0);

		}
	}


}


__global__
void kernel_tensor2parte(HD_Complex_Type* cuSD, Type_var* ND, Type_var h, int Mcx, int Mcy, int Mcz) {

	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

	int n3 = 0, n2 = 0, n1 = 0;
	int nn3 = 0, nn2 = 0, nn1 = 0;
	//double prova1 = fnxx(xn1, xn2, xn3, 1.e0, 1.0e0, h);
	//double prova2 = fnxx(xn2, xn1, xn3, 1.e0, 1.0e0, h);

	int kz2 = (int)Mcz / 2.0;
	int ky2 = (int)Mcy / 2.0;
	int kx2 = (int)Mcx / 2.0;

	if (index < Mcx * Mcy * Mcy) {



		n3 = index / (Mcx) / (Mcy);
		n2 = (index - n3 * (Mcx) * (Mcy)) / (Mcx);
		n1 = index - n3 * (Mcx) * (Mcy)-n2 * (Mcx);

		nn1 = n1 + (kx2 - 1);
		nn2 = n2 + (ky2 - 1);
		nn3 = n3 + (kz2 - 1);

		if (n3 > (kz2)) {
			nn3 = n3 - (kz2 + 1);
		}
		if (n2 > (ky2)) {
			nn2 = n2 - (ky2 + 1);
		}
		if (n1 > (kx2)) {
			nn1 = n1 - (kx2 + 1);
		}

		cuSD[n1 + n2 * Mcx + n3 * Mcx * Mcy] = make_cuDoubleComplex(ND[nn1 + nn2 * Mcx + nn3 * Mcx * Mcy], 0.0e0);

	}



	/*
	for (int n3 = blockIdx.z * blockDim.z + threadIdx.z; n3 <= (Mcz - 1); n3 += blockDim.z * gridDim.z) {
		for (int n2 = blockIdx.y * blockDim.y + threadIdx.y; n2 <= (Mcy - 1); n2 += blockDim.y * gridDim.y) {
			for (int n1 = blockIdx.x * blockDim.x + threadIdx.x; n1 <= (Mcx - 1); n1 += blockDim.x * gridDim.x) {



				int  nn1, nn2, nn3, index1, index2;
				double xn1, xn2, xn3;


				nn1 = n1 + (kx2 - 1);
				nn2 = n2 + (ky2 - 1);
				nn3 = n3 + (kz2 - 1);

				if (n3 > (kz2)) {
					nn3 = n3 - (kz2 + 1);
				}
				if (n2 > (ky2)) {
					nn2 = n2 - (ky2 + 1);
				}
				if (n1 > (kx2)) {
					nn1 = n1 - (kx2 + 1);
				}

				index1 = n1 + n2 * Mcx + n3 * Mcx * Mcy;
				index2 = nn1 + nn2 * Mcx + nn3 * Mcx * Mcy;

				// make_cuDobleComplex, returns a cuDoubleComplex from its
				// real and imaginary part


				cuSDxx[index1] = make_cuDoubleComplex(NDxx[index2], 0.0e0);
				cuSDyy[index1] = make_cuDoubleComplex(NDyy[index2], 0.0e0);
				cuSDzz[index1] = make_cuDoubleComplex(NDzz[index2], 0.0e0);
				cuSDxy[index1] = make_cuDoubleComplex(NDxy[index2], 0.0e0);
				cuSDxz[index1] = make_cuDoubleComplex(NDxz[index2], 0.0e0);
				cuSDyz[index1] = make_cuDoubleComplex(NDyz[index2], 0.0e0);

			}
		}



	}*/

}


