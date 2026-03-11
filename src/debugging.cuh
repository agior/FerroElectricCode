#pragma once
/**
 * @file CudaDebugging.hpp
 * @brief Provides CUDA/Thrust debugging utilities, error checking, and field inspection macros.
 *
 * This header defines:
 *  - Safe CUDA error-checking and assertion macros
 *  - Runtime FFT error decoding for CUFFT
 *  - Device/host constants fetch utilities
 *  - File output functions for saving GPU fields
 *  - Debug comparison macros for validating GPU results against reference data
 *
 * Enabled automatically when **DEBUGGING** is defined.
 */

#include "Debugging.hpp"

#ifdef DEBUGGING

 // ============================================================================
 // Configuration
 // ============================================================================
#define debuggingFolder "debugging"
#define STOP true
// #define STOP false   ///< Uncomment to auto-continue after saveField()

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#if TESTING
#include <iomanip>
#include <fstream>
#endif
#include "parameters.h"

// ============================================================================
// Device Constant Fetch Utility
// ============================================================================
/**
 * @brief Retrieves a spatially varying or constant parameter value for a given grid index.
 *
 * @tparam T Value type (e.g., float, double)
 * @param constant Output constant reference.
 * @param Nxy Number of cells in the XY plane.
 * @param Nz  Number of cells along Z.
 * @param idx Global linear index.
 * @param size Length of the constant array.
 * @param constants Pointer to device/host constant array.
 * @return true on success, false otherwise.
 */
template<typename T>
__host__ __device__ bool GetConstant(T& constant, const int Nxy, const int Nz, const int idx, const int size, const T* constants)
{
    if (size == 1) { constant = constants[0]; }
    else if (size == Nz) { constant = constants[idx / Nxy]; }
    else if (size == Nxy * Nz) { constant = constants[idx]; }
    else { constant = {}; return SHOULD_NOT_GET_HERE; }
    return true;
}

/**
 * @brief Retrieves a constant with a default fallback value.
 */
template<typename T>
__host__ __device__ bool GetConstant(T& constant, const int Nxy, const int Nz, const int idx, const int size, const T* constants, const T defaultValue)
{
    if (size == 1) { constant = defaultValue; }
    else if (size == Nz) { constant = constants[idx / Nxy]; }
    else if (size == Nxy * Nz) { constant = constants[idx]; }
    else { constant = {}; return SHOULD_NOT_GET_HERE; }
    return true;
}

// ============================================================================
// Field Saver Utility
// ============================================================================
/**
 * @brief Saves a field from GPU device memory to a text file for debugging.
 *
 * @param name Field name (file will be saved as `<debuggingFolder>/<name>.txt`)
 * @param field Pointer to device memory.
 * @param N Number of elements in the field.
 * @param stop If true, pauses execution after saving for manual inspection.
 */
template<typename T>
void saveField(const std::string& name, T* field, int N, bool stop = true)
{
    T* temp = new T[N];
    cudaMemcpy(temp, field, N * sizeof(T), cudaMemcpyDeviceToHost);
    ASSERT(createFolder(debuggingFolder));
    std::ofstream writer(std::string(debuggingFolder) + "/" + std::string(name) + ".txt");
    writer.precision(20);
    writer << std::fixed;
    for (int i = 0; i < N; i++)
        writer << temp[i] << '\n';
    writer.close();
    delete[] temp;
    if (stop) {
        std::cout << "stop\n";
        std::cin.get();
    }
}

// ============================================================================
// CUFFT Error Code Translator
// ============================================================================
/**
 * @brief Converts CUFFT error codes to human-readable strings.
 */
inline std::string cufftGetErrorString(cufftResult_t errorCode)
{
    switch (errorCode)
    {
    case CUFFT_SUCCESS:                   return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:              return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:              return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:              return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:             return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:            return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:               return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:              return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:              return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA:            return "CUFFT_UNALIGNED_DATA";
    case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";
    case CUFFT_INVALID_DEVICE:            return "CUFFT_INVALID_DEVICE";
    case CUFFT_PARSE_ERROR:               return "CUFFT_PARSE_ERROR";
    case CUFFT_NO_WORKSPACE:              return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED:           return "CUFFT_NOT_IMPLEMENTED";
    case CUFFT_LICENSE_ERROR:             return "CUFFT_LICENSE_ERROR";
    case CUFFT_NOT_SUPPORTED:             return "CUFFT_NOT_SUPPORTED";
    default:                              return "Unknown CUFFT error";
    }
}

// ============================================================================
// CUDA Error-Checking Macros
// ============================================================================
#define CUDA_ASSERT(errorCode) \
do { \
    (void)((errorCode == cudaSuccess) || (customAssert(cudaGetErrorString(errorCode), __FILE__, (unsigned)(__LINE__)), breakExecution(), 0)); \
} while(0)

#define CUDA_ASSERT_FFT(errorCode) \
do { \
    (void)((errorCode == CUFFT_SUCCESS) || (customAssert(cufftGetErrorString(errorCode), __FILE__, (unsigned)(__LINE__)), breakExecution(), 0)); \
} while(0)

#define CUDA_CHECK(cudaExpr) \
do { \
    cudaError_t errorCode = cudaExpr; \
    CUDA_ASSERT(errorCode); \
} while(0)

#define CUDA_CHECK_FFT(cudaExpr) \
do { \
    cufftResult_t errorCode = cudaExpr; \
    CUDA_ASSERT_FFT(errorCode); \
} while(0)

#define CUDA_CHECK_KERNEL \
do { \
    cudaDeviceSynchronize(); \
    cudaError_t errCode = cudaGetLastError(); \
    CUDA_ASSERT(errCode); \
} while(0)

#define GET_CONSTANT(...) \
do { \
    if (!GetConstant(__VA_ARGS__)) { ASSERT(SHOULD_NOT_GET_HERE); } \
} while(0)

// ============================================================================
// Thrust GPU Result Comparison (TESTING builds only)
// ============================================================================
#if TESTING
#define THRUST_COMPARE_FUNCTION_VERSIONS3(resultOldPtr, resultNewPtr, Nx, Ny, Nz, stop) \
do { \
    Vec3<Type_var>* temp1 = new Vec3<Type_var>[Nx * Ny * Nz]; \
    Vec3<Type_var>* temp2 = new Vec3<Type_var>[Nx * Ny * Nz]; \
    cudaMemcpy(temp1, resultOldPtr, Nx * Ny * Nz * sizeof(Vec3<Type_var>), cudaMemcpyDeviceToHost); \
    cudaMemcpy(temp2, resultNewPtr, Nx * Ny * Nz * sizeof(Vec3<Type_var>), cudaMemcpyDeviceToHost); \
    ASSERT(createFolder(debuggingFolder)); \
    std::ofstream writer1(std::string(debuggingFolder) + "resultOld.txt"); \
    std::ofstream writer2(std::string(debuggingFolder) + "resultNew.txt"); \
    writer1.precision(20); writer2.precision(20); \
    writer1 << std::fixed; writer2 << std::fixed; \
    Vec3<Type_var> means; Vec3<Type_var> stds; \
    for (int ind = 0; ind < Nx * Ny * Nz; ind++) { \
        writer1 << temp1[ind] << "\n"; \
        writer2 << temp2[ind] << "\n"; \
        Vec3<Type_var> diff = temp1[ind] - temp2[ind]; \
        means += diff; \
        int i = ind % Nx; int j = (ind / Nx) % Ny; int k = ind / (Nx * Ny); \
        if (!(temp1[ind] == Vec3<T>())) { \
            Type_var threshold = 5.0e-15; \
            Type_var fracx = abs(diff.x / temp1[ind].x); \
            Type_var fracy = abs(diff.y / temp1[ind].y); \
            Type_var fracz = abs(diff.z / temp1[ind].z); \
            if ((fracx > threshold) || (fracy > threshold) || (fracz > threshold)) { \
                fracx *= 100.0; fracy *= 100.0; fracz *= 100.0; \
                std::cout << fracx << "%  " << fracy << "%  " << fracz << "%  [" << i << "," << j << "," << k << "]\n"; \
            } \
        } else if (((diff.x != 0.0) && (temp1[ind].x == 0.0)) || ((diff.y != 0.0) && (temp1[ind].y == 0.0)) || ((diff.z != 0.0) && (temp1[ind].z == 0.0))) { \
            std::cout << diff.x << "/" << temp1[ind].x << "  " << diff.y << "/" << temp1[ind].y << "  " << diff.z << "/" << temp1[ind].z \
                      << "  [" << i << "," << j << "," << k << "]\n"; \
        } \
    } \
    means /= Type_var(Nx * Ny * Nz); \
    for (int i = 0; i < Nx * Ny * Nz; i++) { \
        Vec3<Type_var> diff = (temp1[i] - temp2[i]) - means; \
        stds += { diff.x * diff.x, diff.y * diff.y, diff.z * diff.z }; \
    } \
    stds /= Type_var(Nx * Ny * Nz); \
    stds = { sqrt(stds.x), sqrt(stds.y), sqrt(stds.z) }; \
    std::cout << "means = " << means << "\nstds = " << stds << '\n'; \
    writer1.close(); writer2.close(); \
    delete[] temp1; delete[] temp2; \
    if (stop) { std::cout << "stop\n"; std::cin.get(); } \
} while(0)

#define SAVE_FIELD(...) saveField(__VA_ARGS__)
#else
#define THRUST_COMPARE_FUNCTION_VERSIONS3(resultOld, resultNew, Nx, Ny, Nz, stop)
#define SAVE_FIELD(...)
#endif

#else // DEBUGGING not defined

#define CUDA_CHECK(cudaExpr) (void)cudaExpr;
#define CUDA_CHECK_FFT(cudaExpr) (void)cudaExpr;
#define CUDA_CHECK_KERNEL
#define THRUST_COMPARE_FUNCTION_VERSIONS3(resultOld, resultNew, Nx, Ny, Nz, stop)
#define SAVE_FIELD(...)
#define GET_CONSTANT(...) GetConstant(__VA_ARGS__)

#endif // DEBUGGING
