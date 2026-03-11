#pragma once
#include "Vec3.h"
#include <fstream>
#include <iostream>
#include <assert.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>

/**
 * @file ab3m2Steps.h
 * @brief Defines templated CUDA-compatible functors for coefficient, shape,
 *        and field computations used in multistep integration and simulation kernels.
 *
 * Each struct here represents a callable unit (`operator()`) suitable for
 * GPU execution (via Thrust or custom CUDA kernels), enabling data-parallel
 * transformations of physical parameters such as α₁–α₆ coefficients, gradients,
 * shape masks, and external fields.
 */

 // ============================================================================
 // PHASE AND SHAPE FUNCTORS
 // ============================================================================

 /**
  * @brief Computes phase components for AC fields.
  * @tparam T Numerical type (float or double).
  */
template <typename T>
struct phaseAc {
    Vec3<T>* phase_ac;    ///< Pointer to phase components.
    int phaseVectorSize;  ///< Size of the phase array.
    int Ncz;              ///< Grid size along z-axis.
    int gridSize;         ///< Total grid size.

    __host__ __device__
        phaseAc(Vec3<T>* phase_ac_, int phaseVectorSize_, int Ncz_, int gridSize_)
        : phase_ac(phase_ac_), phaseVectorSize(phaseVectorSize_), Ncz(Ncz_), gridSize(gridSize_) {
    }

    __host__ __device__ Vec3<T> operator()(int idx) const;
};

/**
 * @brief Defines spatial shape function and geometry masks.
 */
template <typename T>
struct shape {
    int flag_shape;     ///< Shape flag (defines geometry type).
    int* shapeVector;   ///< Pointer to integer mask array.
    int Ncx, Ncy, Ncz;  ///< Grid dimensions along each axis.
    T delta;            ///< Spatial resolution (Δx, Δy, Δz assumed equal).

    __host__ __device__
        shape(int flag_shape_, int* shapeVector_, int Ncx_, int Ncy_, int Ncz_, T delta_)
        : flag_shape(flag_shape_), shapeVector(shapeVector_),
        Ncx(Ncx_), Ncy(Ncy_), Ncz(Ncz_), delta(delta_) {
    }

    __host__ __device__ T operator()(int idx) const;
};

// ============================================================================
// INITIAL POLARIZATION FUNCTORS
// ============================================================================

/**
 * @brief Provides initial polarization field.
 */
template <typename T>
struct initialPolarization {
    Vec3<T>* initialP;   ///< Pointer to initial polarization values.
    int initialPVectorSize;
    int gridSize;
    int Ncz;

    __host__ __device__
        initialPolarization(Vec3<T>* initialP_, int initialPVectorSize_, int gridSize_, int Ncz_)
        : initialP(initialP_), initialPVectorSize(initialPVectorSize_), gridSize(gridSize_), Ncz(Ncz_) {
    }

    __host__ __device__ Vec3<T> operator()(int idx) const;
};

/**
 * @brief Combines shape mask and spontaneous polarization to set initial polarization.
 */
template <typename T>
struct initialPol {
    Vec3<T>* initialP;
    int* shape;
    Vec3<T>* spontenousP;
    int gridSize;
    int Ncz;

    __host__ __device__
        initialPol(Vec3<T>* initialP_, int* shape_, Vec3<T>* spontenousP_, int gridSize_, int Ncz_)
        : initialP(initialP_), shape(shape_), spontenousP(spontenousP_),
        gridSize(gridSize_), Ncz(Ncz_) {
    }

    __host__ __device__ Vec3<T> operator()(int idx) const;
};

// ============================================================================
// ALPHA COEFFICIENT FUNCTORS
// ============================================================================

/**
 * @brief Multiplies α₁ coefficient by temperature-dependent terms.
 */
template <typename T>
struct multiplyAlpha1 {
    T* alpha1;
    T* temperature_val;
    T* transitionTemp_val;
    int alpha1VecSize, tempVectorSize, TransitionTempVectorSize;
    int gridSize, Ncz;

    __host__ __device__
        multiplyAlpha1(T* alpha1_, T* temperature_val_, T* transitionTemp_val_,
            int alpha1VecSize_, int tempVectorSize_,
            int TransitionTempVectorSize_, int gridSize_, int Ncz_)
        : alpha1(alpha1_), temperature_val(temperature_val_),
        transitionTemp_val(transitionTemp_val_),
        alpha1VecSize(alpha1VecSize_), tempVectorSize(tempVectorSize_),
        TransitionTempVectorSize(TransitionTempVectorSize_),
        gridSize(gridSize_), Ncz(Ncz_) {
    }

    __host__ __device__ T operator()(int idx) const;
};

/**
 * @brief Computes variance term based on spontaneous polarization and thermal parameters.
 */
template <typename T>
struct varianceCal {
    Vec3<T>* spontenousP;
    T* L_val_ref;
    T* alpha1_ref;
    T deltaX, deltaY, deltaZ;
    T dTime;
    T kb_val;
    T* temperature_val;
    int temperatureVecSize;
    int gridSize, Ncz;

    __host__ __device__
        varianceCal(Vec3<T>* spontenousP_, T* L_val_ref_, T* alpha1_ref_,
            T deltaX_, T deltaY_, T deltaZ_, T dTime_, T kb_val_,
            T* temperature_val_, int temperatureVecSize_, int gridSize_, int Ncz_)
        : spontenousP(spontenousP_), L_val_ref(L_val_ref_), alpha1_ref(alpha1_ref_),
        deltaX(deltaX_), deltaY(deltaY_), deltaZ(deltaZ_),
        dTime(dTime_), kb_val(kb_val_), temperature_val(temperature_val_),
        temperatureVecSize(temperatureVecSize_), gridSize(gridSize_), Ncz(Ncz_) {
    }

    __host__ __device__ T operator()(int idx) const;
};

// Prime versions for α₁–α₆ coefficients (renormalized or effective parameters)
template <typename T>
struct alphaOnePrime {
    T* alpha1;
    T* alpha1_ref;
    __host__ __device__ alphaOnePrime(T* alpha1_, T* alpha1_ref_) : alpha1(alpha1_), alpha1_ref(alpha1_ref_) {}
    __host__ __device__ T operator()(int idx) const;
};

template <typename T>
struct alphaTwoPrime {
    T* alpha1_ref;
    T* alphaTwo;
    Vec3<T>* spontaneousPol;
    int alphaTwoVecSize, gridSize, Ncz;

    __host__ __device__
        alphaTwoPrime(T* alpha1_ref_, T* alphaTwo_, Vec3<T>* spontaneousPol_,
            int alphaTwoVecSize_, int gridSize_, int Ncz_)
        : alpha1_ref(alpha1_ref_), alphaTwo(alphaTwo_),
        spontaneousPol(spontaneousPol_), alphaTwoVecSize(alphaTwoVecSize_),
        gridSize(gridSize_), Ncz(Ncz_) {
    }

    __host__ __device__ T operator()(int idx) const;
};

// ... (α₃–α₆ follow same pattern, kept consistent)
template <typename T>
struct alphaThreePrime {
    T* alpha1_ref; T* alphaThree; Vec3<T>* spontaneousPol;
    int alphaThreeVecSize, gridSize, Ncz;
    __host__ __device__
        alphaThreePrime(T* a1r, T* a3, Vec3<T>* sp, int sz, int g, int n)
        : alpha1_ref(a1r), alphaThree(a3), spontaneousPol(sp),
        alphaThreeVecSize(sz), gridSize(g), Ncz(n) {
    }
    __host__ __device__ T operator()(int idx) const;
};

template <typename T>
struct alphaFourPrime {
    T* alpha1_ref; T* alphaFour; Vec3<T>* spontaneousPol;
    int alphaFourVecSize, gridSize, Ncz;
    __host__ __device__
        alphaFourPrime(T* a1r, T* a4, Vec3<T>* sp, int sz, int g, int n)
        : alpha1_ref(a1r), alphaFour(a4), spontaneousPol(sp),
        alphaFourVecSize(sz), gridSize(g), Ncz(n) {
    }
    __host__ __device__ T operator()(int idx) const;
};

template <typename T>
struct alphaFivePrime {
    T* alpha1_ref; T* alphaFive; Vec3<T>* spontaneousPol;
    int alphaFiveVecSize, gridSize, Ncz;
    __host__ __device__
        alphaFivePrime(T* a1r, T* a5, Vec3<T>* sp, int sz, int g, int n)
        : alpha1_ref(a1r), alphaFive(a5), spontaneousPol(sp),
        alphaFiveVecSize(sz), gridSize(g), Ncz(n) {
    }
    __host__ __device__ T operator()(int idx) const;
};

template <typename T>
struct alphaSixPrime {
    T* alpha1_ref; T* alphaSix; Vec3<T>* spontaneousPol;
    int alphaSixVecSize, gridSize, Ncz;
    __host__ __device__
        alphaSixPrime(T* a1r, T* a6, Vec3<T>* sp, int sz, int g, int n)
        : alpha1_ref(a1r), alphaSix(a6), spontaneousPol(sp),
        alphaSixVecSize(sz), gridSize(g), Ncz(n) {
    }
    __host__ __device__ T operator()(int idx) const;
};


// Elastic and Electrostrictive Coefficient Structures
// Used in TDGL calculations for ferroelectric materials
// ==============================================

template <typename T>
struct Q11Prime {
    T* alpha1_ref;           // Reference dielectric stiffness coefficient
    T* Q11;                  // Electrostrictive coefficient Q11
    Vec3<T>* spontaneousPol; // Spontaneous polarization vector
    int Q11VecSize;          // Size of Q11 array
    int gridSize;            // Total number of grid points
    int Ncz;                 // Size along z-direction

    __host__ __device__
        Q11Prime(T* alpha1_ref_, T* Q11_, Vec3<T>* spontaneousPol_,
            int Q11VecSize_, int gridSize_, int Ncz_)
        : alpha1_ref(alpha1_ref_), Q11(Q11_),
        spontaneousPol(spontaneousPol_), Q11VecSize(Q11VecSize_),
        gridSize(gridSize_), Ncz(Ncz_) {
    }

    __host__ __device__
        T operator()(int idx) const;
};

template <typename T>
struct Q12Prime {
    T* alpha1_ref;
    T* Q12;                  // Electrostrictive coefficient Q12
    Vec3<T>* spontaneousPol;
    int Q12VecSize;
    int gridSize;
    int Ncz;

    __host__ __device__
        Q12Prime(T* alpha1_ref_, T* Q12_, Vec3<T>* spontaneousPol_,
            int Q12VecSize_, int gridSize_, int Ncz_)
        : alpha1_ref(alpha1_ref_), Q12(Q12_),
        spontaneousPol(spontaneousPol_), Q12VecSize(Q12VecSize_),
        gridSize(gridSize_), Ncz(Ncz_) {
    }

    __host__ __device__
        T operator()(int idx) const;
};

template <typename T>
struct Q44Prime {
    T* alpha1_ref;
    T* Q44;                  // Electrostrictive coefficient Q44
    Vec3<T>* spontaneousPol;
    int Q44VecSize;
    int gridSize;
    int Ncz;

    __host__ __device__
        Q44Prime(T* alpha1_ref_, T* Q44_, Vec3<T>* spontaneousPol_,
            int Q44VecSize_, int gridSize_, int Ncz_)
        : alpha1_ref(alpha1_ref_), Q44(Q44_),
        spontaneousPol(spontaneousPol_), Q44VecSize(Q44VecSize_),
        gridSize(gridSize_), Ncz(Ncz_) {
    }

    __host__ __device__
        T operator()(int idx) const;
};

// ----------------------------
// Elastic constants (C_ij)
// ----------------------------

template <typename T>
struct C11Prime {
    T* alpha1_ref;
    T* C11;                  // Elastic stiffness constant C11
    Vec3<T>* spontaneousPol;
    int C11VecSize;
    int gridSize;
    int Ncz;

    __host__ __device__
        C11Prime(T* alpha1_ref_, T* C11_, Vec3<T>* spontaneousPol_,
            int C11VecSize_, int gridSize_, int Ncz_)
        : alpha1_ref(alpha1_ref_), C11(C11_),
        spontaneousPol(spontaneousPol_), C11VecSize(C11VecSize_),
        gridSize(gridSize_), Ncz(Ncz_) {
    }

    __host__ __device__
        T operator()(int idx) const;
};

template <typename T>
struct C12Prime {
    T* alpha1_ref;
    T* C12;                  // Elastic stiffness constant C12
    Vec3<T>* spontaneousPol;
    int C12VecSize;
    int gridSize;
    int Ncz;

    __host__ __device__
        C12Prime(T* alpha1_ref_, T* C12_, Vec3<T>* spontaneousPol_,
            int C12VecSize_, int gridSize_, int Ncz_)
        : alpha1_ref(alpha1_ref_), C12(C12_),
        spontaneousPol(spontaneousPol_), C12VecSize(C12VecSize_),
        gridSize(gridSize_), Ncz(Ncz_) {
    }

    __host__ __device__
        T operator()(int idx) const;
};

template <typename T>
struct C44Prime {
    T* alpha1_ref;
    T* C44;                  // Elastic stiffness constant C44
    Vec3<T>* spontaneousPol;
    int C44VecSize;
    int gridSize;
    int Ncz;

    __host__ __device__
        C44Prime(T* alpha1_ref_, T* C44_, Vec3<T>* spontaneousPol_,
            int C44VecSize_, int gridSize_, int Ncz_)
        : alpha1_ref(alpha1_ref_), C44(C44_),
        spontaneousPol(spontaneousPol_), C44VecSize(C44VecSize_),
        gridSize(gridSize_), Ncz(Ncz_) {
    }

    __host__ __device__
        T operator()(int idx) const;
};



// ============================================================================
// GRADIENT ENERGY COEFFICIENT FUNCTORS
// ============================================================================

template <typename T>
struct G1Prime {
    T* G0; T* G1; int G1VecSize, gridSize, Ncz;
    __host__ __device__ G1Prime(T* g0, T* g1, int sz, int g, int n)
        : G0(g0), G1(g1), G1VecSize(sz), gridSize(g), Ncz(n) {
    }
    __host__ __device__ T operator()(int idx) const;
};

template <typename T>
struct G2Prime {
    T* G0; T* G2; int G2VecSize, gridSize, Ncz;
    __host__ __device__ G2Prime(T* g0, T* g2, int sz, int g, int n)
        : G0(g0), G2(g2), G2VecSize(sz), gridSize(g), Ncz(n) {
    }
    __host__ __device__ T operator()(int idx) const;
};

template <typename T>
struct G3Prime {
    T* G0; T* G3; int G3VecSize, gridSize, Ncz;
    __host__ __device__ G3Prime(T* g0, T* g3, int sz, int g, int n)
        : G0(g0), G3(g3), G3VecSize(sz), gridSize(g), Ncz(n) {
    }
    __host__ __device__ T operator()(int idx) const;
};

template <typename T>
struct G4Prime {
    T* G0; T* G4; int G4VecSize, gridSize, Ncz;
    __host__ __device__ G4Prime(T* g0, T* g4, int sz, int g, int n)
        : G0(g0), G4(g4), G4VecSize(sz), gridSize(g), Ncz(n) {
    }
    __host__ __device__ T operator()(int idx) const;
};

// ============================================================================
// EXTERNAL FIELD FUNCTORS (ZEEMAN TERMS)
// ============================================================================

/**
 * @brief Computes the interaction between spontaneous polarization
 *        and external electric or magnetic field.
 */
template <typename T>
struct ExternalField {
    int coordinate_flag;     ///< 1 = Cartesian, 2 = Spherical
    T* alpha1_ref;           ///< Magnetization saturation array.
    Vec3<T>* spontaneousP;   ///< Spontaneous polarization vector.
    Vec3<T>* extF;           ///< External field vector.
    int extFVecSize, Ncz, gridSize;

    __host__ __device__
        ExternalField(int coordinate_flag_, T* alpha1_ref_, Vec3<T>* spontaneousP_,
            Vec3<T>* extF_, int extFVecSize_, int Ncz_, int gridSize_)
        : coordinate_flag(coordinate_flag_), alpha1_ref(alpha1_ref_),
        spontaneousP(spontaneousP_), extF(extF_), extFVecSize(extFVecSize_),
        Ncz(Ncz_), gridSize(gridSize_) {
    }

    __host__ __device__ Vec3<T> operator()(int idx) const;
};

/**
 * @brief Computes AC field contribution to spontaneous polarization dynamics.
 */
template <typename T>
struct ExternalFieldAc {
    T* alpha1_ref;
    Vec3<T>* spontaneousP;
    Vec3<T>* extFAc;
    int extFAcVecSize, Ncz, gridSize;

    __host__ __device__
        ExternalFieldAc(T* alpha1_ref_, Vec3<T>* spontaneousP_,
            Vec3<T>* extFAc_, int extFAcVecSize_, int Ncz_, int gridSize_)
        : alpha1_ref(alpha1_ref_), spontaneousP(spontaneousP_),
        extFAc(extFAc_), extFAcVecSize(extFAcVecSize_),
        Ncz(Ncz_), gridSize(gridSize_) {
    }

    __host__ __device__ Vec3<T> operator()(int idx) const;
};

// ============================================================================
// VECTOR OPERATIONS
// ============================================================================

template <typename T>
struct SpontaneousPMag {
    Vec3<T>* spontaneousP;
    __host__ __device__ explicit SpontaneousPMag(Vec3<T>* sp) : spontaneousP(sp) {}
    __host__ __device__ T operator()(int idx) const;
};

template <typename T>
struct addVectors {
    Vec3<T>* vec1; Vec3<T>* vec2;
    __host__ __device__ addVectors(Vec3<T>* v1, Vec3<T>* v2) : vec1(v1), vec2(v2) {}
    __host__ __device__ Vec3<T> operator()(int idx) const;
};

template <typename T>
struct multiplyVectors {
    Vec3<T>* vec1; Vec3<T>* vec2;
    __host__ __device__ multiplyVectors(Vec3<T>* v1, Vec3<T>* v2) : vec1(v1), vec2(v2) {}
    __host__ __device__ T operator()(int idx) const;
};

template <typename T>
struct scalarMultiplication {
    T value; Vec3<T>* vec;
    __host__ __device__ scalarMultiplication(T value_, Vec3<T>* vec_) : value(value_), vec(vec_) {}
    __host__ __device__ Vec3<T> operator()(int idx) const;
};

// Declaration of Normalize functor
template<typename T>
struct getMagnitude {
    __host__ __device__ T operator()(const Vec3<T>& m) const;
};

// Declaration of Normalize functor
template<typename T>
struct Normalize {
    __host__ __device__ Vec3<T> operator()(const Vec3<T>& m) const;
};

// Declaration of RandomInit functor
template<typename T>
struct RandomInit {
    __host__ __device__ Vec3<T> operator()(unsigned int thread_id) const;
};

// ============================================================================
// COEFFICIENT VECTOR IMPLEMENTATION INCLUDE
// ============================================================================
#include "coefficientVector.cu"
