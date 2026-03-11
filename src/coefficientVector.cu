#pragma once
#include "coefficientVector.cuh"
#include "parameters.h"

/**
 * @file coefficientVector.cu
 * @brief Implements templated CUDA-compatible functors for computing field coefficients,
 *        shape masks, and renormalized parameters used in ferroelectric domain simulations.
 *
 * Each operator() is callable on both host and device, supporting Thrust transformations
 * and custom CUDA kernels for data-parallel physics computation.
 */

 // ============================================================================
 // PHASE AND SHAPE OPERATORS
 // ============================================================================

 /**
  * @brief Converts AC phase values from degrees to radians.
  */
template <typename T>
__host__ __device__
Vec3<T> phaseAc<T>::operator()(int idx) const {
    int phase_idx = (phaseVectorSize == 1) ? 0 :
        (phaseVectorSize == Ncz) ? idx % Ncz : idx;

    // Convert phase components from degrees to radians.
    return phase_ac[phase_idx] * (PI / 180.0);
}

/**
 * @brief Evaluates spatial shape mask at the given grid index.
 * @details Supports direct copy or geometric shaping (elliptical mask).
 */
template <typename T>
__host__ __device__
T shape<T>::operator()(int idx) const {
    // Decode 1D index into 3D grid coordinates.
    int z = idx / (Ncx * Ncy);
    int y = (idx / Ncx) % Ncy;
    int x = idx % Ncx;

    // Case 1–2: Return the precomputed mask directly.
    if (flag_shape == 1 || flag_shape == 2)
        return shapeVector[idx];

    // Case 3: Construct an elliptical mask.
    else if (flag_shape == 3) {
        T semiX = (Ncx * delta) / 2;
        T semiY = (Ncy * delta) / 2;

        // Normalized coordinates inside the ellipse.
        T first = ((x + 0.5) * delta - semiX) / semiX;
        T second = ((y + 0.5) * delta - semiY) / semiY;

        // Elliptical condition: (x/a)^2 + (y/b)^2 <= 1
        T shapeMaker = first * first + second * second;

        return shapeMaker <= T(1) ? T(1) : T(0);
    }

    // Default: return zero if undefined.
    return T(0);
}

// ============================================================================
// INITIAL POLARIZATION OPERATORS
// ============================================================================

/**
 * @brief Retrieves the initial polarization vector at the specified grid index.
 */
template <typename T>
__host__ __device__
Vec3<T> initialPolarization<T>::operator()(int idx) const {
    int p_idx = (initialPVectorSize == 1) ? 0 :
        (initialPVectorSize == Ncz) ? idx / (gridSize / Ncz) : idx;

    return Vec3<T>(initialP[p_idx].x, initialP[p_idx].y, initialP[p_idx].z);
}

/**
 * @brief Initializes polarization with shape and spontaneous polarization normalization.
 */
template <typename T>
__host__ __device__
Vec3<T> initialPol<T>::operator()(int idx) const {
    Vec3<T> SP_val;
    T magSP = sqrt(spontenousP[0].x * spontenousP[0].x +
        spontenousP[0].y * spontenousP[0].y +
        spontenousP[0].z * spontenousP[0].z);

    return Vec3<T>(
        initialP[idx].x * shape[idx] / magSP,
        initialP[idx].y * shape[idx] / magSP,
        initialP[idx].z * shape[idx] / magSP
    );
}

// ============================================================================
// ALPHA COEFFICIENT OPERATORS
// ============================================================================

/**
 * @brief Computes α₁ = α₁₀ × (T − T_c).
 */
template <typename T>
__host__ __device__
T multiplyAlpha1<T>::operator()(int idx) const {
    int temp_idx = (tempVectorSize == 1) ? 0 :
        (tempVectorSize == Ncz) ? idx % Ncz : idx;
    int transTemp_idx = (TransitionTempVectorSize == 1) ? 0 :
        (TransitionTempVectorSize == Ncz) ? idx % Ncz : idx;
    int alpha1_idx = (alpha1VecSize == 1) ? 0 :
        (alpha1VecSize == Ncz) ? idx % Ncz : idx;

    return alpha1[alpha1_idx] *
        (temperature_val[temp_idx] - transitionTemp_val[transTemp_idx]);
}

/**
 * @brief Computes stochastic thermal variance scaling.
 */
template <typename T>
__host__ __device__
T varianceCal<T>::operator()(int idx) const {
    int temp_idx = (temperatureVecSize == 1) ? 0 :
        (temperatureVecSize == Ncz) ? idx % Ncz : idx;

    T magSP = sqrt(spontenousP[0].x * spontenousP[0].x +
        spontenousP[0].y * spontenousP[0].y +
        spontenousP[0].z * spontenousP[0].z);

    T value = sqrt(2 * kb_val * (temperature_val[temp_idx])  * L_val_ref[0] /
        (dTime * deltaX * deltaY * deltaZ));

    return (dTime * value) / magSP;
}

/**
 * @brief Normalizes α₁ by its absolute reference.
 */
template <typename T>
__host__ __device__
T alphaOnePrime<T>::operator()(int idx) const {
    T alpha1Abs = abs(alpha1_ref[0]);
    return (alpha1[idx] / alpha1Abs);
}

// Higher-order α' computations (α₂–α₆)
template <typename T>
__host__ __device__
T alphaTwoPrime<T>::operator()(int idx) const {
    int i = (alphaTwoVecSize == 1) ? 0 :
        (alphaTwoVecSize == Ncz) ? idx % Ncz : idx;

    Vec3<T> P = spontaneousPol[0];
    T P2 = P.x * P.x + P.y * P.y + P.z * P.z;
    return (alphaTwo[i] * P2) / abs(alpha1_ref[0]);
}

template <typename T>
__host__ __device__
T alphaThreePrime<T>::operator()(int idx) const {
    int i = (alphaThreeVecSize == 1) ? 0 :
        (alphaThreeVecSize == Ncz) ? idx % Ncz : idx;

    Vec3<T> P = spontaneousPol[0];
    T P2 = P.x * P.x + P.y * P.y + P.z * P.z;
    return (alphaThree[i] * P2) / abs(alpha1_ref[0]);
}

template <typename T>
__host__ __device__
T alphaFourPrime<T>::operator()(int idx) const {
    int i = (alphaFourVecSize == 1) ? 0 :
        (alphaFourVecSize == Ncz) ? idx % Ncz : idx;

    Vec3<T> P = spontaneousPol[0];
    T P2 = P.x * P.x + P.y * P.y + P.z * P.z;
    return (alphaFour[i] * P2 * P2) / abs(alpha1_ref[0]);
}

template <typename T>
__host__ __device__
T alphaFivePrime<T>::operator()(int idx) const {
    int i = (alphaFiveVecSize == 1) ? 0 :
        (alphaFiveVecSize == Ncz) ? idx % Ncz : idx;

    Vec3<T> P = spontaneousPol[0];
    T P2 = P.x * P.x + P.y * P.y + P.z * P.z;
    return (alphaFive[i] * P2 * P2) / abs(alpha1_ref[0]);
}

template <typename T>
__host__ __device__
T alphaSixPrime<T>::operator()(int idx) const {
    int i = (alphaSixVecSize == 1) ? 0 :
        (alphaSixVecSize == Ncz) ? idx % Ncz : idx;

    Vec3<T> P = spontaneousPol[0];
    T P2 = P.x * P.x + P.y * P.y + P.z * P.z;
    return (alphaSix[i] * P2 * P2) / abs(alpha1_ref[0]);
}

// ==============================================
// Electrostrictive coefficient Q11'
// ==============================================
template <typename T>
__host__ __device__
T Q11Prime<T>::operator()(int idx) const {
    // Determine index depending on whether Q11 is uniform, 1D (along z), or full-field
    int i = (Q11VecSize == 1) ? 0 :
        (Q11VecSize == Ncz) ? idx % Ncz : idx;

    // Retrieve spontaneous polarization vector
    Vec3<T> P = spontaneousPol[0];

    // Compute squared polarization magnitude
    T P2 = P.x * P.x + P.y * P.y + P.z * P.z;

    // Return Q11 contribution (field-dependent part)
    return Q11[i] / P2;
}

// ==============================================
// Electrostrictive coefficient Q12'
// ==============================================
template <typename T>
__host__ __device__
T Q12Prime<T>::operator()(int idx) const {
    int i = (Q12VecSize == 1) ? 0 :
        (Q12VecSize == Ncz) ? idx % Ncz : idx;

    Vec3<T> P = spontaneousPol[0];
    T P2 = P.x * P.x + P.y * P.y + P.z * P.z;

    // Return Q12 contribution (cross-coupling term)
    return Q12[i] / P2;
}

// ==============================================
// Electrostrictive coefficient Q44'
// ==============================================
template <typename T>
__host__ __device__
T Q44Prime<T>::operator()(int idx) const {
    int i = (Q44VecSize == 1) ? 0 :
        (Q44VecSize == Ncz) ? idx % Ncz : idx;

    Vec3<T> P = spontaneousPol[0];
    T P2 = P.x * P.x + P.y * P.y + P.z * P.z;

    // Return Q44 contribution (shear coupling term)
    return Q44[i] / P2;
}

// ==============================================
// Elastic stiffness coefficient C11'
// ==============================================
template <typename T>
__host__ __device__
T C11Prime<T>::operator()(int idx) const {
    // Determine index (uniform / z-dependent / full-field)
    int i = (C11VecSize == 1) ? 0 :
        (C11VecSize == Ncz) ? idx % Ncz : idx;

    // Retrieve spontaneous polarization vector
    Vec3<T> P = spontaneousPol[0];

    // Compute polarization magnitude squared
    T P2 = P.x * P.x + P.y * P.y + P.z * P.z;

    // Compute normalized C11
    return C11[i] / (alpha1_ref[0] * P2);
}

// ==============================================
// Elastic stiffness coefficient C12'
// ==============================================
template <typename T>
__host__ __device__
T C12Prime<T>::operator()(int idx) const {
    int i = (C12VecSize == 1) ? 0 :
        (C12VecSize == Ncz) ? idx % Ncz : idx;

    Vec3<T> P = spontaneousPol[0];
    T P2 = P.x * P.x + P.y * P.y + P.z * P.z;

    return C12[i] / (alpha1_ref[0] * P2);
}

// ==============================================
// Elastic stiffness coefficient C44'
// ==============================================
template <typename T>
__host__ __device__
T C44Prime<T>::operator()(int idx) const {
    int i = (C44VecSize == 1) ? 0 :
        (C44VecSize == Ncz) ? idx % Ncz : idx;

    Vec3<T> P = spontaneousPol[0];
    T P2 = P.x * P.x + P.y * P.y + P.z * P.z;

    return C44[i] / (alpha1_ref[0] * P2);
}


// ============================================================================
// GRADIENT ENERGY COEFFICIENT OPERATORS
// ============================================================================

template <typename T>
__host__ __device__ T G1Prime<T>::operator()(int idx) const { return G1[idx] / G0[0]; }

template <typename T>
__host__ __device__ T G2Prime<T>::operator()(int idx) const { return G2[idx] / G0[0]; }

template <typename T>
__host__ __device__ T G3Prime<T>::operator()(int idx) const { return G3[idx] / G0[0]; }

template <typename T>
__host__ __device__ T G4Prime<T>::operator()(int idx) const { return G4[idx] / G0[0]; }

// ============================================================================
// EXTERNAL FIELD OPERATORS (ZEEMAN-LIKE TERMS)
// ============================================================================

/**
 * @brief Computes field coupling depending on coordinate system.
 * @details Converts between spherical and Cartesian coordinates as required.
 */
template <typename T>
__host__ __device__
Vec3<T> ExternalField<T>::operator()(int idx) const {
    int e_idx = (extFVecSize == 1) ? 0 :
        (extFVecSize == Ncz) ? idx % Ncz : idx;

    T magP = sqrt(spontaneousP[0].x * spontaneousP[0].x +
        spontaneousP[0].y * spontaneousP[0].y +
        spontaneousP[0].z * spontaneousP[0].z);

    // Case 1: Cartesian coordinates.
    if (coordinate_flag == 2) {
        return extF[e_idx] / (magP * abs(alpha1_ref[0]));
    }

    // Case 2: Spherical coordinates (convert to Cartesian).
    if (coordinate_flag == 1) {
        Vec3<T> scaled_extF;
        scaled_extF.x = extF[e_idx].x / (magP * abs(alpha1_ref[0]));
        scaled_extF.y = extF[e_idx].y * PI / 180.0;
        scaled_extF.z = extF[e_idx].z * PI / 180.0;

        return Vec3<T>(
            scaled_extF.x * sin(scaled_extF.y) * cos(scaled_extF.z),
            scaled_extF.x * sin(scaled_extF.y) * sin(scaled_extF.z),
            scaled_extF.x * cos(scaled_extF.y)
        );
    }

    return Vec3<T>(0.0, 0.0, 0.0);
}

/**
 * @brief Evaluates AC external field coupling term.
 */
template <typename T>
__host__ __device__
Vec3<T> ExternalFieldAc<T>::operator()(int idx) const {
    int e_idx = (extFAcVecSize == 1) ? 0 :
        (extFAcVecSize == Ncz) ? idx % Ncz : idx;

    T magP = sqrt(spontaneousP[0].x * spontaneousP[0].x +
        spontaneousP[0].y * spontaneousP[0].y +
        spontaneousP[0].z * spontaneousP[0].z);

    return extFAc[e_idx] / (magP * abs(alpha1_ref[0]));
}

// ============================================================================
// VECTOR OPERATIONS
// ============================================================================

/**
 * @brief Computes the magnitude of the spontaneous polarization vector.
 */
template <typename T>
__host__ __device__
T SpontaneousPMag<T>::operator()(int idx) const {
    return sqrt(spontaneousP[0].x * spontaneousP[0].x +
        spontaneousP[0].y * spontaneousP[0].y +
        spontaneousP[0].z * spontaneousP[0].z);
}

/**
 * @brief Adds two 3D vectors elementwise.
 */
template <typename T>
__host__ __device__
Vec3<T> addVectors<T>::operator()(int idx) const {
    vec1[idx].x += vec2[idx].x;
    vec1[idx].y += vec2[idx].y;
    vec1[idx].z += vec2[idx].z;
    return vec1[idx];
}

/**
 * @brief Computes dot product of two 3D vectors.
 */
template <typename T>
__host__ __device__
T multiplyVectors<T>::operator()(int idx) const {
    return vec1[idx].x * vec2[idx].x +
        vec1[idx].y * vec2[idx].y +
        vec1[idx].z * vec2[idx].z;
}

/**
 * @brief Multiplies each component of a vector by a scalar value.
 */
template <typename T>
__host__ __device__
Vec3<T> scalarMultiplication<T>::operator()(int idx) const {
    Vec3<T> v = vec[idx];
    return value * v;
}

// Definition of getMagnitude functor
template<typename T>
__host__ __device__ T getMagnitude<T>::operator()(const Vec3<T>& m) const {
    // Calculate the magnitude of the vector


    // Return a Vec3 where all components are set to the calculated magnitude
    return sqrt(m.x * m.x + m.y * m.y + m.z * m.z);
}

// Definition of Normalize functor
template<typename T>
__host__ __device__ Vec3<T> Normalize<T>::operator()(const Vec3<T>& m) const {
    // Calculate the magnitude of the vector
    T magnitude = sqrt(m.x * m.x + m.y * m.y + m.z * m.z);

    // Normalize the vector if the magnitude is greater than zero
    if (magnitude > 0) {
        return Vec3<T>(m.x / magnitude, m.y / magnitude, m.z / magnitude);
    }
    return Vec3<T>(m.x, m.y, m.z);
}

// Definition of RandomInit functor
template<typename T>
__host__ __device__ Vec3<T> RandomInit<T>::operator()(unsigned int thread_id) const {
    // Use thread_id to create unique seeds for each component
    unsigned int seed_x = thread_id * 2654435761;      // Large prime for X
    unsigned int seed_y = (thread_id + 1) * 16807;     // Different multiplier for Y
    unsigned int seed_z = (thread_id + 2) * 48271;     // Different multiplier for Z

    // Separate RNGs for each axis
    thrust::default_random_engine rng_x(seed_x);
    thrust::default_random_engine rng_y(seed_y);
    thrust::default_random_engine rng_z(seed_z);

    // Uniform distribution for values between -1 and 1
    thrust::uniform_real_distribution<T> dist(-1.0f, 1.0f);

    // Generate random values for each component
    T x = dist(rng_x);
    T y = dist(rng_y);
    T z = dist(rng_z);

    return Vec3<T>(x, y, z);
}
