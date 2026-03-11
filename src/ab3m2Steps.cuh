#pragma once
#ifndef AB3M2STEPS_CUH
#define AB3M2STEPS_CUH

#include "Vec3.h"
#include "ab3m2Steps.h"

/**
 * @file ab3m2Steps.cuh
 * @brief Implements predictor, corrector, and Lagrange interpolation kernels
 *        for the Adams–Bashforth–Moulton 3rd-order (AB3M2) integration scheme.
 *
 * This file defines three CUDA-friendly structures—`predictor`, `corrector`,
 * and `lagrange`—each templated for generic type `T`. All are equipped with
 * host/device callable operators and designed for element-wise GPU execution
 * within parallel kernels.
 */

 // ============================================================================
 // PREDICTOR STRUCT IMPLEMENTATION
 // ============================================================================

 /**
  * @brief Constructs the predictor functor.
  * @tparam T Numerical type (e.g., float or double).
  * @param y_   Pointer to current state vector y(tₙ)
  * @param k_3_ Pointer to derivative at tₙ₋₃
  * @param k_2_ Pointer to derivative at tₙ₋₂
  * @param k_1_ Pointer to derivative at tₙ₋₁
  * @param k_   Pointer to derivative at tₙ
  * @param dt_  Time step Δt
  */
template <typename T>
__host__ __device__ predictor<T>::predictor(
    Vec3<T>* y_, Vec3<T>* k_3_, Vec3<T>* k_2_, Vec3<T>* k_1_, Vec3<T>* k_, T dt_)
    : y(y_), k_3(k_3_), k_2(k_2_), k_1(k_1_), k(k_), dt(dt_) {
}

/**
 * @brief Performs the AB3 predictor step at index `idx`.
 *
 * Uses the 4-point Adams–Bashforth scheme:
 * \f[
 * y_{n+1}^p = y_n + \frac{Δt}{24}(-9k_{n-3} + 37k_{n-2} - 59k_{n-1} + 55k_n)
 * \f]
 */
template <typename T>
__host__ __device__ Vec3<T> predictor<T>::operator()(int idx) const {
    Vec3<T> y_val = y[idx];
    Vec3<T> yk_3 = k_3[idx];
    Vec3<T> yk_2 = k_2[idx];
    Vec3<T> yk_1 = k_1[idx];
    Vec3<T> yk = k[idx];

    Vec3<T> pk;
    pk.x = y_val.x + (dt / static_cast<T>(24)) * (-9 * yk_3.x + 37 * yk_2.x - 59 * yk_1.x + 55 * yk.x);
    pk.y = y_val.y + (dt / static_cast<T>(24)) * (-9 * yk_3.y + 37 * yk_2.y - 59 * yk_1.y + 55 * yk.y);
    pk.z = y_val.z + (dt / static_cast<T>(24)) * (-9 * yk_3.z + 37 * yk_2.z - 59 * yk_1.z + 55 * yk.z);

    return pk;
}

// ============================================================================
// CORRECTOR STRUCT IMPLEMENTATION
// ============================================================================

/**
 * @brief Constructs the corrector functor.
 * @tparam T Numerical type (e.g., float or double).
 * @param y_   Pointer to current state vector y(tₙ)
 * @param k_2_ Pointer to derivative at tₙ₋₂
 * @param k_1_ Pointer to derivative at tₙ₋₁
 * @param k_   Pointer to derivative at tₙ
 * @param k1_  Pointer to derivative at tₙ₊₁ (predicted)
 * @param dt_  Time step Δt
 */
template <typename T>
__host__ __device__ corrector<T>::corrector(
    Vec3<T>* y_, Vec3<T>* k_2_, Vec3<T>* k_1_, Vec3<T>* k_, Vec3<T>* k1_, T dt_)
    : y(y_), k_2(k_2_), k_1(k_1_), k(k_), k1(k1_), dt(dt_) {
}

/**
 * @brief Performs the AB3–M2 corrector step at index `idx`.
 *
 * Uses the 4-point Adams–Moulton (implicit) corrector:
 * \f[
 * y_{n+1} = y_n + \frac{Δt}{24}(k_{n-2} - 5k_{n-1} + 19k_n + 9k_{n+1})
 * \f]
 */
template <typename T>
__host__ __device__ Vec3<T> corrector<T>::operator()(int idx) const {
    Vec3<T> y_val = y[idx];
    Vec3<T> yk_2 = k_2[idx];
    Vec3<T> yk_1 = k_1[idx];
    Vec3<T> yk = k[idx];
    Vec3<T> yk1 = k1[idx];

    Vec3<T> y_next;
    y_next.x = y_val.x + (dt / static_cast<T>(24)) * (yk_2.x - 5 * yk_1.x + 19 * yk.x + 9 * yk1.x);
    y_next.y = y_val.y + (dt / static_cast<T>(24)) * (yk_2.y - 5 * yk_1.y + 19 * yk.y + 9 * yk1.y);
    y_next.z = y_val.z + (dt / static_cast<T>(24)) * (yk_2.z - 5 * yk_1.z + 19 * yk.z + 9 * yk1.z);

    return y_next;
}

// ============================================================================
// LAGRANGE INTERPOLATION STRUCT IMPLEMENTATION
// ============================================================================

/**
 * @brief Constructs the Lagrange interpolation functor.
 *
 * Used for reconstructing intermediate derivatives from 5 known points
 * in a time-centered stencil.
 */
template<typename T>
__host__ __device__ lagrange<T>::lagrange(
    Vec3<T>* k_3_, Vec3<T>* k_2_, Vec3<T>* k_1_, Vec3<T>* k_0_, Vec3<T>* k1_,
    T t_3_, T t_2_, T t_1_, T t_0_, T t1_, T tNew_)
    : k_3(k_3_), k_2(k_2_), k_1(k_1_), k_0(k_0_), k1(k1_),
    t_3(t_3_), t_2(t_2_), t_1(t_1_), t_0(t_0_), t1(t1_), tNew(tNew_) {
}

/**
 * @brief Evaluates the 4th-order Lagrange interpolation polynomial for vector field k(t).
 *
 * Each component (x, y, z) is interpolated independently using the standard
 * Lagrange basis coefficients over five known time points:
 * \f[
 * k(t_{new}) = \sum_{i=-3}^{+1} k_i L_i(t_{new})
 * \f]
 */
template <typename T>
__host__ __device__ Vec3<T> lagrange<T>::operator()(int idx) const {
    // Retrieve stored derivative values
    Vec3<T> fk_3 = k_3[idx];
    Vec3<T> fk_2 = k_2[idx];
    Vec3<T> fk_1 = k_1[idx];
    Vec3<T> fk_0 = k_0[idx];
    Vec3<T> fk1 = k1[idx];

    // Compute Lagrange coefficients (weights)
    T l_3 = ((tNew - t_2) * (tNew - t_1) * (tNew - t_0) * (tNew - t1)) /
        ((t_3 - t_2) * (t_3 - t_1) * (t_3 - t_0) * (t_3 - t1));

    T l_2 = ((tNew - t_3) * (tNew - t_1) * (tNew - t_0) * (tNew - t1)) /
        ((t_2 - t_3) * (t_2 - t_1) * (t_2 - t_0) * (t_3 - t1));

    T l_1 = ((tNew - t_3) * (tNew - t_2) * (tNew - t_0) * (tNew - t1)) /
        ((t_1 - t_3) * (t_1 - t_2) * (t_1 - t_0) * (t_3 - t1));

    T l_0 = ((tNew - t_3) * (tNew - t_2) * (tNew - t_1) * (tNew - t1)) /
        ((t_0 - t_3) * (t_0 - t_2) * (t_0 - t_1) * (t_3 - t1));

    T l1 = ((tNew - t_3) * (tNew - t_2) * (tNew - t_1) * (tNew - t_0)) /
        ((t1 - t_3) * (t1 - t_2) * (t1 - t_1) * (t1 - t_0));

    // Weighted sum of derivative vectors
    Vec3<T> fki;
    fki.x = fk_3.x * l_3 + fk_2.x * l_2 + fk_1.x * l_1 + fk_0.x * l_0 + fk1.x * l1;
    fki.y = fk_3.y * l_3 + fk_2.y * l_2 + fk_1.y * l_1 + fk_0.y * l_0 + fk1.y * l1;
    fki.z = fk_3.z * l_3 + fk_2.z * l_2 + fk_1.z * l_1 + fk_0.z * l_0 + fk1.z * l1;

    return fki;
}

#endif // AB3M2STEPS_CUH
