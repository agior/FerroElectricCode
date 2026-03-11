/**
 * @file rk5Steps.cuh
 * @brief Implements per-stage Runge–Kutta functors for TDGL polarization integration.
 *
 * @details
 * This file defines CUDA functors used in each intermediate stage (y₂, y₃, … y₆)
 * and final state computation (final_y and prime_y) of the 5th-order Runge–Kutta
 * (RK5) or Fehlberg (RK45) method.
 *
 * Each functor computes the temporary intermediate state of the polarization
 * vector `P` given previously calculated derivative vectors `k₁`, `k₂`, …, `k₆`
 * and the time step `dt`. These stages correspond to the Butcher tableau of the
 * RK5/RK45 integration scheme.
 *
 * All functors are compatible with both host and device execution
 * (`__host__ __device__`) and operate on arrays of `Vec3<T>` objects stored
 * on the GPU.
 */

#pragma once
#ifndef RK5STEPS_CUH
#define RK5STEPS_CUH

#include "rk5Steps.h"  // Declarations of Runge–Kutta step functors

 //==============================================================================
 //! @class y_2
 //! @brief Computes the second intermediate state y₂ = y₁ + b21 * dt * k₁.
 //==============================================================================
template <typename T>
__host__ __device__ y_2<T>::y_2(Vec3<T>* d_m_, Vec3<T>* k_1_, T dt_)
    : d_m(d_m_), k_1(k_1_), dt(dt_) {
}

template <typename T>
__host__ __device__ Vec3<T> y_2<T>::operator()(int idx) const {
    Vec3<T> y2 = d_m[idx] + (bb21 * dt) * k_1[idx];
    // y2.Normalize(); // Optional normalization if required by system
    return y2;
}

//==============================================================================
//! @class y_3
//! @brief Computes the third intermediate state y₃ = y₁ + dt * (b31*k₁ + b32*k₂).
//==============================================================================
template <typename T>
__host__ __device__ y_3<T>::y_3(Vec3<T>* d_m_, Vec3<T>* k_1_, Vec3<T>* k_2_, T dt_)
    : d_m(d_m_), k_1(k_1_), k_2(k_2_), dt(dt_) {
}

template <typename T>
__host__ __device__ Vec3<T> y_3<T>::operator()(int idx) const {
    Vec3<T> y3 = d_m[idx]
        + (bb31 * dt) * k_1[idx]
        + (bb32 * dt) * k_2[idx];
    // y3.Normalize();
    return y3;
}

//==============================================================================
//! @class y_4
//! @brief Computes the fourth intermediate state y₄ = y₁ + dt * (b41*k₁ + b42*k₂ + b43*k₃).
//==============================================================================
template <typename T>
__host__ __device__ y_4<T>::y_4(Vec3<T>* d_m_, Vec3<T>* k_1_, Vec3<T>* k_2_, Vec3<T>* k_3_, T dt_)
    : d_m(d_m_), k_1(k_1_), k_2(k_2_), k_3(k_3_), dt(dt_) {
}

template <typename T>
__host__ __device__ Vec3<T> y_4<T>::operator()(int idx) const {
    Vec3<T> y4 = d_m[idx]
        + (bb41 * dt) * k_1[idx]
        + (bb42 * dt) * k_2[idx]
        + (bb43 * dt) * k_3[idx];
    // y4.Normalize();
    return y4;
}

//==============================================================================
//! @class y_5
//! @brief Computes the fifth intermediate state y₅ = y₁ + dt * (b51*k₁ + b52*k₂ + b53*k₃ + b54*k₄).
//==============================================================================
template <typename T>
__host__ __device__ y_5<T>::y_5(Vec3<T>* d_m_, Vec3<T>* k_1_, Vec3<T>* k_2_,
    Vec3<T>* k_3_, Vec3<T>* k_4_, T dt_)
    : d_m(d_m_), k_1(k_1_), k_2(k_2_), k_3(k_3_), k_4(k_4_), dt(dt_) {
}

template <typename T>
__host__ __device__ Vec3<T> y_5<T>::operator()(int idx) const {
    Vec3<T> y5 = d_m[idx]
        + (bb51 * dt) * k_1[idx]
        + (bb52 * dt) * k_2[idx]
        + (bb53 * dt) * k_3[idx]
        + (bb54 * dt) * k_4[idx];
    // y5.Normalize();
    return y5;
}

//==============================================================================
//! @class y_6
//! @brief Computes the sixth intermediate state
//!        y₆ = y₁ + dt * (b61*k₁ + b62*k₂ + b63*k₃ + b64*k₄ + b65*k₅).
//==============================================================================
template <typename T>
__host__ __device__ y_6<T>::y_6(Vec3<T>* d_m_, Vec3<T>* k_1_, Vec3<T>* k_2_,
    Vec3<T>* k_3_, Vec3<T>* k_4_, Vec3<T>* k_5_, T dt_)
    : d_m(d_m_), k_1(k_1_), k_2(k_2_), k_3(k_3_), k_4(k_4_), k_5(k_5_), dt(dt_) {
}

template <typename T>
__host__ __device__ Vec3<T> y_6<T>::operator()(int idx) const {
    Vec3<T> y6 = d_m[idx]
        + (bb61 * dt) * k_1[idx]
        + (bb62 * dt) * k_2[idx]
        + (bb63 * dt) * k_3[idx]
        + (bb64 * dt) * k_4[idx]
        + (bb65 * dt) * k_5[idx];
    // y6.Normalize();
    return y6;
}

//==============================================================================
//! @class final_y
//! @brief Computes the final RK5 integration result using the main coefficients.
//!
//!        y_final = y₁ + dt * Σ(ci * ki), i = 1..6
//==============================================================================
template <typename T>
__host__ __device__ final_y<T>::final_y(Vec3<T>* d_m_, Vec3<T>* k_1_, Vec3<T>* k_2_,
    Vec3<T>* k_3_, Vec3<T>* k_4_, Vec3<T>* k_5_,
    Vec3<T>* k_6_, T dt_)
    : d_m(d_m_), k_1(k_1_), k_2(k_2_), k_3(k_3_), k_4(k_4_),
    k_5(k_5_), k_6(k_6_), dt(dt_) {
}

template <typename T>
__host__ __device__ Vec3<T> final_y<T>::operator()(int idx) const {
    Vec3<T> finalY = d_m[idx]
        + (cc1 * dt) * k_1[idx]
        + (cc2 * dt) * k_2[idx]
        + (cc3 * dt) * k_3[idx]
        + (cc4 * dt) * k_4[idx]
        + (cc5 * dt) * k_5[idx]
        + (cc6 * dt) * k_6[idx];
    // finalY.Normalize();
    return finalY;
}

//==============================================================================
//! @class prime_y
//! @brief Computes the lower-order (4th order) RK estimate used for error control.
//!
//!        y_prime = y₁ + dt * Σ(ci' * ki), i = 1..6
//!
//! This vector is used to estimate the local truncation error between the
//! 5th- and 4th-order Runge–Kutta results, enabling adaptive time stepping.
//==============================================================================
template <typename T>
__host__ __device__ prime_y<T>::prime_y(Vec3<T>* d_m_, Vec3<T>* k_1_, Vec3<T>* k_2_,
    Vec3<T>* k_3_, Vec3<T>* k_4_, Vec3<T>* k_5_,
    Vec3<T>* k_6_, T dt_)
    : d_m(d_m_), k_1(k_1_), k_2(k_2_), k_3(k_3_), k_4(k_4_),
    k_5(k_5_), k_6(k_6_), dt(dt_) {
}

template <typename T>
__host__ __device__ Vec3<T> prime_y<T>::operator()(int idx) const {
    Vec3<T> primeY = d_m[idx]
        + (cc_1 * dt) * k_1[idx]
        + (cc_2 * dt) * k_2[idx]
        + (cc_3 * dt) * k_3[idx]
        + (cc_4 * dt) * k_4[idx]
        + (cc_5 * dt) * k_5[idx]
        + (cc_6 * dt) * k_6[idx];
    // primeY.Normalize();
    return primeY;
}

#endif // RK5STEPS_CUH
