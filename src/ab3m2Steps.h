#pragma once
#ifndef AB3M2STEPS_H
#define AB3M2STEPS_H

/**
 * @file ab3m2Steps.h
 * @brief Declares step functors for the Adams–Bashforth–Moulton 3rd–2nd order (AB3M2) integrator.
 *
 * This header defines callable CUDA functors (`predictor`, `corrector`, and `lagrange`)
 * used in the AB3M2 integration scheme. Each functor operates element-wise over
 * `Vec3<T>` polarization vectors using Thrust parallel algorithms.
 *
 * The predictor computes a provisional state using a 3rd-order Adams–Bashforth step.
 * The corrector refines that prediction with a 2nd-order Adams–Moulton step.
 * The Lagrange functor performs temporal interpolation between previous derivative
 * evaluations for smooth initialization or adaptive corrections.
 *
 * Dependencies:
 *  - Vec3.h : Custom 3D vector class
 *  - thrust/execution_policy.h : Required for Thrust execution contexts
 */

#include "Vec3.h"
#include <thrust/execution_policy.h>

 // ============================================================================
 // Predictor Functor
 // ============================================================================

 /**
  * @struct predictor
  * @brief Computes the predicted next-state vector using the 3rd-order
  *        Adams–Bashforth explicit scheme.
  *
  * The Adams–Bashforth predictor formula is: 
  */
template <typename T>
struct predictor {
    Vec3<T>* y;     ///< Current state vector at step n
    Vec3<T>* k_3;   ///< Derivative term from step n−3
    Vec3<T>* k_2;   ///< Derivative term from step n−2
    Vec3<T>* k_1;   ///< Derivative term from step n−1
    Vec3<T>* k;     ///< Derivative term from current step n
    T dt;           ///< Integration time step

    /**
     * @brief Constructor to initialize predictor functor.
     */
    __host__ __device__
        predictor(Vec3<T>* y_, Vec3<T>* k_3_, Vec3<T>* k_2_, Vec3<T>* k_1_, Vec3<T>* k_, T dt_);

    /**
     * @brief Operator to perform per-element prediction.
     * @param idx Linear index of the grid element
     * @return Predicted `Vec3<T>` value for the next time step
     */
    __host__ __device__
        Vec3<T> operator()(int idx) const;
};

// ============================================================================
// Corrector Functor
// ============================================================================

/**
 * @struct corrector
 * @brief Refines the predicted state using the 2nd-order
 *        Adams–Moulton implicit corrector.
 *
 * The Adams–Moulton corrector formula is:
 * \f[
 *     y_{n+1} = y_n + \frac{\Delta t}{12} (5k_{n+1} + 8k_n - k_{n-1})
 * \f]
 *
 * @tparam T Floating-point precision type (float or double)
 */
template <typename T>
struct corrector {
    Vec3<T>* y;     ///< Current state vector at step n
    Vec3<T>* k_2;   ///< Derivative term from step n−2
    Vec3<T>* k_1;   ///< Derivative term from step n−1
    Vec3<T>* k;     ///< Derivative term from current step n
    Vec3<T>* k1;    ///< Derivative term from predicted step n+1
    T dt;           ///< Integration time step

    /**
     * @brief Constructor to initialize corrector functor.
     */
    __host__ __device__
        corrector(Vec3<T>* y_, Vec3<T>* k_2_, Vec3<T>* k_1_, Vec3<T>* k_, Vec3<T>* k1_, T dt_);

    /**
     * @brief Operator to perform per-element correction.
     * @param idx Linear index of the grid element
     * @return Corrected `Vec3<T>` value after implicit update
     */
    __host__ __device__
        Vec3<T> operator()(int idx) const;
};

// ============================================================================
// Lagrange Interpolator Functor
// ============================================================================

/**
 * @struct lagrange
 * @brief Performs temporal interpolation using the 5-point Lagrange polynomial.
 *
 * This functor is useful for estimating missing derivative values during
 * initialization of multistep schemes (e.g., computing `k_3` or `k_2`
 * from early Runge–Kutta steps).
 *
 * The general Lagrange interpolation is:
 * \f[
 *     f(t_{new}) = \sum_{i=0}^{4} f(t_i) \prod_{j=0, j\neq i}^{4}
 *     \frac{t_{new} - t_j}{t_i - t_j}
 * \f]
 *
 * @tparam T Floating-point precision type (float or double)
 */
template <typename T>
struct lagrange {
    Vec3<T>* k_3;  ///< Function value at tₙ₋₃
    Vec3<T>* k_2;  ///< Function value at tₙ₋₂
    Vec3<T>* k_1;  ///< Function value at tₙ₋₁
    Vec3<T>* k_0;  ///< Function value at tₙ
    Vec3<T>* k1;   ///< Function value at tₙ₊₁

    T t_3;         ///< Time at tₙ₋₃
    T t_2;         ///< Time at tₙ₋₂
    T t_1;         ///< Time at tₙ₋₁
    T t_0;         ///< Time at tₙ
    T t1;          ///< Time at tₙ₊₁
    T tNew;        ///< New interpolation time

    /**
     * @brief Constructor to initialize interpolation parameters.
     */
    __host__ __device__
        lagrange(Vec3<T>* k_3_, Vec3<T>* k_2_, Vec3<T>* k_1_, Vec3<T>* k_0_, Vec3<T>* k1_,
            T t_3_, T t_2_, T t_1_, T t_0_, T t1_, T tNew_);

    /**
     * @brief Computes the interpolated `Vec3<T>` value at `tNew`.
     * @param idx Linear index of the grid element
     * @return Interpolated vector value
     */
    __host__ __device__
        Vec3<T> operator()(int idx) const;
};

// ============================================================================
// Include Implementation
// ============================================================================

#include "ab3m2Steps.cuh"

#endif // AB3M2STEPS_H
