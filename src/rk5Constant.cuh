/**
 * @file rk5Constant.cuh
 * @brief Implements a fixed-step 5th-order Runge–Kutta (RK5) integration method.
 *
 * This function is used to initialize the Adams–Bashforth–Moulton (AB3M2) integrator
 * by providing accurate initial time steps. It evolves the polarization field
 * through six RK stages, leveraging Thrust for parallel vector transformations.
 */

#ifndef RK5CONSTANT_CUH
#define RK5CONSTANT_CUH

#include "rk5Constant.h"
#include <thrust/functional.h>

 //==============================================================================
 //! @brief Fixed-step 5th-order Runge–Kutta (RK5) time integration scheme.
 //!
 //! This integrator advances the polarization field by one full time step using
 //! six intermediate slope evaluations (k1–k6). The method is explicit and fixed-step.
 //!
 //! @tparam T Numeric precision type (float or double).
 //! @param polarization Device vector of polarization field values (updated in place).
 //! @param hFerro Device vector of effective fields.
 //! @param noiseVector Device vector of thermal noise values.
 //! @param gridSize Total number of simulation grid points.
 //! @param Ncz Number of cells along the z-direction.
 //! @param currentTime Current simulation time.
 //! @param stepCounting Global time-step index (used for stochastic noise indexing).
 //==============================================================================
template <typename T>
void rk5FixedStep(
    thrust::device_vector<Vec3<T>>& polarization, //!< Polarization vector (output)
    thrust::device_vector<Vec3<T>>& hFerro,       //!< Effective field (input/output)
    thrust::device_vector<Vec3<T>>& noiseVector,  //!< Random noise vector
    int gridSize,                                 //!< Grid size (number of sites)
    int Ncz,                                      //!< Cells along z-axis
    T currentTime,                                //!< Current simulation time
    int stepCounting                              //!< Step counter for noise seeding
)
{
    //----------------------------------------------------------------------
    // Allocate intermediate Runge–Kutta vectors
    //----------------------------------------------------------------------
    thrust::device_vector<Vec3<T>> k1(gridSize, Vec3<T>(0.0f, 0.0f, 0.0f));
    thrust::device_vector<Vec3<T>> k2(gridSize, Vec3<T>(0.0f, 0.0f, 0.0f));
    thrust::device_vector<Vec3<T>> k3(gridSize, Vec3<T>(0.0f, 0.0f, 0.0f));
    thrust::device_vector<Vec3<T>> k4(gridSize, Vec3<T>(0.0f, 0.0f, 0.0f));
    thrust::device_vector<Vec3<T>> k5(gridSize, Vec3<T>(0.0f, 0.0f, 0.0f));
    thrust::device_vector<Vec3<T>> k6(gridSize, Vec3<T>(0.0f, 0.0f, 0.0f));

    thrust::device_vector<Vec3<T>> y(gridSize, Vec3<T>(0.0f, 0.0f, 0.0f));        //!< Intermediate state vector
    thrust::device_vector<Vec3<T>> finalY(gridSize, Vec3<T>(0.0f, 0.0f, 0.0f));   //!< Final updated polarization
    thrust::device_vector<Vec3<T>> primeY(gridSize, Vec3<T>(0.0f, 0.0f, 0.0f));   //!< (Reserved for future extension)
    thrust::device_vector<Vec3<T>> delta(gridSize);                                //!< Difference vector (unused here)
    thrust::device_vector<T> magnitude_delta(gridSize);                            //!< Magnitude (for diagnostics)

    //----------------------------------------------------------------------
    // Reset effective field before integration step
    //----------------------------------------------------------------------
    thrust::fill(hFerro.begin(), hFerro.end(), Vec3<T>());

    //----------------------------------------------------------------------
    // Stage 1: Compute k1
    //----------------------------------------------------------------------
    k1 = calculate_tdgl<T>(
        polarization, hFerro, noiseVector, gridSize, Ncz,
        currentTime + aa1, stepCounting
    );

    //----------------------------------------------------------------------
    // Stage 2: Compute y1 and k2
    //----------------------------------------------------------------------
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(gridSize)),
        y.begin(),
        y_2<T>(
            thrust::raw_pointer_cast(polarization.data()),
            thrust::raw_pointer_cast(k1.data()),
            FE_geom.dtime
        )
    );

    thrust::fill(hFerro.begin(), hFerro.end(), Vec3<T>(0.0f, 0.0f, 0.0f));
    k2 = calculate_tdgl<T>(
        y, hFerro, noiseVector, gridSize, Ncz,
        currentTime + aa2, stepCounting
    );

    //----------------------------------------------------------------------
    // Stage 3: Compute y2 and k3
    //----------------------------------------------------------------------
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(gridSize)),
        y.begin(),
        y_3<T>(
            thrust::raw_pointer_cast(polarization.data()),
            thrust::raw_pointer_cast(k1.data()),
            thrust::raw_pointer_cast(k2.data()),
            FE_geom.dtime
        )
    );

    thrust::fill(hFerro.begin(), hFerro.end(), Vec3<T>());
    k3 = calculate_tdgl<T>(
        y, hFerro, noiseVector, gridSize, Ncz,
        currentTime + aa3, stepCounting
    );

    //----------------------------------------------------------------------
    // Stage 4: Compute y3 and k4
    //----------------------------------------------------------------------
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(gridSize)),
        y.begin(),
        y_4<T>(
            thrust::raw_pointer_cast(polarization.data()),
            thrust::raw_pointer_cast(k1.data()),
            thrust::raw_pointer_cast(k2.data()),
            thrust::raw_pointer_cast(k3.data()),
            FE_geom.dtime
        )
    );

    thrust::fill(hFerro.begin(), hFerro.end(), Vec3<T>());
    k4 = calculate_tdgl<T>(
        y, hFerro, noiseVector, gridSize, Ncz,
        currentTime + aa4, stepCounting
    );

    //----------------------------------------------------------------------
    // Stage 5: Compute y4 and k5
    //----------------------------------------------------------------------
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(gridSize)),
        y.begin(),
        y_5<T>(
            thrust::raw_pointer_cast(polarization.data()),
            thrust::raw_pointer_cast(k1.data()),
            thrust::raw_pointer_cast(k2.data()),
            thrust::raw_pointer_cast(k3.data()),
            thrust::raw_pointer_cast(k4.data()),
            FE_geom.dtime
        )
    );

    thrust::fill(hFerro.begin(), hFerro.end(), Vec3<T>());
    k5 = calculate_tdgl<T>(
        y, hFerro, noiseVector, gridSize, Ncz,
        currentTime + aa5, stepCounting
    );

    //----------------------------------------------------------------------
    // Stage 6: Compute y5 and k6
    //----------------------------------------------------------------------
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(gridSize)),
        y.begin(),
        y_6<T>(
            thrust::raw_pointer_cast(polarization.data()),
            thrust::raw_pointer_cast(k1.data()),
            thrust::raw_pointer_cast(k2.data()),
            thrust::raw_pointer_cast(k3.data()),
            thrust::raw_pointer_cast(k4.data()),
            thrust::raw_pointer_cast(k5.data()),
            FE_geom.dtime
        )
    );

    thrust::fill(hFerro.begin(), hFerro.end(), Vec3<T>());
    k6 = calculate_tdgl<T>(
        y, hFerro, noiseVector, gridSize, Ncz,
        currentTime + aa6, stepCounting
    );

    //----------------------------------------------------------------------
    // Final stage: Combine all slopes to compute new polarization
    //----------------------------------------------------------------------
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(gridSize)),
        finalY.begin(),
        final_y<T>(
            thrust::raw_pointer_cast(polarization.data()),
            thrust::raw_pointer_cast(k1.data()),
            thrust::raw_pointer_cast(k2.data()),
            thrust::raw_pointer_cast(k3.data()),
            thrust::raw_pointer_cast(k4.data()),
            thrust::raw_pointer_cast(k5.data()),
            thrust::raw_pointer_cast(k6.data()),
            FE_geom.dtime
        )
    );

    //----------------------------------------------------------------------
    // Update polarization with the computed final state
    //----------------------------------------------------------------------
    polarization = finalY;
}

#endif // RK5CONSTANT_CUH
