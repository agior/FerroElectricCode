/**
 * @file ab3m2Integrator.cuh
 * @brief Implements the Adams–Bashforth–Moulton (AB3M2) predictor–corrector integration method
 *        for time-dependent Ginzburg–Landau (TDGL) polarization dynamics.
 *
 * This method combines a 3-step Adams–Bashforth predictor and a 2-step Adams–Moulton corrector.
 * The integrator supports both fixed and adaptive time-stepping modes (based on FE_geom.FLAG_INTEGRATOR),
 * and operates entirely on CUDA device memory via the Thrust library.
 */

#pragma once
#ifndef AB3M3INTEGRATOR_CUH
#define AB3M2INTEGRATOR_CUH

#include "ab3m2Integrator.h"

 //==============================================================================
 //! @brief Adams–Bashforth–Moulton 3/2-step time integration scheme for TDGL equations.
 //!
 //! @tparam T Numeric type (float or double).
 //! @param polarization Device vector of polarization field values.
 //! @param hFerro Device vector of effective fields.
 //! @param noiseVector Device vector of stochastic noise values.
 //! @param gridSize Number of grid points in the simulation domain.
 //! @param Ncz Number of cells along z-direction.
 //! @param totalTime Total simulation time.
 //==============================================================================
template <typename T>
void ab3m2(
    thrust::device_vector<Vec3<T>>& polarization,
    thrust::device_vector<Vec3<T>>& hFerro,
    thrust::device_vector<Vec3<T>>& noiseVector,
    int gridSize,
    int Ncz,
    T totalTime
) {
    //----------------------------------------------------------------------
    // Allocate intermediate vectors and local variables
    //----------------------------------------------------------------------
    thrust::device_vector<Vec3<T>> fk_3(gridSize);     //!< Derivative at t - 3Δt
    thrust::device_vector<Vec3<T>> fk_2(gridSize);     //!< Derivative at t - 2Δt
    thrust::device_vector<Vec3<T>> fk_1(gridSize);     //!< Derivative at t - Δt
    thrust::device_vector<Vec3<T>> fk(gridSize);       //!< Derivative at t
    thrust::device_vector<Vec3<T>> fk1(gridSize);      //!< Derivative at t + Δt (predictor)
    thrust::device_vector<Vec3<T>> fk_predictor(gridSize); //!< Derivative at predictor step
    thrust::device_vector<Vec3<T>> y(gridSize);        //!< Updated polarization (intermediate)
    thrust::device_vector<Vec3<T>> y_predictor(gridSize); //!< Predictor polarization
    thrust::device_vector<Vec3<T>> y_corrector(gridSize); //!< Corrected polarization
    thrust::device_vector<Vec3<T>> delta(gridSize);    //!< Difference between predictor and corrector
    thrust::device_vector<T> magnitude_delta(gridSize);//!< Magnitude of delta vector
    thrust::device_vector<T> magnitude_dp(gridSize);   //!< Magnitude of dp/dt

    thrust::host_vector<Vec3<T>> h_polarization(gridSize); //!< Host copy for output
    std::ofstream output("average_polarization.dat");
    output.precision(16);

    //----------------------------------------------------------------------
    // Local variables for control and averaging
    //----------------------------------------------------------------------

    int counter_print = 0;        //!< Console progress percentage
    int stepCounting = 0;         //!< Global step counter
    int counterPolarization = 1; //!< Counter for polarization output
    int counterSnapshot = 1;      //!< Counter for snapshot output
    T initial_dt = FE_geom.dtime; //!< Initial time step (for adaptive scaling)

    //----------------------------------------------------------------------
    // Initial console message
    //----------------------------------------------------------------------
    printf("[%6.2lf %%] ==> Time %.3e / %.3e s\n", 0.0, 0.0, totalTime);
    counter_print++;

    //----------------------------------------------------------------------
    // Initialization of first three steps using RK5 to seed AB3M2 history
    //----------------------------------------------------------------------
    T currentTime = FE_geom.dtime;
    thrust::fill(hFerro.begin(), hFerro.end(), Vec3<T>());

    // Step 1: Compute fk_3
    fk_3 = calculate_tdgl<T>(polarization, hFerro, noiseVector, gridSize, Ncz, currentTime, stepCounting);
    T t_3 = currentTime;
    currentTime += FE_geom.dtime;

    thrust::fill(hFerro.begin(), hFerro.end(), Vec3<T>());
    stepCounting++;

    // Step 2: Advance one step using RK5
    rk5FixedStep<T>(polarization, hFerro, noiseVector, gridSize, Ncz, currentTime, stepCounting);

    // Console progress
    T percentage = (currentTime + FE_geom.dtime) / totalTime * 100.0;
    if (int(percentage) == counter_print) {
        printf("[%6.2lf %%] ==> Time %.3e / %.3e s\n", percentage, currentTime + FE_geom.dtime, totalTime);
        counter_print++;
    }

    //----------------------------------------------------------------------
    // Step 3: Compute fk_2 via TDGL calculation
    //----------------------------------------------------------------------
    thrust::fill(hFerro.begin(), hFerro.end(), Vec3<T>());
    fk_2 = calculate_tdgl<T>(polarization, hFerro, noiseVector, gridSize, Ncz, currentTime, stepCounting);
    T t_2 = currentTime;
    currentTime += FE_geom.dtime;

    thrust::fill(hFerro.begin(), hFerro.end(), Vec3<T>());
    rk5FixedStep<T>(polarization, hFerro, noiseVector, gridSize, Ncz, currentTime, stepCounting);
    stepCounting++;

    if (int(percentage) == counter_print) {
        printf("[%6.2lf %%] ==> Time %.3e / %.3e s\n", percentage, currentTime + FE_geom.dtime, totalTime);
        counter_print++;
    }

    //----------------------------------------------------------------------
    // Step 4: Compute fk_1 for the final initialization
    //----------------------------------------------------------------------
    thrust::fill(hFerro.begin(), hFerro.end(), Vec3<T>());
    fk_1 = calculate_tdgl<T>(polarization, hFerro, noiseVector, gridSize, Ncz, currentTime, stepCounting);
    T t_1 = currentTime;
    currentTime += FE_geom.dtime;

    thrust::fill(hFerro.begin(), hFerro.end(), Vec3<T>());
    rk5FixedStep<T>(polarization, hFerro, noiseVector, gridSize, Ncz, currentTime, stepCounting);

    if (int(percentage) == counter_print) {
        printf("[%6.2lf %%] ==> Time %.3e / %.3e s\n", percentage, currentTime + FE_geom.dtime, totalTime);
        counter_print++;
    }

    //----------------------------------------------------------------------
    // Main AB3M2 predictor–corrector time-stepping loop
    //----------------------------------------------------------------------
    int stepCounter = 0;

    for (; currentTime < totalTime; currentTime += FE_geom.dtime) {
        stepCounter++;
        T dtOld = FE_geom.dtime;

        //------------------------------------------------------------------
        // Predictor stage (Adams–Bashforth 3)
        //------------------------------------------------------------------
        thrust::fill(hFerro.begin(), hFerro.end(), Vec3<T>());
        fk = calculate_tdgl<T>(polarization, hFerro, noiseVector, gridSize, Ncz, currentTime, stepCounting);
        T t = currentTime;

        thrust::transform(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(static_cast<int>(gridSize)),
            y_predictor.begin(),
            predictor<T>(
                thrust::raw_pointer_cast(polarization.data()),
                thrust::raw_pointer_cast(fk_3.data()),
                thrust::raw_pointer_cast(fk_2.data()),
                thrust::raw_pointer_cast(fk_1.data()),
                thrust::raw_pointer_cast(fk.data()),
                FE_geom.dtime)
        );

        //------------------------------------------------------------------
        // Corrector stage (Adams–Moulton 2)
        //------------------------------------------------------------------
        thrust::fill(hFerro.begin(), hFerro.end(), Vec3<T>());
        fk_predictor = calculate_tdgl<T>(
            y_predictor, hFerro, noiseVector, gridSize, Ncz, currentTime + dtOld, stepCounting
        );

        fk1 = fk_predictor;
        T t1 = currentTime + dtOld;

        thrust::transform(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(static_cast<int>(gridSize)),
            y_corrector.begin(),
            corrector<T>(
                thrust::raw_pointer_cast(polarization.data()),
                thrust::raw_pointer_cast(fk_2.data()),
                thrust::raw_pointer_cast(fk_1.data()),
                thrust::raw_pointer_cast(fk.data()),
                thrust::raw_pointer_cast(fk_predictor.data()),
                FE_geom.dtime)
        );

        //------------------------------------------------------------------
        // Handle integrator mode (fixed or adaptive)
        //------------------------------------------------------------------
        if (FE_geom.FLAG_INTEGRATOR == 4) {
            // --- Fixed-step mode ---
            polarization = y_corrector;
            fk_3 = fk_2;
            fk_2 = fk_1;
            fk_1 = fk;

            T percentage = (currentTime + FE_geom.dtime) / totalTime * 100.0;
            if (int(percentage) == counter_print) {
                printf("[%6.2lf %%] ==> Time %.3e / %.3e s\n",
                    percentage, currentTime + FE_geom.dtime, totalTime);
                counter_print++;
            }
        }

        else if (FE_geom.FLAG_INTEGRATOR == 5) {
            // --- Adaptive mode ---
            if (stepCounter < FE_geom.start_adapting) {
                // Use fixed step for startup period
                polarization = y_corrector;
                fk_3 = fk_2;
                fk_2 = fk_1;
                fk_1 = fk;

                T percentage = (currentTime + FE_geom.dtime) / totalTime * 100.0;
                if (int(percentage) == counter_print) {
                    printf("[%6.2lf %%] ==> Time %.3e / %.3e s\n",
                        percentage, currentTime + FE_geom.dtime, totalTime);
                    counter_print++;
                }
            }
            else
            {
                // =========================================================================
                // Step 15: Compute delta between corrected and predicted results
                // =========================================================================
                thrust::transform(
                    y_corrector.begin(), y_corrector.end(),
                    y_predictor.begin(),
                    delta.begin(),
                    delta_m<T>()
                );

                // =========================================================================
                // Step 16: Compute magnitude of error per grid point
                // =========================================================================
                thrust::transform(
                    delta.begin(), delta.end(),
                    magnitude_delta.begin(),
                    getMagnitude<T>()
                );

                // =========================================================================
                // Step 17: Compute average L2 error over the entire grid
                // =========================================================================
                T sumError = thrust::reduce(magnitude_delta.begin(), magnitude_delta.end(), 0.0f, thrust::plus<T>());
                T errorNorm = (sumError / magnitude_delta.size()) * (19.0 / 270.0); // average scaled error

                // =========================================================================
                // Step 18: Compute adaptive scale factor for time step adjustment
                // =========================================================================
                const T safety = 0.9;
                const T minScale = 0.2;
                const T maxScale = 5.0;

                T scale = safety * pow(FE_geom.error_tolerance / errorNorm, 0.2);
                scale = std::min(std::max(scale, minScale), maxScale);

                // =========================================================================
                // Step 19: Apply scaling and enforce dt limits
                // =========================================================================
                T dt_new = FE_geom.dtime * scale;

                const T dt_min = initial_dt / FE_geom.min_dt_times;
                const T dt_max = initial_dt * FE_geom.max_dt_times;

                // --- Clamp dt_new within allowed bounds ---
                if (dt_new < dt_min)
                {
                    dt_new = dt_min;
                    std::cout << " dt clamped to min: "
                        << std::fixed << std::setprecision(14) << std::scientific << dt_min
                        << "  [Error: " << std::setw(12) << errorNorm << "] ";
                }
                else if (dt_new > dt_max)
                {
                    dt_new = dt_max;
                    std::cout << " dt clamped to max: "
                        << std::fixed << std::setprecision(14) << std::scientific << dt_max
                        << "  [Error: " << std::setw(12) << errorNorm << "] ";
                }
                else
                {
                    std::cout << std::fixed << std::setprecision(16)
                        << "[dt: " << std::setw(20) << FE_geom.dtime << "] "
                        << "[Error: " << std::setw(20) << errorNorm << "] "
                        << (errorNorm < FE_geom.error_tolerance ? " Accepted" : " Rejected");
                }

                // --- Progress logging ---
                T percentage = (currentTime + FE_geom.dtime) / totalTime * 100.0;
                std::cout << "  [" << std::fixed << std::setprecision(4)
                    << std::setw(6) << percentage << " %]"
                    << std::endl << std::endl;

                // =========================================================================
                // Step 20: Accept or reject current step based on error
                // =========================================================================
                if (errorNorm < FE_geom.error_tolerance)
                {
                    // --- Accept step ---
                    polarization = y_corrector;

                    if (FE_geom.dtime != dt_new)
                    {
                        // ---------------------------------------------------------------
                        // Re-interpolate history arrays to new time grid
                        // ---------------------------------------------------------------
                        T t_3New = t_3 + dt_new;
                        thrust::transform(
                            thrust::make_counting_iterator(0),
                            thrust::make_counting_iterator(static_cast<int>(gridSize)),
                            fk_3.begin(),
                            lagrange<T>(
                                thrust::raw_pointer_cast(fk_3.data()),
                                thrust::raw_pointer_cast(fk_2.data()),
                                thrust::raw_pointer_cast(fk_1.data()),
                                thrust::raw_pointer_cast(fk.data()),
                                thrust::raw_pointer_cast(fk1.data()),
                                t_3, t_2, t_1, t, t1, t_3New
                            )
                        );

                        T t_2New = t_2 + dt_new;
                        thrust::transform(
                            thrust::make_counting_iterator(0),
                            thrust::make_counting_iterator(static_cast<int>(gridSize)),
                            fk_2.begin(),
                            lagrange<T>(
                                thrust::raw_pointer_cast(fk_3.data()),
                                thrust::raw_pointer_cast(fk_2.data()),
                                thrust::raw_pointer_cast(fk_1.data()),
                                thrust::raw_pointer_cast(fk.data()),
                                thrust::raw_pointer_cast(fk1.data()),
                                t_3, t_2, t_1, t, t1, t_2New
                            )
                        );

                        T t_1New = t_1 + dt_new;
                        thrust::transform(
                            thrust::make_counting_iterator(0),
                            thrust::make_counting_iterator(static_cast<int>(gridSize)),
                            fk_1.begin(),
                            lagrange<T>(
                                thrust::raw_pointer_cast(fk_3.data()),
                                thrust::raw_pointer_cast(fk_2.data()),
                                thrust::raw_pointer_cast(fk_1.data()),
                                thrust::raw_pointer_cast(fk.data()),
                                thrust::raw_pointer_cast(fk1.data()),
                                t_3, t_2, t_1, t, t1, t_1New
                            )
                        );

                        T tNew = t + dt_new;
                        thrust::transform(
                            thrust::make_counting_iterator(0),
                            thrust::make_counting_iterator(static_cast<int>(gridSize)),
                            fk.begin(),
                            lagrange<T>(
                                thrust::raw_pointer_cast(fk_3.data()),
                                thrust::raw_pointer_cast(fk_2.data()),
                                thrust::raw_pointer_cast(fk_1.data()),
                                thrust::raw_pointer_cast(fk.data()),
                                thrust::raw_pointer_cast(fk1.data()),
                                t_3, t_2, t_1, t, t1, tNew
                            )
                        );

                        // --- Update history and time step ---
                        t_3 = t_2;
                        t_2 = t_1;
                        t_1 = t;

                        fk_3 = fk_2;
                        fk_2 = fk_1;
                        fk_1 = fk;

                        FE_geom.dtime = dt_new;
                    }
                }
                else // errorNorm > tolerance → reject step
                {
                    // --- Reject step: update fk history and retry ---
                    if (FE_geom.dtime != dt_new)
                    {
                        // Re-interpolate to new time spacing (same as above)
                        T t_3New = t_3 + dt_new;
                        thrust::transform(
                            thrust::make_counting_iterator(0),
                            thrust::make_counting_iterator(static_cast<int>(gridSize)),
                            fk_3.begin(),
                            lagrange<T>(
                                thrust::raw_pointer_cast(fk_3.data()),
                                thrust::raw_pointer_cast(fk_2.data()),
                                thrust::raw_pointer_cast(fk_1.data()),
                                thrust::raw_pointer_cast(fk.data()),
                                thrust::raw_pointer_cast(fk1.data()),
                                t_3, t_2, t_1, t, t1, t_3New
                            )
                        );

                        T t_2New = t_2 + dt_new;
                        thrust::transform(
                            thrust::make_counting_iterator(0),
                            thrust::make_counting_iterator(static_cast<int>(gridSize)),
                            fk_2.begin(),
                            lagrange<T>(
                                thrust::raw_pointer_cast(fk_3.data()),
                                thrust::raw_pointer_cast(fk_2.data()),
                                thrust::raw_pointer_cast(fk_1.data()),
                                thrust::raw_pointer_cast(fk.data()),
                                thrust::raw_pointer_cast(fk1.data()),
                                t_3, t_2, t_1, t, t1, t_2New
                            )
                        );

                        T t_1New = t_1 + dt_new;
                        thrust::transform(
                            thrust::make_counting_iterator(0),
                            thrust::make_counting_iterator(static_cast<int>(gridSize)),
                            fk_1.begin(),
                            lagrange<T>(
                                thrust::raw_pointer_cast(fk_3.data()),
                                thrust::raw_pointer_cast(fk_2.data()),
                                thrust::raw_pointer_cast(fk_1.data()),
                                thrust::raw_pointer_cast(fk.data()),
                                thrust::raw_pointer_cast(fk1.data()),
                                t_3, t_2, t_1, t, t1, t_1New
                            )
                        );

                        T tNew = t + dt_new;
                        thrust::transform(
                            thrust::make_counting_iterator(0),
                            thrust::make_counting_iterator(static_cast<int>(gridSize)),
                            fk.begin(),
                            lagrange<T>(
                                thrust::raw_pointer_cast(fk_3.data()),
                                thrust::raw_pointer_cast(fk_2.data()),
                                thrust::raw_pointer_cast(fk_1.data()),
                                thrust::raw_pointer_cast(fk.data()),
                                thrust::raw_pointer_cast(fk1.data()),
                                t_3, t_2, t_1, t, t1, tNew
                            )
                        );

                        // --- Update history and retry ---
                        t_3 = t_2;
                        t_2 = t_1;
                        t_1 = t;

                        fk_3 = fk_2;
                        fk_2 = fk_1;
                        fk_1 = fk;

                        currentTime -= FE_geom.dtime;  // rollback
                        FE_geom.dtime = dt_new;
                    }

                    continue;  // Retry the step
                }
            }
        }

        //------------------------------------------------------------------
        // Output handling and stopping criteria
        //------------------------------------------------------------------
        h_polarization = polarization;
        thrust::host_vector<Vec3<T>> h_Force = fk_predictor;

        // Compute energy if enabled
        FE_geom.ferro_energy.resize(gridSize);
        if (FE_geom.FLAG_ENERGY == 1)
            FE_geom.ferro_energy = landau_energy(polarization);

        thrust::host_vector<Type_var> total_energy = FE_geom.ferro_energy;

        handleAllOutputs(
            counterPolarization, counterSnapshot, stepCounting,
            currentTime, h_polarization, h_Force, total_energy
        );

        counterPolarization++;
        counterSnapshot++;
        stepCounting++;

        // Check maximum derivative to stop simulation
        thrust::transform(fk_predictor.begin(), fk_predictor.end(),
            magnitude_dp.begin(), getMagnitude<T>());

        T max_dt = thrust::reduce(magnitude_dp.begin(), magnitude_dp.end(),
            T(0), thrust::maximum<T>());

        if (max_dt <= FE_geom.stop_sim) {
            std::cout << "Simulation stopped due to threshold condition.\n";
            std::cout << "Maximum derivative observed: " << max_dt << '\n';
            std::cout << "Total steps completed before termination: " << counter_print << '\n';
            break;
        }
    }
}

#endif // AB3M2INTEGRATOR_CUH
