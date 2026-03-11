#ifndef RK5INTEGRATOR_CUH
#define RK5INTEGRATOR_CUH

#include "rk5Integrator.h"
#include <thrust/functional.h>

// ============================================================================
// Runge–Kutta 5th Order (RK5) Integrator
// -----------------------------------------------------------------------------
// This CUDA/Thrust-based routine numerically integrates the polarization
// vector field over time using a 5th-order Runge–Kutta scheme. It supports
// both fixed and adaptive time-stepping, depending on the configuration
// flag (FE_geom.FLAG_INTEGRATOR).
// ============================================================================

template <typename T>
void rk5(
    thrust::device_vector<Vec3<T>>& polarization,      // Polarization vector to be updated
    thrust::device_vector<Vec3<T>>& hFerro,            // Effective field vector
    thrust::device_vector<Vec3<T>>& noiseVector,       // Noise (thermal/random) vector
    int gridSize,                                      // Total number of grid points
    int Ncz,                                           // Number of cells along z-axis
    T totalTime                                        // Total simulation time
)
{
    // ------------------------------------------------------------------------
    // Allocate intermediate RK5 vectors on the device
    // ------------------------------------------------------------------------
    thrust::device_vector<Vec3<T>> k1(gridSize, Vec3<T>(0.0f, 0.0f, 0.0f));
    thrust::device_vector<Vec3<T>> y(gridSize, Vec3<T>(0.0f, 0.0f, 0.0f));
    thrust::device_vector<Vec3<T>> k2(gridSize, Vec3<T>(0.0f, 0.0f, 0.0f));
    thrust::device_vector<Vec3<T>> k3(gridSize, Vec3<T>(0.0f, 0.0f, 0.0f));
    thrust::device_vector<Vec3<T>> k4(gridSize, Vec3<T>(0.0f, 0.0f, 0.0f));
    thrust::device_vector<Vec3<T>> k5(gridSize, Vec3<T>(0.0f, 0.0f, 0.0f));
    thrust::device_vector<Vec3<T>> k6(gridSize, Vec3<T>(0.0f, 0.0f, 0.0f));
    thrust::device_vector<Vec3<T>> finalY(gridSize, Vec3<T>(0.0f, 0.0f, 0.0f));
    thrust::device_vector<Vec3<T>> primeY(gridSize, Vec3<T>(0.0f, 0.0f, 0.0f));
    thrust::device_vector<Vec3<T>> delta(gridSize);
    thrust::device_vector<T> magnitude_delta(gridSize);
    thrust::device_vector<T> magnitude_dp(gridSize);

    // ------------------------------------------------------------------------
    // Simulation counters and control variables
    // ------------------------------------------------------------------------
    int stepCounting = 0;
    int counter_print = 0;
    int counterPolarization = 1;
    int counterSnapshot = 1;

    Vec3<T> avgForce;
    thrust::host_vector<Vec3<T>> h_polarization((int)gridSize);
    std::ofstream output("average_polarization.dat");
    output.precision(16);

    int counter_adapter = 0;
    Vec3<T> avg;

    // ------------------------------------------------------------------------
    // Simulation start message
    // ------------------------------------------------------------------------
    printf("[%6.2lf %%] ==> Time %.3e / %.3e s\n", 0.0 / totalTime * 100.0, 0.0, totalTime);

    // Store the initial time step for adaptive scaling
    T initial_dt = FE_geom.dtime;

    T maxError;
    int count = 0;

    // ========================================================================
    // Main RK5 Integration Loop
    // ========================================================================
    for (T currentTime = 0.0; currentTime <= totalTime; currentTime += FE_geom.dtime)
    {
        // Reset the effective field for this time step
        thrust::fill(hFerro.begin(), hFerro.end(), Vec3<T>());

        // --------------------------------------------------------------------
        // Stage 1: Compute k1
        // --------------------------------------------------------------------
        k1 = calculate_tdgl<T>(polarization, hFerro, noiseVector, gridSize, Ncz, currentTime + aa1, stepCounting);

        // Compute y1 intermediate state
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

        // --------------------------------------------------------------------
        // Stage 2: Compute k2
        // --------------------------------------------------------------------
        thrust::fill(hFerro.begin(), hFerro.end(), Vec3<T>());
        k2 = calculate_tdgl<T>(y, hFerro, noiseVector, gridSize, Ncz, currentTime + aa2, stepCounting);

        // Compute y2 intermediate state
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

        // --------------------------------------------------------------------
        // Stage 3: Compute k3
        // --------------------------------------------------------------------
        thrust::fill(hFerro.begin(), hFerro.end(), Vec3<T>());
        k3 = calculate_tdgl<T>(y, hFerro, noiseVector, gridSize, Ncz, currentTime + aa3, stepCounting);

        // Compute y3 intermediate state
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

        // --------------------------------------------------------------------
        // Stage 4: Compute k4
        // --------------------------------------------------------------------
        thrust::fill(hFerro.begin(), hFerro.end(), Vec3<T>());
        k4 = calculate_tdgl<T>(y, hFerro, noiseVector, gridSize, Ncz, currentTime + aa4, stepCounting);

        // Compute y4 intermediate state
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

        // --------------------------------------------------------------------
        // Stage 5: Compute k5
        // --------------------------------------------------------------------
        thrust::fill(hFerro.begin(), hFerro.end(), Vec3<T>());
        k5 = calculate_tdgl<T>(y, hFerro, noiseVector, gridSize, Ncz, currentTime + aa5, stepCounting);

        // Compute y5 intermediate state
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

        // --------------------------------------------------------------------
        // Stage 6: Compute k6
        // --------------------------------------------------------------------
        thrust::fill(hFerro.begin(), hFerro.end(), Vec3<T>());
        k6 = calculate_tdgl<T>(y, hFerro, noiseVector, gridSize, Ncz, currentTime + aa6, stepCounting);

        // --------------------------------------------------------------------
        // Compute Final Y (RK5 update)
        // --------------------------------------------------------------------
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

        // --------------------------------------------------------------------
        // Compute Prime Y (for adaptive step control)
        // --------------------------------------------------------------------
        thrust::transform(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(static_cast<int>(gridSize)),
            primeY.begin(),
            prime_y<T>(
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

        // ====================================================================
        // Handle Integration Mode (Fixed vs Adaptive)
        // ====================================================================

        // ------------------- Fixed Step Size -------------------
        if (FE_geom.FLAG_INTEGRATOR == 2)
        {
            polarization = finalY;

            T percentage = (currentTime + FE_geom.dtime) / totalTime * 100.0;
            if (int(percentage) == counter_print)
            {
                printf("[%6.2lf %%] ==> Time %.3e / %.3e s\n",
                    percentage, currentTime + FE_geom.dtime, totalTime);
                counter_print++;
            }
        }

        // ------------------- Adaptive Step Size -------------------
        else if (FE_geom.FLAG_INTEGRATOR == 3)
        {
            counter_adapter++;

            // Run initial fixed steps before adapting
            if (counter_adapter <= FE_geom.start_adapting)
            {
                polarization = finalY;

                T percentage = (currentTime + FE_geom.dtime) / totalTime * 100.0;
                if (int(percentage) == counter_print)
                {
                    printf("[%6.2lf %%] ==> Time %.3e / %.3e s\n",
                        percentage, currentTime + FE_geom.dtime, totalTime);
                    counter_print++;
                }
            }
            else
            {
                // Compute local error estimate: delta = |finalY - primeY|
                thrust::transform(finalY.begin(), finalY.end(), primeY.begin(), delta.begin(), delta_m<T>());

                // Compute magnitude of error for each grid point
                thrust::transform(delta.begin(), delta.end(), magnitude_delta.begin(), getMagnitude<T>());

                // Average error across the grid
                T sumError = thrust::reduce(magnitude_delta.begin(), magnitude_delta.end(), 0.0f, thrust::plus<T>());
                T errorNorm = sumError / magnitude_delta.size();

                // Adaptive time-step scaling parameters
                const T safety = 0.9;
                const T minScale = 0.2;
                const T maxScale = 5.0;

                // Compute adaptive scale factor
                T scale = safety * pow(FE_geom.error_tolerance / errorNorm, 0.2);
                scale = std::min(std::max(scale, minScale), maxScale);

                // Apply scale to compute new dt
                T dt_new = FE_geom.dtime * scale;

                // Clamp dt between allowed limits
                const T dt_min = initial_dt / FE_geom.min_dt_times;
                const T dt_max = initial_dt * FE_geom.max_dt_times;

                if (dt_new < dt_min)
                {
                    dt_new = dt_min;
                    std::cout << " dt clamped to min: " << std::scientific << dt_min
                        << "  [Error: " << std::setw(12) << errorNorm << "] ";
                }
                else if (dt_new > dt_max)
                {
                    dt_new = dt_max;
                    std::cout << " dt clamped to max: " << std::scientific << dt_max
                        << "  [Error: " << std::setw(12) << errorNorm << "] ";
                }
                else
                {
                    std::cout << std::fixed << std::setprecision(16)
                        << "[dt: " << std::setw(20) << FE_geom.dtime << "] "
                        << "[Error: " << std::setw(20) << errorNorm << "] "
                        << (errorNorm < FE_geom.error_tolerance ? " Accepted" : " Rejected");
                }

                T percentage = (currentTime + FE_geom.dtime) / totalTime * 100.0;
                std::cout << "  [" << std::fixed << std::setprecision(4)
                    << std::setw(6) << percentage << " %]" << std::endl
                    << std::endl;

                // Accept or reject current step
                if (errorNorm < FE_geom.error_tolerance)
                {
                    polarization = finalY;
                    FE_geom.dtime = dt_new;
                }
                else
                {
                    FE_geom.dtime = dt_new;
                    currentTime -= FE_geom.dtime;
                    continue;  // Retry same step
                }
            }
        }

        // ====================================================================
        // Data Output and Termination Conditions
        // ====================================================================
        h_polarization = polarization;
        thrust::host_vector<Vec3<T>> h_Force = k6;

        // Compute Landau energy (if enabled)
        FE_geom.ferro_energy.resize(gridSize);
        if (FE_geom.FLAG_ENERGY == 1)
            FE_geom.ferro_energy = landau_energy(polarization);

        // Copy energy to host
        thrust::host_vector<Type_var> total_energy = FE_geom.ferro_energy;

        // Handle all output files and visualization snapshots
        handleAllOutputs(
            counterPolarization,
            counterSnapshot,
            stepCounting,
            currentTime,
            h_polarization,
            h_Force,
            total_energy
        );

        counterPolarization++;
        counterSnapshot++;
        stepCounting++;

        // Compute maximum field magnitude for stop condition
        thrust::transform(k6.begin(), k6.end(), magnitude_dp.begin(), getMagnitude<T>());
        T max_dt = thrust::reduce(magnitude_dp.begin(), magnitude_dp.end(), T(0), thrust::maximum<T>());

        // Stop simulation if below threshold
        if (max_dt <= FE_geom.stop_sim)
        {
            std::cout << "Simulation stopped due to threshold condition.\n";
            std::cout << "Maximum derivative observed: " << max_dt << '\n';
            std::cout << "Total steps completed before termination: " << counter_print << '\n';
            break;
        }
    }
}

#endif  // RK5INTEGRATOR_CUH
