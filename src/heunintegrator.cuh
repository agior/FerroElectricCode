/**
 * @file heun.cu
 * @brief Implements the Heun (predictor-corrector) integration method for TDGL polarization dynamics.
 *
 * This file performs time integration of the polarization field using Heun’s method
 * (a second-order explicit Runge–Kutta method). It supports CUDA device computation
 * and optional performance timing.
 */

#include <fstream>
#include <iomanip>
#include <chrono>

#include "heunSteps.h"  // Step calculation methods for Heun's method

 //------------------------------------------------------------------------------
 // Timing macro control
 //------------------------------------------------------------------------------
#define timing 0
#if timing
#define timer(x)                                                           \
    do {                                                                   \
        auto start = std::chrono::high_resolution_clock::now();            \
        x;                                                                 \
        auto stop = std::chrono::high_resolution_clock::now();             \
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); \
        printf("%s time (ms): %f\n", #x, elapsed.count());                 \
    } while (0)
#else
#define timer(x) x;
#endif

//------------------------------------------------------------------------------
// External variables
//------------------------------------------------------------------------------
extern set_output set_out;

//==============================================================================
//! @brief Heun integration scheme for polarization dynamics in TDGL equations.
//!
//! @tparam T Numeric precision type (e.g., float or double).
//! @param polarization Device vector holding current polarization values (Vec3<T>).
//! @param hFerro Device vector for effective field values (Vec3<T>).
//! @param noiseVector Device vector storing stochastic noise contributions.
//! @param gridSize Total number of spatial elements in the simulation grid.
//! @param Ncz Number of cells along the z-axis (depth).
//! @param totalTime Total simulation time (normalized).
//==============================================================================
template <typename T>
void heun(
    thrust::device_vector<Vec3<T>>& polarization,
    thrust::device_vector<Vec3<T>>& hFerro,
    thrust::device_vector<Vec3<T>>& noiseVector,
    int gridSize,
    int Ncz,
    T totalTime
) {
    //--------------------------------------------------------------------------
    // Allocate intermediate vectors for Heun’s method calculations
    //--------------------------------------------------------------------------
    thrust::device_vector<Vec3<T>> k1(gridSize);       // First slope
    thrust::device_vector<Vec3<T>> y1(gridSize);       // Intermediate state
    thrust::device_vector<Vec3<T>> k2(gridSize);       // Second slope
    thrust::device_vector<Vec3<T>> y2(gridSize);       // (Reserved, not used)
    thrust::device_vector<Vec3<T>> ki_avg(gridSize);   // Averaged slope
    thrust::device_vector<T> magnitude_dp(gridSize);   // Magnitude of dP/dt

    //--------------------------------------------------------------------------
    // Step counters and control variables
    //--------------------------------------------------------------------------
    int counter_print = 0;
    int stepCounting = 0;
    int counterMagnetization = 1;
    int counterSnapshot = 1;

    //--------------------------------------------------------------------------
    // Host vectors for optional data extraction and logging
    //--------------------------------------------------------------------------
    thrust::host_vector<Vec3<T>> h_polarization((int)gridSize);
    thrust::host_vector<Vec3<T>> h_Force((int)gridSize);
    std::ofstream output("average_polarization.dat");
    output.precision(16);

    Vec3<T> avg;
    Vec3<T> avgForce;
    T sumEnergy;

    //--------------------------------------------------------------------------
    // Initial console output
    //--------------------------------------------------------------------------
    printf("[%6.2lf %%] ==> Time %.3e / %.3e s\n", 0.0 / totalTime * 100.0, 0.0, totalTime);
    counter_print++;

    totalTime = FE_geom.time_sim;  // Use normalized total simulation time

    //--------------------------------------------------------------------------
    // Main time integration loop
    //--------------------------------------------------------------------------
    T currentTime = 0.0;
    while (currentTime <= totalTime) {

        //----------------------------------------------------------------------
        // Step 1: Reset effective field for the first Heun step
        //----------------------------------------------------------------------
        timer(thrust::fill(hFerro.begin(), hFerro.end(), Vec3<T>()));

        //----------------------------------------------------------------------
        // Step 2: Compute first derivative (k1) using TDGL equation
        //----------------------------------------------------------------------
        timer(k1 = calculate_tdgl<T>(
            polarization, hFerro, noiseVector, gridSize, Ncz, currentTime, stepCounting
        ));

        stepCounting++;

        //----------------------------------------------------------------------
        // Step 3: Compute intermediate state y1 = P + dt * k1
        //----------------------------------------------------------------------
        thrust::transform(
            polarization.begin(), polarization.end(),
            k1.begin(), y1.begin(),
            updated_y<T>(FE_geom.dtime)
        );

        //----------------------------------------------------------------------
        // Step 4: Reset field and compute second derivative (k2)
        //----------------------------------------------------------------------
        thrust::fill(hFerro.begin(), hFerro.end(), Vec3<T>());

        timer(k2 = calculate_tdgl<T>(
            y1, hFerro, noiseVector, gridSize, Ncz, currentTime + FE_geom.dtime, stepCounting
        ));

        //----------------------------------------------------------------------
        // Step 5: Average slopes and update polarization
        //----------------------------------------------------------------------
        thrust::transform(
            k1.begin(), k1.end(),
            k2.begin(), ki_avg.begin(),
            k_avg<T>()  // average (k1 + k2) / 2
        );

        timer(thrust::transform(
            polarization.begin(), polarization.end(),
            ki_avg.begin(), polarization.begin(),
            updated_y<T>(FE_geom.dtime)
        ));

        //----------------------------------------------------------------------
        // Step 6: Print progress information
        //----------------------------------------------------------------------
        T percentage = (currentTime + FE_geom.dtime) / totalTime * 100.0;
        if (int(percentage) >= counter_print) {
            printf("[%6.2lf %%] ==> Time %.3e / %.3e s\n", percentage, currentTime + FE_geom.dtime, totalTime);
            fflush(stdout);
            counter_print = int(percentage) + 1;
        }

        //----------------------------------------------------------------------
        // Step 7: Copy data back to host for output
        //----------------------------------------------------------------------
        h_polarization = polarization;
        h_Force = ki_avg;

        //----------------------------------------------------------------------
        // Step 8: Optional energy computation
        //----------------------------------------------------------------------
        FE_geom.ferro_energy.resize(gridSize);
        if (FE_geom.FLAG_ENERGY == 1) {
            FE_geom.ferro_energy = landau_energy(polarization);
        }

        thrust::host_vector<Type_var> total_energy = FE_geom.ferro_energy;

        //----------------------------------------------------------------------
        // Step 9: Handle magnetization and snapshot outputs
        //----------------------------------------------------------------------
        handleAllOutputs(
            counterMagnetization,
            counterSnapshot,
            stepCounting,
            currentTime,
            h_polarization,
            h_Force,
            total_energy
        );

        counterMagnetization++;
        counterSnapshot++;

        //----------------------------------------------------------------------
        // Step 10: Check for stop condition based on derivative magnitude
        //----------------------------------------------------------------------
        thrust::transform(
            ki_avg.begin(), ki_avg.end(),
            magnitude_dp.begin(),
            getMagnitude<T>()
        );

        T max_dt = thrust::reduce(
            magnitude_dp.begin(), magnitude_dp.end(),
            T(0), thrust::maximum<T>()
        );

        if (max_dt <= FE_geom.stop_sim) {
            std::cout << "Simulation stopped due to threshold condition.\n";
            std::cout << "Maximum derivative observed: " << max_dt << '\n';
            std::cout << "Total steps completed before termination: " << counter_print << '\n';
            break;
        }

        //----------------------------------------------------------------------
        // Step 11: Advance simulation time
        //----------------------------------------------------------------------
        currentTime += FE_geom.dtime;
    }
}
