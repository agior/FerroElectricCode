// ============================================================================
// main.cu
// ============================================================================
// High-performance CUDA + Thrust simulation driver for ferroelectric modeling.
// Handles configuration loading, simulation setup, and integration kernel launch.
// ============================================================================

/* ========================= CUDA and Thrust Headers ========================= */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

/* ============================ Standard C++ Headers ========================= */
#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <filesystem>

/* =========================== Project-Specific Headers ====================== */
#include "Vec3.h"
#include "heunIntegrator.h"
#include "rk5Integrator.h"
#include "constants.h"
#include "output.h"
#include "ab3m2Integrator.h"
#include "landauEnergy.cu"
#include "landauFreeEnergy.cuh"
#include "setValues.cuh"
#include "output.hpp"

/* =============================== Global Variables ========================== */
Type_var field_ext;
std::ofstream file_log, file_conf;
landau_free landau_free_param;
FE_geometry FE_geom;
set_output set_out;
initial_polarization initial_pol;
field_external external_field;
field_external_normalized external_field_vectors;
gradient gradient_field_param;
elastic elastic_field_param;
landau_free_final landau_vectors;
gradient_final gradient_vectors;
SET_parameters conf;
set_tensor conf_tens;
std::string outputFolder = "output";
std::ofstream output;

/**
 * @brief Loads all configuration files and initializes global simulation parameters.
 *
 * This function reads configuration files for:
 * - Finite element geometry
 * - Landau free energy parameters
 * - External field setup
 * - Initial polarization
 * - Output parameters
 * - Gradient field and electrostatic tensors
 *
 * It prints the geometry configuration to console for verification.
 */
void configurationFunc() {
    // Load finite difference geometry configuration
    load_geometry_configuration(FE_geom);

    // Extract geometry parameters
    int N_d = FE_geom.N_d;
    int Ncx = FE_geom.Ncx;
    int Ncy = FE_geom.Ncy;
    int Ncz = FE_geom.Ncz;
    int N = Ncx * Ncy * Ncz * N_d;

    // Display geometry summary
    printf("\n Ncx= %d Ncy= %d Ncz= %d N_d= %d  number cells=%d \n ",
        Ncx, Ncy, Ncz, N_d, N);

    // Load remaining configuration files and material parameters
    load_Landau_configuration(landau_free_param, N);
    load_external_configuration(external_field, N);
    load_initialP_configuration(initial_pol, N);
    load_output_configuration(set_out);
    gradient_field_param = load_Gradient_configuration("./file_configuration/gradient_field.dat", N);
    conf_tens = conf_tensor("./file_configuration/electrostatic.dat");
    conf = set_configuration(FE_geom, conf_tens);

    printf("\n ...End read file configuration!\n ");
}

/**
 * @brief Main simulation entry point.
 *
 * Handles:
 * - Resetting output directories
 * - Loading configurations
 * - Selecting and running numerical integrators (Heun, RK5, AB3M2)
 * - Measuring device runtime
 * - Writing output and diagnostic data
 *
 * @return int Exit code (0 = success)
 */
int main() {

    // Force use of GPU 1
    cudaError_t err = cudaSetDevice(1);
    if (err != cudaSuccess) {
        printf("Failed to set CUDA device: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Running on GPU 1\n");

    // Reset output folder (deletes/creates as needed)
    resetOutputFolders();

    // Load all simulation configuration files and constants
    configurationFunc();

    // Initialize constant parameters (e.g., normalization constants)
    set_values<Type_var>();

    // Print simulation overview
    std::cout << "Normalized simulation time: " << FE_geom.time_sim
        << " | Normalized dt: " << FE_geom.dtime
        << " | Total steps: " << FE_geom.time_sim / FE_geom.dtime << "\n";

    // Compute grid size (total number of simulation cells)
    int gridSize = FE_geom.Ncx * FE_geom.Ncy * FE_geom.Ncz * FE_geom.N_d;

    // Start timing GPU computation
    auto device_start = std::chrono::high_resolution_clock::now();

    // Allocate device vectors
    thrust::device_vector<Vec3<Type_var>> polarization = initial_pol.initial_pol_vector;
    thrust::device_vector<Vec3<Type_var>> hFerro(gridSize);
    thrust::device_vector<Vec3<Type_var>> noiseVector(gridSize);

    // ------------------------------------------------------------------------
    // Select and execute numerical integrator
    // ------------------------------------------------------------------------
    if (FE_geom.FLAG_INTEGRATOR == 1) {
        // Heun’s method (predictor-corrector)
        heun<Type_var>(polarization, hFerro, noiseVector, gridSize, FE_geom.Ncz, FE_geom.time_sim);
    }
    else if (FE_geom.FLAG_INTEGRATOR == 2 || FE_geom.FLAG_INTEGRATOR == 3) {
        // Runge-Kutta 5th-order method (fixed or adaptive)
        rk5<Type_var>(polarization, hFerro, noiseVector, gridSize, FE_geom.Ncz, FE_geom.time_sim);
    }
    else if (FE_geom.FLAG_INTEGRATOR == 4 || FE_geom.FLAG_INTEGRATOR == 5) {
        // Adams-Bashforth 3rd order / Moulton 2-step method
        ab3m2<Type_var>(polarization, hFerro, noiseVector, gridSize, FE_geom.Ncz, FE_geom.time_sim);
    }

    // Stop device timing
    auto device_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<Type_var> device_elapsed = device_end - device_start;
    Type_var timeTaken = device_elapsed.count();

    // Save normalized Landau and gradient coefficients for reference
    saveAlphaVectorsToFile<Type_var>();
    saveGradientVectorsToFile<Type_var>();

    // Output results (final polarization fields and runtime statistics)
    getOutput<Type_var>(
        polarization,
        FE_geom.Ncx,
        FE_geom.Ncy,
        FE_geom.Ncz,
        FE_geom.N_d,
        timeTaken
    );

    return 0;
}
