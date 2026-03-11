/**
 * @file constants.cuh
 * @brief Implements configuration loading, initialization, and file I/O helper routines
 *        for TDGL-based CUDA C++ simulations.
 *
 * This file defines the implementation of functions declared in `constants.h`.
 * It handles:
 *   - Reading scalar and vector data into device vectors.
 *   - Loading simulation geometry, polarization, and configuration files.
 *   - Managing initialization of polarization and shape fields.
 *
 * @note All operations are performed on the host; data are transferred to GPU memory
 *       via Thrust device vectors where applicable.
 */

#pragma once
#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

//------------------------------------------------------------------------------
// Includes
//------------------------------------------------------------------------------
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>

#include "constants.h"

//------------------------------------------------------------------------------
// Global variables (file handles and configuration structures)
//------------------------------------------------------------------------------

// Input file streams for configuration files
std::ifstream fileiic, filei1c, fileii;

// Output logging file stream
std::ofstream file_logc;
extern std::ofstream file_log;

// Core configuration structures shared across modules
SET_parameters configuration;
extern landau_free landau_free_param;
extern FE_geometry FE_geom;
extern set_output set_out;
extern initial_polarization initial_pol;
extern field_external external_field;
extern gradient gradient_field_param;
extern elastic elastic_field_param;
extern set_tensor conf_tens;

//==============================================================================
//! @name File Input Utility Functions
//! These functions handle reading scalar and vector data into Thrust device vectors.
//==============================================================================

/**
 * @brief Loads scalar values from a text file into a Thrust device vector.
 *
 * @tparam T Numeric type (e.g., float, double).
 * @param filename Path to input file containing scalar data.
 * @param vec Thrust device vector to be populated.
 *
 * @details
 * - Each line of the file should contain one numeric value.
 * - The function stops reading once the required number of entries is reached.
 * - Prints information messages and exits upon error.
 */
template <typename T>
void inputValues(const std::string& filename, thrust::device_vector<T>& vec) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        exit(-1);
        return;
    }

    T value;
    size_t count = 0;
    int size = vec.size();
    std::vector<double> host_vector;

    // Read values up to specified vector size
    while (infile >> value && count < size) {
        host_vector.push_back(value);
        count++;
    }

    // Warn if file had insufficient entries
    if (count < size) {
        std::cerr << "Warning: File " << filename
                  << " contains fewer values than needed." << std::endl;
        exit(-1);
    }

    // Transfer from host to device
    vec = host_vector;

    infile.close();
    std::cerr << " Info: Successfully loaded " << host_vector.size()
              << " values from " << filename << std::endl;
}

/**
 * @brief Loads 3-component vector values (x, y, z) from file into a device vector of Vec3<T>.
 *
 * @tparam T Numeric type (e.g., float, double).
 * @param filename Path to input file.
 * @param vec Thrust device vector to be populated with Vec3<T> elements.
 *
 * @details
 * - Each line in the file must contain three numeric values (x, y, z).
 * - Exits on failure or insufficient data.
 */
template <typename T>
void inputValuesVec3(const std::string& filename, thrust::device_vector<Vec3<T>>& vec) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        exit(-1);
        return;
    }

    T x, y, z;
    size_t count = 0;
    int size = vec.size();
    std::vector<Vec3<T>> host_vector;

    // Read triplets from file
    while (infile >> x >> y >> z && count < size) {
        host_vector.push_back(Vec3<T>(x, y, z));
        count++;
    }

    // Warn if insufficient entries
    if (count < size) {
        std::cerr << "Warning: File " << filename
                  << " contains fewer sets of values than needed." << std::endl;
        exit(-1);
    }

    vec = host_vector;

    infile.close();
    std::cerr << " Info: Successfully loaded " << host_vector.size()
              << " Vec3 values from " << filename << std::endl;
}

/**
 * @brief Loads paired scalar values from file into two Thrust device vectors.
 *
 * @tparam T Numeric type (e.g., float, double).
 * @param filename Path to input file.
 * @param vec1 First Thrust device vector to fill.
 * @param vec2 Second Thrust device vector to fill.
 *
 * @details
 * - Each line must contain two numeric values (x, y).
 * - Useful for coupled parameter fields or 2D datasets.
 */
template <typename T>
void inputValuesVec2(const std::string& filename,
                     thrust::device_vector<T>& vec1,
                     thrust::device_vector<T>& vec2) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        exit(-1);
        return;
    }

    T x, y;
    size_t count = 0;
    int size = vec1.size();
    std::vector<T> host_vector1;
    std::vector<T> host_vector2;

    // Read paired values
    while (infile >> x >> y && count < size) {
        host_vector1.push_back(T(x));
        host_vector2.push_back(T(y));
        count++;
    }

    if (count < size) {
        std::cerr << "Warning: File " << filename
                  << " contains fewer sets of values than needed." << std::endl;
        exit(-1);
    }

    vec1 = host_vector1;
    vec2 = host_vector2;

    infile.close();
    std::cerr << " Info: Successfully loaded " << host_vector1.size()
              << " paired values from " << filename << std::endl;
}

//==============================================================================
//! @name Shape and Geometry
//! Reads shape configuration for the simulation domain.
//==============================================================================

/**
 * @brief Loads the geometric shape configuration ("shape.dat") into a device vector.
 * @tparam T Numeric type (float, double, etc.)
 * @param shapeVector Thrust device vector to hold shape data.
 */
template <typename T>
void shapeIn(thrust::device_vector<T>& shapeVector) {
    const std::string file_path = "shape.dat";
    inputValues<T>(file_path, shapeVector);
}

//==============================================================================
//! @name Initial Polarization Setup
//! Loads and configures the initial polarization distribution.
//==============================================================================

/**
 * @brief Loads initial polarization configuration from predefined file.
 *
 * @param initial_pol Struct containing initial polarization data.
 * @param N Total number of grid points.
 *
 * @details
 * Reads "./file_configuration/initial_polarization.dat" and initializes
 * the polarization vector according to the specified FLAG_INICIALP value:
 *   - 1 → Uniform polarization.
 *   - 2 → Layer-wise polarization (from file).
 *   - 3 → Non-uniform full-field polarization (from file).
 */
void load_initialP_configuration(initial_polarization& initial_pol, int N) {
    const char* filename = "./file_configuration/initial_polarization.dat";
    try {
        fileiic.open(filename, std::ifstream::in);

        if (!fileiic.is_open()) {
            printf("errore nell'apertura file %s \n", filename);
            file_logc << "\n";
            file_logc << "errore nell apertura file" << filename;
            exit(-1);
        } else {
            load_initial_polarization(initial_pol, fileiic, N);
            fileiic.close();
        }
    } catch (std::exception& e) {
        printf("%s", e.what());
    }
}

/**
 * @brief Loads and interprets the initial polarization parameters from an open file stream.
 *
 * @param initial_pol Struct holding polarization configuration.
 * @param filei Input file stream.
 * @param N Total number of grid points.
 *
 * @details
 * Reads configuration blocks marked by `//Pinicial` from the provided stream and
 * fills the `initial_polarization` structure accordingly.
 */
void load_initial_polarization(initial_polarization& initial_pol,
                               std::ifstream& filei,
                               int N) {
    char pinicial_z_variable[N_BUFFER];
    char pinicial_Non_Uniform[N_BUFFER];
    Vec3<Type_var> polarization_val(0.0, 0.0, 0.0);
    char temp[N_BUFFER], temp2[N_BUFFER];
    int i = 0, j, check;

    do {
        filei.getline(temp2, N_BUFFER);

        if (strcmp(temp2, "//Pinicial") == 0) {
            //--- Read FLAG_INICIALP ---
            filei.getline(temp, N_BUFFER, '!');
            initial_pol.FLAG_INICIALP = atoi(temp);
            filei.getline(temp, N_BUFFER);

            //--- Read X-component ---
            filei.getline(temp, N_BUFFER, '!');
            polarization_val.x = atof(temp);
            filei.getline(temp, N_BUFFER);

            //--- Read Y-component ---
            filei.getline(temp, N_BUFFER, '!');
            polarization_val.y = atof(temp);
            filei.getline(temp, N_BUFFER);

            //--- Read Z-component ---
            filei.getline(temp, N_BUFFER, '!');
            polarization_val.z = atof(temp);
            filei.getline(temp, N_BUFFER);

            //--- Read z-variable filename ---
            filei.getline(temp, N_BUFFER, '!');
            i = 0;
            while (temp[i] != '\0') {
                check = 0;
                if (temp[i] == ' ' || temp[i] == '\t') {
                    j = i;
                    while (temp[j - 1] != '\0') {
                        temp[j] = temp[j + 1];
                        j++;
                    }
                    check = 1;
                }
                if (check == 0) i++;
            }
            strncpy(pinicial_z_variable, temp, N_BUFFER);

            //--- Read non-uniform filename ---
            filei.getline(temp, N_BUFFER);
            filei.getline(temp, N_BUFFER, '!');
            i = 0;
            while (temp[i] != '\0') {
                check = 0;
                if (temp[i] == ' ' || temp[i] == '\t') {
                    j = i;
                    while (temp[j - 1] != '\0') {
                        temp[j] = temp[j + 1];
                        j++;
                    }
                    check = 1;
                }
                if (check == 0) i++;
            }
            strncpy(pinicial_Non_Uniform, temp, N_BUFFER);

            //--- Assign polarization according to mode ---
            if (initial_pol.FLAG_INICIALP == 1) {
                // Uniform polarization for all cells
                initial_pol.initial_pol_vector.resize(1);
                initial_pol.initial_pol_vector[0] = polarization_val;
            } else if (initial_pol.FLAG_INICIALP == 2) {
                // z-dependent polarization from file
                initial_pol.initial_pol_vector.resize(FE_geom.Ncz);
                inputValuesVec3<Type_var>(pinicial_z_variable, initial_pol.initial_pol_vector);
            } else if (initial_pol.FLAG_INICIALP == 3) {
                // Fully non-uniform polarization from file
                initial_pol.initial_pol_vector.resize(N);
                inputValuesVec3<Type_var>(pinicial_Non_Uniform, initial_pol.initial_pol_vector);
            } else {
                printf("\n The value of FLAG_INITIAL_P is not valid!\n");
                file_logc << "\n";
                file_logc << "The value of FLAG_INITIAL_P is not valid!";
                exit(-1);
            }
        }
    } while (!filei.eof());
}


/**
 * @brief Loads geometry configuration from predefined input file.
 *
 * This function opens and reads the "ferro_geometry.dat" file, which contains
 * information about the finite element geometry (domain size, spacing, shape,
 * and simulation time parameters). It calls `load_geometry()` to parse and
 * store these parameters into the `FE_geometry` structure.
 *
 * @param FE_geom Reference to the FE_geometry structure where data will be stored.
 */
void load_geometry_configuration(FE_geometry& FE_geom) {
	const char* filename = "./file_configuration/ferro_geometry.dat";
	try {
		fileiic.open(filename, std::ifstream::in);

		if (fileiic.is_open() == false) {
			printf("errore nell'apertura file %s \n", filename);
			file_logc << "\n";
			file_logc << "errore nell apertura file" << filename;
			exit(-1);
		}
		else {
			load_geometry(FE_geom, fileiic);
			fileiic.close();
		}
	}
	catch (std::exception& e) {
		printf("%s", e.what());
	}
}

/**
 * @brief Loads geometry parameters from an input stream.
 *
 * Reads detailed geometry data such as grid dimensions, shape flag, cell size,
 * and time integration settings. Optionally loads the shape mask from a file
 * if FLAG_SHAPE == 2.
 *
 * @param FE_geom Reference to FE_geometry structure to populate.
 * @param filei   Input file stream containing geometry parameters.
 */
void load_geometry(FE_geometry& FE_geom, std::ifstream& filei) {

	char shapeFile[N_BUFFER];
	char temp[N_BUFFER], temp2[N_BUFFER];
	int i = 0, j, check;

	do {
		filei.getline(temp2, N_BUFFER);

		if (strcmp(temp2, "//Geometry Parameters") == 0) {

			filei.getline(temp, N_BUFFER, '!');  // Read FLAG_STUDY
			FE_geom.FLAG_STUDY = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Read N_d (domain dimension multiplier)
			FE_geom.N_d = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Read Ncx
			FE_geom.Ncx = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Read Ncy
			FE_geom.Ncy = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Read Ncz
			FE_geom.Ncz = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Read FLAG_SHAPE
			FE_geom.FLAG_SHAPE = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Read shape filename or value

			// Remove spaces/tabs from filename string
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(shapeFile, temp, N_BUFFER);

			int N = FE_geom.Ncx * FE_geom.Ncy * FE_geom.Ncz * FE_geom.N_d;

			// Shape configuration based on FLAG_SHAPE
			if (FE_geom.FLAG_SHAPE == 1) {
				// Uniform shape
				FE_geom.shape.resize(N);
				thrust::fill(FE_geom.shape.begin(), FE_geom.shape.end(), 1);
			}
			else if (FE_geom.FLAG_SHAPE == 2) {
				// Shape loaded from file
				FE_geom.shape.resize(N);
				inputValues<int>(shapeFile, FE_geom.shape);
			}
			else if (FE_geom.FLAG_SHAPE == 3) {
				// Empty shape allocation
				FE_geom.shape.resize(N);
			}
			else {
				printf("\n The value of FLAG_SHAPE is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_SHAPE is not valid!";
				exit(-1);
			}

			// Spatial step sizes
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			FE_geom.delta_x = atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			FE_geom.delta_y = atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			FE_geom.delta_z = atof(temp);

			// Thickness
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			FE_geom.h = atof(temp);

			// Simulation time and timestep
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			FE_geom.time_sim = atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			FE_geom.dtime = atof(temp);

			// Stop condition
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			FE_geom.stop_sim = atof(temp);

			// Integration scheme flag
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			FE_geom.FLAG_INTEGRATOR = atoi(temp);

			// Validate integrator flag
			if (!(FE_geom.FLAG_INTEGRATOR == 0 || FE_geom.FLAG_INTEGRATOR == 1 ||
				FE_geom.FLAG_INTEGRATOR == 2 || FE_geom.FLAG_INTEGRATOR == 3 ||
				FE_geom.FLAG_INTEGRATOR == 4 || FE_geom.FLAG_INTEGRATOR == 5)) {
				printf("\n The value of FLAG_INTEGRATOR is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_INTEGRATOR is not valid!";
				exit(-1);
			}

			// Adaptive timestep start flag
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			FE_geom.start_adapting = atoi(temp);

			// Energy computation flag
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			FE_geom.FLAG_ENERGY = atoi(temp);

			// Validate energy flag
			if (!(FE_geom.FLAG_ENERGY == 0 || FE_geom.FLAG_ENERGY == 1)) {
				printf("\n The value of FLAG_ENERGY is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_ENERGY is not valid!";
				exit(-1);
			}

			// Energy logging interval
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			FE_geom.interval_energy = atoi(temp);

			// Adaptive integration error tolerance
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			FE_geom.error_tolerance = atof(temp);

			// Maximum time scaling for dt
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			FE_geom.max_dt_times = atof(temp);

			// Minimum time scaling for dt
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			FE_geom.min_dt_times = atof(temp);

			// Total number of elements
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			FE_geom.n_elements = atoi(temp);
		}

	} while (!filei.eof());
}


/**
 * @brief Loads Landau free energy configuration from predefined input file.
 *
 * This function opens the "Landau_free.dat" configuration file, which defines
 * the Landau–Ginzburg energy expansion coefficients, noise parameters, and
 * study flags for parallel simulations. It calls `load_Landau_free()` to parse
 * and populate the `landau_free` structure accordingly.
 *
 * @param landau_free_param Reference to the landau_free structure where parameters are stored.
 * @param N                 Total number of simulation elements.
 */
void load_Landau_configuration(landau_free& landau_free_param, int N) {
	const char* filename = "./file_configuration/Landau_free.dat";
	try {
		fileiic.open(filename, std::ifstream::in);

		if (fileiic.is_open() == false) {
			printf("errore nell'apertura file %s \n", filename);
			file_logc << "\n";
			file_logc << "errore nell apertura file" << filename;
			exit(-1);
		}
		else {
			load_Landau_free(landau_free_param, fileiic, N);
			fileiic.close();
		}
	}
	catch (std::exception& e) {
		printf("%s", e.what());
	}
}

/**
 * @brief Loads Landau free energy parameters from an input stream.
 *
 * Reads coefficients α₀–α₆, spontaneous polarization, temperature dependence,
 * relaxation, and related study/noise configuration flags. Depending on the
 * selected flag, parameters may be uniform, vary along z, or be spatially
 * non-uniform (read from external data files).
 *
 * @param landau_free_param Reference to the landau_free structure to populate.
 * @param filei             Input file stream containing Landau parameter data.
 * @param N                 Total number of simulation elements.
 */
void load_Landau_free(landau_free& landau_free_param, std::ifstream& filei, int N) {

	// --- Declare and initialize Landau coefficients and auxiliary buffers ---
	Type_var alpha0_constant = 0.0;
	char alpha0_z_variable[N_BUFFER];
	char alpha0_Non_Uniform[N_BUFFER];
	char alpha0_study[N_BUFFER];

	Type_var alpha2_constant = 0.0;
	char alpha2_z_variable[N_BUFFER];
	char alpha2_Non_Uniform[N_BUFFER];
	char alpha2_study[N_BUFFER];

	Type_var alpha3_constant = 0.0;
	char alpha3_z_variable[N_BUFFER];
	char alpha3_Non_Uniform[N_BUFFER];
	char alpha3_study[N_BUFFER];

	Type_var alpha4_constant = 0.0;
	char alpha4_z_variable[N_BUFFER];
	char alpha4_Non_Uniform[N_BUFFER];
	char alpha4_study[N_BUFFER];

	Type_var alpha5_constant = 0.0;
	char alpha5_z_variable[N_BUFFER];
	char alpha5_Non_Uniform[N_BUFFER];
	char alpha5_study[N_BUFFER];

	Type_var alpha6_constant = 0.0;
	char alpha6_z_variable[N_BUFFER];
	char alpha6_Non_Uniform[N_BUFFER];
	char alpha6_study[N_BUFFER];

	Vec3<Type_var> spontanaeousP = Vec3<Type_var>(0.0, 0.0, 0.0);

	Type_var alpha1_ref = 0.0;

	Type_var susciptibilityWeight = 0.0;
	char susciptibilityWeight_z_variable[N_BUFFER];
	char susciptibilityWeight_non_uniform[N_BUFFER];

	Type_var temperature_val = 0.0;
	char temperature_z_variable[N_BUFFER];
	char temperature_non_uniform[N_BUFFER];

	Type_var temperature_noise_val = 0.0;
	char temperature_noise_z_variable[N_BUFFER];
	char temperature_noise_non_uniform[N_BUFFER];

	Type_var TransitionTemp_val = 0.0;
	char transitionTemp_z_variable[N_BUFFER];
	char transitionTemp_non_uniform[N_BUFFER];

	Type_var relaxation_val;
	char relaxation_z_variable[N_BUFFER];
	char relaxation_non_uniform[N_BUFFER];

	Type_var uni_anis_z;
	char anis_z_variable[N_BUFFER];
	char anis_non_uniform[N_BUFFER];
	char anis_study[N_BUFFER];

	char temp[N_BUFFER], temp2[N_BUFFER];
	int i = 0, j, check;

	do {
		filei.getline(temp2, N_BUFFER);

		if (strcmp(temp2, "//Landau Free parameters") == 0) {

			// --- General flags and noise setup ---
			filei.getline(temp, N_BUFFER, '!');  // FLAG_LANDAU
			landau_free_param.FLAG_LANDAU = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // FLAG_NOISE
			landau_free_param.FLAG_NOISE = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // steps_noise
			landau_free_param.steps_noise = atoi(temp);

			// --- Noise mode selection ---
			if (landau_free_param.FLAG_NOISE == 0) {
				printf("\n Simulation is Without initial or dynamic noise \n");
			}
			else if (landau_free_param.FLAG_NOISE == 1) {
				printf("\n Simulation is considering noise at first step \n");
			}
			else if (landau_free_param.FLAG_NOISE == 2) {
				printf("\n Simulation is considering dynamic noise \n");
			}
			else {
				printf("\n The value of FLAG_NOISE is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_NOISE is not valid!";
				exit(-1);
			}

			// --- Study mode flag (controls parametric simulation runs) ---
			if (FE_geom.FLAG_STUDY == 0) {
				printf("\n Running a single simulation \n");
			}
			else if (FE_geom.FLAG_STUDY == 1) {
				printf("\n Running multiple simulations in parallel to study alpha0 \n");
			}
			else if (FE_geom.FLAG_STUDY == 2) {
				printf("\n Running multiple simulations in parallel to study alpha11 \n");
			}
			else if (FE_geom.FLAG_STUDY == 3) {
				printf("\n Running multiple simulations in parallel to study alpha12 \n");
			}
			else if (FE_geom.FLAG_STUDY == 4) {
				printf("\n Running multiple simulations in parallel to study alpha111 \n");
			}
			else if (FE_geom.FLAG_STUDY == 5) {
				printf("\n Running multiple simulations in parallel to study alpha112 \n");
			}
			else if (FE_geom.FLAG_STUDY == 6) {
				printf("\n Running multiple simulations in parallel to study alpha123 \n");
			}
			else {
				printf("\n The value of FLAG_STUDY is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_STUDY is not valid!";
				exit(-1);
			}

			// --- α0 coefficient setup ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // FLAG_ALPHA0
			landau_free_param.FLAG_ALPHA0 = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // α0 constant
			alpha0_constant = (Type_var)atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // z-variable file name

			// Remove spaces/tabs from file name
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(alpha0_z_variable, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Non-uniform file name

			// Remove spaces/tabs again
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(alpha0_Non_Uniform, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Study file name

			// Remove spaces/tabs once more
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(alpha0_study, temp, N_BUFFER);

			// --- Load α0 according to flag type ---
			if (landau_free_param.FLAG_ALPHA0 == 1) {
				// Uniform constant
				landau_free_param.d_alpha0.resize(1);
				landau_free_param.d_alpha0[0] = alpha0_constant;
			}
			else if (landau_free_param.FLAG_ALPHA0 == 2) {
				// z-dependent
				landau_free_param.d_alpha0.resize(FE_geom.Ncz);
				inputValues<Type_var>(alpha0_z_variable, landau_free_param.d_alpha0);
			}
			else if (landau_free_param.FLAG_ALPHA0 == 3) {
				// Fully non-uniform (spatial map)
				landau_free_param.d_alpha0.resize(N);
				inputValues<Type_var>(alpha0_Non_Uniform, landau_free_param.d_alpha0);
			}
			else if (landau_free_param.FLAG_ALPHA0 == 4) {
				// Study parameter (multiple simulations)
				landau_free_param.d_alpha0.resize(FE_geom.N_d);
				inputValues<Type_var>(alpha0_study, landau_free_param.d_alpha0);
			}
			else {
				printf("\n The value of FLAG_ALPHA0 is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_ALPHA0 is not valid!";
				exit(-1);
			}
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			landau_free_param.FLAG_ALPHA2 = atoi(temp);  // Read alpha2 flag

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			alpha2_constant = atof(temp);  // Read alpha2 constant value

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Remove whitespace characters (spaces, tabs) from input string for z-variable filename */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(alpha2_z_variable, temp, N_BUFFER);  // Copy cleaned z-variable filename

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Remove whitespace for non-uniform filename */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(alpha2_Non_Uniform, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Remove whitespace for study filename */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(alpha2_study, temp, N_BUFFER);

			/** Assign values to alpha2 parameter based on mode flag */
			if (landau_free_param.FLAG_ALPHA2 == 1) {
				landau_free_param.d_alpha2.resize(1);
				landau_free_param.d_alpha2[0] = alpha2_constant;  // Constant mode
			}
			else if (landau_free_param.FLAG_ALPHA2 == 2) {
				landau_free_param.d_alpha2.resize(FE_geom.Ncz);
				inputValues<Type_var>(alpha2_z_variable, landau_free_param.d_alpha2);  // z-dependent mode
			}
			else if (landau_free_param.FLAG_ALPHA2 == 3) {
				landau_free_param.d_alpha2.resize(N);
				inputValues<Type_var>(alpha2_Non_Uniform, landau_free_param.d_alpha2);  // Non-uniform mode
			}
			else if (landau_free_param.FLAG_ALPHA2 == 4) {
				landau_free_param.d_alpha2.resize(FE_geom.N_d);
				inputValues<Type_var>(alpha2_study, landau_free_param.d_alpha2);  // Study mode
			}
			else {
				printf("\n The value of FLAG_ALPHA2 is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_ALPHA2 is not valid!";
				exit(-1);
			}

			/**
			 * @brief Reads alpha3 parameter configuration from file and assigns corresponding values
			 *        to landau_free_param based on FLAG_ALPHA3 mode.
			 */
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			landau_free_param.FLAG_ALPHA3 = atoi(temp);  // Read alpha3 flag

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			alpha3_constant = atof(temp);  // Read alpha3 constant value

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Remove whitespace characters from input string for z-variable filename */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(alpha3_z_variable, temp, N_BUFFER);  // Copy cleaned z-variable filename

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Remove whitespace for non-uniform filename */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(alpha3_Non_Uniform, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Remove whitespace for study filename */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(alpha3_study, temp, N_BUFFER);

			/** Assign values to alpha3 parameter based on mode flag */
			if (landau_free_param.FLAG_ALPHA3 == 1) {
				landau_free_param.d_alpha3.resize(1);
				landau_free_param.d_alpha3[0] = alpha3_constant;  // Constant mode
			}
			else if (landau_free_param.FLAG_ALPHA3 == 2) {
				landau_free_param.d_alpha3.resize(FE_geom.Ncz);
				inputValues<Type_var>(alpha3_z_variable, landau_free_param.d_alpha3);  // z-dependent mode
			}
			else if (landau_free_param.FLAG_ALPHA3 == 3) {
				landau_free_param.d_alpha3.resize(N);
				inputValues<Type_var>(alpha3_Non_Uniform, landau_free_param.d_alpha3);  // Non-uniform mode
			}
			else if (landau_free_param.FLAG_ALPHA3 == 4) {
				landau_free_param.d_alpha3.resize(FE_geom.N_d);
				inputValues<Type_var>(alpha3_study, landau_free_param.d_alpha3);  // Study mode
			}
			else {
				printf("\n The value of FLAG_ALPHA3 is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_ALPHA3 is not valid!";
				exit(-1);
			}

			/**
			 * @brief Reads alpha4 parameter configuration header and constant value (setup continues below).
			 */
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			landau_free_param.FLAG_ALPHA4 = atoi(temp);  // Read alpha4 flag

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			alpha4_constant = atof(temp);  // Read alpha4 constant value

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Continue alpha4 parsing (next section)

			/**
 * @brief Clean input string for alpha4 z-variable filename (remove whitespace)
 */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(alpha4_z_variable, temp, N_BUFFER);  // Store cleaned z-variable name

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/**
			 * @brief Clean input string for alpha4 non-uniform filename (remove whitespace)
			 */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(alpha4_Non_Uniform, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/**
			 * @brief Clean input string for alpha4 study filename (remove whitespace)
			 */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(alpha4_study, temp, N_BUFFER);

			/**
			 * @brief Assign alpha4 values based on mode flag
			 */
			if (landau_free_param.FLAG_ALPHA4 == 1) {
				landau_free_param.d_alpha4.resize(1);
				landau_free_param.d_alpha4[0] = alpha4_constant;  // Constant mode
			}
			else if (landau_free_param.FLAG_ALPHA4 == 2) {
				landau_free_param.d_alpha4.resize(FE_geom.Ncz);
				inputValues<Type_var>(alpha4_z_variable, landau_free_param.d_alpha4);  // z-dependent mode
			}
			else if (landau_free_param.FLAG_ALPHA4 == 3) {
				landau_free_param.d_alpha4.resize(N);
				inputValues<Type_var>(alpha4_Non_Uniform, landau_free_param.d_alpha4);  // Non-uniform mode
			}
			else if (landau_free_param.FLAG_ALPHA4 == 4) {
				landau_free_param.d_alpha4.resize(FE_geom.N_d);
				inputValues<Type_var>(alpha4_study, landau_free_param.d_alpha4);  // Study mode
			}
			else {
				printf("\n The value of FLAG_ALPHA4 is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_ALPHA4 is not valid!";
				exit(-1);
			}

			/**
			 * @brief Reads alpha5 parameter configuration and assigns based on mode flag
			 */
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			landau_free_param.FLAG_ALPHA5 = atoi(temp);  // Read alpha5 flag

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			alpha5_constant = atof(temp);  // Read constant alpha5 value

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Clean z-variable filename for alpha5 */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(alpha5_z_variable, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Clean non-uniform filename for alpha5 */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(alpha5_Non_Uniform, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Clean study filename for alpha5 */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(alpha5_study, temp, N_BUFFER);

			/** Assign alpha5 values based on mode flag */
			if (landau_free_param.FLAG_ALPHA5 == 1) {
				landau_free_param.d_alpha5.resize(1);
				landau_free_param.d_alpha5[0] = alpha5_constant;  // Constant mode
			}
			else if (landau_free_param.FLAG_ALPHA5 == 2) {
				landau_free_param.d_alpha5.resize(FE_geom.Ncz);
				inputValues<Type_var>(alpha5_z_variable, landau_free_param.d_alpha5);  // z-dependent mode
			}
			else if (landau_free_param.FLAG_ALPHA5 == 3) {
				landau_free_param.d_alpha5.resize(N);
				inputValues<Type_var>(alpha5_Non_Uniform, landau_free_param.d_alpha5);  // Non-uniform mode
			}
			else if (landau_free_param.FLAG_ALPHA5 == 4) {
				landau_free_param.d_alpha5.resize(FE_geom.N_d);
				inputValues<Type_var>(alpha5_study, landau_free_param.d_alpha5);  // Study mode
			}
			else {
				printf("\n The value of FLAG_ALPHA5 is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_ALPHA5 is not valid!";
				exit(-1);
			}

			/**
			 * @brief Reads alpha6 parameter configuration and assigns based on mode flag
			 */
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			landau_free_param.FLAG_ALPHA6 = atoi(temp);  // Read alpha6 flag

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			alpha6_constant = atof(temp);  // Read alpha6 constant

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Clean z-variable filename for alpha6 */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(alpha6_z_variable, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Clean non-uniform filename for alpha6 */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(alpha6_Non_Uniform, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Clean study filename for alpha6 */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(alpha6_study, temp, N_BUFFER);
			/**
			 * @brief Assign alpha6 values based on mode flag
			 */
			if (landau_free_param.FLAG_ALPHA6 == 1) {
				landau_free_param.d_alpha6.resize(1);
				landau_free_param.d_alpha6[0] = alpha6_constant;  // Constant mode
			}
			else if (landau_free_param.FLAG_ALPHA6 == 2) {
				landau_free_param.d_alpha6.resize(FE_geom.Ncz);
				inputValues<Type_var>(alpha6_z_variable, landau_free_param.d_alpha6);  // z-dependent mode
			}
			else if (landau_free_param.FLAG_ALPHA6 == 3) {
				landau_free_param.d_alpha6.resize(N);
				inputValues<Type_var>(alpha6_Non_Uniform, landau_free_param.d_alpha6);  // Non-uniform mode
			}
			else if (landau_free_param.FLAG_ALPHA6 == 4) {
				landau_free_param.d_alpha6.resize(FE_geom.N_d);
				inputValues<Type_var>(alpha6_Non_Uniform, landau_free_param.d_alpha6);  // Study mode
			}
			else {
				printf("\n The value of FLAG_ALPHA6 is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_ALPHA6 is not valid!";
				exit(-1);
			}

			/**
			 * @brief Load spontaneous polarization vector components
			 */
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			spontanaeousP.x = atof(temp);  // Px

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			spontanaeousP.y = atof(temp);  // Py

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			spontanaeousP.z = atof(temp);  // Pz

			landau_free_param.d_spontaneousp_ref.resize(1);
			landau_free_param.d_spontaneousp_ref[0] = spontanaeousP;

			/**
			 * @brief Load reference alpha1 value
			 */
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			alpha1_ref = atof(temp);

			landau_free_param.d_alpha1_ref.resize(1);
			landau_free_param.d_alpha1_ref[0] = alpha1_ref;

			/**
			 * @brief Load susceptibility weight configuration
			 */
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			landau_free_param.FLAG_SUSCIPTIBILITY_WEIGHTS = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			susciptibilityWeight = atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!'); //modifica del 15 Dicembre 2021

			/** Clean z-variable filename for susceptibility weights */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(susciptibilityWeight_z_variable, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Clean non-uniform filename for susceptibility weights */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(susciptibilityWeight_non_uniform, temp, N_BUFFER);

			/** Assign susceptibility weights based on mode flag */
			if (landau_free_param.FLAG_SUSCIPTIBILITY_WEIGHTS == 1) {
				landau_free_param.d_susciptibilityWeights.resize(1);
				landau_free_param.d_susciptibilityWeights[0] = susciptibilityWeight;  // Constant mode
			}
			else if (landau_free_param.FLAG_SUSCIPTIBILITY_WEIGHTS == 2) {
				landau_free_param.d_susciptibilityWeights.resize(FE_geom.Ncz);
				inputValues<Type_var>(susciptibilityWeight_z_variable, landau_free_param.d_susciptibilityWeights);  // z-dependent mode
			}
			else if (landau_free_param.FLAG_SUSCIPTIBILITY_WEIGHTS == 3) {
				landau_free_param.d_susciptibilityWeights.resize(N);
				inputValues<Type_var>(susciptibilityWeight_non_uniform, landau_free_param.d_susciptibilityWeights);  // Non-uniform mode
			}
			else {
				printf("\n The value of FLAG_SUSCIPTIBILITY_WEIGHTS is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_SUSCIPTIBILITY_WEIGHTS is not valid!";
				exit(-1);
			}

			/**
			 * @brief Load temperature configuration
			 */
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			landau_free_param.FLAG_TEMPERATURE = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			temperature_val = atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!'); //modifica del 15 Dicembre 2021

			/** Clean z-variable filename for temperature */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(temperature_z_variable, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Clean non-uniform filename for temperature */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(temperature_non_uniform, temp, N_BUFFER);

			/** Assign temperature values based on mode flag */
			if (landau_free_param.FLAG_TEMPERATURE == 1) {
				landau_free_param.d_temperature.resize(1);
				landau_free_param.d_temperature[0] = temperature_val;
			}
			else if (landau_free_param.FLAG_TEMPERATURE == 2) {
				landau_free_param.d_temperature.resize(FE_geom.Ncz);
				inputValues<Type_var>(temperature_z_variable, landau_free_param.d_temperature);
			}
			else if (landau_free_param.FLAG_TEMPERATURE == 3) {
				landau_free_param.d_temperature.resize(N);
				inputValues<Type_var>(temperature_non_uniform, landau_free_param.d_temperature);
			}
			else {
				printf("\n The value of FLAG_TEMPERATURE is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_TEMPERATURE is not valid!";
				exit(-1);
			}

			/**
		 * @brief Load temperature_noise configuration
		 */
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			landau_free_param.FLAG_TEMPERATURE_NOISE = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			temperature_noise_val = atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!'); //modifica del 15 Dicembre 2021

			/** Clean z-variable filename for temperature */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(temperature_noise_z_variable, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Clean non-uniform filename for temperature */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(temperature_noise_non_uniform, temp, N_BUFFER);

			/** Assign temperature values based on mode flag */
			if (landau_free_param.FLAG_TEMPERATURE_NOISE == 1) {
				landau_free_param.d_temperature_noise.resize(1);
				landau_free_param.d_temperature_noise[0] = temperature_noise_val;
			}
			else if (landau_free_param.FLAG_TEMPERATURE_NOISE == 2) {
				landau_free_param.d_temperature_noise.resize(FE_geom.Ncz);
				inputValues<Type_var>(temperature_noise_z_variable, landau_free_param.d_temperature_noise);
			}
			else if (landau_free_param.FLAG_TEMPERATURE_NOISE == 3) {
				landau_free_param.d_temperature_noise.resize(N);
				inputValues<Type_var>(temperature_noise_non_uniform, landau_free_param.d_temperature_noise);
			}
			else {
				printf("\n The value of FLAG_TEMPERATURE_NOISE is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_TEMPERATURE_NOISE is not valid!";
				exit(-1);
			}

			/**
			 * @brief Load transition temperature configuration
			 */
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			landau_free_param.FLAG_TRANSITION_TEMP = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			TransitionTemp_val = atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Clean z-variable filename for transition temperature */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(transitionTemp_z_variable, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Clean non-uniform filename for transition temperature */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(transitionTemp_non_uniform, temp, N_BUFFER);

			/** Assign transition temperature based on mode flag */
			if (landau_free_param.FLAG_TRANSITION_TEMP == 1) {
				landau_free_param.d_transition_temp.resize(1);
				landau_free_param.d_transition_temp[0] = TransitionTemp_val;
			}
			else if (landau_free_param.FLAG_TRANSITION_TEMP == 2) {
				landau_free_param.d_transition_temp.resize(FE_geom.Ncz);
				inputValues<Type_var>(transitionTemp_z_variable, landau_free_param.d_transition_temp);
			}
			else if (landau_free_param.FLAG_TEMPERATURE == 3) {
				landau_free_param.d_transition_temp.resize(N);
				inputValues<Type_var>(transitionTemp_non_uniform, landau_free_param.d_transition_temp);
			}
			else {
				printf("\n The value of FLAG_TRANSITION_TEMP is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_TRANSITION_TEMP is not valid!";
				exit(-1);
			}

			/**
			 * @brief Load relaxation configuration
			 */
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			landau_free_param.FLAG_RELAXATION = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			relaxation_val = atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Clean z-variable filename for relaxation */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(relaxation_z_variable, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Clean non-uniform filename for relaxation */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(relaxation_non_uniform, temp, N_BUFFER);

			/** Assign relaxation values based on mode flag */
			if (landau_free_param.FLAG_RELAXATION == 1) {
				landau_free_param.d_relaxation.resize(1);
				landau_free_param.d_relaxation[0] = relaxation_val;
			}
			else if (landau_free_param.FLAG_RELAXATION == 2) {
				landau_free_param.d_relaxation.resize(FE_geom.Ncz);
				inputValues<Type_var>(relaxation_z_variable, landau_free_param.d_relaxation);
			}
			else if (landau_free_param.FLAG_RELAXATION == 3) {
				landau_free_param.d_relaxation.resize(N);
				inputValues<Type_var>(relaxation_non_uniform, landau_free_param.d_relaxation);
			}
			else {
				printf("\n The value of FLAG_RELAXATION is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_RELAXATION is not valid!";
				exit(-1);
			}

			/**
			 * @brief Load relaxation configuration
			 */
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			landau_free_param.FLAG_UNI_ANIS_Z = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			uni_anis_z = atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Clean z-variable filename for relaxation */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(anis_z_variable, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Clean non-uniform filename for relaxation */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(anis_non_uniform, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Clean study filename for alpha5 */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(anis_study, temp, N_BUFFER);

			/** Assign relaxation values based on mode flag */
			if (landau_free_param.FLAG_UNI_ANIS_Z == 1) {
				landau_free_param.uni_anistropy_z.resize(1);
				landau_free_param.uni_anistropy_z[0] = uni_anis_z;
			}
			else if (landau_free_param.FLAG_UNI_ANIS_Z == 2) {
				landau_free_param.uni_anistropy_z.resize(FE_geom.Ncz);
				inputValues<Type_var>(anis_z_variable, landau_free_param.uni_anistropy_z);
			}
			else if (landau_free_param.FLAG_UNI_ANIS_Z == 3) {
				landau_free_param.uni_anistropy_z.resize(N);
				inputValues<Type_var>(anis_non_uniform, landau_free_param.uni_anistropy_z);
			}
			else if (landau_free_param.FLAG_UNI_ANIS_Z == 4) {
				landau_free_param.uni_anistropy_z.resize(FE_geom.N_d);
				inputValues<Type_var>(anis_study, landau_free_param.uni_anistropy_z);
			}
			else {
				printf("\n The value of FLAG_UNI_ANIS_Z is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_UNI_ANIS_Z is not valid!";
				exit(-1);
			}

			/**
			 * @brief Load reference relaxation value
			 */
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			landau_free_param.d_relaxation_ref.resize(1);
			landau_free_param.d_relaxation_ref[0] = atof(temp);
		}


	} while (!filei.eof());
}



/**
 * @brief Loads elastic field configuration from predefined input file.
 *
 * This function opens the "elasic_field.dat" configuration file
 */
void load_elastic_configuration(elastic &elastic_field_param, int N) {
	const char* filename = "./file_configuration/elastic_field.dat";
	try {
		fileiic.open(filename, std::ifstream::in);

		if (fileiic.is_open() == false) {
			printf("errore nell'apertura file %s \n", filename);
			file_logc << "\n";
			file_logc << "errore nell apertura file" << filename;
			exit(-1);
		}
		else {
			load_elastic(elastic_field_param, fileiic, N);
			fileiic.close();
		}
	}
	catch (std::exception& e) {
		printf("%s", e.what());
	}
}




void load_elastic(elastic& elastic_field_param, std::ifstream& filei, int N) {

	// --- Declare and initialize elastic coefficients and auxiliary buffers ---
	Type_var Q_11 = 0.0;
	char Q_11_z_variable[N_BUFFER];
	char Q_11_Non_Uniform[N_BUFFER];
	char Q_11_study[N_BUFFER];

	Type_var Q_12 = 0.0;
	char Q_12_z_variable[N_BUFFER];
	char Q_12_Non_Uniform[N_BUFFER];
	char Q_12_study[N_BUFFER];

	Type_var Q_44 = 0.0;
	char Q_44_z_variable[N_BUFFER];
	char Q_44_Non_Uniform[N_BUFFER];
	char Q_44_study[N_BUFFER];

	Type_var C_11 = 0.0;
	char C_11_z_variable[N_BUFFER];
	char C_11_Non_Uniform[N_BUFFER];
	char C_11_study[N_BUFFER];

	Type_var C_12 = 0.0;
	char C_12_z_variable[N_BUFFER];
	char C_12_Non_Uniform[N_BUFFER];
	char C_12_study[N_BUFFER];

	Type_var C_44 = 0.0;
	char C_44_z_variable[N_BUFFER];
	char C_44_Non_Uniform[N_BUFFER];
	char C_44_study[N_BUFFER];

	Type_var sigma_xx = 0.0;
	char sigma_xx_z_variable[N_BUFFER];
	char sigma_xx_Non_Uniform[N_BUFFER];
	char sigma_xx_study[N_BUFFER];

	Type_var sigma_yy = 0.0;
	char sigma_yy_z_variable[N_BUFFER];
	char sigma_yy_Non_Uniform[N_BUFFER];
	char sigma_yy_study[N_BUFFER];

	Type_var sigma_zz = 0.0;
	char sigma_zz_z_variable[N_BUFFER];
	char sigma_zz_Non_Uniform[N_BUFFER];
	char sigma_zz_study[N_BUFFER];

	Type_var sigma_xy = 0.0;
	char sigma_xy_z_variable[N_BUFFER];
	char sigma_xy_Non_Uniform[N_BUFFER];
	char sigma_xy_study[N_BUFFER];


	Type_var sigma_xz = 0.0;
	char sigma_xz_z_variable[N_BUFFER];
	char sigma_xz_Non_Uniform[N_BUFFER];
	char sigma_xz_study[N_BUFFER];

	Type_var sigma_yz = 0.0;
	char sigma_yz_z_variable[N_BUFFER];
	char sigma_yz_Non_Uniform[N_BUFFER];
	char sigma_yz_study[N_BUFFER];

	

	char temp[N_BUFFER], temp2[N_BUFFER];
	int i = 0, j, check;

	do {
		filei.getline(temp2, N_BUFFER);

		if (strcmp(temp2, "//Elastic Field parameters") == 0) {

			// --- General flags setup ---
			filei.getline(temp, N_BUFFER, '!');  // FLAG_ELASTIC
			elastic_field_param.FLAG_ELASTIC = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // FLAG_DISPLACEMENT_FIELD
			elastic_field_param.FLAG_DISPLACEMENT_FIELD = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // FLAG_STRESS 
			elastic_field_param.FLAG_POLARIZATION_FIELD = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // FLAG_STRESS 
			elastic_field_param.FLAG_STRESS = atoi(temp);

			
			// --- Q11 coefficient setup ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // FLAG_Q11
			elastic_field_param.FLAG_Q11 = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Q_11 constant
			Q_11 = (Type_var)atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // z-variable file name

			// Remove spaces/tabs from file name
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(Q_11_z_variable, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Non-uniform file name

			// Remove spaces/tabs again
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(Q_11_Non_Uniform, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Study file name

			// Remove spaces/tabs once more
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(Q_11_study, temp, N_BUFFER);

			// --- Load Q11 according to flag type ---
			if (elastic_field_param.FLAG_Q11 == 1) {
				// Uniform constant
				elastic_field_param.Q11.resize(1);
				elastic_field_param.Q11[0] = Q_11;
			}
			else if (elastic_field_param.FLAG_Q11 == 2) {
				// z-dependent
				elastic_field_param.Q11.resize(FE_geom.Ncz);
				inputValues<Type_var>(Q_11_z_variable, elastic_field_param.Q11);
			}
			else if (elastic_field_param.FLAG_Q11 == 3) {
				// Fully non-uniform (spatial map)
				elastic_field_param.Q11.resize(N);
				inputValues<Type_var>(Q_11_Non_Uniform, elastic_field_param.Q11);
			}
			else if (elastic_field_param.FLAG_Q11 == 4) {
				// Study parameter (multiple simulations)
				elastic_field_param.Q11.resize(FE_geom.N_d);
				inputValues<Type_var>(Q_11_study, elastic_field_param.Q11);
			}
			else {
				printf("\n The value of FLAG_Q11 is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_Q11 is not valid!";
				exit(-1);
			}


			// --- Q12 coefficient setup ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // FLAG_Q12
			elastic_field_param.FLAG_Q12 = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Q_12 constant
			Q_12 = (Type_var)atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // z-variable file name

			// Remove spaces/tabs from file name
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(Q_12_z_variable, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Non-uniform file name

			// Remove spaces/tabs again
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(Q_12_Non_Uniform, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Study file name

			// Remove spaces/tabs once more
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(Q_12_study, temp, N_BUFFER);

			// --- Load Q12 according to flag type ---
			if (elastic_field_param.FLAG_Q12 == 1) {
				// Uniform constant
				elastic_field_param.Q12.resize(1);
				elastic_field_param.Q12[0] = Q_11;
			}
			else if (elastic_field_param.FLAG_Q12 == 2) {
				// z-dependent
				elastic_field_param.Q12.resize(FE_geom.Ncz);
				inputValues<Type_var>(Q_12_z_variable, elastic_field_param.Q12);
			}
			else if (elastic_field_param.FLAG_Q12 == 3) {
				// Fully non-uniform (spatial map)
				elastic_field_param.Q12.resize(N);
				inputValues<Type_var>(Q_12_Non_Uniform, elastic_field_param.Q12);
			}
			else if (elastic_field_param.FLAG_Q12 == 4) {
				// Study parameter (multiple simulations)
				elastic_field_param.Q12.resize(FE_geom.N_d);
				inputValues<Type_var>(Q_12_study, elastic_field_param.Q12);
			}
			else {
				printf("\n The value of FLAG_Q12 is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_Q12 is not valid!";
				exit(-1);
			}

			// --- Q44 coefficient setup ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // FLAG_Q44
			elastic_field_param.FLAG_Q44 = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Q_44 constant
			Q_44 = (Type_var)atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // z-variable file name

			// Remove spaces/tabs from file name
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(Q_44_z_variable, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Non-uniform file name

			// Remove spaces/tabs again
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(Q_44_Non_Uniform, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Study file name

			// Remove spaces/tabs once more
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(Q_44_study, temp, N_BUFFER);

			// --- Load Q44 according to flag type ---
			if (elastic_field_param.FLAG_Q44 == 1) {
				// Uniform constant
				elastic_field_param.Q44.resize(1);
				elastic_field_param.Q44[0] = Q_44;
			}
			else if (elastic_field_param.FLAG_Q44 == 2) {
				// z-dependent
				elastic_field_param.Q44.resize(FE_geom.Ncz);
				inputValues<Type_var>(Q_44_z_variable, elastic_field_param.Q44);
			}
			else if (elastic_field_param.FLAG_Q44 == 3) {
				// Fully non-uniform (spatial map)
				elastic_field_param.Q44.resize(N);
				inputValues<Type_var>(Q_44_Non_Uniform, elastic_field_param.Q44);
			}
			else if (elastic_field_param.FLAG_Q44 == 4) {
				// Study parameter (multiple simulations)
				elastic_field_param.Q44.resize(FE_geom.N_d);
				inputValues<Type_var>(Q_44_study, elastic_field_param.Q44);
			}
			else {
				printf("\n The value of FLAG_Q44 is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_Q44 is not valid!";
				exit(-1);
			}

			// --- C11 coefficient setup ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // FLAG_C11
			elastic_field_param.FLAG_C11 = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // C_11 constant
			C_11 = (Type_var)atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // z-variable file name

			// Remove spaces/tabs from file name
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(C_11_z_variable, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Non-uniform file name

			// Remove spaces/tabs again
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(C_11_Non_Uniform, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Study file name

			// Remove spaces/tabs once more
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(C_11_study, temp, N_BUFFER);

			// --- Load Q11 according to flag type ---
			if (elastic_field_param.FLAG_C11 == 1) {
				// Uniform constant
				elastic_field_param.C11.resize(1);
				elastic_field_param.C11[0] = C_11;
			}
			else if (elastic_field_param.FLAG_C11 == 2) {
				// z-dependent
				elastic_field_param.C11.resize(FE_geom.Ncz);
				inputValues<Type_var>(C_11_z_variable, elastic_field_param.C11);
			}
			else if (elastic_field_param.FLAG_C11 == 3) {
				// Fully non-uniform (spatial map)
				elastic_field_param.C11.resize(N);
				inputValues<Type_var>(C_11_Non_Uniform, elastic_field_param.C11);
			}
			else if (elastic_field_param.FLAG_C11 == 4) {
				// Study parameter (multiple simulations)
				elastic_field_param.C11.resize(FE_geom.N_d);
				inputValues<Type_var>(C_11_study, elastic_field_param.C11);
			}
			else {
				printf("\n The value of FLAG_C11 is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_C11 is not valid!";
				exit(-1);
			}

			// --- C12 coefficient setup ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // FLAG_C12
			elastic_field_param.FLAG_C12 = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // C_12 constant
			C_12 = (Type_var)atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // z-variable file name

			// Remove spaces/tabs from file name
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(C_12_z_variable, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Non-uniform file name

			// Remove spaces/tabs again
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(C_12_Non_Uniform, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Study file name

			// Remove spaces/tabs once more
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(C_12_study, temp, N_BUFFER);

			// --- Load C12 according to flag type ---
			if (elastic_field_param.FLAG_C12 == 1) {
				// Uniform constant
				elastic_field_param.C12.resize(1);
				elastic_field_param.C12[0] = C_11;
			}
			else if (elastic_field_param.FLAG_C12 == 2) {
				// z-dependent
				elastic_field_param.C12.resize(FE_geom.Ncz);
				inputValues<Type_var>(C_12_z_variable, elastic_field_param.C12);
			}
			else if (elastic_field_param.FLAG_C12 == 3) {
				// Fully non-uniform (spatial map)
				elastic_field_param.C12.resize(N);
				inputValues<Type_var>(C_12_Non_Uniform, elastic_field_param.C12);
			}
			else if (elastic_field_param.FLAG_C12 == 4) {
				// Study parameter (multiple simulations)
				elastic_field_param.C12.resize(FE_geom.N_d);
				inputValues<Type_var>(C_12_study, elastic_field_param.C12);
			}
			else {
				printf("\n The value of FLAG_C12 is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_C12 is not valid!";
				exit(-1);
			}

			// --- C44 coefficient setup ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // FLAG_C44
			elastic_field_param.FLAG_C44 = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // C_44 constant
			C_44 = (Type_var)atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // z-variable file name

			// Remove spaces/tabs from file name
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(C_44_z_variable, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Non-uniform file name

			// Remove spaces/tabs again
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(C_44_Non_Uniform, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Study file name

			// Remove spaces/tabs once more
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(C_44_study, temp, N_BUFFER);

			// --- Load C44 according to flag type ---
			if (elastic_field_param.FLAG_C44 == 1) {
				// Uniform constant
				elastic_field_param.C44.resize(1);
				elastic_field_param.C44[0] = C_44;
			}
			else if (elastic_field_param.FLAG_C44 == 2) {
				// z-dependent
				elastic_field_param.C44.resize(FE_geom.Ncz);
				inputValues<Type_var>(C_44_z_variable, elastic_field_param.C44);
			}
			else if (elastic_field_param.FLAG_C44 == 3) {
				// Fully non-uniform (spatial map)
				elastic_field_param.C44.resize(N);
				inputValues<Type_var>(C_44_Non_Uniform, elastic_field_param.C44);
			}
			else if (elastic_field_param.FLAG_C44 == 4) {
				// Study parameter (multiple simulations)
				elastic_field_param.C44.resize(FE_geom.N_d);
				inputValues<Type_var>(C_44_study, elastic_field_param.C44);
			}
			else {
				printf("\n The value of FLAG_C44 is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_C44 is not valid!";
				exit(-1);
			}


			// --- sigma_xx_ext setup ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // FLAG_SIGMA_XX
			elastic_field_param.FLAG_SIGMA_XX = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // sigma_xx_ext
			sigma_xx = (Type_var)atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // z-variable file name

			// Remove spaces/tabs from file name
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(sigma_xx_z_variable, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Non-uniform file name

			// Remove spaces/tabs again
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(sigma_xx_Non_Uniform, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Study file name

			// Remove spaces/tabs once more
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(sigma_xx_study, temp, N_BUFFER);

			// --- Load sigma_xx_ext according to flag type ---
			if (elastic_field_param.FLAG_SIGMA_XX == 1) {
				// Uniform constant
				elastic_field_param.sigma_xx_ext.resize(1);
				elastic_field_param.sigma_xx_ext[0] = sigma_xx;
			}
			else if (elastic_field_param.FLAG_SIGMA_XX == 2) {
				// z-dependent
				elastic_field_param.sigma_xx_ext.resize(FE_geom.Ncz);
				inputValues<Type_var>(sigma_xx_z_variable, elastic_field_param.sigma_xx_ext);
			}
			else if (elastic_field_param.FLAG_SIGMA_XX == 3) {
				// Fully non-uniform (spatial map)
				elastic_field_param.sigma_xx_ext.resize(N);
				inputValues<Type_var>(sigma_xx_Non_Uniform, elastic_field_param.sigma_xx_ext);
			}
			else if (elastic_field_param.FLAG_SIGMA_XX == 4) {
				// Study parameter (multiple simulations)
				elastic_field_param.sigma_xx_ext.resize(FE_geom.N_d);
				inputValues<Type_var>(sigma_xx_study, elastic_field_param.sigma_xx_ext);
			}
			else {
				printf("\n The value of FLAG_SIGMA_XX is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_SIGMA_XX is not valid!";
				exit(-1);
			}


			


			// --- sigma_yy_ext setup ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // FLAG_SIGMA_YY
			elastic_field_param.FLAG_SIGMA_YY = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // sigma_yy_ext
			sigma_yy = (Type_var)atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // z-variable file name

			// Remove spaces/tabs from file name
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(sigma_yy_z_variable, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Non-uniform file name

			// Remove spaces/tabs again
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(sigma_yy_Non_Uniform, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Study file name

			// Remove spaces/tabs once more
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(sigma_yy_study, temp, N_BUFFER);

			// --- Load sigma_yy_ext according to flag type ---
			if (elastic_field_param.FLAG_SIGMA_YY == 1) {
				// Uniform constant
				elastic_field_param.sigma_yy_ext.resize(1);
				elastic_field_param.sigma_yy_ext[0] = sigma_yy;
			}
			else if (elastic_field_param.FLAG_SIGMA_YY == 2) {
				// z-dependent
				elastic_field_param.sigma_yy_ext.resize(FE_geom.Ncz);
				inputValues<Type_var>(sigma_yy_z_variable, elastic_field_param.sigma_yy_ext);
			}
			else if (elastic_field_param.FLAG_SIGMA_YY == 3) {
				// Fully non-uniform (spatial map)
				elastic_field_param.sigma_yy_ext.resize(N);
				inputValues<Type_var>(sigma_yy_Non_Uniform, elastic_field_param.sigma_yy_ext);
			}
			else if (elastic_field_param.FLAG_SIGMA_YY == 4) {
				// Study parameter (multiple simulations)
				elastic_field_param.sigma_yy_ext.resize(FE_geom.N_d);
				inputValues<Type_var>(sigma_yy_study, elastic_field_param.sigma_yy_ext);
			}
			else {
				printf("\n The value of FLAG_SIGMA_YY is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_SIGMA_YY is not valid!";
				exit(-1);
			}


			// --- sigma_zz_ext setup ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // FLAG_SIGMA_ZZ
			elastic_field_param.FLAG_SIGMA_ZZ = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // sigma_zz_ext
			sigma_zz = (Type_var)atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // z-variable file name

			// Remove spaces/tabs from file name
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(sigma_zz_z_variable, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Non-uniform file name

			// Remove spaces/tabs again
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(sigma_zz_Non_Uniform, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Study file name

			// Remove spaces/tabs once more
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(sigma_zz_study, temp, N_BUFFER);

			// --- Load sigma_zz_ext according to flag type ---
			if (elastic_field_param.FLAG_SIGMA_ZZ == 1) {
				// Uniform constant
				elastic_field_param.sigma_zz_ext.resize(1);
				elastic_field_param.sigma_zz_ext[0] = sigma_zz;
			}
			else if (elastic_field_param.FLAG_SIGMA_ZZ == 2) {
				// z-dependent
				elastic_field_param.sigma_zz_ext.resize(FE_geom.Ncz);
				inputValues<Type_var>(sigma_zz_z_variable, elastic_field_param.sigma_zz_ext);
			}
			else if (elastic_field_param.FLAG_SIGMA_ZZ == 3) {
				// Fully non-uniform (spatial map)
				elastic_field_param.sigma_zz_ext.resize(N);
				inputValues<Type_var>(sigma_zz_Non_Uniform, elastic_field_param.sigma_zz_ext);
			}
			else if (elastic_field_param.FLAG_SIGMA_ZZ == 4) {
				// Study parameter (multiple simulations)
				elastic_field_param.sigma_zz_ext.resize(FE_geom.N_d);
				inputValues<Type_var>(sigma_zz_study, elastic_field_param.sigma_zz_ext);
			}
			else {
				printf("\n The value of FLAG_SIGMA_ZZ is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_SIGMA_ZZ is not valid!";
				exit(-1);
			}


			// --- sigma_xy_ext setup ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // FLAG_SIGMA_XY
			elastic_field_param.FLAG_SIGMA_XY = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // sigma_xy_ext
			sigma_xy = (Type_var)atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // z-variable file name

			// Remove spaces/tabs from file name
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(sigma_xy_z_variable, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Non-uniform file name

			// Remove spaces/tabs again
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(sigma_xy_Non_Uniform, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Study file name

			// Remove spaces/tabs once more
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(sigma_xy_study, temp, N_BUFFER);

			// --- Load sigma_xx_ext according to flag type ---
			if (elastic_field_param.FLAG_SIGMA_XY == 1) {
				// Uniform constant
				elastic_field_param.sigma_xy_ext.resize(1);
				elastic_field_param.sigma_xy_ext[0] = sigma_xy;
			}
			else if (elastic_field_param.FLAG_SIGMA_XY == 2) {
				// z-dependent
				elastic_field_param.sigma_xx_ext.resize(FE_geom.Ncz);
				inputValues<Type_var>(sigma_xy_z_variable, elastic_field_param.sigma_xy_ext);
			}
			else if (elastic_field_param.FLAG_SIGMA_XY == 3) {
				// Fully non-uniform (spatial map)
				elastic_field_param.sigma_xy_ext.resize(N);
				inputValues<Type_var>(sigma_xy_Non_Uniform, elastic_field_param.sigma_xy_ext);
			}
			else if (elastic_field_param.FLAG_SIGMA_XY == 4) {
				// Study parameter (multiple simulations)
				elastic_field_param.sigma_xy_ext.resize(FE_geom.N_d);
				inputValues<Type_var>(sigma_xy_study, elastic_field_param.sigma_xy_ext);
			}
			else {
				printf("\n The value of FLAG_SIGMA_XY is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_SIGMA_XY is not valid!";
				exit(-1);
			}


			// --- sigma_xz_ext setup ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // FLAG_SIGMA_XZ
			elastic_field_param.FLAG_SIGMA_XZ = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // sigma_xz_ext
			sigma_xz = (Type_var)atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // z-variable file name

			// Remove spaces/tabs from file name
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(sigma_xz_z_variable, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Non-uniform file name

			// Remove spaces/tabs again
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(sigma_xz_Non_Uniform, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Study file name

			// Remove spaces/tabs once more
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(sigma_xz_study, temp, N_BUFFER);

			// --- Load sigma_zz_ext according to flag type ---
			if (elastic_field_param.FLAG_SIGMA_XZ == 1) {
				// Uniform constant
				elastic_field_param.sigma_xz_ext.resize(1);
				elastic_field_param.sigma_xz_ext[0] = sigma_xz;
			}
			else if (elastic_field_param.FLAG_SIGMA_XZ == 2) {
				// z-dependent
				elastic_field_param.sigma_xz_ext.resize(FE_geom.Ncz);
				inputValues<Type_var>(sigma_xz_z_variable, elastic_field_param.sigma_xz_ext);
			}
			else if (elastic_field_param.FLAG_SIGMA_XZ == 3) {
				// Fully non-uniform (spatial map)
				elastic_field_param.sigma_xz_ext.resize(N);
				inputValues<Type_var>(sigma_xz_Non_Uniform, elastic_field_param.sigma_xz_ext);
			}
			else if (elastic_field_param.FLAG_SIGMA_XZ == 4) {
				// Study parameter (multiple simulations)
				elastic_field_param.sigma_xz_ext.resize(FE_geom.N_d);
				inputValues<Type_var>(sigma_xz_study, elastic_field_param.sigma_xz_ext);
			}
			else {
				printf("\n The value of FLAG_SIGMA_XZ is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_SIGMA_XZ is not valid!";
				exit(-1);
			}


			// --- sigma_yz_ext setup ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // FLAG_SIGMA_YZ
			elastic_field_param.FLAG_SIGMA_YZ = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // sigma_yz_ext
			sigma_yz = (Type_var)atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // z-variable file name

			// Remove spaces/tabs from file name
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(sigma_yz_z_variable, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Non-uniform file name

			// Remove spaces/tabs again
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(sigma_yz_Non_Uniform, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');  // Study file name

			// Remove spaces/tabs once more
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0)
					i++;
			}
			strncpy(sigma_yz_study, temp, N_BUFFER);

			// --- Load sigma_zz_ext according to flag type ---
			if (elastic_field_param.FLAG_SIGMA_YZ == 1) {
				// Uniform constant
				elastic_field_param.sigma_yz_ext.resize(1);
				elastic_field_param.sigma_yz_ext[0] = sigma_yz;
			}
			else if (elastic_field_param.FLAG_SIGMA_YZ == 2) {
				// z-dependent
				elastic_field_param.sigma_yz_ext.resize(FE_geom.Ncz);
				inputValues<Type_var>(sigma_yz_z_variable, elastic_field_param.sigma_yz_ext);
			}
			else if (elastic_field_param.FLAG_SIGMA_YZ == 3) {
				// Fully non-uniform (spatial map)
				elastic_field_param.sigma_yz_ext.resize(N);
				inputValues<Type_var>(sigma_yz_Non_Uniform, elastic_field_param.sigma_yz_ext);
			}
			else if (elastic_field_param.FLAG_SIGMA_YZ == 4) {
				// Study parameter (multiple simulations)
				elastic_field_param.sigma_yz_ext.resize(FE_geom.N_d);
				inputValues<Type_var>(sigma_yz_study, elastic_field_param.sigma_yz_ext);
			}
			else {
				printf("\n The value of FLAG_SIGMA_YZ is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_SIGMA_YZ is not valid!";
				exit(-1);
			}
		
		}

	} while (!filei.eof());
}


/**
 * @brief Loads the external field configuration from file.
 *
 * This function opens the configuration file "./file_configuration/external_field.dat",
 * reads its contents, and calls load_external() to populate the field_external structure.
 *
 * @param external_field Reference to the external field data structure.
 * @param N Number of field elements or related data entries.
 */
void load_external_configuration(field_external& external_field, int N) {
	const char* filename = "./file_configuration/external_field.dat";

	try {
		fileiic.open(filename, std::ifstream::in);   // Open input file stream

		if (fileiic.is_open() == false) {
			// Error handling if file cannot be opened
			printf("errore nell'apertura file %s \n", filename);
			file_logc << "\n";
			file_logc << "errore nell apertura file" << filename;
			exit(-1);
		}
		else {
			// Successfully opened file → proceed to load data
			load_external(external_field, fileiic, N);
			fileiic.close();
		}
	}
	catch (std::exception& e) {
		// Print any exceptions that occur during file operations
		printf("%s", e.what());
	}
}

/**
 * @brief Parses and loads external field parameters from an open file stream.
 *
 * This function reads numerical and text data from the configuration file and
 * fills the provided field_external structure accordingly. It supports various
 * input formats depending on FLAG_FIELD and FLAG_INPUT_TYPE.
 *
 * @param external_field Reference to the field_external data structure to fill.
 * @param filei Open input file stream.
 * @param N Number of total elements or mesh points to load (for specific FLAG modes).
 */
void load_external(field_external& external_field, std::ifstream& filei, int N) {

	// --- Field components and buffer variables ---
	Vec3<Type_var> field_ext_spherical;
	Vec3<Type_var> field_ext_cartesian;
	char field_z_variable[N_BUFFER];
	char field_Non_Uniform[N_BUFFER];
	char field_study[N_BUFFER];

	// --- AC field parameters ---
	Vec3<Type_var> h_ac;
	char AC_z_variable[N_BUFFER];
	char AC_Non_Uniform[N_BUFFER];
	char AC_study[N_BUFFER];

	// --- Frequency-related variables ---
	Type_var frequency = 0.0;
	char frequency_z_variable[N_BUFFER];
	char frequency_Non_Uniform[N_BUFFER];
	char frequency_study[N_BUFFER];

	// --- Phase parameters ---
	Vec3<Type_var> phase_fi;
	char phase_z_variable[N_BUFFER];
	char phase_Non_Uniform[N_BUFFER];
	char phase_study[N_BUFFER];

	// --- Pulse field parameters ---
	int nfield;
	char file_pulse_field[N_BUFFER];
	char file_pulse_time[N_BUFFER];
	Type_var H_theta_pulse;
	Type_var H_Phi_pulse;

	// --- Temporary variables used for parsing ---
	char temp[N_BUFFER], temp2[N_BUFFER];
	int i = 0, j, check;

	// --- Main read loop ---
	do {
		filei.getline(temp2, N_BUFFER);   // Read line into temp2

		// Section identifier
		if (strcmp(temp2, "//Field external") == 0) {

			// --- Read external field parameters ---
			filei.getline(temp, N_BUFFER, '!');     // Read FLAG_FIELD
			external_field.FLAG_FIELD = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');     // Read FLAG_INPUT_TYPE
			external_field.FLAG_INPUT_TYPE = atoi(temp);

			// --- Read spherical components ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			field_ext_spherical.x = atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			field_ext_spherical.y = atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			field_ext_spherical.z = atof(temp);

			// --- Read cartesian components ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			field_ext_cartesian.x = atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			field_ext_cartesian.y = atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			field_ext_cartesian.z = atof(temp);

			// --- Read variable name for z-direction field ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			// Remove whitespace from variable name
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(field_z_variable, temp, N_BUFFER);

			// --- Read Non-Uniform field variable ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(field_Non_Uniform, temp, N_BUFFER);

			// --- Read study field variable ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(field_study, temp, N_BUFFER);

			// --- Initialize field based on FLAG_FIELD ---
			if (external_field.FLAG_FIELD == 0) {
				external_field.d_field.resize(1);
				external_field.d_field[0] = Vec3<Type_var>(0.0, 0.0, 0.0);
			}
			else if (external_field.FLAG_FIELD == 1) {
				if (external_field.FLAG_INPUT_TYPE == 1) {
					external_field.d_field.resize(1);
					external_field.d_field[0] = field_ext_spherical;
				}
				else if (external_field.FLAG_INPUT_TYPE == 2) {
					external_field.d_field.resize(1);
					external_field.d_field[0] = field_ext_cartesian;
				}
				else {
					printf("\n The value of FLAG_INPUT_TYPE is not valid!\n");
					file_logc << "\n";
					file_logc << "The value of FLAG_INPUT_TYPE is not valid!";
					exit(-1);
				}
			}
			else if (external_field.FLAG_FIELD == 2) {
				external_field.d_field.resize(FE_geom.Ncz);
				inputValuesVec3<Type_var>(field_z_variable, external_field.d_field);
			}
			else if (external_field.FLAG_FIELD == 3) {
				external_field.d_field.resize(N);
				inputValuesVec3<Type_var>(field_Non_Uniform, external_field.d_field);
			}
			else if (external_field.FLAG_FIELD == 4) {
				external_field.d_field.resize(FE_geom.N_d);
				inputValuesVec3<Type_var>(field_study, external_field.d_field);
			}
			else {
				printf("\n The value of FLAG_FIELD is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_FIELD is not valid!";
				exit(-1);
			}

			// --- Read AC field section ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			external_field.FLAG_FIELD_AC = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			h_ac.x = atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			h_ac.y = atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			h_ac.z = atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			// Remove whitespace in AC variable name
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(AC_z_variable, temp, N_BUFFER);
			AC_z_variable[strcspn(AC_z_variable, "\r\n")] = 0;  // Remove newline
			filei.getline(temp, N_BUFFER);                 // Read next line (possibly delimiter)
			filei.getline(temp, N_BUFFER, '!');            // Read until '!' character

			/** Remove spaces and tabs from variable name (AC Non-Uniform). */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(AC_Non_Uniform, temp, N_BUFFER);

			// --- Read AC study variable name ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Remove spaces and tabs from variable name (AC Study). */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(AC_study, temp, N_BUFFER);

			/**
			 * @brief Initialize the AC field based on FLAG_FIELD_AC.
			 * FLAG_FIELD_AC determines how the AC field values are set:
			 * 0: Zero field
			 * 1: Constant vector h_ac
			 * 2: z-variable field
			 * 3: Non-uniform field
			 * 4: Study-dependent field
			 */
			if (external_field.FLAG_FIELD_AC == 0) {
				external_field.d_AC_field.resize(1);
				external_field.d_AC_field[0] = Vec3<Type_var>(0.0, 0.0, 0.0);
			}
			else if (external_field.FLAG_FIELD_AC == 1) {
				external_field.d_AC_field.resize(1);
				external_field.d_AC_field[0] = h_ac;
			}
			else if (external_field.FLAG_FIELD_AC == 2) {
				external_field.d_AC_field.resize(FE_geom.Ncz);
				inputValuesVec3<Type_var>(AC_z_variable, external_field.d_AC_field);
			}
			else if (external_field.FLAG_FIELD_AC == 3) {
				external_field.d_AC_field.resize(N);
				inputValuesVec3<Type_var>(AC_Non_Uniform, external_field.d_AC_field);
			}
			else if (external_field.FLAG_FIELD_AC == 4) {
				external_field.d_AC_field.resize(FE_geom.N_d);
				inputValuesVec3<Type_var>(AC_study, external_field.d_AC_field);
			}
			else {
				printf("\n The value of FLAG_FIELD_AC is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_FIELD_AC is not valid!";
				exit(-1);
			}

			/**
			 * @section Frequency Configuration
			 * @brief Reads and configures the AC field frequency data.
			 */

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			external_field.FLAG_FREQUENCY = atoi(temp);    // Frequency mode flag

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			frequency = atof(temp);                        // Frequency value (Hz)

			// --- Frequency variable name (z-dependent) ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Remove whitespace in frequency variable name (z-variable). */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(frequency_z_variable, temp, N_BUFFER);

			// --- Frequency Non-Uniform variable ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Remove whitespace in frequency Non-Uniform variable name. */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(frequency_Non_Uniform, temp, N_BUFFER);

			// --- Frequency Study variable ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Remove whitespace in frequency Study variable name. */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(frequency_study, temp, N_BUFFER);

			/**
			 * @brief Initialize AC frequency based on FLAG_FREQUENCY.
			 */
			if (external_field.FLAG_FREQUENCY == 1) {
				external_field.d_AC_frequency.resize(1);
				external_field.d_AC_frequency[0] = frequency;
			}
			else if (external_field.FLAG_FREQUENCY == 2) {
				external_field.d_AC_frequency.resize(FE_geom.Ncz);
				inputValues<Type_var>(frequency_z_variable, external_field.d_AC_frequency);
			}
			else if (external_field.FLAG_FREQUENCY == 3) {
				external_field.d_AC_frequency.resize(N);
				inputValues<Type_var>(frequency_Non_Uniform, external_field.d_AC_frequency);
			}
			else if (external_field.FLAG_FREQUENCY == 4) {
				external_field.d_AC_frequency.resize(FE_geom.N_d);
				inputValues<Type_var>(frequency_study, external_field.d_AC_frequency);
			}
			else {
				printf("\n The value of FLAG_FREQUENCY is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_FREQUENCY is not valid!";
				exit(-1);
			}

			/**
			 * @section Phase Configuration
			 * @brief Reads and initializes phase data for the AC field.
			 */

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			external_field.FLAG_PHASE = atoi(temp);        // Phase mode flag

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			phase_fi.x = atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			phase_fi.y = atof(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			phase_fi.z = atof(temp);

			// --- Phase variable names ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Remove spaces from phase z-variable name. */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(phase_z_variable, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Remove spaces from phase Non-Uniform variable name. */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(phase_Non_Uniform, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Remove spaces from phase study variable name. */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(phase_study, temp, N_BUFFER);

			/**
			 * @brief Initialize phase values based on FLAG_PHASE.
			 */
			if (external_field.FLAG_PHASE == 1) {
				external_field.d_AC_phase.resize(1);
				external_field.d_AC_phase[0] = phase_fi;
			}
			else if (external_field.FLAG_PHASE == 2) {
				external_field.d_AC_phase.resize(FE_geom.Ncz);
				inputValuesVec3<Type_var>(phase_z_variable, external_field.d_AC_phase);
			}
			else if (external_field.FLAG_PHASE == 3) {
				external_field.d_AC_phase.resize(N);
				inputValuesVec3<Type_var>(phase_Non_Uniform, external_field.d_AC_phase);
			}
			else if (external_field.FLAG_PHASE == 4) {
				external_field.d_AC_phase.resize(FE_geom.N_d);
				inputValuesVec3<Type_var>(phase_study, external_field.d_AC_phase);
			}
			else {
				printf("\n The value of FLAG_PHASE is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_PHASE is not valid!";
				exit(-1);
			}

			/**
			 * @section Pulse Field Configuration
			 * @brief Reads and initializes pulse field information.
			 */

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			external_field.FLAG_FIELD_PULSE = atoi(temp);  // Pulse field flag

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			nfield = atoi(temp);                            // Number of field values

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Remove spaces in pulse field filename. */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(file_pulse_field, temp, N_BUFFER);

			/**
			 * @brief Initialize the pulse field and time vectors.
			 */
			if (external_field.FLAG_FIELD_PULSE == 0) {
				external_field.d_pulse_field.resize(1);
				external_field.d_pulse_time.resize(1);
				external_field.d_pulse_field[0] = 0.0;
				external_field.d_pulse_time[0] = 0.0;
			}
			else if (external_field.FLAG_FIELD_PULSE == 1) {
				external_field.d_pulse_field.resize(nfield);
				external_field.d_pulse_time.resize(nfield);
				inputValuesVec2(file_pulse_field, external_field.d_pulse_field, external_field.d_pulse_time);
			}
			else {
				printf("\n The value of FLAG_FIELD_PULSE is not valid!\n");
				file_logc << "\n";
				file_logc << "The value of FLAG_FIELD_PULSE is not valid!";
				exit(-1);
			}

			// --- Read pulse angular parameters ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			external_field.H_theta_pulse = atof(temp);       // Polar angle component

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			external_field.H_phi_pulse = atof(temp);         // Azimuthal angle component
		}


	} while (!filei.eof());
}

/**
 * @brief Loads the gradient field configuration from the given file.
 *
 * @param filename  Path to the gradient configuration file.
 * @param N         Number of mesh points (or elements).
 * @return gradient Structure containing gradient field parameters.
 */
gradient load_Gradient_configuration(const char* filename, int N) {
	gradient gradient_field_param;

	try {
		fileiic.open(filename, std::ifstream::in);

		if (fileiic.is_open() == false) {
			printf("errore nell'apertura file %s \n", filename);
			file_logc << "\n";
			file_logc << "errore nell apertura file" << filename;
			exit(-1);  // Fatal: configuration file cannot be opened
		}
		else {
			/**
			 * @brief File successfully opened; load all gradient parameters.
			 */
			gradient_field_param = load_gradient_field(fileiic, N);
			fileiic.close();
		}
	}
	catch (std::exception& e) {
		printf("%s", e.what());  // Print exception message to console
	}

	return gradient_field_param;
}


/**
 * @brief Reads and initializes gradient field parameters from an open configuration file.
 *
 * This function parses multiple flags (FLAG_G1, FLAG_G2, etc.) to determine
 * whether each gradient component is constant, z-dependent, or non-uniform.
 *
 * @param filei   Reference to open input stream (gradient configuration file).
 * @param N       Number of spatial points or mesh elements.
 * @return gradient Structure filled with parsed and allocated gradient data.
 */
gradient load_gradient_field(std::ifstream& filei, int N) {

	Type_var G0_constant = 0.0;

	Type_var G1_constant = 0.0;
	char G1_z_variable[N_BUFFER];
	char G1_non_uniform[N_BUFFER];

	Type_var G2_constant = 0.0;
	char G2_z_variable[N_BUFFER];
	char G2_non_uniform[N_BUFFER];

	Type_var G3_constant = 0.0;
	char G3_z_variable[N_BUFFER];
	char G3_non_uniform[N_BUFFER];

	Type_var G4_constant = 0.0;
	char G4_z_variable[N_BUFFER];
	char G4_non_uniform[N_BUFFER];

	char temp[N_BUFFER], temp2[N_BUFFER];
	int i = 0, j, check;

	do {
		filei.getline(temp2, N_BUFFER);  // Read next header line

		if (strcmp(temp2, "//Gradient field parameters") == 0) {

			/**
			 * @section Gradient G0
			 * @brief Load G0 (base gradient constant)
			 */
			filei.getline(temp, N_BUFFER, '!');                 // Read until '!' delimiter
			gradient_field_param.FLAG_GRADIENT = atoi(temp);    // Global gradient flag

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			G0_constant = (Type_var)atof(temp);                 // Constant G0 value

			gradient_field_param.d_G0.resize(1);
			gradient_field_param.d_G0[0] = G0_constant;

			/**
			 * @section Gradient G1
			 * @brief Load configuration for G1 component.
			 */
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			gradient_field_param.FLAG_G1 = atoi(temp);          // Flag for G1 type

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			G1_constant = (Type_var)atof(temp);                 // Constant G1 value

			// --- G1 z-variable name ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Remove spaces/tabs from G1_z_variable name. */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(G1_z_variable, temp, N_BUFFER);

			// --- G1 Non-Uniform variable name ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Remove spaces/tabs from G1_non_uniform name. */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(G1_non_uniform, temp, N_BUFFER);

			/**
			 * @brief Initialize G1 according to FLAG_G1.
			 * 1: constant, 2: z-variable, 3: non-uniform.
			 */
			if (gradient_field_param.FLAG_G1 == 1) {
				gradient_field_param.d_G1.resize(1);
				gradient_field_param.d_G1[0] = G1_constant;
			}
			else if (gradient_field_param.FLAG_G1 == 2) {
				gradient_field_param.d_G1.resize(FE_geom.Ncz);
				inputValues(G1_z_variable, gradient_field_param.d_G1);
			}
			else if (gradient_field_param.FLAG_G1 == 3) {
				gradient_field_param.d_G1.resize(N);
				inputValues(G1_non_uniform, gradient_field_param.d_G1);
			}
			else {
				printf("\n The value of FLAG_G1 is not valid!\n");
				file_logc << "\nThe value of FLAG_G1 is not valid!";
				exit(-1);
			}

			/**
			 * @section Gradient G2
			 * @brief Load configuration for G2 component.
			 */
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			gradient_field_param.FLAG_G2 = atoi(temp);          // Flag for G2 type

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			G2_constant = atof(temp);                           // Constant G2 value

			// --- G2 z-variable name ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Remove spaces/tabs from G2_z_variable name. */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(G2_z_variable, temp, N_BUFFER);

			// --- G2 Non-Uniform variable name ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Remove spaces/tabs from G2_non_uniform name. */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(G2_non_uniform, temp, N_BUFFER);

			/**
			 * @brief Initialize G2 based on FLAG_G2 value.
			 */
			if (gradient_field_param.FLAG_G2 == 1) {
				gradient_field_param.d_G2.resize(1);
				gradient_field_param.d_G2[0] = G2_constant;
			}
			else if (gradient_field_param.FLAG_G2 == 2) {
				gradient_field_param.d_G2.resize(FE_geom.Ncz);
				inputValues(G2_z_variable, gradient_field_param.d_G2);
			}
			else if (landau_free_param.FLAG_ALPHA2 == 3) {  // ⚠ Possibly typo — check variable name
				gradient_field_param.d_G2.resize(N);
				inputValues(G2_non_uniform, gradient_field_param.d_G2);
			}
			else {
				printf("\n The value of FLAG_G2 is not valid!\n");
				file_logc << "\nThe value of FLAG_G2 is not valid!";
				exit(-1);
			}

			/**
			 * @section Gradient G3
			 * @brief Load configuration for G3 component.
			 */
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			gradient_field_param.FLAG_G3 = atoi(temp);          // Flag for G3 type

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			G3_constant = atof(temp);                           // Constant G3 value

			// --- G3 z-variable name ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/** Remove spaces/tabs from G3_z_variable name. */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(G3_z_variable, temp, N_BUFFER);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			/**
			 * @brief Remove spaces/tabs from G3_non_uniform variable name.
			 */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(G3_non_uniform, temp, N_BUFFER);

			/**
			 * @brief Initialize G3 according to FLAG_G3.
			 * 1: constant, 2: z-variable, 3: non-uniform.
			 */
			if (gradient_field_param.FLAG_G3 == 1) {
				gradient_field_param.d_G3.resize(1);
				gradient_field_param.d_G3[0] = G3_constant;
			}
			else if (gradient_field_param.FLAG_G3 == 2) {
				gradient_field_param.d_G3.resize(FE_geom.Ncz);
				inputValues(G3_z_variable, gradient_field_param.d_G3);
			}
			else if (landau_free_param.FLAG_ALPHA3 == 3) {
				gradient_field_param.d_G3.resize(N);
				inputValues(G3_non_uniform, gradient_field_param.d_G3);
			}
			else {
				printf("\n The value of FLAG_G3 is not valid!\n");
				file_logc << "\nThe value of FLAG_G3 is not valid!";
				exit(-1);
			}


			/**
			 * @section Gradient G4
			 * @brief Load configuration for G4 component.
			 */
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			gradient_field_param.FLAG_G4 = atoi(temp);          // Flag for G4 type

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			G4_constant = atof(temp);                           // Constant G4 value

			// --- G4 z-variable name ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/**
			 * @brief Remove spaces/tabs from G4_z_variable name.
			 */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(G4_z_variable, temp, N_BUFFER);

			// --- G4 Non-Uniform variable name ---
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');

			/**
			 * @brief Remove spaces/tabs from G4_non_uniform variable name.
			 */
			i = 0;
			while (temp[i] != '\0') {
				check = 0;
				if (temp[i] == ' ' || temp[i] == '\t') {
					j = i;
					while (temp[j - 1] != '\0') {
						temp[j] = temp[j + 1];
						j++;
					}
					check = 1;
				}
				if (check == 0) i++;
			}
			strncpy(G4_non_uniform, temp, N_BUFFER);

			/**
			 * @brief Initialize G4 according to FLAG_G4.
			 * 1: constant, 2: z-variable, 3: non-uniform.
			 */
			if (gradient_field_param.FLAG_G4 == 1) {
				gradient_field_param.d_G4.resize(1);
				gradient_field_param.d_G4[0] = G4_constant;
			}
			else if (gradient_field_param.FLAG_G4 == 2) {
				gradient_field_param.d_G4.resize(FE_geom.Ncz);
				inputValues(G4_z_variable, gradient_field_param.d_G4);
			}
			else if (landau_free_param.FLAG_ALPHA4 == 3) {
				gradient_field_param.d_G4.resize(N);
				inputValues(G4_non_uniform, gradient_field_param.d_G4);
			}
			else {
				printf("\n The value of FLAG_G4 is not valid!\n");
				file_logc << "\nThe value of FLAG_G4 is not valid!";
				exit(-1);
			}

			/**
			 * @section Boundary Conditions
			 * @brief Read boundary condition flag for gradient field.
			 */
			Type_var spacing_val;
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			gradient_field_param.FLAG_BC = atoi(temp);  // Boundary condition flag value
		}

	} while (!filei.eof());

	/**
	 * @brief Return fully initialized gradient field structure.
	 */
	return gradient_field_param;
}

/**
 * @brief Loads electrostatic tensor configuration from file.
 *
 * @param filename Path to configuration file.
 * @return set_tensor Struct containing tensor parameters and flags.
 */
set_tensor conf_tensor(const char* filename) {
	set_tensor set;

	try {
		fileii.open(filename, std::ifstream::in);

		// Check if file opened successfully
		if (fileii.is_open() == false) {
			printf("errore nell'apertura file set_tensor %s\n", filename);
			file_logc << "\n";
			file_logc << "errore nell'apertura file set_tensor \n" << filename;
			exit(-1);
		}
		else {
			char temp[N_BUFFER], temp2[N_BUFFER];

			do {
				fileii.getline(temp2, N_BUFFER);

				// Identify correct section in configuration file
				if (strcmp(temp2, "//Configuration file for electrostatic field") == 0) {

					fileii.getline(temp, N_BUFFER, '!');	// Read until '!'
					set.FLAG_TENSOR = atoi(temp);			// Tensor configuration flag

					// Validate FLAG_TENSOR value
					if (set.FLAG_TENSOR > 4) {
						printf("\n The value of FLAG_TENSOR is not valid!\n");
						file_logc << "\n";
						file_logc << "The value of FLAG_TENSOR is not valid!";
						exit(-1);
					}
					/**
					 * @brief FLAG_TENSOR == 3 → load Dx, Dy, Dz parameters from file.
					 */
					else if (set.FLAG_TENSOR == 3) {

						fileii.getline(temp, N_BUFFER);
						fileii.getline(temp, N_BUFFER, '!');
						set.Dx = atof(temp);

						fileii.getline(temp, N_BUFFER);
						fileii.getline(temp, N_BUFFER, '!');
						set.Dy = atof(temp);

						fileii.getline(temp, N_BUFFER);
						fileii.getline(temp, N_BUFFER, '!');
						set.Dz = atof(temp);
					}
					/**
					 * @brief Other FLAG_TENSOR values → initialize Dx, Dy, Dz to zero
					 * and read padding configuration.
					 */
					else {
						fileii.getline(temp, N_BUFFER);
						fileii.getline(temp, N_BUFFER, '!');
						set.Dx = 0.0;

						fileii.getline(temp, N_BUFFER);
						fileii.getline(temp, N_BUFFER, '!');
						set.Dy = 0.0;

						fileii.getline(temp, N_BUFFER);
						fileii.getline(temp, N_BUFFER, '!');
						set.Dz = 0.0;

						fileii.getline(temp, N_BUFFER);
						fileii.getline(temp, N_BUFFER, '!');
						set.FLAG_PADDING = atoi(temp);	// Padding flag

						// Validate padding flag value
						if (set.FLAG_PADDING >= 2) {
							printf("\n The value of FLAG_PADDING is not valid!\n");
							file_logc << "\n";
							file_logc << "The value of FLAG_PADDING is not valid!";
							exit(-1);
						}

						// Read padding dimensions
						fileii.getline(temp, N_BUFFER);
						fileii.getline(temp, N_BUFFER, '!');
						set.PAD_cx = atoi(temp);

						fileii.getline(temp, N_BUFFER);
						fileii.getline(temp, N_BUFFER, '!');
						set.PAD_cy = atoi(temp);

						fileii.getline(temp, N_BUFFER);
						fileii.getline(temp, N_BUFFER, '!');
						set.PAD_cz = atoi(temp);
					}
				} // end if section match

			} while (!fileii.eof());
		}

		fileii.close();
		return set; // Return fully initialized tensor configuration
	}
	catch (std::exception& e) {
		printf("%s", e.what());
	}
}


/**
 * @brief Sets global simulation parameters based on geometry and tensor configuration.
 *
 * @param FE_geom Finite element geometry (domain size, mesh parameters, etc.)
 * @param conf_tens Tensor configuration structure.
 * @return SET_parameters Complete configuration for electrostatic setup.
 */
SET_parameters set_configuration(FE_geometry FE_geom, set_tensor conf_tens) {
	SET_parameters configuration;

	// Total number of grid points
	int N = FE_geom.Ncx * FE_geom.Ncy * FE_geom.Ncz;

	/**
	 * @brief Determine padded mesh dimensions based on FLAG_PADDING.
	 * 0 → no padding
	 * 1 → use specified padding sizes
	 */
	if (conf_tens.FLAG_PADDING == 0) {
		configuration.Mcx = FE_geom.Ncx;
		configuration.Mcy = FE_geom.Ncy;
		configuration.Mcz = FE_geom.Ncz;
		printf("\n %d %d %d \n", configuration.Mcx, configuration.Mcy, configuration.Mcz);
	}
	else if (conf_tens.FLAG_PADDING == 1) {
		configuration.Mcx = conf_tens.PAD_cx + FE_geom.Ncx;
		configuration.Mcy = conf_tens.PAD_cy + FE_geom.Ncy;
		configuration.Mcz = conf_tens.PAD_cz + FE_geom.Ncz;
		printf("\n %d %d %d \n", configuration.Mcx, configuration.Mcy, configuration.Mcz);
	}

	/**
	 * @brief Setup demagnetization (electrostatic tensor) if required.
	 */
	if (conf_tens.FLAG_TENSOR == 1 || conf_tens.FLAG_TENSOR == 2) {
		configuration.demag_param = set_demagnetization(
			configuration.Mcx,
			configuration.Mcy,
			configuration.Mcz,
			FE_geom.h,
			conf_tens
		);
	}

	// Return final configuration parameters
	return configuration;
}
/**
 * @brief Allocates and initializes electrostatic demagnetization tensors on the GPU.
 *
 * This function allocates GPU memory for the components of the electrostatic tensor
 * (xx, yy, zz, xy, yz, xz) and computes them via FFT using the `electroStaticTensor` kernel.
 *
 * @param Mcx Number of cells in x-direction.
 * @param Mcy Number of cells in y-direction.
 * @param Mcz Number of cells in z-direction.
 * @param h   Grid spacing (cell size).
 * @param conf_tens Tensor configuration parameters.
 * @return set_DEMAG Structure containing GPU pointers to the electrostatic tensor components.
 */
set_DEMAG set_demagnetization(int Mcx, int Mcy, int Mcz, Type_var h, set_tensor conf_tens) {

	set_DEMAG set;
	set.FLAG_CALC = 1; // Enable calculation flag

	// Allocate GPU memory for tensor components
	check(cudaMalloc((void**)&(set.cuSDxx), sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz));
	check(cudaMalloc((void**)&(set.cuSDyy), sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz));
	check(cudaMalloc((void**)&(set.cuSDzz), sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz));

	check(cudaMalloc((void**)&(set.cuSDxy), sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz));
	check(cudaMalloc((void**)&(set.cuSDyz), sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz));
	check(cudaMalloc((void**)&(set.cuSDxz), sizeof(HD_Complex_Type) * Mcx * Mcy * Mcz));

	// ----------------------------------------------------------
	// Compute electrostatic tensor and perform FFT internally
	// ----------------------------------------------------------
	electroStaticTensor(
		set.cuSDxx,
		set.cuSDyy,
		set.cuSDzz,
		set.cuSDxy,
		set.cuSDxz,
		set.cuSDyz,
		h,
		Mcx, Mcy, Mcz,
		conf_tens
	);

	return set;
}


/**
 * @brief Loads the output configuration from the predefined file `output.dat`.
 *
 * @param set_out Reference to output configuration structure to be filled.
 */
void load_output_configuration(set_output& set_out) {
	const char* filename = "./file_configuration/output.dat";

	try {
		fileiic.open(filename, std::ifstream::in);

		// Verify file opened successfully
		if (fileiic.is_open() == false) {
			printf("errore nell'apertura file %s \n", filename);
			file_logc << "\n";
			file_logc << "errore nell apertura file" << filename;
			exit(-1);
		}
		else {
			// Parse file contents into structure
			load_output(set_out, fileiic);
			fileiic.close();
		}
	}
	catch (std::exception& e) {
		printf("%s", e.what());
	}
}


/**
 * @brief Reads and loads simulation output parameters from a file stream.
 *
 * Expected input format:
 * ```
 * //Parameters output
 * <num_iteration>!
 * <interval_output>!
 * <interval_polarization>!
 * <FLAG_SPIN>!
 * ...
 * ```
 *
 * @param set_out Output configuration structure to populate.
 * @param filei   Input file stream reference.
 */
void load_output(set_output& set_out, std::ifstream& filei) {

	char shapeFile[N_BUFFER];
	char temp[N_BUFFER], temp2[N_BUFFER];
	int i = 0, j, check;

	do {
		filei.getline(temp2, N_BUFFER);

		// Locate correct section header in file
		if (strcmp(temp2, "//Parameters output") == 0) {

			// ----------------------------------------------------------
			// Core output control parameters
			// ----------------------------------------------------------
			filei.getline(temp, N_BUFFER, '!');
			set_out.num_iteration = atoi(temp); // Total simulation iterations

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			set_out.interval_output = atoi(temp); // Interval for output files

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			set_out.interval_polarization = atoi(temp); // Polarization output frequency

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			set_out.FLAG_SPIN = atoi(temp); // Enable/disable spin output

			// ----------------------------------------------------------
			// Local polarization output
			// ----------------------------------------------------------
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			set_out.FLAG_LOCAL_POLARIZATION = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			set_out.interval_local_polarization = atoi(temp);

			// ----------------------------------------------------------
			// Average polarization output
			// ----------------------------------------------------------
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			set_out.FLAG_AVERAGE_POLARIZATION = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			set_out.interval_average_polarization = atoi(temp);

			// ----------------------------------------------------------
			// Local energy output
			// ----------------------------------------------------------
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			set_out.FLAG_LOCAL_ENERGY = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			set_out.interval_local_energy = atoi(temp);

			// ----------------------------------------------------------
			// Average energy output
			// ----------------------------------------------------------
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			set_out.FLAG_AVERAGE_ENERGY = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			set_out.interval_average_energy = atoi(temp);

			// ----------------------------------------------------------
			// Local electric field output
			// ----------------------------------------------------------
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			set_out.FLAG_LOCAL_FIELD = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			set_out.interval_local_field = atoi(temp);

			// ----------------------------------------------------------
			// Average electric field output
			// ----------------------------------------------------------
			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			set_out.FLAG_AVERAGE_FIELD = atoi(temp);

			filei.getline(temp, N_BUFFER);
			filei.getline(temp, N_BUFFER, '!');
			set_out.interval_average_field = atoi(temp);
		}

	} while (!filei.eof());
}


#endif // CONSTANTS_CUH
