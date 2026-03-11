#ifndef OUTPUT_CUH
#define OUTPUT_CUH

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "output.h"

// -----------------------------------------------------------------------------
// External global variables
// -----------------------------------------------------------------------------
extern landau_free               landau_free_param;
extern FE_geometry               FE_geom;
extern initial_polarization      initial_pol;
extern field_external            external_field;
extern field_external_normalized external_field_vectors;
extern gradient                  gradient_field_param;
extern elastic                   elastic_field_param;
extern landau_free_final         landau_vectors;
extern gradient_final            gradient_vectors;

// =============================================================================
// Output Utility Functions
// =============================================================================

/**
 * @brief Writes final polarization vectors to output files for each simulation.
 *
 * This function transfers polarization vectors from device to host memory and
 * writes them to text files with high precision. A separate file is created
 * for each domain or simulation (`N_d`).
 *
 * @tparam T         Numeric type (e.g., float or double).
 * @param d_vec      Device vector holding polarization vectors.
 * @param Ncx        Grid size in x-direction.
 * @param Ncy        Grid size in y-direction.
 * @param Ncz        Grid size in z-direction.
 * @param N_d        Number of domains or simulations.
 * @param time       Execution time to be recorded (currently commented out in file output).
 */
template <typename T>
void getOutput(thrust::device_vector<Vec3<T>>& d_vec,
    int& Ncx,
    int& Ncy,
    int& Ncz,
    int& N_d,
    T& time)
{
    // Transfer polarization data from device to host
    thrust::host_vector<Vec3<T>> h_vec = d_vec;

    // Total number of cells per domain
    int spatialSize = Ncx * Ncy * Ncz;

    for (int d = 0; d < N_d; ++d) {
        // Construct output filename
        std::ostringstream filename;
        filename << "output/simulation_output_" << d << ".txt";

        // Open output file
        std::ofstream output_file(filename.str());
        if (!output_file.is_open()) {
            std::cerr << "Failed to open output file: " << filename.str() << std::endl;
            continue;
        }

        // Optional header:
        //output_file << Ncx << "      " << Ncy << "      " << Ncz << "\n";
        // output_file << "Final Polarization Vectors (Simulation " << d << "):\n";

        // Write polarization values
        for (int z = 0; z < Ncz; ++z) {
            for (int y = 0; y < Ncy; ++y) {
                for (int x = 0; x < Ncx; ++x) {
                    int spatialIndex = z * Ncx * Ncy + y * Ncx + x;
                    int index = d * spatialSize + spatialIndex;

                    Vec3<T> m = h_vec[index];

                    output_file << std::fixed << std::setprecision(16)
                        // << "Cell (" << x << ", " << y << ", " << z << "): "
                        << m.x << "      "
                        << m.y << "      "
                        << m.z << std::endl;
                }
            }
        }

        output_file.close();
    }
}

/**
 * @brief Writes constant scalar values stored in a device vector to a file.
 *
 * @tparam T     Numeric type.
 * @param d_vec  Device vector containing constant values.
 */
template <typename T>
void getConstant(thrust::device_vector<T>& d_vec)
{
    // Transfer device vector to host
    thrust::host_vector<T> h_vec = d_vec;

    // Open file
    std::ofstream output_file("constant.txt");
    if (!output_file.is_open()) {
        std::cerr << "Failed to open constant.txt for writing.\n";
        return;
    }

    // Write constant values with their indices
    for (int i = 0; i < static_cast<int>(h_vec.size()); i++) {
        output_file << "value(" << i << "): " << h_vec[i] << std::endl;
    }

    output_file.close();
}

/**
 * @brief Writes shape mask or material mask values to a file.
 *
 * @tparam T                Numeric or integer type.
 * @param d_shapeVector     Device vector containing shape mask values.
 */
template <typename T>
void getShape(thrust::device_vector<T>& d_shapeVector)
{
    // Transfer device vector to host
    thrust::host_vector<T> h_shapeVector = d_shapeVector;

    // Open file
    std::ofstream output_file("shapeOutput.dat");
    if (!output_file.is_open()) {
        std::cerr << "Failed to open shapeOutput.dat for writing.\n";
        return;
    }

    // Write shape values with their indices
    for (int i = 0; i < static_cast<int>(h_shapeVector.size()); i++) {
        output_file << "value(" << i << "): " << h_shapeVector[i] << std::endl;
    }

    output_file.close();
}

/**
 * @brief Writes the global alpha coefficient vectors (α₁…α₆) to a file.
 *
 * Uses global variables `landau_vectors`. Handles coefficient vectors of
 * different lengths by leaving blank spaces when a coefficient is missing
 * at a particular index.
 *
 * @tparam T Numeric type.
 */
template <typename T>
void saveAlphaVectorsToFile()
{
    std::ofstream outFile("alpha_vectors.dat");
    if (!outFile.is_open()) {
        std::cerr << "Failed to open alpha_vectors.dat for writing.\n";
        return;
    }

    // Determine the largest vector length
    size_t max_size = std::max({
        landau_vectors.vector_alpha1.size(),
        landau_vectors.vector_alpha2.size(),
        landau_vectors.vector_alpha3.size(),
        landau_vectors.vector_alpha4.size(),
        landau_vectors.vector_alpha5.size(),
        landau_vectors.vector_alpha6.size()
        });

    // Write all vectors side by side
    for (size_t i = 0; i < max_size; ++i) {
        outFile << std::fixed << std::setprecision(16);

        if (i < landau_vectors.vector_alpha1.size()) outFile << landau_vectors.vector_alpha1[i]; else outFile << "    ";
        outFile << "     ";
        if (i < landau_vectors.vector_alpha2.size()) outFile << landau_vectors.vector_alpha2[i]; else outFile << "    ";
        outFile << "     ";
        if (i < landau_vectors.vector_alpha3.size()) outFile << landau_vectors.vector_alpha3[i]; else outFile << "    ";
        outFile << "     ";
        if (i < landau_vectors.vector_alpha4.size()) outFile << landau_vectors.vector_alpha4[i]; else outFile << "    ";
        outFile << "     ";
        if (i < landau_vectors.vector_alpha5.size()) outFile << landau_vectors.vector_alpha5[i]; else outFile << "    ";
        outFile << "     ";
        if (i < landau_vectors.vector_alpha6.size()) outFile << landau_vectors.vector_alpha6[i]; else outFile << "    ";

        outFile << "\n";
    }

    outFile.close();
    std::cout << "Alpha vectors written to alpha_vectors.dat using global variables.\n";
}

/**
 * @brief Writes the global gradient energy coefficient vectors to a file.
 *
 * Uses global variable `gradient_vectors`. Handles different vector lengths.
 *
 * @tparam T Numeric type.
 */
template <typename T>
void saveGradientVectorsToFile()
{
    std::ofstream outFile("gradient_vectors.dat");
    if (!outFile.is_open()) {
        std::cerr << "Failed to open gradient_vectors.dat for writing.\n";
        return;
    }

    // Determine the largest vector length
    size_t max_size = std::max({
        gradient_vectors.vector_G1.size(),
        gradient_vectors.vector_G2.size(),
        gradient_vectors.vector_G3.size(),
        gradient_vectors.vector_G4.size()
        });

    // Write all vectors side by side
    for (size_t i = 0; i < max_size; ++i) {
        outFile << std::fixed << std::setprecision(16);

        if (i < gradient_vectors.vector_G1.size()) outFile << gradient_vectors.vector_G1[i]; else outFile << "    ";
        outFile << "     ";
        if (i < gradient_vectors.vector_G2.size()) outFile << gradient_vectors.vector_G2[i]; else outFile << "    ";
        outFile << "     ";
        if (i < gradient_vectors.vector_G3.size()) outFile << gradient_vectors.vector_G3[i]; else outFile << "    ";
        outFile << "     ";
        if (i < gradient_vectors.vector_G4.size()) outFile << gradient_vectors.vector_G4[i]; else outFile << "    ";

        outFile << "\n";
    }

    outFile.close();
    std::cout << "Gradient vectors written to gradient_vectors.dat using global variables.\n";
}

#endif // OUTPUT_CUH
