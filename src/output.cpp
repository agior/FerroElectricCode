// ============================================================================
// output.cpp
// ============================================================================
// Handles all simulation output, logging, and file generation for polarization,
// field, and energy data. Creates structured directories and writes per-device
// and averaged results in various formats (plain text, OOMMF, etc.).
// ============================================================================

#include "output.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <numeric>
#include <iomanip> // for std::setw and std::setfill
#include "debugging.hpp"
#include <thrust/reduce.h>

/* ========================== External Declarations ========================== */
extern std::string outputFolder;
extern std::ofstream file_log;
extern std::ofstream output;
extern FE_geometry FE_geom;
extern set_output set_out;

/**
 * @brief Resets and prepares output folder hierarchy.
 *
 * Deletes any existing output directory and recreates the folder structure:
 * - /output/
 * - /output/energy/
 * - /output/Spin_Complete_i/
 * - /output/energy/localEnergy_device_i/
 *
 * Initializes log and main output files.
 */
void resetOutputFolders() {
    if (std::filesystem::exists(outputFolder)) {
        try {
            std::filesystem::remove_all(outputFolder);
        }
        catch (std::exception& e) {
            reportFatalSettingsError(
                "ERROR::ONE OR MORE FILES INSIDE \"" + outputFolder +
                "\" ARE OPEN IN OTHER PROGRAMS. CLOSE THEM AND RETRY\n");
        }
    }

    // Base directories
    std::filesystem::create_directories(outputFolder);
    std::filesystem::create_directories(outputFolder + "/energy");

    // Per-device directories
    for (int sim = 0; sim < FE_geom.N_d; ++sim) {
        std::filesystem::create_directories(outputFolder + "/Spin_Complete_" + std::to_string(sim));
        std::filesystem::create_directories(outputFolder + "/energy/localEnergy_device_" + std::to_string(sim));
    }

    // Initialize log and output files
    file_log.open(outputFolder + "/log.txt");
    output.open(outputFolder + "/output.txt");

    if (!file_log.is_open() || !output.is_open()) {
        reportFatalSettingsError(
            "ERROR::FAILED TO OPEN LOG OR OUTPUT FILES IN \"" + outputFolder + "\"\n");
    }
}

/**
 * @brief Reports a non-fatal settings or runtime message to console and log.
 * @param message The message string to log.
 */
void reportSettingsError(const std::string& message) {
    std::cout << message;
    file_log << message;
}

/**
 * @brief Reports a fatal error and terminates the program.
 * @param message The error message.
 */
void reportFatalSettingsError(const std::string& message) {
    reportSettingsError(message);
    exit(-1);
}

/**
 * @brief Generates OOMMF (.omf) file representing magnetization configuration.
 * @param magnetization Pointer to magnetization array.
 * @param sim_id Device or simulation ID.
 * @param local_counter Snapshot index (used for file numbering).
 */
void writeOommfFile(const Vec3<Type_var>* magnetization, int sim_id, int local_counter) {
    const int N = FE_geom.Ncx * FE_geom.Ncy * FE_geom.Ncz;
    std::string folder = outputFolder + "/Spin_Complete_" + std::to_string(sim_id);

    std::filesystem::create_directories(folder); // Ensure folder exists

    std::ofstream writer(generateOommfFileName(sim_id, local_counter));
    ASSERT(writer.is_open() && !writer.fail());

    static std::string OOMMFsetup =
        "# OOMMF: rectangular mesh v0.99\n"
        "# Segment count: 1\n"
        "# Begin: Segment\n"
        "# Begin: Header\n"
        "# Title: model.omf\n"
        "# Desc:\n"
        "# meshunit: unknown\n"
        "# xbase: 0\n"
        "# ybase: 0\n"
        "# zbase: 0\n"
        "# xstepsize: 1\n"
        "# ystepsize: 1\n"
        "# zstepsize: 1\n"
        "# xnodes: " + std::to_string(FE_geom.Ncx) + "\n"
        "# ynodes: " + std::to_string(FE_geom.Ncy) + "\n"
        "# znodes: " + std::to_string(FE_geom.Ncz) + "\n"
        "# xmin: -0.5\n"
        "# ymin: -0.5\n"
        "# zmin: -0.5\n"
        "# xmax: " + std::to_string(float(FE_geom.Ncx) - 0.5) + "\n"
        "# ymax: " + std::to_string(float(FE_geom.Ncy) - 0.5) + "\n"
        "# zmax: " + std::to_string(float(FE_geom.Ncz) - 0.5) + "\n"
        "# valueunit: Ms\n"
        "# valuemultiplier: 1\n"
        "# ValueRangeMaxMag:  1.\n"
        "# ValueRangeMinMag:  1e-8\n"
        "# End: Header\n"
        "# Begin: data text\n";

    writer << OOMMFsetup;
    for (int i = 0; i < N; i++) {
        writer << magnetization[i] << '\n';
    }
    writer << "# End: data text \n# End: segment \n";
    writer.close();
}

/**
 * @brief Builds standardized OOMMF file name string.
 */
std::string generateOommfFileName(int sim_id, int local_counter) {
    std::ostringstream ss;
    ss << outputFolder << "/Spin_Complete_" << sim_id << "/SL"
        << std::setw(5) << std::setfill('0') << local_counter << ".omf";
    return ss.str();
}

/**
 * @brief Saves magnetization vector field to plain text file (m_simX.txt).
 */
void save_m(const Vec3<Type_var>* magnetization, int sim_id) {
    const int N = FE_geom.Ncx * FE_geom.Ncy * FE_geom.Ncz;
    std::filesystem::create_directories(outputFolder);

    std::ostringstream filename;
    filename << outputFolder << "/m_sim" << sim_id << ".txt";

    std::ofstream writer(filename.str(), std::ios::app);
    ASSERT(writer.is_open());

    writer.precision(16);
    for (int i = 0; i < N; i++) {
        writer << magnetization[i] << '\n';
    }
    writer.close();
}

/**
 * @brief Writes averaged magnetization to output stream and periodic snapshots to file.
 */
void writeOutput(int& counterMagnetization, int& counterSnapshot, Type_var time,
    const Vec3<Type_var>* magnetization) {
    const int N_local = FE_geom.Ncx * FE_geom.Ncy * FE_geom.Ncz;

    // Write global averaged polarization at defined interval
    if (counterMagnetization == set_out.interval_output) {
        output.precision(6);
        output << std::fixed << std::scientific << time << " ";

        for (int sim = 0; sim < FE_geom.N_d; ++sim) {
            const Vec3<Type_var>* simMag = magnetization + sim * N_local;
            Vec3<Type_var> avg =
                thrust::reduce(simMag, simMag + N_local, Vec3<Type_var>()) /
                Type_var(FE_geom.n_elements);

            output.precision(16);
            output << std::fixed << avg;
            if (sim != FE_geom.N_d - 1)
                output << " ";
            else
                output << "\n";
        }
        counterMagnetization = 0;
    }

    // Save snapshots of polarization fields
    if (counterSnapshot == set_out.interval_polarization) {
        static int local_counter = 0;
        for (int sim = 0; sim < FE_geom.N_d; ++sim) {
            const Vec3<Type_var>* simMag = magnetization + sim * N_local;

            switch (set_out.FLAG_SPIN) {
            case 0:
                save_m(simMag, sim);
                break;
            case 1:
                writeOommfFile(simMag, sim, local_counter);
                break;
            case 2:
                save_m(simMag, sim);
                writeOommfFile(simMag, sim, local_counter);
                break;
            default:
                ASSERT(SHOULD_NOT_GET_HERE);
            }
        }
        counterSnapshot = 0;
        local_counter++;
    }
}

/**
 * @brief Writes per-device local energy arrays to disk for each simulation step.
 */
void writeEnergyToFile(const Type_var* energy, int size, int stepIndex) {
    const int N_devices = FE_geom.N_d;

    if (N_devices <= 0) {
        std::cerr << "Error: FE_geom.N_d <= 0, cannot write energy files.\n";
        return;
    }

    // Validate array size divisibility
    if (size % N_devices != 0) {
        std::cerr << "Error: Energy array size (" << size
            << ") not divisible by number of devices (" << N_devices << ").\n";
        return;
    }

    const int N_local_energy = size / N_devices;
    std::filesystem::create_directories(outputFolder + "/energy");

    for (int sim_id = 0; sim_id < N_devices; ++sim_id) {
        std::string deviceFolder = outputFolder + "/energy/localEnergy_device_" +
            std::to_string(sim_id);
        std::filesystem::create_directories(deviceFolder);

        std::ostringstream filename;
        filename << deviceFolder << "/energy" << stepIndex << ".dat";

        std::ofstream fout(filename.str());
        if (!fout.is_open()) {
            std::cerr << "Error: Unable to open " << filename.str() << "\n";
            continue;
        }

        int offset = sim_id * N_local_energy;
        for (int i = 0; i < N_local_energy; ++i) {
            fout << energy[offset + i] << "\n";
        }

        fout.close();
    }
}

/**
 * @brief Writes time evolution of average polarization per device.
 */


void writeAvgPolarizationToFile(Type_var time, const Vec3<Type_var>* magnetization)
{
    const int Nx = FE_geom.Ncx;
    const int Ny = FE_geom.Ncy;
    const int Nz = FE_geom.Ncz;
    const int sliceSize = Nx * Ny;
    const int N_local = sliceSize * Nz;

    // Updated z regions (no vacuum)
    int substrate_z_start = 0;
    int substrate_z_end = 12; // exclusive
    int film_z_start = 12;
    int film_z_end = 32; // exclusive

    std::filesystem::create_directories(outputFolder + "/polarization_avg");

    // static step counter
    static int step_counter = 0;

    // Only write every step (change modulus to control frequency)
    step_counter++;
    if (step_counter % 1 != 0) {
        return;
    }

    for (int sim = 0; sim < FE_geom.N_d; ++sim)
    {
        const Vec3<Type_var>* simMag = magnetization + sim * N_local;

        auto compute_avg = [&](int z_start, int z_end) {
            Type_var sum_x = 0, sum_y = 0, sum_z = 0;
            int count = (z_end - z_start) * sliceSize;

            for (int z = z_start; z < z_end; ++z) {
                int base = z * sliceSize;
                for (int i = 0; i < sliceSize; ++i) {
                    const Vec3<Type_var>& v = simMag[base + i];
                    sum_x += v.x;
                    sum_y += v.y;
                    sum_z += v.z;
                }
            }

            Vec3<Type_var> avg(0, 0, 0);
            if (count > 0) {
                avg.x = sum_x / count;
                avg.y = sum_y / count;
                avg.z = sum_z / count;
            }
            return avg;
            };

        Vec3<Type_var> avg_substrate = compute_avg(substrate_z_start, substrate_z_end);
        Vec3<Type_var> avg_film = compute_avg(film_z_start, film_z_end);

        // Build filename
        std::ostringstream fname;
        fname << outputFolder << "/polarization_avg/avg_polarization_device_" << sim << ".dat";

        // Append to file
        std::ofstream fout(fname.str(), std::ios::app);
        if (!fout.is_open()) {
            std::cerr << "Error: Unable to open " << fname.str() << "\n";
            continue;
        }

        fout << std::scientific << std::setprecision(6) << time << " "
            << std::fixed << std::setprecision(16)
            // substrate
            << avg_substrate.x << " " << avg_substrate.y << " " << avg_substrate.z << " "
            // film
            << avg_film.x << " " << avg_film.y << " " << avg_film.z
            << "\n";

        fout.close();
    }
}

/**
 * @brief Writes time evolution of average energy per device.
 */
void writeAvgEnergyToFile(Type_var time, const Type_var* energy) {
    const int N_local = FE_geom.Ncx * FE_geom.Ncy * FE_geom.Ncz;
    const int N_devices = FE_geom.N_d;

    if (N_devices <= 0 || N_local <= 0) {
        std::cerr << "Error: Invalid FE_geom configuration in writeAvgEnergyToFile.\n";
        return;
    }

    std::filesystem::create_directories(outputFolder + "/energy/averageEnergy");

    for (int sim = 0; sim < N_devices; ++sim) {
        Type_var sum = 0.0;
        for (int i = 0; i < N_local; ++i)
            sum += energy[sim * N_local + i];

        Type_var avg = sum / static_cast<Type_var>(N_local);

        std::ostringstream fname;
        fname << outputFolder << "/energy/averageEnergy/avg_energy_device_" << sim << ".dat";

        std::ofstream fout(fname.str(), std::ios::app);
        if (!fout.is_open()) {
            std::cerr << "Error: Unable to open " << fname.str() << "\n";
            continue;
        }

        fout << std::scientific << std::setprecision(6) << time << " "
            << std::fixed << std::setprecision(16) << avg << "\n";
        fout.close();
    }
}

/**
 * @brief Writes time evolution of average electric field per device.
 */
void writeAvgFieldToFile(Type_var time, const Vec3<Type_var>* field) {
    const int N_local = FE_geom.Ncx * FE_geom.Ncy * FE_geom.Ncz;
    const int N_devices = FE_geom.N_d;

    if (N_devices <= 0 || N_local <= 0) {
        std::cerr << "Error: Invalid FE_geom configuration in writeAvgFieldToFile.\n";
        return;
    }

    std::filesystem::create_directories(outputFolder + "/field/averageField");

    for (int sim = 0; sim < N_devices; ++sim) {
        Type_var sum_x = 0, sum_y = 0, sum_z = 0;

        for (int i = 0; i < N_local; ++i) {
            const Vec3<Type_var>& f = field[sim * N_local + i];
            sum_x += f.x;
            sum_y += f.y;
            sum_z += f.z;
        }

        Vec3<Type_var> avg{ sum_x / N_local, sum_y / N_local, sum_z / N_local };

        std::ostringstream fname;
        fname << outputFolder << "/field/averageField/avg_field_device_" << sim << ".dat";

        std::ofstream fout(fname.str(), std::ios::app);
        if (!fout.is_open()) {
            std::cerr << "Error: Unable to open " << fname.str() << "\n";
            continue;
        }

        fout << std::scientific << std::setprecision(6) << time << " "
            << std::fixed << std::setprecision(16)
            << avg.x << " " << avg.y << " " << avg.z << "\n";
        fout.close();
    }
}

/**
 * @brief Writes full local polarization field per device for current step.
 */
void writeLocalPolarizationToFile(const Vec3<Type_var>* polarization, int stepIndex) {
    const int N_devices = FE_geom.N_d;
    const int N_local = FE_geom.Ncx * FE_geom.Ncy * FE_geom.Ncz;

    if (N_devices <= 0 || N_local <= 0) {
        std::cerr << "Error: Invalid FE_geom configuration in writeLocalPolarizationToFile.\n";
        return;
    }

    std::filesystem::create_directories(outputFolder + "/polarization");

    for (int sim = 0; sim < N_devices; ++sim) {
        std::string deviceFolder =
            outputFolder + "/polarization/localPolarization_device_" + std::to_string(sim);
        std::filesystem::create_directories(deviceFolder);

        std::ostringstream fname;
        fname << deviceFolder << "/polarization_" << stepIndex << ".dat";

        std::ofstream fout(fname.str());
        if (!fout.is_open()) {
            std::cerr << "Error: Unable to open " << fname.str() << "\n";
            continue;
        }

        int offset = sim * N_local;
        fout << std::scientific << std::setprecision(8);
        for (int i = 0; i < N_local; ++i) {
            const Vec3<Type_var>& p = polarization[offset + i];
            fout << std::fixed << std::setprecision(16)
                << p.x << " " << p.y << " " << p.z << "\n";
        }

        fout.close();
    }
}

/**
 * @brief Writes full local electric field vectors per device for current step.
 */
void writeLocalFieldToFile(const Vec3<Type_var>* localField, int stepIndex) {
    const int N_devices = FE_geom.N_d;
    const int N_local = FE_geom.Ncx * FE_geom.Ncy * FE_geom.Ncz;

    if (N_devices <= 0 || N_local <= 0) {
        std::cerr << "Error: Invalid FE_geom configuration in writeLocalFieldToFile.\n";
        return;
    }

    std::filesystem::create_directories(outputFolder + "/field");

    for (int sim = 0; sim < N_devices; ++sim) {
        std::string deviceFolder =
            outputFolder + "/field/localField_device_" + std::to_string(sim);
        std::filesystem::create_directories(deviceFolder);

        std::ostringstream fname;
        fname << deviceFolder << "/localField_" << stepIndex << ".dat";

        std::ofstream fout(fname.str());
        if (!fout.is_open()) {
            std::cerr << "Error: Unable to open " << fname.str() << "\n";
            continue;
        }

        int offset = sim * N_local;
        fout << std::scientific << std::setprecision(8);
        for (int i = 0; i < N_local; ++i) {
            const Vec3<Type_var>& E = localField[offset + i];
            fout << std::fixed << std::setprecision(16)
                << E.x << " " << E.y << " " << E.z << "\n";
        }

        fout.close();
    }
}
