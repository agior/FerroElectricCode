#pragma once                      // Prevent multiple inclusions of this header file

#ifndef HEUNINTEGRATOR_H          // Start include guard for HeunIntegrator header
#define HEUNINTEGRATOR_H

#include <thrust/device_vector.h>                // Include Thrust library for GPU device vector operations
#include "Vec3.h"                                // Include Vec3 class for handling 3D vectors
#include <thrust/device_vector.h>                // Include Thrust for GPU vector operations
#include <thrust/transform.h>                    // Include Thrust for transformations
#include <thrust/fill.h>                         // Include Thrust for filling vectors
#include <thrust/iterator/counting_iterator.h>   // Iterator for counting indices
#include "heunIntegrator.h"                      // Include Heun integrator functions
#include "coefficientVector.cuh"                       // Include normalization operations
#include "externalField.h"                       // Include external field calculations
#include "tdgl.cuh"
#include "coefficientVector.cuh"
#include <thrust/reduce.h>
#include "output.hpp"
#include <thrust/execution_policy.h>
#include "output_handle.h"


// Declaration of the Heun integrator function 
template <typename T>
void heun(
    thrust::device_vector<Vec3<T>>& polarization,      // Result polarization vector
    thrust::device_vector<Vec3<T>>& hFerro,            // Effective field vector
    thrust::device_vector<Vec3<T>>& noiseVector,       // Vector to store noise
    int gridSize,                                      // Size of the grid
    int Ncz,                                           // Number of cells along z direction
    T totalTime                                        // Time of simulation
);

#include "heunintegrator.cuh"     // Include the Heun integrator implementation
#endif // HEUNINTEGRATOR_H         // End include guard
