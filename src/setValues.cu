#pragma once
#include "setValues.cuh"
#include "tdgl.cu"

// ===== External Declarations =====
extern landau_free landau_free_param;
extern FE_geometry FE_geom;
extern initial_polarization initial_pol;
extern field_external external_field;
extern field_external_normalized external_field_vectors;
extern gradient gradient_field_param;
extern elastic elastic_field_param;
extern elastic elastic_field_param;
extern landau_free_final landau_vectors;
extern gradient_final gradient_vectors;

/**
 * @brief Sets up and normalizes all simulation parameters for the TDGL system.
 *
 * Performs normalization of Landau coefficients, gradient parameters,
 * external fields, grid spacing, and time step. Initializes CURAND states
 * for stochastic processes and prepares grid-based data structures.
 *
 * @tparam T Floating-point precision type (float or double).
 */
template <typename T>
void set_values()
{
    // ---- Fundamental constants ----
    Type_var Kb_val = 1.380649e-23;   // Boltzmann constant

    // ---- Derived grid size ----
    int gridSize = FE_geom.Ncx * FE_geom.Ncy * FE_geom.Ncz * FE_geom.N_d;

    // ---- Alpha1 vector sizing ----
    int vectorSizeTemp = std::max(
        static_cast<int>(landau_free_param.d_temperature.size()),
        static_cast<int>(landau_free_param.d_transition_temp.size())
    );

    int alpha1PrimeVectorSize = std::max(
        static_cast<int>(landau_free_param.d_alpha0.size()),
        vectorSizeTemp
    );

    // ---- Compute ??(T) values ----
    landau_free_param.d_alpha1.resize(alpha1PrimeVectorSize);
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(landau_free_param.d_alpha1.size())),
        landau_free_param.d_alpha1.begin(),
        multiplyAlpha1<Type_var>(
            thrust::raw_pointer_cast(landau_free_param.d_alpha0.data()),
            thrust::raw_pointer_cast(landau_free_param.d_temperature.data()),
            thrust::raw_pointer_cast(landau_free_param.d_transition_temp.data()),
            landau_free_param.d_alpha0.size(),
            landau_free_param.d_temperature.size(),
            landau_free_param.d_transition_temp.size(),
            gridSize,
            FE_geom.Ncz
        )
    );

    // ---- Variance vector (thermal fluctuations) ----
    landau_vectors.vector_variance.resize(landau_free_param.d_temperature.size());

    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(landau_free_param.d_temperature_noise.size())),
        landau_vectors.vector_variance.begin(),
        varianceCal<Type_var>(
            thrust::raw_pointer_cast(landau_free_param.d_spontaneousp_ref.data()),
            thrust::raw_pointer_cast(landau_free_param.d_relaxation_ref.data()),
            thrust::raw_pointer_cast(landau_free_param.d_alpha1_ref.data()),
            FE_geom.delta_x, FE_geom.delta_y, FE_geom.delta_z,
            FE_geom.dtime,
            Kb_val,
            thrust::raw_pointer_cast(landau_free_param.d_temperature_noise.data()),
            landau_free_param.d_temperature_noise.size(),
            gridSize,
            FE_geom.Ncz
        )
    );

    // ---- Normalize gradient coefficients (G?–G?) ----
    gradient_vectors.vector_G1.resize(gradient_field_param.d_G1.size());
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(gradient_vectors.vector_G1.size())),
        gradient_vectors.vector_G1.begin(),
        G1Prime<Type_var>(
            thrust::raw_pointer_cast(gradient_field_param.d_G0.data()),
            thrust::raw_pointer_cast(gradient_field_param.d_G1.data()),
            gradient_field_param.d_G1.size(),
            gridSize,
            FE_geom.Ncz
        )
    );

    gradient_vectors.vector_G2.resize(gradient_field_param.d_G2.size());
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(gradient_vectors.vector_G2.size())),
        gradient_vectors.vector_G2.begin(),
        G2Prime<Type_var>(
            thrust::raw_pointer_cast(gradient_field_param.d_G0.data()),
            thrust::raw_pointer_cast(gradient_field_param.d_G2.data()),
            gradient_field_param.d_G2.size(),
            gridSize,
            FE_geom.Ncz
        )
    );

    gradient_vectors.vector_G3.resize(gradient_field_param.d_G3.size());
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(gradient_vectors.vector_G3.size())),
        gradient_vectors.vector_G3.begin(),
        G3Prime<Type_var>(
            thrust::raw_pointer_cast(gradient_field_param.d_G0.data()),
            thrust::raw_pointer_cast(gradient_field_param.d_G3.data()),
            gradient_field_param.d_G3.size(),
            gridSize,
            FE_geom.Ncz
        )
    );

    gradient_vectors.vector_G4.resize(gradient_field_param.d_G4.size());
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(gradient_vectors.vector_G4.size())),
        gradient_vectors.vector_G4.begin(),
        G4Prime<Type_var>(
            thrust::raw_pointer_cast(gradient_field_param.d_G0.data()),
            thrust::raw_pointer_cast(gradient_field_param.d_G4.data()),
            gradient_field_param.d_G4.size(),
            gridSize,
            FE_geom.Ncz
        )
    );

    // ---- Normalize Landau ?-coefficients ----
    landau_vectors.vector_alpha1.resize(landau_free_param.d_alpha1.size());
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(landau_free_param.d_alpha1.size())),
        landau_vectors.vector_alpha1.begin(),
        alphaOnePrime<Type_var>(
            thrust::raw_pointer_cast(landau_free_param.d_alpha1.data()),
            thrust::raw_pointer_cast(landau_free_param.d_alpha1_ref.data())
        )
    );

    // Repeat normalization for ??–??
    landau_vectors.vector_alpha2.resize(landau_free_param.d_alpha2.size());
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(landau_vectors.vector_alpha2.size())),
        landau_vectors.vector_alpha2.begin(),
        alphaTwoPrime<Type_var>(
            thrust::raw_pointer_cast(landau_free_param.d_alpha1_ref.data()),
            thrust::raw_pointer_cast(landau_free_param.d_alpha2.data()),
            thrust::raw_pointer_cast(landau_free_param.d_spontaneousp_ref.data()),
            landau_free_param.d_alpha2.size(),
            gridSize,
            FE_geom.Ncz
        )
    );

    // Repeat normalization for ??–??
    landau_vectors.vector_alpha2.resize(landau_free_param.d_alpha2.size());

    landau_vectors.vector_alpha3.resize(landau_free_param.d_alpha3.size());
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(landau_vectors.vector_alpha3.size())),
        landau_vectors.vector_alpha3.begin(),
        alphaThreePrime<Type_var>(
            thrust::raw_pointer_cast(landau_free_param.d_alpha1_ref.data()),
            thrust::raw_pointer_cast(landau_free_param.d_alpha3.data()),
            thrust::raw_pointer_cast(landau_free_param.d_spontaneousp_ref.data()),
            landau_free_param.d_alpha3.size(),
            gridSize,
            FE_geom.Ncz
        )
    );

    landau_vectors.vector_alpha4.resize(landau_free_param.d_alpha4.size());
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(landau_vectors.vector_alpha4.size())),
        landau_vectors.vector_alpha4.begin(),
        alphaFourPrime<Type_var>(
            thrust::raw_pointer_cast(landau_free_param.d_alpha1_ref.data()),
            thrust::raw_pointer_cast(landau_free_param.d_alpha4.data()),
            thrust::raw_pointer_cast(landau_free_param.d_spontaneousp_ref.data()),
            landau_free_param.d_alpha4.size(),
            gridSize,
            FE_geom.Ncz
        )
    );

    landau_vectors.vector_alpha5.resize(landau_free_param.d_alpha5.size());
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(landau_vectors.vector_alpha5.size())),
        landau_vectors.vector_alpha5.begin(),
        alphaFivePrime<Type_var>(
            thrust::raw_pointer_cast(landau_free_param.d_alpha1_ref.data()),
            thrust::raw_pointer_cast(landau_free_param.d_alpha5.data()),
            thrust::raw_pointer_cast(landau_free_param.d_spontaneousp_ref.data()),
            landau_free_param.d_alpha5.size(),
            gridSize,
            FE_geom.Ncz
        )
    );

    landau_vectors.vector_alpha6.resize(landau_free_param.d_alpha6.size());
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(landau_vectors.vector_alpha6.size())),
        landau_vectors.vector_alpha6.begin(),
        alphaSixPrime<Type_var>(
            thrust::raw_pointer_cast(landau_free_param.d_alpha1_ref.data()),
            thrust::raw_pointer_cast(landau_free_param.d_alpha6.data()),
            thrust::raw_pointer_cast(landau_free_param.d_spontaneousp_ref.data()),
            landau_free_param.d_alpha6.size(),
            gridSize,
            FE_geom.Ncz
        )
    );
    // ============================================================
    // Q11Prime Call
    // ============================================================
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(elastic_field_param.Q11.size())),
        elastic_field_param.Q11.begin(),
        Q11Prime<Type_var>(
            thrust::raw_pointer_cast(landau_free_param.d_alpha1_ref.data()),
            thrust::raw_pointer_cast(elastic_field_param.Q11.data()),
            thrust::raw_pointer_cast(landau_free_param.d_spontaneousp_ref.data()),
            elastic_field_param.Q11.size(),
            gridSize,
            FE_geom.Ncz
        )
    );

    // ============================================================
    // Q12Prime Call
    // ============================================================
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(elastic_field_param.Q12.size())),
        elastic_field_param.Q12.begin(),
        Q12Prime<Type_var>(
            thrust::raw_pointer_cast(landau_free_param.d_alpha1_ref.data()),
            thrust::raw_pointer_cast(elastic_field_param.Q12.data()),
            thrust::raw_pointer_cast(landau_free_param.d_spontaneousp_ref.data()),
            elastic_field_param.Q12.size(),
            gridSize,
            FE_geom.Ncz
        )
    );

    // ============================================================
    // Q44Prime Call
    // ============================================================
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(elastic_field_param.Q44.size())),
        elastic_field_param.Q44.begin(),
        Q44Prime<Type_var>(
            thrust::raw_pointer_cast(landau_free_param.d_alpha1_ref.data()),
            thrust::raw_pointer_cast(elastic_field_param.Q44.data()),
            thrust::raw_pointer_cast(landau_free_param.d_spontaneousp_ref.data()),
            elastic_field_param.Q44.size(),
            gridSize,
            FE_geom.Ncz
        )
    );

    // ============================================================
    // C11Prime Call
    // ============================================================
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(elastic_field_param.C11.size())),
        elastic_field_param.C11.begin(),
        C11Prime<Type_var>(
            thrust::raw_pointer_cast(landau_free_param.d_alpha1_ref.data()),
            thrust::raw_pointer_cast(elastic_field_param.C11.data()),
            thrust::raw_pointer_cast(landau_free_param.d_spontaneousp_ref.data()),
            elastic_field_param.C11.size(),
            gridSize,
            FE_geom.Ncz
        )
    );

    // ============================================================
    // C12Prime Call
    // ============================================================
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(elastic_field_param.C12.size())),
        elastic_field_param.C12.begin(),
        C12Prime<Type_var>(
            thrust::raw_pointer_cast(landau_free_param.d_alpha1_ref.data()),
            thrust::raw_pointer_cast(elastic_field_param.C12.data()),
            thrust::raw_pointer_cast(landau_free_param.d_spontaneousp_ref.data()),
            elastic_field_param.C12.size(),
            gridSize,
            FE_geom.Ncz
        )
    );

    // ============================================================
    // C44Prime Call
    // ============================================================
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(elastic_field_param.C44.size())),
        elastic_field_param.C44.begin(),
        C44Prime<Type_var>(
            thrust::raw_pointer_cast(landau_free_param.d_alpha1_ref.data()),
            thrust::raw_pointer_cast(elastic_field_param.C44.data()),
            thrust::raw_pointer_cast(landau_free_param.d_spontaneousp_ref.data()),
            elastic_field_param.C44.size(),
            gridSize,
            FE_geom.Ncz
        )
    );


    // ---- Normalize external fields ----
    external_field_vectors.vector_field.resize(external_field.d_field.size());
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(external_field_vectors.vector_field.size())),
        external_field_vectors.vector_field.begin(),
        ExternalField<Type_var>(
            external_field.FLAG_INPUT_TYPE,
            thrust::raw_pointer_cast(landau_free_param.d_alpha1_ref.data()),
            thrust::raw_pointer_cast(landau_free_param.d_spontaneousp_ref.data()),
            thrust::raw_pointer_cast(external_field.d_field.data()),
            external_field.d_field.size(),
            FE_geom.Ncz,
            gridSize
        )
    );

    // ---- AC field normalization ----
    external_field_vectors.vector_AC_phase.resize(external_field.d_AC_phase.size());
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(external_field.d_AC_phase.size())),
        external_field_vectors.vector_AC_phase.begin(),
        phaseAc<Type_var>(
            thrust::raw_pointer_cast(external_field.d_AC_phase.data()),
            external_field.d_AC_phase.size(),
            FE_geom.Ncz,
            gridSize
        )
    );

    external_field_vectors.vector_Ac_field.resize(external_field.d_AC_field.size());
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(external_field_vectors.vector_Ac_field.size())),
        external_field_vectors.vector_Ac_field.begin(),
        ExternalFieldAc<Type_var>(
            thrust::raw_pointer_cast(landau_free_param.d_alpha1_ref.data()),
            thrust::raw_pointer_cast(landau_free_param.d_spontaneousp_ref.data()),
            thrust::raw_pointer_cast(external_field.d_AC_field.data()),
            external_field.d_AC_field.size(),
            FE_geom.Ncz,
            gridSize
        )
    );

    // ---- Spontaneous polarization magnitude ----
    external_field_vectors.MagSp.resize(1);
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(landau_free_param.d_spontaneousp_ref.size())),
        external_field_vectors.MagSp.begin(),
        SpontaneousPMag<Type_var>(
            thrust::raw_pointer_cast(landau_free_param.d_spontaneousp_ref.data())
        )
    );

    // ---- Normalize simulation time and steps ----
    Type_var simulationTime = FE_geom.time_sim;
    Type_var step_size = FE_geom.dtime;

    FE_geom.time_sim = simulationTime *
        landau_free_param.d_relaxation_ref[0] *
        abs(landau_free_param.d_alpha1_ref[0]);

    FE_geom.dtime = step_size *
        landau_free_param.d_relaxation_ref[0] *
        abs(landau_free_param.d_alpha1_ref[0]);

    FE_geom.dtime_min = 0.2 * FE_geom.dtime;
    FE_geom.dtime_max = 5 * FE_geom.dtime;

    // ---- Normalize grid spacing ----
    Type_var distance_x = FE_geom.delta_x;
    Type_var distance_y = FE_geom.delta_y;
    Type_var distance_z = FE_geom.delta_z;

    FE_geom.delta_x = sqrt(abs(landau_free_param.d_alpha1_ref[0]) / gradient_field_param.d_G0[0]) * distance_x;
    FE_geom.delta_y = sqrt(abs(landau_free_param.d_alpha1_ref[0]) / gradient_field_param.d_G0[0]) * distance_y;
    FE_geom.delta_z = sqrt(abs(landau_free_param.d_alpha1_ref[0]) / gradient_field_param.d_G0[0]) * distance_z;

    // ---- Apply shape transformation to the mesh ----
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(gridSize)),
        FE_geom.shape.begin(),
        shape<Type_var>(
            FE_geom.FLAG_SHAPE,
            thrust::raw_pointer_cast(FE_geom.shape.data()),
            FE_geom.Ncx, FE_geom.Ncy, FE_geom.Ncz, FE_geom.delta_x
        )
    );

    // ---- Initialize polarization field ----
    int polarizationValuesSize = initial_pol.initial_pol_vector.size();
    initial_pol.initial_pol_vector.resize(gridSize);

    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(gridSize)),
        initial_pol.initial_pol_vector.begin(),
        initialPolarization<Type_var>(
            thrust::raw_pointer_cast(initial_pol.initial_pol_vector.data()),
            polarizationValuesSize,
            gridSize,
            FE_geom.Ncz
        )
    );

    // Normalize and apply shape factor to polarization
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(gridSize)),
        initial_pol.initial_pol_vector.begin(),
        initialPol<Type_var>(
            thrust::raw_pointer_cast(initial_pol.initial_pol_vector.data()),
            thrust::raw_pointer_cast(FE_geom.shape.data()),
            thrust::raw_pointer_cast(landau_free_param.d_spontaneousp_ref.data()),
            gridSize,
            FE_geom.Ncz
        )
    );

    // ---- Initialize CURAND RNG states ----
    cudaMalloc(&landau_free_param.states, gridSize * sizeof(curandState));
    init_curand_states << <(gridSize + 255) / 256, 256 >> > (landau_free_param.states, time(0), gridSize);
    cudaDeviceSynchronize();

    // ---- Finalize mesh shape count ----
    getShape(FE_geom.shape);
    FE_geom.n_elements = 0;
    for (int i = 0; i < FE_geom.shape.size(); ++i)
    {
        if (FE_geom.shape[i] != 0)
            FE_geom.n_elements++;
    }
}
