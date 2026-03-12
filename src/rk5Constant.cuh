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
