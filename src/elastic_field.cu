#include "elastic_field.cuh"

// ============================================================================
//                                Helpers
// ============================================================================

inline int idx3(int i, int j, int k, int Nx, int Ny, int Nz) {
    return (i * Ny + j) * Nz + k;
}

void save_slice(
    const std::string& fname,
    const thrust::host_vector<double>& arr,
    int Nx, int Ny, int Nz, int zslice
) {
    std::ofstream of(fname);
    of.setf(std::ios::scientific);
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            of << arr[idx3(i, j, zslice, Nx, Ny, Nz)];
            if (j < Ny - 1) of << ",";
        }
        of << "\n";
    }
    of.close();
}

// ============================================================================
//                          Kernel Implementations
// ============================================================================

__global__ void solve_kernel(
    const cuC* exx, const cuC* eyy, const cuC* ezz,
    const cuC* exy, const cuC* exz, const cuC* eyz,
    cuC* ux_hat, cuC* uy_hat, cuC* uz_hat,
    int Nx, int Ny, int Nz,
    double Lx, double Ly, double Lz,
    double C11, double C12, double C44
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= Nx || j >= Ny || k >= Nz) return;

    int id = (i * Ny + j) * Nz + k;

    double kx = 2 * M_PI * ((i <= Nx / 2) ? i : i - Nx) / Lx;
    double ky = 2 * M_PI * ((j <= Ny / 2) ? j : j - Ny) / Ly;
    double kz = 2 * M_PI * ((k <= Nz / 2) ? k : k - Nz) / Lz;

    if (i == 0 && j == 0 && k == 0) {
        ux_hat[id].x = uy_hat[id].x = uz_hat[id].x = 0;
        ux_hat[id].y = uy_hat[id].y = uz_hat[id].y = 0;
        return;
    }

    cuC Exx = exx[id], Eyy = eyy[id], Ezz = ezz[id];
    cuC Exy = exy[id], Exz = exz[id], Eyz = eyz[id];

    auto mult_minus_i = [&](cuC t) { return cuC{ t.y, -t.x }; };

    // -------------------- b(k): anisotropic RHS --------------------
    cuC bx = mult_minus_i({
        C11 * kx * Exx.x + C12 * kx * Eyy.x + C12 * kx * Ezz.x
        + C44 * ky * Exy.x + C44 * kz * Exz.x,
        C11 * kx * Exx.y + C12 * kx * Eyy.y + C12 * kx * Ezz.y
        + C44 * ky * Exy.y + C44 * kz * Exz.y
        });

    cuC by = mult_minus_i({
        C44 * kx * Exy.x + C11 * ky * Eyy.x + C12 * ky * Exx.x + C12 * ky * Ezz.x
        + C44 * kz * Eyz.x,
        C44 * kx * Exy.y + C11 * ky * Eyy.y + C12 * ky * Exx.y + C12 * ky * Ezz.y
        + C44 * kz * Eyz.y
        });

    cuC bz = mult_minus_i({
        C44 * kx * Exz.x + C44 * ky * Eyz.x + C11 * kz * Ezz.x
        + C12 * kz * Exx.x + C12 * kz * Eyy.x,
        C44 * kx * Exz.y + C44 * ky * Eyz.y + C11 * kz * Ezz.y
        + C12 * kz * Exx.y + C12 * kz * Eyy.y
        });

    // -------------------- A(k): anisotropic stiffness matrix --------------------
    double Axx = C11 * kx * kx + C44 * (ky * ky + kz * kz);
    double Ayy = C11 * ky * ky + C44 * (kx * kx + kz * kz);
    double Azz = C11 * kz * kz + C44 * (kx * kx + ky * ky);
    double Axy = (C12 + C44) * kx * ky;
    double Axz = (C12 + C44) * kx * kz;
    double Ayz = (C12 + C44) * ky * kz;

    double det = Axx * (Ayy * Azz - Ayz * Ayz)
        - Axy * (Axy * Azz - Ayz * Axz)
        + Axz * (Axy * Ayz - Ayy * Axz);

    if (fabs(det) < 1e-30) {
        ux_hat[id].x = uy_hat[id].x = uz_hat[id].x = 0;
        ux_hat[id].y = uy_hat[id].y = uz_hat[id].y = 0;
        return;
    }

    double invdet = 1.0 / det;
    double inv_xx = (Ayy * Azz - Ayz * Ayz) * invdet;
    double inv_xy = (Axz * Ayz - Axy * Azz) * invdet;
    double inv_xz = (Axy * Ayz - Axz * Ayy) * invdet;
    double inv_yy = (Axx * Azz - Axz * Axz) * invdet;
    double inv_yz = (Axy * Axz - Axx * Ayz) * invdet;
    double inv_zz = (Axx * Ayy - Axy * Axy) * invdet;

    // -------------------- Solve u? = A⁻¹ b --------------------
    ux_hat[id].x = bx.x * inv_xx + by.x * inv_xy + bz.x * inv_xz;
    ux_hat[id].y = bx.y * inv_xx + by.y * inv_xy + bz.y * inv_xz;
    uy_hat[id].x = bx.x * inv_xy + by.x * inv_yy + bz.x * inv_yz;
    uy_hat[id].y = bx.y * inv_xy + by.y * inv_yy + bz.y * inv_yz;
    uz_hat[id].x = bx.x * inv_xz + by.x * inv_yz + bz.x * inv_zz;
    uz_hat[id].y = bx.y * inv_xz + by.y * inv_yz + bz.y * inv_zz;
}

__global__ void deriv_kernel(
    const cuC* u_hat,
    cuC* out_x, cuC* out_y, cuC* out_z,
    int Nx, int Ny, int Nz,
    double Lx, double Ly, double Lz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= Nx || j >= Ny || k >= Nz) return;

    int id = (i * Ny + j) * Nz + k;

    double kx = 2 * M_PI * ((i <= Nx / 2) ? i : i - Nx) / Lx;
    double ky = 2 * M_PI * ((j <= Ny / 2) ? j : j - Ny) / Ly;
    double kz = 2 * M_PI * ((k <= Nz / 2) ? k : k - Nz) / Lz;

    cuC t = u_hat[id];
    out_x[id].x = -kx * t.y; out_x[id].y = kx * t.x;
    out_y[id].x = -ky * t.y; out_y[id].y = ky * t.x;
    out_z[id].x = -kz * t.y; out_z[id].y = kz * t.x;
}

// ============================================================================
//                          Functor Implementations
// ============================================================================

__host__ __device__
ConvertFromCuCToVec3Functor::ConvertFromCuCToVec3Functor(
    cuC* ux_hat_, cuC* uy_hat_, cuC* uz_hat_,
    cuC* ux_x_hat_, cuC* ux_y_hat_, cuC* ux_z_hat_,
    cuC* uy_x_hat_, cuC* uy_y_hat_, cuC* uy_z_hat_,
    cuC* uz_x_hat_, cuC* uz_y_hat_, cuC* uz_z_hat_,
    Vec3<double>* U_out_, Vec3<double>* Ux_deriv_out_,
    Vec3<double>* Uy_deriv_out_, Vec3<double>* Uz_deriv_out_,
    double invN_, int N_
)
    : ux_hat(ux_hat_), uy_hat(uy_hat_), uz_hat(uz_hat_),
    ux_x_hat(ux_x_hat_), ux_y_hat(ux_y_hat_), ux_z_hat(ux_z_hat_),
    uy_x_hat(uy_x_hat_), uy_y_hat(uy_y_hat_), uy_z_hat(uy_z_hat_),
    uz_x_hat(uz_x_hat_), uz_y_hat(uz_y_hat_), uz_z_hat(uz_z_hat_),
    U_out(U_out_), Ux_deriv_out(Ux_deriv_out_),
    Uy_deriv_out(Uy_deriv_out_), Uz_deriv_out(Uz_deriv_out_),
    invN(invN_), N(N_) {
}

__host__ __device__
void ConvertFromCuCToVec3Functor::operator()(const int& idx) const {
    if (idx < 0 || idx >= N) return;

    // Real part contains real-space values after inverse FFT; multiply by invN
    Vec3<double> u;
    u.x = ux_hat[idx].x * invN;
    u.y = uy_hat[idx].x * invN;
    u.z = uz_hat[idx].x * invN;
    U_out[idx] = u;

    Vec3<double> ux_d, uy_d, uz_d;
    ux_d.x = ux_x_hat[idx].x * invN;
    ux_d.y = ux_y_hat[idx].x * invN;
    ux_d.z = ux_z_hat[idx].x * invN;
    Ux_deriv_out[idx] = ux_d;

    uy_d.x = uy_x_hat[idx].x * invN;
    uy_d.y = uy_y_hat[idx].x * invN;
    uy_d.z = uy_z_hat[idx].x * invN;
    Uy_deriv_out[idx] = uy_d;

    uz_d.x = uz_x_hat[idx].x * invN;
    uz_d.y = uz_y_hat[idx].x * invN;
    uz_d.z = uz_z_hat[idx].x * invN;
    Uz_deriv_out[idx] = uz_d;
}

__host__ __device__
FullPhysicsFunctor::FullPhysicsFunctor(
    Vec3<double>* P_, Vec3<double>* U_, Vec3<double>* Ux_deriv_,
    Vec3<double>* Uy_deriv_, Vec3<double>* Uz_deriv_,
    Vec3<double>* eps_norm_out_, Vec3<double>* eps_shear_out_,
    double* fE1_out_, double* fE2_out_, double* fE3_out_,
    Vec3<double>* field_P_out_, double* field_mag_out_, Vec3<double>* heff_, double* elastic_energy_,
    double Q11_, double Q12_, double Q44_,
    double C11_, double C12_, double C44_,
    double q11_, double q12_, double q44_,
    double b11_, double b12_, int N_,
    double eps_ext_xx_, double eps_ext_yy_, double eps_ext_zz_,
    double eps_ext_xy_, double eps_ext_xz_, double eps_ext_yz_,
    int FLAG_USE_DISPLACEMENT_FIELD_, int FLAG_USE_POLARIZATION_FIELD_
)
    : P(P_), U(U_), Ux_deriv(Ux_deriv_), Uy_deriv(Uy_deriv_), Uz_deriv(Uz_deriv_),
    eps_norm_out(eps_norm_out_), eps_shear_out(eps_shear_out_),
    fE1_out(fE1_out_), fE2_out(fE2_out_), fE3_out(fE3_out_),
    field_P_out(field_P_out_), field_mag_out(field_mag_out_), heff(heff_), elastic_energy(elastic_energy_),
    Q11(Q11_), Q12(Q12_), Q44(Q44_),
    C11(C11_), C12(C12_), C44(C44_),
    q11(q11_), q12(q12_), q44(q44_),
    b11(b11_), b12(b12_), N(N_),
    eps_ext_xx(eps_ext_xx_), eps_ext_yy(eps_ext_yy_), eps_ext_zz(eps_ext_zz_),
    eps_ext_xy(eps_ext_xy_), eps_ext_xz(eps_ext_xz_), eps_ext_yz(eps_ext_yz_),
    FLAG_USE_DISPLACEMENT_FIELD(FLAG_USE_DISPLACEMENT_FIELD_), 
    FLAG_USE_POLARIZATION_FIELD(FLAG_USE_POLARIZATION_FIELD_){
}

__host__ __device__
void FullPhysicsFunctor::operator()(const int& idx) const {
    if (idx < 0 || idx >= N) return;

    // -------------------- Read polarization --------------------
    Vec3<double> p = P[idx];
    double p1 = p.x, p2 = p.y, p3 = p.z;

    // -------------------- Eigenstrain (local) --------------------
    double exx_r = Q11 * p1 * p1 + Q12 * (p2 * p2 + p3 * p3);
    double eyy_r = Q11 * p2 * p2 + Q12 * (p1 * p1 + p3 * p3);
    double ezz_r = Q11 * p3 * p3 + Q12 * (p1 * p1 + p2 * p2);
    double exy_r = Q44 * p1 * p2;
    double exz_r = Q44 * p1 * p3;
    double eyz_r = Q44 * p2 * p3;

    // -------------------- Strain calculation --------------------
    double eps_xx = 0.0, eps_yy = 0.0, eps_zz = 0.0;
    double eps_xy = 0.0, eps_xz = 0.0, eps_yz = 0.0;

    if (FLAG_USE_DISPLACEMENT_FIELD) {
        // Use displacement derivatives from Ux_deriv, Uy_deriv, Uz_deriv
        Vec3<double> ux_d = Ux_deriv[idx];
        Vec3<double> uy_d = Uy_deriv[idx];
        Vec3<double> uz_d = Uz_deriv[idx];

        eps_xx = ux_d.x;
        eps_yy = uy_d.y;
        eps_zz = uz_d.z;
        eps_xy = 0.5 * (ux_d.y + uy_d.x);
        eps_xz = 0.5 * (ux_d.z + uz_d.x);
        eps_yz = 0.5 * (uy_d.z + uz_d.y);
    }
    else {
        // Skip displacement-dependent fields: start from zero strain (we'll add eps_ext below)
        eps_xx = eps_yy = eps_zz = 0.0;
        eps_xy = eps_xz = eps_yz = 0.0;
    }

    // -------------------- Add externally-induced homogeneous strain --------------------
    eps_xx += eps_ext_xx;
    eps_yy += eps_ext_yy;
    eps_zz += eps_ext_zz;
    eps_xy += eps_ext_xy;
    eps_xz += eps_ext_xz;
    eps_yz += eps_ext_yz;

    // store eps as Vec3
    eps_norm_out[idx] = Vec3<double>(eps_xx, eps_yy, eps_zz);
    eps_shear_out[idx] = Vec3<double>(eps_xy, eps_xz, eps_yz);

    // -------------------- Elastic energy --------------------
    double fE1 = 0.5 * (
        C11 * (eps_xx * eps_xx + eps_yy * eps_yy + eps_zz * eps_zz) +
        C12 * (eps_xx * eps_yy + eps_yy * eps_zz + eps_xx * eps_zz) +
        2.0 * C44 * (eps_xy * eps_xy + eps_yz * eps_yz + eps_xz * eps_xz)
        );
    fE1_out[idx] = fE1;

    // -------------------- Landau free energy --------------------
    double fE2 =
        b11 * (p1 * p1 * p1 * p1 + p2 * p2 * p2 * p2 + p3 * p3 * p3 * p3) +
        b12 * (p1 * p1 * p2 * p2 + p2 * p2 * p3 * p3 + p1 * p1 * p3 * p3);
    fE2_out[idx] = fE2;

    // -------------------- Electromechanical coupling --------------------
    double fE3 = (
        -(q11 * eps_xx + q12 * eps_yy + q12 * eps_zz) * p1 * p1
        - (q11 * eps_yy + q12 * eps_xx + q12 * eps_zz) * p2 * p2
        - (q11 * eps_zz + q12 * eps_xx + q12 * eps_yy) * p3 * p3
        - 2.0 * q44 * (eps_xy * p1 * p2 + eps_yz * p2 * p3 + eps_xz * p1 * p3)
        );
    fE3_out[idx] = fE3;

    elastic_energy[idx] = fE1 + fE2 + fE3;

    // -------------------- Derivatives --------------------
    double dfE2_p1 = 4.0 * b11 * p1 * p1 * p1 + 2.0 * b12 * p1 * (p2 * p2 + p3 * p3);
    double dfE2_p2 = 4.0 * b11 * p2 * p2 * p2 + 2.0 * b12 * p2 * (p1 * p1 + p3 * p3);
    double dfE2_p3 = 4.0 * b11 * p3 * p3 * p3 + 2.0 * b12 * p3 * (p1 * p1 + p2 * p2);

    double pref1 = (q11 * eps_xx + q12 * eps_yy + q12 * eps_zz);
    double pref2 = (q11 * eps_yy + q12 * eps_xx + q12 * eps_zz);
    double pref3 = (q11 * eps_zz + q12 * eps_xx + q12 * eps_yy);

    double dfE3_p1 = -2.0 * pref1 * p1 - 2.0 * q44 * (eps_xy * p2 + eps_xz * p3);
    double dfE3_p2 = -2.0 * pref2 * p2 - 2.0 * q44 * (eps_xy * p1 + eps_yz * p3);
    double dfE3_p3 = -2.0 * pref3 * p3 - 2.0 * q44 * (eps_yz * p2 + eps_xz * p1);

    // -------------------- Total effective field --------------------
    double fx;
    double fy;
    double fz;

    if (FLAG_USE_POLARIZATION_FIELD == 1) {
        fx = (dfE2_p1 + dfE3_p1);
        fy = (dfE2_p2 + dfE3_p2);
        fz = (dfE2_p3 + dfE3_p3);
    }
    else {
        fx = dfE3_p1;
        fy = dfE3_p2;
        fz = dfE3_p3;
    }

    field_P_out[idx] = Vec3<double>(fx, fy, fz);
    heff[idx].x += fx;
    heff[idx].y += fy;
    heff[idx].z += fz;
    field_mag_out[idx] = sqrt(fx * fx + fy * fy + fz * fz);
}

// ============================================================================
//                        compute_elastic_field (host)
// ============================================================================

void compute_elastic_field(
    thrust::device_vector<Vec3<double>>& d_P,      ///< Input polarization field
    thrust::device_vector<Vec3<double>>& d_heff,   ///< Effective field (output)
    thrust::device_vector<double>& d_elastic_energy,
    int Nx, int Ny, int Nz,                        ///< Grid dimensions
    double Lx, double Ly, double Lz,               ///< Domain lengths
    double Q11, double Q12, double Q44,            ///< Electrostrictive constants
    double C11, double C12, double C44,            ///< Elastic constants
    int FLAG_OUTPUT_FILES,                          ///< Output control flag
    double sigma_xx_ext, double sigma_yy_ext, double sigma_zz_ext,
    double sigma_xy_ext, double sigma_xz_ext, double sigma_yz_ext,
    int FLAG_USE_DISPLACEMENT_FIELD,
    int FLAG_POLARIZATION_FIELD
) {
    const long N = static_cast<long>(Nx) * Ny * Nz;

    // ---------------- Compute cubic compliance scalars ----------------
    // Inverse of stiffness for cubic symmetry (block inverse formulas)
    // Note: ensure denominator is not zero (material must be physically valid)
    double D = (C11 - C12) * (C11 + 2.0 * C12);
    double S11 = 0.0, S12 = 0.0, S44 = 0.0;
    if (fabs(D) > 1e-30) {
        S11 = (C11 + C12) / D;
        S12 = -C12 / D;
    }
    if (fabs(C44) > 1e-30) S44 = 1.0 / C44;

    // ---------------- external homogeneous strain from stress ----------------
    double eps_ext_xx = S11 * sigma_xx_ext + S12 * (sigma_yy_ext + sigma_zz_ext);
    double eps_ext_yy = S11 * sigma_yy_ext + S12 * (sigma_xx_ext + sigma_zz_ext);
    double eps_ext_zz = S11 * sigma_zz_ext + S12 * (sigma_xx_ext + sigma_yy_ext);
    double eps_ext_xy = S44 * sigma_xy_ext;
    double eps_ext_xz = S44 * sigma_xz_ext;
    double eps_ext_yz = S44 * sigma_yz_ext;

    // ---------------- Copy polarization to host ----------------
    thrust::host_vector<Vec3<double>> P_host = d_P;

    // ---------------- Build eigenstrain components ----------------
    thrust::host_vector<double> exx_h(N), eyy_h(N), ezz_h(N);
    thrust::host_vector<double> exy_h(N), exz_h(N), eyz_h(N);

    for (long i = 0; i < N; ++i) {
        double p1 = P_host[i].x, p2 = P_host[i].y, p3 = P_host[i].z;
        exx_h[i] = Q11 * p1 * p1 + Q12 * (p2 * p2 + p3 * p3);
        eyy_h[i] = Q11 * p2 * p2 + Q12 * (p1 * p1 + p3 * p3);
        ezz_h[i] = Q11 * p3 * p3 + Q12 * (p1 * p1 + p2 * p2);
        exy_h[i] = Q44 * p1 * p2;
        exz_h[i] = Q44 * p1 * p3;
        eyz_h[i] = Q44 * p2 * p3;
    }

    // ---------------- Convert to complex form ----------------
    thrust::host_vector<cuC> exx_hc(N), eyy_hc(N), ezz_hc(N);
    thrust::host_vector<cuC> exy_hc(N), exz_hc(N), eyz_hc(N);

    for (long i = 0; i < N; ++i) {
        exx_hc[i].x = exx_h[i]; exx_hc[i].y = 0.0;
        eyy_hc[i].x = eyy_h[i]; eyy_hc[i].y = 0.0;
        ezz_hc[i].x = ezz_h[i]; ezz_hc[i].y = 0.0;
        exy_hc[i].x = exy_h[i]; exy_hc[i].y = 0.0;
        exz_hc[i].x = exz_h[i]; exz_hc[i].y = 0.0;
        eyz_hc[i].x = eyz_h[i]; eyz_hc[i].y = 0.0;
    }

    // ---------------- Copy complex fields to device ----------------
    thrust::device_vector<cuC> d_exx = exx_hc;
    thrust::device_vector<cuC> d_eyy = eyy_hc;
    thrust::device_vector<cuC> d_ezz = ezz_hc;
    thrust::device_vector<cuC> d_exy = exy_hc;
    thrust::device_vector<cuC> d_exz = exz_hc;
    thrust::device_vector<cuC> d_eyz = eyz_hc;

    // ---------------- Allocate FFT buffers ----------------
    thrust::device_vector<cuC> d_ux_hat(N), d_uy_hat(N), d_uz_hat(N);
    thrust::device_vector<cuC> d_ux_x_hat(N), d_ux_y_hat(N), d_ux_z_hat(N);
    thrust::device_vector<cuC> d_uy_x_hat(N), d_uy_y_hat(N), d_uy_z_hat(N);
    thrust::device_vector<cuC> d_uz_x_hat(N), d_uz_y_hat(N), d_uz_z_hat(N);

    // ---------------- cuFFT plan ----------------
    cufftHandle plan;
    CHECK_CUFFT(cufftPlan3d(&plan, Nx, Ny, Nz, CUFFT_Z2Z));

    // Raw pointers for CUDA calls
    cuC* d_exx_ptr = thrust::raw_pointer_cast(d_exx.data());
    cuC* d_eyy_ptr = thrust::raw_pointer_cast(d_eyy.data());
    cuC* d_ezz_ptr = thrust::raw_pointer_cast(d_ezz.data());
    cuC* d_exy_ptr = thrust::raw_pointer_cast(d_exy.data());
    cuC* d_exz_ptr = thrust::raw_pointer_cast(d_exz.data());
    cuC* d_eyz_ptr = thrust::raw_pointer_cast(d_eyz.data());

    cuC* d_ux_hat_ptr = thrust::raw_pointer_cast(d_ux_hat.data());
    cuC* d_uy_hat_ptr = thrust::raw_pointer_cast(d_uy_hat.data());
    cuC* d_uz_hat_ptr = thrust::raw_pointer_cast(d_uz_hat.data());

    // ---------------- Forward FFTs ----------------
    CHECK_CUFFT(cufftExecZ2Z(plan, d_exx_ptr, d_exx_ptr, CUFFT_FORWARD));
    CHECK_CUFFT(cufftExecZ2Z(plan, d_eyy_ptr, d_eyy_ptr, CUFFT_FORWARD));
    CHECK_CUFFT(cufftExecZ2Z(plan, d_ezz_ptr, d_ezz_ptr, CUFFT_FORWARD));
    CHECK_CUFFT(cufftExecZ2Z(plan, d_exy_ptr, d_exy_ptr, CUFFT_FORWARD));
    CHECK_CUFFT(cufftExecZ2Z(plan, d_exz_ptr, d_exz_ptr, CUFFT_FORWARD));
    CHECK_CUFFT(cufftExecZ2Z(plan, d_eyz_ptr, d_eyz_ptr, CUFFT_FORWARD));
    CHECK_CUDA(cudaDeviceSynchronize());

    // ---------------- Solve displacement fields (optional) ----------------
    dim3 blockDim(4, 4, 4);
    dim3 gridDim(
        (Nx + blockDim.x - 1) / blockDim.x,
        (Ny + blockDim.y - 1) / blockDim.y,
        (Nz + blockDim.z - 1) / blockDim.z
    );

    if (FLAG_USE_DISPLACEMENT_FIELD) {
        solve_kernel << <gridDim, blockDim >> > (
            d_exx_ptr, d_eyy_ptr, d_ezz_ptr,
            d_exy_ptr, d_exz_ptr, d_eyz_ptr,
            d_ux_hat_ptr, d_uy_hat_ptr, d_uz_hat_ptr,
            Nx, Ny, Nz, Lx, Ly, Lz, C11, C12, C44
            );
        CHECK_CUDA(cudaDeviceSynchronize());

        // ---------------- Derivative kernels ----------------
        deriv_kernel << <gridDim, blockDim >> > (d_ux_hat_ptr, /*out*/ thrust::raw_pointer_cast(d_ux_x_hat.data()),
            thrust::raw_pointer_cast(d_ux_y_hat.data()), thrust::raw_pointer_cast(d_ux_z_hat.data()),
            Nx, Ny, Nz, Lx, Ly, Lz);
        deriv_kernel << <gridDim, blockDim >> > (d_uy_hat_ptr, thrust::raw_pointer_cast(d_uy_x_hat.data()),
            thrust::raw_pointer_cast(d_uy_y_hat.data()), thrust::raw_pointer_cast(d_uy_z_hat.data()),
            Nx, Ny, Nz, Lx, Ly, Lz);
        deriv_kernel << <gridDim, blockDim >> > (d_uz_hat_ptr, thrust::raw_pointer_cast(d_uz_x_hat.data()),
            thrust::raw_pointer_cast(d_uz_y_hat.data()), thrust::raw_pointer_cast(d_uz_z_hat.data()),
            Nx, Ny, Nz, Lx, Ly, Lz);
        CHECK_CUDA(cudaDeviceSynchronize());

        // ---------------- Inverse FFTs ----------------
        CHECK_CUFFT(cufftExecZ2Z(plan, d_ux_hat_ptr, d_ux_hat_ptr, CUFFT_INVERSE));
        CHECK_CUFFT(cufftExecZ2Z(plan, d_uy_hat_ptr, d_uy_hat_ptr, CUFFT_INVERSE));
        CHECK_CUFFT(cufftExecZ2Z(plan, d_uz_hat_ptr, d_uz_hat_ptr, CUFFT_INVERSE));

        CHECK_CUFFT(cufftExecZ2Z(plan, thrust::raw_pointer_cast(d_ux_x_hat.data()), thrust::raw_pointer_cast(d_ux_x_hat.data()), CUFFT_INVERSE));
        CHECK_CUFFT(cufftExecZ2Z(plan, thrust::raw_pointer_cast(d_ux_y_hat.data()), thrust::raw_pointer_cast(d_ux_y_hat.data()), CUFFT_INVERSE));
        CHECK_CUFFT(cufftExecZ2Z(plan, thrust::raw_pointer_cast(d_ux_z_hat.data()), thrust::raw_pointer_cast(d_ux_z_hat.data()), CUFFT_INVERSE));
        CHECK_CUFFT(cufftExecZ2Z(plan, thrust::raw_pointer_cast(d_uy_x_hat.data()), thrust::raw_pointer_cast(d_uy_x_hat.data()), CUFFT_INVERSE));
        CHECK_CUFFT(cufftExecZ2Z(plan, thrust::raw_pointer_cast(d_uy_y_hat.data()), thrust::raw_pointer_cast(d_uy_y_hat.data()), CUFFT_INVERSE));
        CHECK_CUFFT(cufftExecZ2Z(plan, thrust::raw_pointer_cast(d_uy_z_hat.data()), thrust::raw_pointer_cast(d_uy_z_hat.data()), CUFFT_INVERSE));
        CHECK_CUFFT(cufftExecZ2Z(plan, thrust::raw_pointer_cast(d_uz_x_hat.data()), thrust::raw_pointer_cast(d_uz_x_hat.data()), CUFFT_INVERSE));
        CHECK_CUFFT(cufftExecZ2Z(plan, thrust::raw_pointer_cast(d_uz_y_hat.data()), thrust::raw_pointer_cast(d_uz_y_hat.data()), CUFFT_INVERSE));
        CHECK_CUFFT(cufftExecZ2Z(plan, thrust::raw_pointer_cast(d_uz_z_hat.data()), thrust::raw_pointer_cast(d_uz_z_hat.data()), CUFFT_INVERSE));
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else {
        // We still performed forward FFTs above, but now skip the solve/deriv/inverse.
        // Ensure derivative & displacement arrays are zero-initialized:
        thrust::fill(d_ux_hat.begin(), d_ux_hat.end(), cuC{ 0.0, 0.0 });
        thrust::fill(d_uy_hat.begin(), d_uy_hat.end(), cuC{ 0.0, 0.0 });
        thrust::fill(d_uz_hat.begin(), d_uz_hat.end(), cuC{ 0.0, 0.0 });

        thrust::fill(d_ux_x_hat.begin(), d_ux_x_hat.end(), cuC{ 0.0, 0.0 });
        thrust::fill(d_ux_y_hat.begin(), d_ux_y_hat.end(), cuC{ 0.0, 0.0 });
        thrust::fill(d_ux_z_hat.begin(), d_ux_z_hat.end(), cuC{ 0.0, 0.0 });
        thrust::fill(d_uy_x_hat.begin(), d_uy_x_hat.end(), cuC{ 0.0, 0.0 });
        thrust::fill(d_uy_y_hat.begin(), d_uy_y_hat.end(), cuC{ 0.0, 0.0 });
        thrust::fill(d_uy_z_hat.begin(), d_uy_z_hat.end(), cuC{ 0.0, 0.0 });
        thrust::fill(d_uz_x_hat.begin(), d_uz_x_hat.end(), cuC{ 0.0, 0.0 });
        thrust::fill(d_uz_y_hat.begin(), d_uz_y_hat.end(), cuC{ 0.0, 0.0 });
        thrust::fill(d_uz_z_hat.begin(), d_uz_z_hat.end(), cuC{ 0.0, 0.0 });
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // ---------------- Convert to Vec3 format ----------------
    double invN = 1.0 / static_cast<double>(N);
    thrust::device_vector<Vec3<double>> d_U(N);
    thrust::device_vector<Vec3<double>> d_Ux_deriv(N);
    thrust::device_vector<Vec3<double>> d_Uy_deriv(N);
    thrust::device_vector<Vec3<double>> d_Uz_deriv(N);

    // If FLAG_USE_DISPLACEMENT_FIELD was used we must convert inverse-FFTs to real-space vectors.
    // If FLAG_USE_DISPLACEMENT_FIELD==0 we have zero cuC arrays — conversion will produce zeros.
    ConvertFromCuCToVec3Functor convFun(
        thrust::raw_pointer_cast(d_ux_hat.data()), thrust::raw_pointer_cast(d_uy_hat.data()), thrust::raw_pointer_cast(d_uz_hat.data()),
        thrust::raw_pointer_cast(d_ux_x_hat.data()), thrust::raw_pointer_cast(d_ux_y_hat.data()), thrust::raw_pointer_cast(d_ux_z_hat.data()),
        thrust::raw_pointer_cast(d_uy_x_hat.data()), thrust::raw_pointer_cast(d_uy_y_hat.data()), thrust::raw_pointer_cast(d_uy_z_hat.data()),
        thrust::raw_pointer_cast(d_uz_x_hat.data()), thrust::raw_pointer_cast(d_uz_y_hat.data()), thrust::raw_pointer_cast(d_uz_z_hat.data()),
        thrust::raw_pointer_cast(d_U.data()), thrust::raw_pointer_cast(d_Ux_deriv.data()),
        thrust::raw_pointer_cast(d_Uy_deriv.data()), thrust::raw_pointer_cast(d_Uz_deriv.data()),
        invN, static_cast<int>(N)
    );

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>((int)N), convFun);
    CHECK_CUDA(cudaDeviceSynchronize());

    // ---------------- Run full physics functor ----------------
    thrust::device_vector<Vec3<double>> d_eps_norm(N), d_eps_shear(N), d_field_P(N);
    thrust::device_vector<double> d_fE1(N), d_fE2(N), d_fE3(N), d_field_mag(N);

    double b11 = 0.5 * C11 * (Q11 * Q11 + 2.0 * Q12 * Q12) + C12 * Q12 * (2.0 * Q11 + Q12);
    double b12 = C11 * Q12 * (2.0 * Q11 + Q12) + C12 * (Q11 * Q11 + 3.0 * Q12 * Q12 + 2.0 * Q11 * Q12) + 2.0 * C44 * Q44 * Q44;
    double q11 = C11 * Q11 + 2.0 * C12 * Q12;
    double q12 = C11 * Q12 + C12 * (Q11 + Q12);
    double q44 = 2.0 * C44 * Q44;

    FullPhysicsFunctor fullFun(
        thrust::raw_pointer_cast(d_P.data()), thrust::raw_pointer_cast(d_U.data()),
        thrust::raw_pointer_cast(d_Ux_deriv.data()), thrust::raw_pointer_cast(d_Uy_deriv.data()), 
        thrust::raw_pointer_cast(d_Uz_deriv.data()),
        thrust::raw_pointer_cast(d_eps_norm.data()), thrust::raw_pointer_cast(d_eps_shear.data()),
        thrust::raw_pointer_cast(d_fE1.data()), thrust::raw_pointer_cast(d_fE2.data()),
        thrust::raw_pointer_cast(d_fE3.data()),
        thrust::raw_pointer_cast(d_field_P.data()), thrust::raw_pointer_cast(d_field_mag.data()),
        thrust::raw_pointer_cast(d_heff.data()), thrust::raw_pointer_cast(d_elastic_energy.data()),
        Q11, Q12, Q44, C11, C12, C44, q11, q12, q44, b11, b12, static_cast<int>(N),
        eps_ext_xx, eps_ext_yy, eps_ext_zz, eps_ext_xy, eps_ext_xz, eps_ext_yz,
        FLAG_USE_DISPLACEMENT_FIELD, FLAG_POLARIZATION_FIELD
    );

    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>((int)N), fullFun);
    CHECK_CUDA(cudaDeviceSynchronize());

    // ---------------- Output results (optional) ----------------
    if (FLAG_OUTPUT_FILES == 1) {
        // Copy outputs to host and save central slices
        thrust::host_vector<Vec3<double>> h_U = d_U;
        thrust::host_vector<Vec3<double>> h_eps_norm = d_eps_norm;
        thrust::host_vector<Vec3<double>> h_eps_shear = d_eps_shear;
        thrust::host_vector<Vec3<double>> h_field_P = d_field_P;
        thrust::host_vector<double> h_field_mag = d_field_mag;

        // scalar arrays for ux, uy, uz and strain components
        thrust::host_vector<double> ux_scalar(N), uy_scalar(N), uz_scalar(N);
        thrust::host_vector<double> eps_xx_h(N), eps_yy_h(N), eps_zz_h(N), eps_xy_h(N), eps_xz_h(N), eps_yz_h(N);
        thrust::host_vector<double> field_P1_h(N), field_P2_h(N), field_P3_h(N);

        for (long i = 0; i < N; ++i) {
            ux_scalar[i] = h_U[i].x;
            uy_scalar[i] = h_U[i].y;
            uz_scalar[i] = h_U[i].z;

            eps_xx_h[i] = h_eps_norm[i].x;
            eps_yy_h[i] = h_eps_norm[i].y;
            eps_zz_h[i] = h_eps_norm[i].z;

            eps_xy_h[i] = h_eps_shear[i].x;
            eps_xz_h[i] = h_eps_shear[i].y;
            eps_yz_h[i] = h_eps_shear[i].z;

            field_P1_h[i] = h_field_P[i].x;
            field_P2_h[i] = h_field_P[i].y;
            field_P3_h[i] = h_field_P[i].z;
        }

        int zslice = Nz / 2;

        save_slice("ux_slice_cuda.txt", ux_scalar, Nx, Ny, Nz, zslice);
        save_slice("uy_slice_cuda.txt", uy_scalar, Nx, Ny, Nz, zslice);
        save_slice("uz_slice_cuda.txt", uz_scalar, Nx, Ny, Nz, zslice);
        save_slice("eps_xx_slice_cuda.txt", eps_xx_h, Nx, Ny, Nz, zslice);
        save_slice("eps_yy_slice_cuda.txt", eps_yy_h, Nx, Ny, Nz, zslice);
        save_slice("eps_zz_slice_cuda.txt", eps_zz_h, Nx, Ny, Nz, zslice);
        save_slice("eps_xy_slice_cuda.txt", eps_xy_h, Nx, Ny, Nz, zslice);
        save_slice("eps_xz_slice_cuda.txt", eps_xz_h, Nx, Ny, Nz, zslice);
        save_slice("eps_yz_slice_cuda.txt", eps_yz_h, Nx, Ny, Nz, zslice);

        save_slice("field_P1_slice_cuda.txt", field_P1_h, Nx, Ny, Nz, zslice);
        save_slice("field_P2_slice_cuda.txt", field_P2_h, Nx, Ny, Nz, zslice);
        save_slice("field_P3_slice_cuda.txt", field_P3_h, Nx, Ny, Nz, zslice);
        save_slice("field_mag_slice_cuda.txt", h_field_mag, Nx, Ny, Nz, zslice);

        // Save full results (text)
        std::ofstream fout("pbtio3_cuda_results_one_functor.dat");
        fout.setf(std::ios::scientific);
        for (long id = 0; id < N; ++id) {
            fout << ux_scalar[id] << " " << uy_scalar[id] << " " << uz_scalar[id] << " ";
            fout << eps_xx_h[id] << " " << eps_yy_h[id] << " " << eps_zz_h[id] << " ";
            fout << eps_xy_h[id] << " " << eps_xz_h[id] << " " << eps_yz_h[id] << " ";
            fout << field_P1_h[id] << " " << field_P2_h[id] << " " << field_P3_h[id] << "\n";
        }
        fout.close();
        std::cout << "Saved central slices and full data file (one-functor variant).\n";
    }

    // ---------------- Cleanup ----------------
    CHECK_CUFFT(cufftDestroy(plan));
}
