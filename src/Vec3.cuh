#pragma once
#ifndef VEC3_CUH
#define VEC3_CUH
#include "Vec3.h"

// Default constructor
template<typename T>
__host__ __device__
Vec3<T>::Vec3() : x(T(0)), y(T(0)), z(T(0)) {}

// Parameterized constructor
template<typename T>
__host__ __device__
Vec3<T>::Vec3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}

// Copy constructor
template<typename T>
__host__ __device__
Vec3<T>::Vec3(const Vec3<T>& other) : x(other.x), y(other.y), z(other.z) {}

// Assignment operator
template<typename T>
__host__ __device__
Vec3<T>& Vec3<T>::operator=(const Vec3<T>& other) {
    if (this != &other) { // Check for self-assignment
        x = other.x;
        y = other.y;
        z = other.z;
    }
    return *this;
}


// Unary plus operator
template<typename T>
__host__ __device__
Vec3<T> Vec3<T>::operator+() const {
    return *this;
}

// Vector addition
template<typename T>
__host__ __device__
Vec3<T> Vec3<T>::operator+(const Vec3<T>& other) const {
    return Vec3<T>(x + other.x, y + other.y, z + other.z);
}

// Add and assign
template<typename T>
__host__ __device__
void Vec3<T>::operator+=(const Vec3<T>& other) {
    x += other.x;
    y += other.y;
    z += other.z;
}

// Unary minus operator
template<typename T>
__host__ __device__
Vec3<T> Vec3<T>::operator-() const {
    return { -x, -y, -z };
}

// Vector subtraction
template<typename T>
__host__ __device__
Vec3<T> Vec3<T>::operator-(const Vec3<T>& other) const {
    return Vec3<T>(x - other.x, y - other.y, z - other.z);
}

// Subtract and assign
template<typename T>
__host__ __device__
void Vec3<T>::operator-=(const Vec3<T>& other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
}

// Scale vector by scalar
template<typename T>
__host__ __device__
Vec3<T> Vec3<T>::operator*(const T scalar) const {
    return Vec3<T>(x * scalar, y * scalar, z * scalar);
}

// Scale and assign
template<typename T>
__host__ __device__
void Vec3<T>::operator*=(const T scalar) {
    x *= scalar;
    y *= scalar;
    z *= scalar;
}

// Dot product
template<typename T>
__host__ __device__
T Vec3<T>::operator*(const Vec3<T>& other) const {
    return (x * other.x + y * other.y + z * other.z);
}

template<typename T>
__host__ __device__ Vec3<T> Vec3<T>::operator/(const T scalar) const
{
    return { x / scalar, y / scalar, z / scalar };
}

template<typename T>
__host__ __device__ void Vec3<T>::operator/=(const T scalar)
{
    T inv = 1.0 / scalar;
    x *= inv;
    y *= inv;
    z *= inv;
}

// Maximum component of the vector
template<typename T>
__host__ __device__
T Vec3<T>::maxComponent() const {
    T max_val = (x > y) ? x : y;
    return (max_val > z) ? max_val : z;
}

// Cross product
template<typename T>
__host__ __device__
Vec3<T> Vec3<T>::Cross(const Vec3<T>& other) {
    return Vec3<T>(
        y * other.z - z * other.y,
        z * other.x - x * other.z,
        x * other.y - y * other.x
    );
}

// Square of the vector's magnitude
template<typename T>
__host__ __device__
T Vec3<T>::SquareModulus() {
    return (*(this) * *(this));
}

// Magnitude of the vector
template<typename T>
__host__ __device__
T Vec3<T>::Modulus() {
    return T(sqrt(this->SquareModulus()));
}

// Normalize the vector
template<typename T>
__host__ __device__
void Vec3<T>::Normalize() {
    T mod = Modulus();
    if (mod > T(0))
    {
        T inverseModulus = T(1.0) / mod;
        *this *= inverseModulus;
    }
}

// Scale vector by scalar (commutative operator)
template<typename T>
__host__ __device__
Vec3<T> operator*(const T scalar, const Vec3<T>& vector) {
    return (vector * scalar);
}

template<typename T>
std::ostream& operator<<(std::ostream& out, const Vec3<T>& vec)
{
    out << vec.x << ' ' << vec.y << ' ' << vec.z;
    return out;
}

template<typename T>
struct ToVec3
{
    T* a;
    T* b;
    T* c;

    ToVec3(T* a, T* b, T* c)
        : a(a), b(b), c(c) {
    }

    __host__ __device__  Vec3<T> operator()(const int i) { return { a[i], b[i], c[i] }; }
};

template<typename T>
struct FromVec3
{
    T* a;
    T* b;
    T* c;

    FromVec3(T* a, T* b, T* c)
        : a(a), b(b), c(c) {
    }

    __host__ __device__ Vec3<T> operator()(const int i, const Vec3<T>& src)
    {
        a[i] = src.x;
        b[i] = src.y;
        c[i] = src.z;

        return src;
    }
};
#endif // VEC3_CUH
