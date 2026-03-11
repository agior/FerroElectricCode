#ifndef _DEMAGTENSOR_H
#define _DEMAGTENSOR_H

#include "parameters.h"
extern int threadsPerBlock;

void electroStaticTensor( HD_Complex_Type *cuSDxx,  HD_Complex_Type *cuSDyy,  HD_Complex_Type *cuSDzz,  
				  HD_Complex_Type *cuSDxy,  HD_Complex_Type *cuSDxz,  HD_Complex_Type *cuSDyz, 
				  Type_var h, int Mcx, int Mcy, int Mcz, set_tensor conf_tens);

DT_double host_fnxx(DT_double, DT_double, DT_double, DT_double, DT_double, DT_double );
DT_double host_fnxy(DT_double, DT_double, DT_double, DT_double, DT_double, DT_double);
DT_double host_f2(DT_double, DT_double, DT_double);
DT_double host_g2(DT_double, DT_double, DT_double);
DT_double host_f(DT_double, DT_double, DT_double);
DT_double host_g(DT_double, DT_double, DT_double);
DT_double host_fi(DT_double);
DT_double host_Dz(DT_double, DT_double, DT_double);


__global__ void kernel_tensor1partea(Type_var* NDxx, Type_var h, int Mcx, int Mcy, int Mcz);
__global__ void kernel_tensor1parteb(Type_var* NDxx, Type_var h, int Mcx, int Mcy, int Mcz);

__global__ void kernel_tensor1partec(Type_var* NDxx, Type_var h, int Mcx, int Mcy, int Mcz);

__global__ void kernel_tensor1parted(Type_var* NDxx, Type_var h, int Mcx, int Mcy, int Mcz);

__global__ void kernel_tensor1partee(Type_var* NDxx, Type_var h, int Mcx, int Mcy, int Mcz);

__global__ void kernel_tensor1partef(Type_var* NDxx,Type_var h, int Mcx, int Mcy, int Mcz);



__global__
void kernel_tensor2parte(HD_Complex_Type* cuSD, Type_var* ND, Type_var h, int Mcx, int Mcy, int Mcz);

/*static __device__  double fnxx(double, double, double, double, double, double);
static __device__  double fnxx2(double x, double y, double z, double dx, double dy, double dz);
static __device__  double fnxx3(double x, double y, double z, double dx, double dy, double dz);
static __device__  double fnxy(double, double, double, double, double, double);
static __device__  double fnxy2(double, double, double, double, double, double);
static __device__  double fnxy3(double, double, double, double, double, double);*/
__device__   double f2(double, double, double);
__device__  double g2(double, double, double);
__device__  double f(double, double, double);
 __device__  double g(double, double, double);
__device__  double fi(double);
//static __device__  double Dz(double, double, double);
//static __device__  double Dz2(double, double, double);
//static __device__  double Dz3(double, double, double);


#endif