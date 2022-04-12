#include <stdint.h>
#include <stdio.h>
#include <hip/hip_runtime.h>

extern double (* dci0)[2][3];
extern double (* dci1)[2][3];
extern double (* dco0)[2][3];


__global__ void device_kernel(double X[2][3], double Y[2][3], double Z[2][3]) {


	int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    Z[i][j] = X[i][j] + Y[i][j];

}

extern "C" int64_t runkernel() {
    int64_t res;

    device_kernel<<<1, dim3(2,3)>>>(*dci0, *dci1, *dco0);

	res = 0;

    return res;
}
