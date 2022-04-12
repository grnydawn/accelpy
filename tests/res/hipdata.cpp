#include <stdint.h>
#include <stdio.h>
#include <hip/hip_runtime.h>

double (*dci0)[2][3];
double (*dci1)[2][3];
double (*dco0)[2][3];

extern "C" int64_t dataenter(void * X, void * Y, void * Z) {

    int64_t res;

	hipMalloc((void **)&dci0, 6 * sizeof(double));
	hipMalloc((void **)&dci1, 6 * sizeof(double));
	hipMalloc((void **)&dco0, 6 * sizeof(double));

	hipMemcpyHtoD(dci0, X, 6 * sizeof(double));
	hipMemcpyHtoD(dci1, Y, 6 * sizeof(double));

    res = 0;

    return res;

}

extern "C" int64_t dataexit(void * Z) {

    int64_t res;

	hipMemcpyDtoH(Z, dco0, 6 * sizeof(double));

	hipFree(dci0);
	hipFree(dci1);
	hipFree(dco0);

    res = 0;

    return res;

}
