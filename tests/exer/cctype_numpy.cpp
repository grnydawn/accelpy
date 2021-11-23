extern "C" int add1d(int * n, double * x, double * y, double *z) {

	for (int i=0; i < *n; i++) {
		z[i] = x[i] + y[i];
	}

	return 0;
}
