extern "C" int add1d(int * nattr, int attr[],  double x[], double y[], double z[]) {

	for (int i=0; i < attr[0]; i++) {
		z[i] = x[i] + y[i];
	}

	return 0;
}
