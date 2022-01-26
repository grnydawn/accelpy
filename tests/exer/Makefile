FC := gfortran
CC := g++

fctype_accelpy:
	${FC} -dynamiclib -fPIC -o fctype_accelpy.dylib fctype_accelpy.F90
	python3 ctype_accelpy.py fctype_accelpy

cctype_accelpy:
	${FC} -dynamiclib -fPIC -o cctype_accelpy.dylib cctype_accelpy.cpp
	python3 ctype_accelpy.py cctype_accelpy

fctype_numpy:
	${FC} -dynamiclib -fPIC -o fctype_numpy.dylib fctype_numpy.F90
	python3 ctype_numpy.py fctype_numpy

cctype_numpy:
	${FC} -dynamiclib -fPIC -o cctype_numpy.dylib cctype_numpy.cpp
	python3 ctype_numpy.py cctype_numpy

fctype_scalar:
	${FC} -dynamiclib -fPIC -o fctype_scalar.dylib fctype_scalar.F90
	python3 ctype_scalar.py fctype_scalar

cctype_scalar:
	${CC} -dynamiclib -fPIC -o cctype_scalar.dylib cctype_scalar.cpp
	python3 ctype_scalar.py cctype


fbound:
	${FC} -o fbound.exe fbound.F90
	./fbound.exe

recast:
	${CC} -o recast.exe recast.cpp
	./recast.exe

clean:
	rm -rf *.dylib *.so *.o *.mod *.exe core
