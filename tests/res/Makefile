runhip: hipdriver.py
	python hipdriver.py

runomp: ompdriver.py
	python ompdriver.py

runacc: accdriver.py
	python accdriver.py

profomp:
	#rocprof --hsa-trace python ompdriver.py
	rocprof -i rocprof.txt python ompdriver.py

profacc:
	rocprof -i rocprof.txt python accdriver.py

clean:
	rm -f *.mod *.o *.so *.s results.* core
