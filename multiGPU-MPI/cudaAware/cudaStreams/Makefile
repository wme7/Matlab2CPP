all: test1 test1_wflag test2 test2_wflag test3 test4 test5 test6 test7 test8 test9

test1: example1.cu
	nvcc $< -o $@.run

test1_wflag: example1.cu
	nvcc -default-stream per-thread $< -o $@.run

test2: example2.cu
	nvcc $< -o $@.run

test2_wflag: example2.cu
	nvcc -default-stream per-thread $< -o $@.run

test3: example3.cu
	nvcc $< -o $@.run

test4: example4.cu
	nvcc $< -o $@.run

test5: example5.cu
	nvcc $< -o $@.run

test6: example6.cu
	nvcc $< -o $@.run

test7: example7.cu
	nvcc $< -o $@.run

test8: example8.cu
	nvcc $< -o $@.run

test9: example9.cu
	nvcc $< -o $@.run

clean: 
	rm *.run
