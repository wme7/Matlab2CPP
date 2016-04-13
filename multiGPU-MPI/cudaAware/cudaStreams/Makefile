all: example1 example2

example1: main.cu
	nvcc -default-stream per-thread main.cu -o test1.run

example2: main2.cu
	nvcc -default-stream per-thread main2.cu -o test2.run

clean: 
	rm *.run
