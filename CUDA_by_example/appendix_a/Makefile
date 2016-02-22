all: dot hashtable_cpu hashtable_gpu

dot: dot.cu lock.h
	nvcc -O3 -arch sm_11  -o dot dot.cu 

hashtable_cpu: hashtable_cpu.cu
	nvcc -O3 -o hashtable_cpu hashtable_cpu.cu 

hashtable_gpu: hashtable_gpu.cu lock.h
	nvcc -O3 -arch sm_11  -o hashtable_gpu hashtable_gpu.cu 

clean:
	rm dot hashtable_cpu hashtable_gpu
