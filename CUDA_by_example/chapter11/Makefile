all: multidevice portable time_zero2

multidevice: multidevice.cu
	nvcc -O3  -o multidevice multidevice.cu 

portable: portable.cu
	nvcc -O3 -o portable  portable.cu 

time_zero2: time_zero2.cu
	nvcc -O3 -o time_zero2  time_zero2.cu 

clean:
	rm multidevice portable time_zero2
