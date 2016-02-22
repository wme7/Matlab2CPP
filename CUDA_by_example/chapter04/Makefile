all: julia_gpu julia_cpu add_loop_cpu add_loop_gpu

julia_gpu: julia_gpu.cu
	nvcc -o julia_gpu julia_gpu.cu -Xlinker -framework,GLUT -Xlinker -framework,OpenGL

julia_cpu: julia_cpu.cu
	nvcc -o julia_cpu julia_cpu.cu -Xlinker -framework,GLUT -Xlinker -framework,OpenGL

add_loop_cpu: add_loop_cpu.cu
	nvcc -o add_loop_cpu add_loop_cpu.cu

add_loop_gpu: add_loop_gpu.cu
	nvcc -o add_loop_gpu add_loop_gpu.cu

clean:
	rm julia_gpu julia_cpu add_loop_cpu add_loop_gpu
