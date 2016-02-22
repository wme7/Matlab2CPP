all: basic_interop heat ripple

heat: heat.cu
	nvcc -o heat heat.cu -Xlinker -framework,GLUT -Xlinker -framework,OpenGL

basic_interop: basic_interop.cu
	nvcc -o basic_interop basic_interop.cu -Xlinker -framework,GLUT -Xlinker -framework,OpenGL

ripple: ripple.cu
	nvcc -o ripple ripple.cu -Xlinker -framework,GLUT -Xlinker -framework,OpenGL

clean:
	rm heat ripple basic_interop
