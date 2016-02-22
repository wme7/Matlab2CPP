all: heat heat_2d

heat: heat.cu
	nvcc -o heat heat.cu -Xlinker -framework,GLUT -Xlinker -framework,OpenGL

heat_2d: heat_2d.cu
	nvcc -o heat_2d heat_2d.cu -Xlinker -framework,GLUT -Xlinker -framework,OpenGL


clean:
	rm heat heat_2d
