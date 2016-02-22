all: ray ray_noconst

ray: ray.cu
	nvcc -o ray ray.cu -Xlinker -framework,GLUT -Xlinker -framework,OpenGL

ray_noconst: ray_noconst.cu
	nvcc -o ray_noconst ray_noconst.cu -Xlinker -framework,GLUT -Xlinker -framework,OpenGL

clean:
	rm ray ray_noconst
