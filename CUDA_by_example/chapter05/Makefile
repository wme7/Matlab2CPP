all: add_loop_blocks add_loop_long_blocks dot ripple shared_bitmap

add_loop_blocks: add_loop_blocks.cu
	nvcc -o add_loop_blocks add_loop_blocks.cu

add_loop_long_blocks: add_loop_long_blocks.cu
	nvcc -o add_loop_long_blocks add_loop_long_blocks.cu

dot: dot.cu
	nvcc -o dot dot.cu

ripple: ripple.cu
	nvcc -o ripple ripple.cu -Xlinker -framework,GLUT -Xlinker -framework,OpenGL

shared_bitmap: shared_bitmap.cu
	nvcc -o shared_bitmap shared_bitmap.cu -Xlinker -framework,GLUT -Xlinker -framework,OpenGL 

clean:
	rm add_loop_blocks dot ripple add_loop_long_blocks shared_bitmap
