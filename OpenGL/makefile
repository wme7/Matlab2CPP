all: MAIN

MAIN: EXAMPLE1 EXAMPLE2 EXAMPLE3 EXAMPLE4

EXAMPLE1: main_OpenGL.cpp
	g++ main_OpenGL.cpp -lGL -lGLU -lglut -o openGL.run

EXAMPLE2: main_Shapes.cpp
	g++ main_Shapes.cpp -lGL -lGLU -lglut -o shapes.run

EXAMPLE3: main_lighting2.cpp
	g++ main_lighting2.cpp -lGL -lGLU -lglut -o lighting2.run

EXAMPLE4: main_GLUTexample.cpp
	g++ main_OpenGL.cpp -lGLEW -lGL -lGLU -lglut -o glutExample.run

clean: 
	rm *.run
