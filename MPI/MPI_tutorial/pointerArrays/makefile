
all: pointerArray1 pointerArray2 pointerArray3 pointerArray4

pointerArray1: singlePointer.cpp
	g++ singlePointer.cpp -o singlePointer.run

pointerArray2: pointerArray.cpp
	g++ pointerArray.cpp -o pointerArray.run

pointerArray3: pointer2pointer.cpp
	g++ pointer2pointer.cpp -o pointer2pointer.run

pointerArray4: doublePointer.cpp
	g++ doublePointer.cpp -o doublePointer.run

clean:
	rm *.run
	rm *.txt
