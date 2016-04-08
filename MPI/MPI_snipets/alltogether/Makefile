all: SendReceive Scatter ScatterGather

SendReceive: main.cpp
	mpic++ main.cpp -o test1.run

Scatter: main2.cpp
	mpic++ main2.cpp -o test2.run

ScatterGather: main3.cpp
	mpic++ main3.cpp -o test3.run

clean:
	rm *.run


