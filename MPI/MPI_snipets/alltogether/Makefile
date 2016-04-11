all: SendReceive Scatter ScatterGather withBoundaries Generalized

SendReceive: main1.cpp
	mpic++ main1.cpp -o test1.run

Scatter: main2.cpp
	mpic++ main2.cpp -o test2.run

ScatterGather: main3.cpp
	mpic++ main3.cpp -o test3.run

withBoundaries: main4.cpp
	mpic++ main4.cpp -o test4.run

Generalized: main5.cpp
	mpic++ main5.cpp -o test5.run

clean:
	rm *.run


