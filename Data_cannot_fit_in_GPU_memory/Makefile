CC=nvcc

all: main

main: main.cu reductionsum.o 
	$(CC) reductionsum.o main.cu -o main

reductionsum.o: reductionsum.cu reductionsum.h
	$(CC) -c reductionsum.cu -o reductionsum.o

clean:
	rm -f main reductionsum.o
