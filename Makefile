CC=nvcc

all: main main1

main: main.cu reductionsum.o 
	$(CC) reductionsum.o main.cu -o main

main1: main1.cu reductionsum.o 
	$(CC) reductionsum.o main1.cu -o main1

reductionsum.o: reductionsum.cu reductionsum.h
	$(CC) -c reductionsum.cu -o reductionsum.o

clean:
	rm -f main reductionsum.o
