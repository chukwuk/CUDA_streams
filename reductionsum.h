

#ifndef __REDUCTIONSUM_H
#define __REDUCTIONSUM_H

__global__  void reductionSum(int* reduceData, int* sumData, unsigned long int numData, unsigned int nCols, int offset); 

__global__  void reductionSumDefaultStream(int* reduceData, int* sumData, unsigned long int numData, unsigned int nCols);

#endif
