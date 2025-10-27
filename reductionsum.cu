#include <math.h>
#include "reductionsum.h"    
#include <stdio.h>

using namespace std;


__global__  void reductionSum(int* reduceData, int* sumData, unsigned long int numData, unsigned int nCols, int offset ) {

   int gid = offset + (blockIdx.x *  blockDim.x +  threadIdx.x);
   unsigned long int shift = (unsigned long int)gid *  (unsigned long int) nCols;
   if (gid < (offset + numData)) {
      int sum = 0;
      for ( unsigned long int i = 0; i < (unsigned long int) nCols; i++) {
          sum += reduceData[shift + i];
      }
      sumData[gid] = sum;
   }

}





