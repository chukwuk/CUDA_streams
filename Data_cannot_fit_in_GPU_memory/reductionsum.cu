#include <math.h>
#include "reductionsum.h"    
#include <stdio.h>

using namespace std;


__global__  void reductionSum(int* reduceData, int* sumData, unsigned long int numData, unsigned int nCols, int offset ) {

   int gid = offset + (blockIdx.x *  blockDim.x +  threadIdx.x);
   size_t shift = (size_t)gid *  (size_t) nCols;
   if (gid < (offset + numData)) {
      int sum = 0;
      for (size_t i = 0; i < nCols; i++) {
          sum += reduceData[shift + i];
      }
      sumData[gid] = sum;
   }

}





