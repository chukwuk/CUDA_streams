#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <cmath>
#include <string>
#include <cuda_runtime.h>

#include "reductionsum.h"

#define IDX2C(i,j,ld) (((i)*(ld))+(j))

#define c(x) #x
#define stringify(x) c(x)

#define t(s1,s2) s1##s2
#define tg(s1,s2) t(s1,s2)

#define tgg(s1,s2,s3) tg(tg(s1,s2),s3)
#define tggg(s1,s2,s3,s4) tg(tgg(s1,s2,s3),s4)


using namespace std;

inline
cudaError_t checkCudaErrors(cudaError_t result, string functioncall = "")
{
//#if defined(DEBUG) || defined(_DEBUG)
  //fprintf(stderr, "CUDA Runtime Error: %d\n", result);
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error for this function call ( %s ) : %s\n", 
            functioncall.c_str(), cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
//#endif
  return result;
}

int
main( int argc, char* argv[ ] )
{ 
  //srand(time(0));
//  fprintf (stderr, "Amount of data transfered to the device is %lld GB\n", bytes4euc/1000000000);
 
  long int numData = 1000000000;
  long int sumNumData =  (numData+1)/2;
  double* reduceData = new double[numData];
  double* sumData = new double[sumNumData];   

  for (int i = 0; i < numData; i++) {
      reduceData[i] = (rand() % 8)/1.0; 
  }
   
  int BLOCKSIZE = 128;
  int NUMBLOCKS = (sumNumData + BLOCKSIZE-1)/BLOCKSIZE;
    
  fprintf (stderr, "NUMBER OF BLOCKS is %d\n", NUMBLOCKS);
  
 
  // allocate memory on device

  double* reduceDataDev;
  double* sumDataDev; 

  // Create CUDA events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  

  // Record the start event
  cudaEventRecord(start, 0); 
  
  cudaError_t status;
  long int reduceDataSize = (sizeof(double) * numData);
  fprintf (stderr, "Amount of data transfered to the device is %ld Bytes\n", reduceDataSize);
   
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&reduceDataDev), reduceDataSize);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&reduceDataDev), reduceDataSize) ");

  long int sumDataSize = (sizeof(double) * sumNumData) ;
  fprintf (stderr, "Amount of data transfered to the device is %ld Bytes\n", sumDataSize);
   
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&sumDataDev), sumDataSize);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&sumDataDev), sumDataSize); ");  

  // copy data from host memory to the device:

  status = cudaMemcpy(reduceDataDev, reduceData, reduceDataSize, cudaMemcpyHostToDevice );
  // checks for cuda errors
  checkCudaErrors( status,"cudaMemcpy(reduceDataDev, reduceData, reduceDataSize, cudaMemcpyHostToDevice );");  
 

  // allocate number of threads in a block  
  dim3 threads(BLOCKSIZE, 1, 1 );

  // allocate number of blocks
  dim3 grid(NUMBLOCKS, 1, 1 );
  
 
  
  // call the kernel
  reductionSum<<< grid, threads >>>( reduceDataDev, sumDataDev, numData);
  
  status = cudaDeviceSynchronize( );
  
   
  checkCudaErrors( status," reductionSum<<< grid, threads >>>( reduceDataDev, sumDataDev, numData); ");
 
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");  

  // copy data device memory to host:
  cudaMemcpy(sumData, sumDataDev,  sumDataSize , cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status, " cudaMemcpy(sumData, sumDataDev,  sumDataSize , cudaMemcpyDeviceToHost);"); 
  
  // Record the stop event
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop); 
  
  // Calculate elapsed time
  float GpuTime = 0;
  cudaEventElapsedTime(&GpuTime, start, stop); 
  
  printf("  GPU time: %f milliseconds\n", GpuTime);

  printf(" summation values: %f \n", reduceData[(numData-2)]);
  printf(" summation values: %f \n", reduceData[(numData-1)]);
  printf(" summation values: %f \n", sumData[(sumNumData-1)]); 
  
  double sum = 0.00;
  
  
  // Record the start event
  cudaEventRecord(start, 0); 
     
  for (int i = 0; i < numData; i++) {
       sum+=reduceData[i]; 
  }
  
  // Record the stop event
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop); 
  
  // Calculate elapsed time
  cudaEventElapsedTime(&GpuTime, start, stop); 
  printf("  GPU time: %f milliseconds\n", GpuTime);
  
  // free device memory 
  cudaFree( sumDataDev );
  cudaFree( reduceDataDev );
  delete[] sumData;
  delete[] reduceData;
  

  return 0;
};	
