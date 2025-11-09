#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <cmath>
#include <string>
#include <fstream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>


#include "reductionsum.h"


#ifndef NUMDATA

 #define NUMDATA 3000000000

#endif

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
   
  unsigned long int numData;

  if (argc > 1) {
      numData = std::stoi(argv[1]);
  
  } else {

       numData = NUMDATA;
  }
  
  fprintf (stderr, "NUMBER OF DATA is %lu\n", numData);

  unsigned int nCols = 6;

  unsigned long int sumNumData =  numData/nCols;
  
  int* reduceData;
  int* sumData; 
   
  cudaError_t status;
  unsigned long int reduceDataSize = (sizeof(int) * numData);
  fprintf (stderr, "Amount of reduceData data transfered to the device is %lu GB\n", reduceDataSize/1000000000);
  
  unsigned long int sumDataSize = (sizeof(int) * sumNumData) ;
  fprintf (stderr, "Amount of sumData data transfered to the device is %lu GB\n", sumDataSize/1000000000);
  
  // pinned data
  
  cudaMallocHost((void**)&reduceData, reduceDataSize);
  cudaMallocHost((void**)&sumData, sumDataSize);

  
  //memset(reduceData, 1, reduceDataSize); 
  
  for (unsigned int i = 0; i < numData; i++) {
       reduceData[i] = 1;
  } 

 
  // allocate memory on device

  int* reduceDataDev;
  int* sumDataDev; 

  // Create CUDA events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  

  
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&reduceDataDev), reduceDataSize);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&reduceDataDev), reduceDataSize) ");
 
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&sumDataDev), sumDataSize);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&sumDataDev), sumDataSize); ");  

  
  
  float GpuTime = 0;
 
  
  // Record the start event
  cudaEventRecord(start, 0); 

  // copy data from host memory to the device:

  status = cudaMemcpy(reduceDataDev, reduceData, reduceDataSize, cudaMemcpyHostToDevice );
  // checks for cuda errors
  checkCudaErrors( status,"cudaMemcpy(reduceDataDev, reduceData, reduceDataSize, cudaMemcpyHostToDevice );");  
  
  
  // Record the stop event
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop); 
  
  // Calculate elapsed time
  cudaEventElapsedTime(&GpuTime, start, stop); 
  
   
  printf("Time for memory copy from host to device for the sequential execution: %f milliseconds\n", GpuTime);


  int BLOCKSIZE;
  int NUMBLOCKS;
  int MINGRIDSIZE;  
  
  
  cudaOccupancyMaxPotentialBlockSize( &MINGRIDSIZE, &BLOCKSIZE, 
                                      reductionSum, 0, 0); 
   
  BLOCKSIZE = 128;
  NUMBLOCKS = (sumNumData + BLOCKSIZE-1)/BLOCKSIZE;
   
  
  // allocate number of threads in a block  
  dim3 threads(BLOCKSIZE, 1, 1 );

  // allocate number of blocks
  dim3 grid(NUMBLOCKS, 1, 1 );
  
     
  // Record the start event
  cudaEventRecord(start, 0); 
      
  reductionSum<<< grid, threads >>>( reduceDataDev, sumDataDev, sumNumData, nCols, 0);

  
  // Record the stop event
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop); 
  
  // Calculate elapsed time
  cudaEventElapsedTime(&GpuTime, start, stop); 
  
  printf("Time for reductionSum kernel execution for the sequential execution: %f milliseconds\n", GpuTime);

  
  status = cudaDeviceSynchronize( );
   
  checkCudaErrors( status," reductionSum<<< grid, threads >>>( reduceDataDev, sumDataDev, numData, sumNumData); ");
 
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");   


   // copy data from device memory to host 
  cudaMemcpy(sumData, sumDataDev, sumDataSize, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status, " cudaMemcpy(sumData, sumDataDev,  sumDataSize , cudaMemcpyDeviceToHost);"); 
  
  printf(" summation values: %i \n", sumData[0]); 
  

  for (unsigned long int i = 0; i < sumNumData; i++) {
      if (sumData[i] - 6 != 0) {
         printf(" The value that is wrong is: %lu, %i\n",i, sumData[i]);
	 break; 
      }  
  }

  cudaFree( sumDataDev );
  cudaFree( reduceDataDev );
   
  cudaFreeHost( sumData );
  cudaFreeHost( reduceData );

  int devId = 0;
  if (argc > 1) devId = atoi(argv[1]);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, devId);
  printf("Device : %s\n", prop.name);
  cudaSetDevice(devId); 


   
  unsigned long int nStreams = 4;
  unsigned long int streamSize = numData / nStreams;
  unsigned long int streamBytes = streamSize * sizeof(int); 
  unsigned long int streamSizeResult = sumNumData / nStreams;
  unsigned long int streamBytesResult = streamSizeResult * sizeof(int);
 // const int bytes = n * sizeof(int);
  

   // create streams
  cudaStream_t stream[nStreams];
  for (int i = 0; i < nStreams; ++i) {
    cudaStreamCreate(&stream[i]);
  }
   
  int* reduceStrData;
  int* sumStrData; 
  
  
  // pinned data
  
  cudaMallocHost((void**)&reduceStrData, reduceDataSize);
  cudaMallocHost((void**)&sumStrData, sumDataSize);

    
  //memset(reduceData, 1, reduceDataSize); 
  
  for (unsigned int i = 0; i < numData; i++) {
       reduceStrData[i] = 1;
  } 

    
  int* reduceStrDataDev;
  int* sumStrDataDev; 
  
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&reduceStrDataDev), reduceDataSize);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&reduceStrDataDev), reduceDataSize) ");
 
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&sumStrDataDev), sumDataSize);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&sumStrDataDev), sumDataSize); ");  



  


  
  NUMBLOCKS = (streamSizeResult + BLOCKSIZE-1)/BLOCKSIZE;
  grid.x = NUMBLOCKS;
  for (int i = 0; i < nStreams; ++i) { 
    unsigned long int offset = i * streamSize; 
    int offsetResult = i * streamSizeResult; 


    // Record the start event
    cudaEventRecord(start, 0); 

    cudaMemcpyAsync(&reduceStrDataDev[offset], &reduceStrData[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);  
    
        
    // Record the stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 
  
    // Calculate elapsed time
    cudaEventElapsedTime(&GpuTime, start, stop); 
   
    printf("Time for Asynchronous memory copy from host to device: %f milliseconds\n", GpuTime);
    
    
    // Record the start event
    cudaEventRecord(start, 0); 
    reductionSum<<<grid, threads, 0, stream[i]>>>( reduceStrDataDev, sumStrDataDev, streamSizeResult, nCols, offsetResult);
    
    // Record the stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 
  
    // Calculate elapsed time
  
    cudaEventElapsedTime(&GpuTime, start, stop); 
  
    printf("Time for the reductionSum kernel excution (ms): %f milliseconds\n", GpuTime);
    cudaMemcpyAsync(&sumStrData[offsetResult], &sumStrDataDev[offsetResult], streamBytesResult, cudaMemcpyDeviceToHost, stream[i]);
  }
  
  
  printf(" summation values: %i \n", sumStrData[0]); 
 
  for (unsigned long int i = 0; i < sumNumData; i++) {
      if (sumStrData[i] - 6 != 0) {
         printf(" The value that is wrong is: %lu, %i\n",i, sumStrData[i]);
	 break; 
      }  
  }
  
  
  cudaFree( sumStrDataDev );
  cudaFree( reduceStrDataDev );
   
  cudaFreeHost( sumStrData );
  cudaFreeHost( reduceStrData );
 
  for (int i = 0; i < nStreams; ++i) {
     cudaStreamDestroy(stream[i]) ;
  }
  
  return 0;
};	
