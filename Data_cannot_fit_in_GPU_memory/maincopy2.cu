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

 #define NUMDATA 4500000000

#endif


#ifndef GPUNUMDATA

 #define GPUNUMDATA 3000000000

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

  
  unsigned long int numGPUData;
  
  if (argc > 1) {
      numData = std::stoi(argv[1]);
  
  } else {

       numData = NUMDATA;
       numGPUData = GPUNUMDATA;
  }
  
  fprintf (stderr, "NUMBER OF DATA is %lu\n", numData);

  unsigned long int nCols = 6;

  unsigned long int sumNumData =  numData/nCols;
  
  unsigned long int sumGPUNumData =  numGPUData/nCols;


  cudaError_t status;
  unsigned long int reduceDataSize = (sizeof(int) * numData);
  fprintf (stderr, "Amount of reduceData data transfered to the device is %lu GB\n", reduceDataSize/1000000000);
  
  unsigned long int sumDataSize = (sizeof(int) * sumNumData) ;
  fprintf (stderr, "Amount of sumData data transfered to the device is %lu GB\n", sumDataSize/1000000000);
  
  
  unsigned long int reduceGPUDataSize = (sizeof(int) * numGPUData);
  fprintf (stderr, "Amount of reduceData data transfered to the device is %lu GB\n", reduceGPUDataSize/1000000000);
  
  unsigned long int sumGPUDataSize = (sizeof(int) * sumGPUNumData) ;
  fprintf (stderr, "Amount of sumData data transfered to the device is %lu GB\n", sumGPUDataSize/1000000000);


  
  // pinned data
  

  
  //memset(reduceData, 1, reduceDataSize); 

  
  // Create CUDA events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  
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
  
   
       

  int devId = 0;
  if (argc > 1) devId = atoi(argv[1]);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, devId);
  printf("Device : %s\n", prop.name);
  cudaSetDevice(devId); 


   
  unsigned long int nStreams = 6;
  unsigned long int nStreamsFitGPU = 4;
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
   
   /*  
   int* reduceStrData;
   int* sumStrData; 
 
   
  
  // pinned data
  
  status =  cudaMallocHost((void**)&reduceStrData, reduceDataSize);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&reduceStrData), reduceDataSize) ");
 
  status = cudaMallocHost((void**)&sumStrData, sumDataSize);
  checkCudaErrors( status, "  cudaMallocHost((void**)&sumStrData, sumDataSize); ");
   */
   
  // cudaMallocHost could not allocate more than 16GB on physical memory
  
  
  int* reduceStrData = new int [numData];
  int* sumStrData  = new int [sumNumData]; 


  //memset(reduceData, 1, reduceDataSize); 
  
  
  for (unsigned long int i = 0; i < numData; i++) {
      reduceStrData[i] = (rand() % 100000) + 1;
  }
 /*  
  
  for (unsigned long int i = 0; i < numGPUData; i++) {
       reduceStrData[i] = 1;
  } 
  
 
  for (unsigned long int i = numGPUData; i < numData; i++) {
       reduceStrData[i] = 2;
  } 
  */

   
  int* reduceStrDataDev;
  int* sumStrDataDev; 
  
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&reduceStrDataDev), reduceGPUDataSize);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&reduceStrDataDev), reduceGPUDataSize) ");
 
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&sumStrDataDev), sumGPUDataSize);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&sumStrDataDev), sumGPUDataSize); ");  



  

  // Record the start event
  cudaEventRecord(start, 0); 

  
  NUMBLOCKS = (streamSizeResult + BLOCKSIZE-1)/BLOCKSIZE;
  grid.x = NUMBLOCKS;
  for (int i = 0; i < nStreams; ++i) { 
    unsigned long int offset = i * streamSize; 
    unsigned long int offsetResult = i * streamSizeResult;
    if (i >= nStreamsFitGPU) {
      cudaStreamSynchronize(stream[i-nStreamsFitGPU]); 
    }  
    cudaMemcpyAsync(&reduceStrDataDev[offset % numGPUData], &reduceStrData[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);  
    reductionSum<<<grid, threads, 0, stream[i]>>>( reduceStrDataDev, sumStrDataDev, streamSizeResult, nCols, offsetResult % sumGPUNumData);
    cudaMemcpyAsync(&sumStrData[offsetResult ], &sumStrDataDev[offsetResult % sumGPUNumData ], streamBytesResult, cudaMemcpyDeviceToHost, stream[i]);
  }
  
  
  // Record the stop event
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop); 
  
  // Calculate elapsed time
  float GpuTime; 
  cudaEventElapsedTime(&GpuTime, start, stop); 
  
  printf("Time for asynchronous V1 transfer and execute (ms): %f milliseconds\n", GpuTime);
  //printf(" summation values: %i \n", sumStrData[0]); 
 
  /*  
  for (unsigned long int i = 0; i < sumGPUNumData; i++) {
      if (sumStrData[i] - 6 != 0) {
         printf(" The value that is wrong is: %lu, %i\n",i, sumStrData[i]);
	 break; 
      }  
  }
  
    
  for (unsigned long int i = sumGPUNumData; i < sumNumData; i++) {
      if (sumStrData[i] - 12 != 0) {
         printf(" The value that is wrong is: %lu, %i\n",i, sumStrData[i]);
	 break; 
      }  
  }
  */
 
  
  unsigned long int sum = 0;
  for (unsigned long int i = 0;  i < sumNumData; i++) {
      sum = 0;
      for (unsigned long int j = 0; j < nCols; j++) {
          sum+=reduceStrData[IDX2C(i,j,nCols)];
      }
      if (sum != sumStrData[i]) {
         printf(" The value that is wrong is: %lu, %i\n",i, sumStrData[i]);
	 break;
      }
  }
  
  cudaFree( sumStrDataDev );
  cudaFree( reduceStrDataDev );
   
  delete[] sumStrData;
  delete[] reduceStrData;
  
  //cudaFreeHost( sumStrData );
  //cudaFreeHost( reduceStrData );
  
  

  
  // second asynchronous  transfer 
  /*
  int* reduceStrOneData;
  int* sumStrOneData; 
  
 
    
  // pinned data
  
  cudaMallocHost((void**)&reduceStrOneData, reduceDataSize);
  cudaMallocHost((void**)&sumStrOneData, sumDataSize);
  
    
  //memset(reduceData, 1, reduceDataSize); 
  
  for (unsigned int i = 0; i < numData; i++) {
       reduceStrOneData[i] = 1;
  }
  */
   
  
  // cudaMallocHost could not allocate more than 16GB on physical memory
  int* reduceStrOneData = new int [numData];
  int* sumStrOneData  = new int [sumNumData]; 


  //memset(reduceData, 1, reduceDataSize); 
  
  
  for (unsigned long int i = 0; i < numData; i++) {
      reduceStrOneData[i] = (rand() % 100000) + 1;
  }
  

  int* reduceStrOneDataDev;
  int* sumStrOneDataDev; 
    
    
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&reduceStrOneDataDev), reduceGPUDataSize);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&reduceStrDataDev), reduceDataSize) ");
 
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&sumStrOneDataDev), sumGPUDataSize);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&sumStrDataDev), sumDataSize); ");  
 

  // Record the start event
  cudaEventRecord(start, 0); 

    
  
  for (int i = 0; i < nStreams; i+= nStreamsFitGPU) { 
    for (int j = 0, k = i; k < nStreams && j < nStreamsFitGPU; ++j, k++ ) { 
      unsigned long int offset = k * streamSize;
      cudaMemcpyAsync(&reduceStrOneDataDev[offset % numGPUData], &reduceStrOneData[offset], streamBytes, cudaMemcpyHostToDevice, stream[j]);   
    }

    for (int j = 0, k = i; k < nStreams && j < nStreamsFitGPU; ++j, k++ ) { 
      unsigned long int offsetResult = k * streamSizeResult;
      reductionSum<<<grid, threads, 0, stream[j]>>>( reduceStrOneDataDev, sumStrOneDataDev, streamSizeResult, nCols, offsetResult % sumGPUNumData);
    }
  
    for (int j = 0, k = i; k < nStreams && j < nStreamsFitGPU; ++j, k++ ) { 
      unsigned long int offsetResult = k * streamSizeResult;
      cudaMemcpyAsync(&sumStrOneData[offsetResult], &sumStrOneDataDev[offsetResult % sumGPUNumData ], streamBytesResult, cudaMemcpyDeviceToHost, stream[j]);
    }
  }
  
   
  // Record the stop event
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop); 
  
  // Calculate elapsed time
  
  cudaEventElapsedTime(&GpuTime, start, stop); 
  
  printf("Time for asynchronous V2 transfer and execute (ms): %f milliseconds\n", GpuTime);
  printf(" summation values: %i \n", sumStrOneData[0]); 
 
   
  for (unsigned long int i = 0;  i < sumNumData; i++) {
      sum = 0;
      for (unsigned long int j = 0; j < nCols; j++) {
          sum+=reduceStrOneData[IDX2C(i,j,nCols)];
      }
      if (sum != sumStrOneData[i]) {
	 printf("The value of sum :%lu \n", sum);
         printf("The value of :%i\n", reduceStrOneData[IDX2C(i,0,nCols)]);

         printf("The value of :%i\n", reduceStrOneData[IDX2C(i,1,nCols)]);
         
	 
         printf("The value of :%i \n", reduceStrOneData[IDX2C(i,2,nCols)]);
	 
         printf("The value of :%i \n", reduceStrOneData[IDX2C(i,3,nCols)]);

         printf("The value of :%i \n", reduceStrOneData[IDX2C(i,4,nCols)]);

         printf("The value of :%i \n", reduceStrOneData[IDX2C(i,5,nCols)]);
	 printf(" The value that is wrong is: %lu, %i\n",i, sumStrOneData[i]);
	 break;
      }
  }
  
   
 
  /*
  for (int i = 0; i < sumNumData; i++) {
      if (sumStrOneData[i] - 6 != 0) {
         printf(" The value that is wrong is: %i, %i\n",i, sumStrOneData[i]);
	 break; 
      }  
  }
  */
  
 cudaFree( sumStrOneDataDev );
 cudaFree( reduceStrOneDataDev );
  
  for (int i = 0; i < nStreams; ++i) {
     cudaStreamDestroy(stream[i]) ;
  }
 
  
  delete[] sumStrOneData;
  delete[] reduceStrOneData;
  //cudaFreeHost( sumStrOneData );
 //cudaFreeHost( reduceStrOneData );

 

  

  return 0;
};	
