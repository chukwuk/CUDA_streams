# CUDA Streams.

# Introduction.
CUDA Stream application for CUDA kernel implementation for the summation of 2D array along the rows.

# Compile code
This is used for the compilation of code for two cases. The first case involves data that both fit in the CPU (host) and GPU (device) memory and it is located in the root directory. The second case involves data that fit in the host memory but does not fit in the device memory and it is located in this directory (./Data_cannot_fit_in_GPU_memory).
```
$ make
```
# Run code
```
$ ./main
```
# References
* [How to Overlap Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)
* [CUDA Streams and Concurrency](https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf/)
* [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)


