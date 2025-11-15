# CUDA Streams.

# Introduction.
CUDA Stream application for CUDA kernel implementation for the summation of 2D array along the rows.

# Compile code
This is used for compilation of code for both case in which the data either fit OR not fit into GPU memory.
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


