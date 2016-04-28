nvcc example3.cu -o test3.run
nvprof --export-profile test3.nvprof ./test3.run 