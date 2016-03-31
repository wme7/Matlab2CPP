//mpi gpu ping pong with GPU-aware MPI implementations:
// node0/cpu -> node0/gpu --> node1/gpu -> node1/gpu 
// node0/cpu <- node0/gpu <--------------- node1/gpu
 
// compilation:                                                                     
// mpicxx ../../../scratch/mpi-pingpong-gpu.cpp \
//        -I <path to cuda include dir> \
//        -L <path to cuda lib dir> \
//        -lcudart

#include <iostream>
#include <algorithm>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h> // required if *not* compiling with nvcc
#include <host_allocator.h>
#define CHECK_CUDA_ERROR( err ) \
  if( err != cudaSuccess ) { \
    std::cerr << __LINE__ << " - ERROR: " \
              << cudaGetErrorString( err ) << std::endl; \
    exit( -1 ); \
  }

int main(int argc, char** argv) {
    if(argc != 2) {
        std::cout << "usage: " << argv[0]
                  << " <number of elements>" << std::endl;
    }
    MPI_Init(&argc, &argv);
    int task = -1;
    const size_t size = atoi(argv[1]);
    const size_t byte_size = size * sizeof(double);
    MPI_Comm_rank(MPI_COMM_WORLD, &task);
    double* device_data_send = 0;
    double* device_data_recv = 0;
    CHECK_CUDA_ERROR(cudaMalloc(&device_data_send, byte_size));
    CHECK_CUDA_ERROR(cudaMalloc(&device_data_recv, byte_size));
    const int tag0to1 = 0x01;
    const int tag1to0 = 0x10;
    int source = -1;
    int dest = -1;
    MPI_Status status;
#ifdef PAGE_LOCKED
    std::vector< double, host_allocator<double> > host_data(size, 0);
    std::vector< double, host_allocator<double> > device_host_data(size, 1);
#else
    std::vector< double > host_data(size, 0);
    std::vector< double > device_host_data(size, 1);
#endif
    if(task == 0) {
        dest = 1;
        source = 1;  
        for(int i = 0; i != int(host_data.size()); ++i) host_data[i] = i;
        CHECK_CUDA_ERROR(cudaMemcpy(device_data_send, &host_data[0],
                         byte_size, cudaMemcpyHostToDevice));
        MPI_Request send_req;
        MPI_Request recv_req;
        const double start_time = MPI_Wtime();
#ifdef HOST_COPY
        cudaMemcpy(&host_data[0], device_data_send, byte_size, cudaMemcpyDeviceToHost);
        MPI_Isend(&host_data[0], size, MPI_DOUBLE, dest, tag0to1, MPI_COMM_WORLD, &send_req);
        MPI_Irecv(&device_host_data[0], size, MPI_DOUBLE, source, tag1to0, MPI_COMM_WORLD, &recv_req);
#else
    	MPI_Isend(device_data_send, size, MPI_DOUBLE, dest, tag0to1, MPI_COMM_WORLD, &send_req);
        MPI_Irecv(device_data_recv, size, MPI_DOUBLE, source, tag1to0, MPI_COMM_WORLD, &recv_req);
#endif
        MPI_Wait(&recv_req, &status);
#ifdef HOST_COPY
        cudaMemcpy(device_data_recv, &device_host_data[0], byte_size, cudaMemcpyHostToDevice);
#endif
        const double exchange_time = MPI_Wtime(); 
    	  CHECK_CUDA_ERROR(cudaMemcpy(&device_host_data[0], device_data_recv, byte_size,
    	                   cudaMemcpyDeviceToHost));
        const double device_host_time = MPI_Wtime();
        const bool passed = std::equal(host_data.begin(), host_data.end(),
                                device_host_data.begin());
        if(passed) {
          std::cout << "PASSED\n";
          if( byte_size < 1024 * 1024 ) 
            std::cout << "Message size(bytes): " << byte_size << std::endl;
          else    
            std::cout << "Message size(MB): " << (byte_size/(1024*1024.0)) << std::endl;  
          std::cout << "Round-trip time(ms): " << 1000 * (exchange_time - start_time) << std::endl
                    << "Device to host transfer time(ms): "
                    << 1000 * (device_host_time - exchange_time) << std::endl;
        } else {
          std::cout << "FAILED" << std::endl;
        }     
    } else {
      dest = 0;
      source = 0;
      MPI_Request send_req;
      MPI_Request recv_req;
#ifdef HOST_COPY
      MPI_Irecv(&host_data[0], size, MPI_DOUBLE, source, tag0to1, MPI_COMM_WORLD, &recv_req);
      MPI_Wait(&recv_req, &status);
      cudaMemcpy(device_data_recv, &host_data[0], byte_size, cudaMemcpyHostToDevice);
      cudaMemcpy(&host_data[0], device_data_recv, byte_size, cudaMemcpyDeviceToHost);
      MPI_Isend(&host_data[0], size, MPI_DOUBLE, dest, tag1to0, MPI_COMM_WORLD, &send_req);
      MPI_Wait(&send_req, &status);
#else
      MPI_Irecv(device_data_recv, size, MPI_DOUBLE, source, tag0to1, MPI_COMM_WORLD, &recv_req);
      MPI_Wait(&recv_req, &status);
      MPI_Isend(device_data_recv, size, MPI_DOUBLE, dest, tag1to0, MPI_COMM_WORLD, &send_req);
      MPI_Wait(&send_req, &status);
#endif
    }
    CHECK_CUDA_ERROR(cudaFree(device_data_send));
    CHECK_CUDA_ERROR(cudaFree(device_data_recv));
    MPI_Finalize();
    CHECK_CUDA_ERROR(cudaDeviceReset());
    return 0;
}




