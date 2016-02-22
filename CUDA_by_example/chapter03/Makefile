all: enum_gpu set_gpu simple_kernel hello_world simple_device_call simple_kernel_params

enum_gpu: enum_gpu.cu
	nvcc -o enum_gpu enum_gpu.cu

set_gpu: set_gpu.cu
	nvcc -o set_gpu set_gpu.cu

simple_kernel: simple_kernel.cu
	nvcc -o simple_kernel simple_kernel.cu

hello_world: hello_world.cu
	nvcc -o hello_world hello_world.cu

simple_device_call: simple_device_call.cu
	nvcc -o simple_device_call simple_device_call.cu

simple_kernel_params: simple_kernel_params.cu
	nvcc -o simple_kernel_params simple_kernel_params.cu

clean:
	rm enum_gpu set_gpu simple_kernel hello_world simple_device_call simple_kernel_params
