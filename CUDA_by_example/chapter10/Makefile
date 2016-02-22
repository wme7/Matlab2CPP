all: basic_double_stream basic_double_stream_correct basic_single_stream copy_timed

basic_double_stream: basic_double_stream.cu
	nvcc -O3  -o basic_double_stream basic_double_stream.cu 

basic_double_stream_correct: basic_double_stream_correct.cu
	nvcc -O3 -o basic_double_stream_correct  basic_double_stream_correct.cu 

basic_single_stream: basic_single_stream.cu
	nvcc -O3 -o basic_single_stream  basic_single_stream.cu 

copy_timed: copy_timed.cu
	nvcc -O3 -o copy_timed  copy_timed.cu 

clean:
	rm basic_double_stream basic_double_stream_correct basic_single_stream copy_timed 
