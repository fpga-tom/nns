matriced:matriced.cu
	#g++ -g -std=c++0x -o matriced matriced.cc
	nvcc -O3  -gencode arch=compute_30,code=sm_30 -o matriced matriced.cu -I./inc -lcublas
cublas:cublas.cu
	nvcc -O3 -gencode arch=compute_30,code=sm_30 -o cublas cublas.cu -I./inc -lcublas
#	nvcc -O0 -G -g -gencode arch=compute_30,code=sm_30 --compiler-options "-std=c++0x" -o matriced matriced.cu -I./inc
ber:ber.cc
	nvcc -O3 -gencode arch=compute_30,code=sm_30 -o ber ber.cc -I./inc
bch_decoder_test:bch_decoder_test.cu layered.h
	nvcc -O3 -gencode arch=compute_30,code=sm_30 -o bch_decoder_test bch_decoder_test.cu -I./inc
bch_encoder_test:bch_encoder_test.cu layered.h
	nvcc -O3 -gencode arch=compute_30,code=sm_30 -o bch_encoder_test bch_encoder_test.cu -I./inc
awgn:awgn.cc
	nvcc -O3 -gencode arch=compute_30,code=sm_30 -o awgn awgn.cc -I./inc
bch_decoder:bch_decoder.cu layered.h
	nvcc -O3 -gencode arch=compute_30,code=sm_30 -o bch_decoder bch_decoder.cu -I./inc
bch_encoder:bch_encoder.cu layered.h
	nvcc -O3 -gencode arch=compute_30,code=sm_30 -o bch_encoder bch_encoder.cu -I./inc
random:random.cc
	nvcc -O3 -gencode arch=compute_30,code=sm_30 -o random random.cc
read_io:read_io.cc
	nvcc -O3 -gencode arch=compute_30,code=sm_30 -o read_io read_io.cc
layered:layered.cu layered.h
	nvcc -O3 -gencode arch=compute_30,code=sm_30 -o layered layered.cu -I./inc 
gabp:gabp.cu
	nvcc -O3  -gencode arch=compute_30,code=sm_30 -o gabp gabp.cu
sort:bs.cu
	nvcc -O0 -G -g -gencode arch=compute_30,code=sm_30 -o bs bs.cu
gpu1:
	nvcc -O3 -gencode arch=compute_30,code=sm_30 -o nns gpu.cu bitonicSort.cu -I./inc
gpu:
	nvcc -O3 -gencode arch=compute_30,code=sm_30 -o nns genome.cu
dynpar:dynpar.cu
	nvcc -O3 -arch=sm_35 -rdc=true dynpar.cu -o dynpar -lcudadevrt -L/usr/lib/x86_64-linux-gnu/
debug:
	nvcc -G -g -O0 -gencode arch=compute_30,code=sm_30 -o nns genome.cu
sum:sum.cu
	nvcc -G -g -O0 -gencode arch=compute_30,code=sm_30 -o sum sum.cu
all:layered random bch_encoder bch_decoder awgn bch_encoder_test bch_decoder_test
	$(CXX) -O3 -mtune=corei7-avx -o nns genome.cc -pthread
#	$(CXX) -g -o nns genome.cc -pthread
