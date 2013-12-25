#!/bin/sh

nvcc -DNR_INPUTS=5 -DNR_OUTPUTS=15 -DMAXE=16 -DSAMPLES=32 -O3 -gencode arch=compute_30,code=sm_30 -o layered layered.cu -I./inc
./layered -i bch.i -o bch.o -g enc.bin
./bch_encoder_test -i bch.i -o bch.o -g enc.bin
nvcc -DNR_INPUTS=15 -DNR_OUTPUTS=5 -DMAXE=16 -DSAMPLES=32 -O3 -gencode arch=compute_30,code=sm_30 -o layered layered.cu -I./inc
./layered -i bch.o -o bch.i -g dec.bin

./random -w 5 -c 1 -o binary.dat
./bch_encoder -g enc.bin -i binary.dat > encoded.dat
./awgn  -i encoded.dat -o awgn.dat
./bch_decoder -g dec.bin -i encoded.dat > decoded.dat
#./ber -i binary.dat -o decoded.dat

cat binary.dat decoded.dat
