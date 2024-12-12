all:
	nvcc -std=c++11 -rdc=true -G  -Xptxas=-maxrregcount=32 --ptxas-options=-v -lineinfo -c flash_decoding_local_test.cu -o flash_decoding_local_test.o
	nvcc -std=c++11 -o flash_decoding_local_test flash_decoding_local_test.o -g -lineinfo
