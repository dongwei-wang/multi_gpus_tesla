# Author      : Dongwei Wang
# wdw828@gmail.com
# opencv + cuda makefile

BIN  		:= multi_gpus
CXXFLAGS	:= -std=c++11 -g
NVCCFLAGS 	:= -O3 \
		-Wno-deprecated-gpu-targets \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_20,code=sm_20

CUDA_INSTALL_PATH := /usr/local/cuda
LIBS 	= -L$(CUDA_INSTALL_PATH)/lib64 -lcuda -lcudart
LDFLAGS = `pkg-config --libs opencv`
CFLAGS 	= `pkg-config --cflags opencv`

CXX 	= g++

all: $(BIN)
$(BIN): main.o multi_gpus.o
	$(CXX) $(CXXFLAGS) -o $(BIN) main.o multi_gpus.o $(LDFLAGS) $(LIBS)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp $(CFLAGS)

multi_gpus.o: multi_gpus.cu
	nvcc $(NVCCFLAGS) -c multi_gpus.cu $(CFLAGS)

echo_install_path:
	echo $(CUDA_INSTALL_PATH)

clean:
	rm -f $(BIN) *.o tags

cleangrays:
	cd tar; rm -rf *.jpg; cd ..;
