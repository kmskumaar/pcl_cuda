#include "cuda_base.h"

#define CUDA_CHECK(cudaStatus) if((cudaStatus)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
	printf("CUDA Error Code: %d -> %s\n",static_cast<int>(cudaStatus), cudaGetErrorString(cudaStatus));\
    }

pclcuda::CudaBase::CudaBase() {

	int deviceIndex = -1;
	cudaError_t status = cudaGetDevice(&deviceIndex);

	std::cout << deviceIndex << std::endl;
	int deviceCount = 0;
	CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

	std::cout << "Number of CUDA Devices Found: " << deviceCount << std::endl;	
	this->setCudaDevice(0);
}

pclcuda::CudaBase::~CudaBase() {
	CUDA_CHECK(cudaDeviceReset());
}

template <typename T>
T* pclcuda::CudaBase::allocateDeviceMem(const unsigned int size) {
	T* D;
	CUDA_CHECK(cudaMalloc((void**)&D, size * sizeof(T)));
	return D;
}

template <typename T>
T* pclcuda::CudaBase::copyHost2DevMem(T *H, unsigned int size) {
	T *D;
	CUDA_CHECK(cudaMalloc((void**)&D, size * sizeof(T)));
	CUDA_CHECK(cudaMemcpy(D, H, size * sizeof(T), cudaMemcpyHostToDevice));
	return D;	
}

void pclcuda::CudaBase::setCudaDevice(const int deviceIndex) {
	CUDA_CHECK(cudaGetDeviceProperties(&this->deviceProp, deviceIndex));
	std::cout << "Setting " << this->deviceProp.name << std::endl;
	CUDA_CHECK(cudaSetDevice(deviceIndex));
}


template int* pclcuda::CudaBase::allocateDeviceMem<int>(const unsigned int size);
template int* pclcuda::CudaBase::copyHost2DevMem<int>(int *H, unsigned int size);

template float* pclcuda::CudaBase::allocateDeviceMem<float>(const unsigned int size);
template float* pclcuda::CudaBase::copyHost2DevMem<float>(float *H, unsigned int size);

template float4* pclcuda::CudaBase::allocateDeviceMem<float4>(const unsigned int size);
template float4* pclcuda::CudaBase::copyHost2DevMem<float4>(float4 *H, unsigned int size);

template double* pclcuda::CudaBase::allocateDeviceMem<double>(const unsigned int size);
template double* pclcuda::CudaBase::copyHost2DevMem<double>(double *H, unsigned int size);

template double4* pclcuda::CudaBase::allocateDeviceMem<double4>(const unsigned int size);
template double4* pclcuda::CudaBase::copyHost2DevMem<double4>(double4 *H, unsigned int size);
