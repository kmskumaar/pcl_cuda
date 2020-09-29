#pragma once

#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"

namespace pclcuda {
	class CudaBase
	{
	public:
		CudaBase();

		~CudaBase();

		/*
		Allocates for the data type T in the device memory 
		[in] size - size of the allocation
		[return] - pointer to the device memory
		*/
		template <typename T>
		T* allocateDeviceMem(const unsigned int size);

		/*
		Allocates for the data type T in the device memory and copies the content of the host memory into device memory
		[in] H - Pointer to the host memory
		[in] size - size of the allocation and data to copy
		[return] - pointer to the device memory
		*/
		template <class T>
		T* copyHost2DevMem(T *H, unsigned int size);

	private:
		cudaDeviceProp deviceProp;

		void setCudaDevice(const int deviceIndex);

		void resetCudaDevices();

	};
}


