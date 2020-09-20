#pragma once
#include "../pcl/pcl_base.h"
#include <thrust/host_vector.h>
#include <vector_functions.hpp>


namespace pclcuda {
	template<typename T>
	using  PointCloudHost = thrust::host_vector<T>;

}

namespace pclcuda {
	/*
	Copies the content of the point cloud into the host memory thrust vector. 
	[in] inCloud - Pointer to the cloud 
	[out] outCloudHost - Pointer to the cloud in the thrust host vector
	[return] boolean - true is process succeeded/ else false
	*/
	template <typename T1, typename T2>
	bool copyToHostMem(pcl::PointCloud<T1>& inCloud, pclcuda::PointCloudHost<T2>& outCloudHost);
}
