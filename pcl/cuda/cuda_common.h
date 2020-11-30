#pragma once
#include "../pcl/pcl_base.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector_functions.hpp>


namespace pclcuda {
	template<typename T>
	using  PointCloudHost = thrust::host_vector<T>;

	template<typename T>
	using  PlaneParamHost = thrust::host_vector<T>;

	template<typename T>
	using  PointCloudDevice = thrust::device_vector<T>;

	template<typename T>
	using  PlaneParamDevice = thrust::device_vector<T>;

}

namespace pclcuda {
	/*
	Copies the content of the point cloud into the host memory thrust vector. 
	[in] inCloud - Pointer to the cloud 
	[out] outCloudHost - Pointer to the cloud in the thrust host vector
	[return] boolean - true is process succeeded/ else false
	*/
	template <typename T1, typename T2>
	bool copyPtCldToHostMem(pcl::PointCloud<T1>& inCloud, pclcuda::PointCloudHost<T2>& outCloudHost);

	/*
	Copies the content of the mesh into the host memory thrust vector.
	[in] inCloud - Pointer to the cloud
	[out] outCloudHost - Pointer to the cloud in the thrust host vector
	[return] boolean - true is process succeeded/ else false
	*/
	template <typename T1, typename T2>
	bool copyMeshToHostMem(pcl::PolygonMesh<T1>& inMesh, pclcuda::PointCloudHost<T2>& vertices_1, pclcuda::PointCloudHost<T2>& vertices_2, pclcuda::PointCloudHost<T2>& vertices_3,
		pclcuda::PlaneParamHost<T2>& planeParam);
}
