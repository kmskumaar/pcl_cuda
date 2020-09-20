#pragma once

#include "pcl_base.h"
//#include <thrust/host_vector.h>
//#include <vector_functions.hpp>
//
//namespace pcl {
//	/*
//	Copies the content of the point cloud into the host memory thrust vector. 
//	[in] inCloud - Pointer to the cloud 
//	[out] outCloudHost - Pointer to the cloud in the thrust host vector
//	[return] boolean - true is process succeeded/ else false
//	*/
//	template <typename T1, typename T2>
//	bool copyToHostMem(pcl::PointCloud<T1>& inCloud, pcl::cuda::PointCloudHost<T2>& outCloudHost); 
//}
//
//template<>
//bool pcl::copyToHostMem(pcl::PointCloud<float>& inCloud, pcl::cuda::PointCloudHost<float4>& outCloudHost) {
//		size_t noOfPts = inCloud.size();
//		outCloudHost.resize(noOfPts);
//
//		for (size_t i = 0; i < noOfPts; i++)
//			outCloudHost[i] = make_float4(inCloud[i].x, inCloud[i].y, inCloud[i].z, 1.0);
//
//		return true;
//}
//
//template<>
//bool pcl::copyToHostMem(pcl::PointCloud<double>& inCloud, pcl::cuda::PointCloudHost<double4>& outCloudHost) {
//	size_t noOfPts = inCloud.size();
//	outCloudHost.resize(noOfPts);
//
//	for (size_t i = 0; i < noOfPts; i++)
//		outCloudHost[i] = make_double4(inCloud[i].x, inCloud[i].y, inCloud[i].z, 1.0);
//
//	return true;
//}
