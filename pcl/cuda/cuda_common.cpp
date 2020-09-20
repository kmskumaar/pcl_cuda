#include "cuda_common.h"

template<>
bool pclcuda::copyToHostMem(pcl::PointCloud<float>& inCloud, pclcuda::PointCloudHost<float4>& outCloudHost) {
	size_t noOfPts = inCloud.size();
	outCloudHost.resize(noOfPts);

	for (size_t i = 0; i < noOfPts; i++)
		outCloudHost[i] = make_float4(inCloud[i].x, inCloud[i].y, inCloud[i].z, 1.0);

	return true;
}

template<>
bool pclcuda::copyToHostMem(pcl::PointCloud<double>& inCloud, pclcuda::PointCloudHost<double4>& outCloudHost) {
	size_t noOfPts = inCloud.size();
	outCloudHost.resize(noOfPts);

	for (size_t i = 0; i < noOfPts; i++)
		outCloudHost[i] = make_double4(inCloud[i].x, inCloud[i].y, inCloud[i].z, 1.0);

	return true;
}