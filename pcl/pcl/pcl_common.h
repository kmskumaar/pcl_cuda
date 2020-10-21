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

namespace pcl {

	/*
	Computes the centroid for the given indices in the point cloud 
	[in] inCloud - Pointer to the input cloud 
	[in] indices - Pointer to the indices	
	[return] pcl::PointXYZ - centroid for the points mentioned in the indices
	*/
	template<typename T>
	pcl::PointXYZ<T> get3DCentroid(pcl::PointCloud<T>& inCloud, pcl::Indices &indices) {

		pcl::PointXYZ<double> pt{ 0.0,0.0,0.0 };
		pcl::PointXYZ<T> returnCentroid{ 0.0,0.0,0.0 };
		pcl::PointXYZ<double> centroidDouble;	//Precision is lost with float when the sum of points is large
		for (size_t idx = 0; idx < indices.indices.size(); idx++) {
			pt.x = pt.x + inCloud[indices.indices[idx]].x;
			pt.y = pt.y + inCloud[indices.indices[idx]].y;
			pt.z = pt.z + inCloud[indices.indices[idx]].z;
		}			
		
		centroidDouble = pt / indices.indices.size();
		returnCentroid.x = centroidDouble.x;
		returnCentroid.y = centroidDouble.y;
		returnCentroid.z = centroidDouble.z;
		return returnCentroid;
	}

	/*
	Computes the centroid for the entire point cloud
	[in] inCloud - Pointer to the input cloud
	[return] pcl::PointXYZ - centroid for the point cloud
	*/
	template<typename T>
	pcl::PointXYZ<T> get3DCentroid(pcl::PointCloud<T>& inCloud) {

		pcl::PointXYZ<double> pt{ 0.0,0.0,0.0 };
		pcl::PointXYZ<T> returnCentroid{ 0.0,0.0,0.0 };
		pcl::PointXYZ<double> centroidDouble;	//Precision is lost with float when the sum of points is large
		for (size_t idx = 0; idx < inCloud.size(); idx++) {
			pt.x = pt.x + inCloud[idx].x;
			pt.y = pt.y + inCloud[idx].y;
			pt.z = pt.z + inCloud[idx].z;
		}

		centroidDouble = pt / inCloud.size();
		returnCentroid.x = centroidDouble.x;
		returnCentroid.y = centroidDouble.y;
		returnCentroid.z = centroidDouble.z;
		return returnCentroid;
	}


	template<typename T>
	pcl::PointCloud<T> demeanCloud(pcl::PointCloud<T>& inCloud) {

		pcl::PointCloud<T> outCloud;
		outCloud.resize(inCloud.size());
		pcl::PointXYZ<T> centroid = get3DCentroid<T>(inCloud);

		for (size_t idx = 0; idx < inCloud.size(); idx++)
			outCloud[idx] = inCloud[idx] - centroid;

		return outCloud;
	}

	/*
	Demeans only the points meantioned in the indices 
	[in] inCloud - Pointer to the input cloud
	[in] indices - Pointer to the indices
	[in] useCldCentroid - Uses the centroid of the entire point cloud. When false, uses only the centroid of the points in indices
	[return] pcl::PointXYZ - centroid for the points mentioned in the indices
	*/
	template<typename T>
	pcl::PointCloud<T> demeanCloud(pcl::PointCloud<T>& inCloud, pcl::Indices indices, const bool useCldCentroid = false) {

		pcl::PointCloud<T> outCloud;
		outCloud.resize(indices.indices.size());
		pcl::PointXYZ<T> centroid;
		if (useCldCentroid)
			centroid = get3DCentroid<T>(inCloud);
		else
			centroid = get3DCentroid<T>(inCloud, indices);

		for (size_t idx = 0; idx < indices.indices.size(); idx++)
			outCloud[idx] = inCloud[indices.indices[idx]] - centroid;

		return outCloud;
	}

	/*
	Flips the normal vector to the view point
	[in] pt - input cloud point
	[in] viewPt - viewpoint
	[out] normal - Pointer to the Normal vector that has to be flipped	
	*/
	template<typename T>
	void flipNormalToViewPoint(pcl::PointXYZ<T> pt, pcl::PointXYZ<T> viewPt, pcl::Normal<T>& normal) {
	
		pcl::Normal<T> losVec = reinterpret_cast<pcl::Normal<T>&>(viewPt - pt); //Line of Sight

		losVec = losVec.normalize();

		pcl::Normal<T> crossPro = losVec.cross(normal);
		float angle = atan2(crossPro.norm2(), losVec.dot(normal));
		if (angle > M_PI_2 || angle < -M_PI_2)
			normal = normal * (-1);
	}
	
}

