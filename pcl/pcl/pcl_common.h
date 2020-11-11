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
	pcl::PointCloud<T> demeanCloud(pcl::PointCloud<T>& inCloud, pcl::PointXYZ<T>& outCentroid) {

		pcl::PointCloud<T> outCloud;
		outCloud.resize(inCloud.size());
		outCentroid = get3DCentroid<T>(inCloud);

		for (size_t idx = 0; idx < inCloud.size(); idx++)
			outCloud[idx] = inCloud[idx] - outCentroid;

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
	pcl::PointCloud<T> demeanCloud(pcl::PointCloud<T>& inCloud, pcl::Indices indices, pcl::PointXYZ<T>& outCentroid, const bool useCldCentroid = false) {

		pcl::PointCloud<T> outCloud;
		outCloud.resize(indices.indices.size());
		if (useCldCentroid)
			outCentroid = get3DCentroid<T>(inCloud);
		else
			outCentroid = get3DCentroid<T>(inCloud, indices);

		for (size_t idx = 0; idx < indices.indices.size(); idx++)
			outCloud[idx] = inCloud[indices.indices[idx]] - outCentroid;

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

	/*
	Fit a plane to the given set of points
	[in] inCloud - Pointer to the input cloud
	[in] indices - Pointer to the indices
	[return] pcl::Plane - Parameters of the plane 
	*/
	template<typename T>
	pcl::Plane<T> fitPlane(pcl::PointCloud<T>& inCloud, pcl::Indices indices, T &rms) {

		pcl::Normal<T> ptNormal{ 0.0,0.0,0.0 };
		pcl::PointXYZ<T> centroid;
		pcl::PointCloud<T> demeanCld;

		pcl::Plane<T> outPlane = { 0.0,0.0,0.0,0.0 };
		if ((indices.indices.size() != 0) && (indices.indices.size() < 3))
			return outPlane;
		if (indices.indices.size() == 0)
			demeanCld = pcl::demeanCloud(inCloud, centroid);
		else
			demeanCld = pcl::demeanCloud(inCloud, indices, centroid, false);

		Eigen::Matrix< T, Eigen::Dynamic, 3> matEig;
		matEig.resize(demeanCld.size(), 3);
		for (size_t idx = 0; idx < demeanCld.size(); idx++)
		{
			matEig(idx, 0) = demeanCld[idx].x;
			matEig(idx, 1) = demeanCld[idx].y;
			matEig(idx, 2) = demeanCld[idx].z;
		}
		Eigen::JacobiSVD<Eigen::MatrixXf> svd(matEig.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::Matrix<T, 3, 1> ptNormalVec = svd.matrixU().col(2);
		T variance = pow(svd.singularValues()(2, 0), 2);
		ptNormal.i = ptNormalVec(0, 0);
		ptNormal.j = ptNormalVec(1, 0);
		ptNormal.k = ptNormalVec(2, 0);

		pcl::flipNormalToViewPoint(centroid, { 0.0,0.0,0.0 }, ptNormal);

		outPlane.A = ptNormal.i;
		outPlane.B = ptNormal.j;
		outPlane.C = ptNormal.k;
		outPlane.D = (-1.0)*ptNormal.dot(*reinterpret_cast<pcl::Normal<T>*>(&centroid));
		rms = 10.0;	// TODO rms calculation using the singular values
		return outPlane;
	}

	/*
	Fit a plane to the given set of points
	[in] inCloud - Pointer to the input cloud
	[return] pcl::Plane - Parameters of the plane
	*/
	template<typename T>
	pcl::Plane<T> fitPlane(pcl::PointCloud<T>& inCloud, T &rms) {
		pcl::Indices ind;
		ind.indices.resize(0);	// Empty indices

		return pcl::fitPlane(inCloud, ind, rms);
	}
	
	template<typename T>
	T distanceToPlane(pcl::Plane<T> planeParam, pcl::PointXYZ<T> point) {
		pcl::Normal<T> planeNormal = (*reinterpret_cast<pcl::Normal<T>*>(&planeParam));
		return (planeParam.D + planeNormal.dot(*reinterpret_cast<pcl::Normal<T>*>(&point)));
	}
}

