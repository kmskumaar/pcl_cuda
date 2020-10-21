#pragma once

#include "pcl_common.h"
#include "pcl_nn.h"
#include <Eigen/SVD>

namespace pcl {
	template <class T>
	class PreProcess
	{
	public:
		/*
		Sorts the input point cloud into clusters based on the euclidean distance (Euclidean Clustering)
		[in] inCloud - pointer to the input point cloud
		[in] clusterTolerance - maximum distance from the point that belongs to the cluster. 
								Should be choosen based on the point cloud density. Also refers to the radius for the nearest neighbor search
		[in] max_nn - Maximum number of nearest neighbors returned for radius search. 
					  Higher values can affect the performance of the NN search
		[return] - Indices of the individual clusters
		*/
		pcl::Clusters euclideanClustering(pcl::PointCloud<T> &inCloud, const float clusterTolrence, const int max_nn = 100);

		/*
		Computes the normal vectors for individual points in the cloud. PCA is used to find the normal vectors
		[in] inCloud - pointer to the input point cloud
		[out] outNormal - pointer to the normal vectors
		[in] neighbors - Number of NN to use for finding the normal
		*/
		void normalEstimation(pcl::PointCloud<T> &inCloud, pcl::NormalCloud<T> &outNormal, const int neighbors);
		
		/*
		Computes the normal vectors for individual points in the cloud. PCA is used to find the normal vectors
		[in] inCloud - pointer to the input point cloud
		[out] outNormal - pointer to the normal vectors
		[in] radius - Radius for the NN search
		*/
		void normalEstimation(pcl::PointCloud<T> &inCloud, pcl::Normal<T> &outNormal, const float radius);

		/*
		Computes the normal vector for a point. PCA is used to find the normal vectors
		[in] point - pointer to the point
		[in] inCloud - pointer to the input Cloud
		[in] indices - Indices of the nearest points that shall be used for computing the normal
		[return] - Normal vector of the point
		*/
		pcl::Normal<T> computePointNormal(pcl::PointXYZ<T> &point, pcl::PointCloud<T> &inCloud, pcl::Indices indices);
	};

}


