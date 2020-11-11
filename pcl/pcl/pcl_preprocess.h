#pragma once

#include <thread>
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
		Sorts the input point cloud into clusters that form a plane within the planarTolerance
		[in] inCloud - pointer to the input point cloud
		[in] planarTolerance - maximum distance between the point and the plane to include the point into the current cluster
		[in] neighbors - Number of neighbors to choose in nn search. The same will be used to create a seed plane
						 Higher values can affect the performance of the NN search
		[return] - Indices of the individual clusters
		*/
		pcl::Clusters planarClustering(pcl::PointCloud<T> &inCloud, const T planarTolerance, const float radius, const int max_nn);

		/*
		Computes the normal vectors for individual points in the cloud. PCA is used to find the normal vectors
		[in] inCloud - pointer to the input point cloud
		[out] outNormal - pointer to the normal vectors
		[in] neighbors - Number of NN to use for finding the normal
		[in] threadsToUse - Number of threads to use. Default: maximum possible threads
		*/
		void normalEstimation(pcl::PointCloud<T> &inCloud, pcl::NormalCloud<T> &outNormal, const int neighbors, const short threadToUse = 0);

		/*
		Computes the normal vectors for individual points in the cloud. PCA is used to find the normal vectors
		[in] inCloud - pointer to the input point cloud
		[out] outNormal - pointer to the normal vectors
		[in] radius - Radius for the NN search
		[in] max_nn - Maximum number of neighbors
		[in] threadsToUse - Number of threads to use. Default: maximum possible threads
		*/
		void normalEstimation(pcl::PointCloud<T> &inCloud, pcl::NormalCloud<T> &outNormal, const float radius, const int max_nn = 20, const short threadToUse = 0);

		/*
		Computes the normal vector for a point. PCA is used to find the normal vectors
		[in] point - pointer to the point
		[in] inCloud - pointer to the input Cloud
		[in] indices - Indices of the nearest points that shall be used for computing the normal
		[return] - Normal vector of the point
		*/
		pcl::Normal<T> computePointNormal(pcl::PointXYZ<T> &point, pcl::PointCloud<T> &inCloud, pcl::Indices indices);

	private:
		void normalEstimation_parallel(const int start, const int end, pcl::PointCloud<T> &inCloud, flann::Matrix<int> flannIndices, int max_nn, pcl::NormalCloud<T> &outNormal);
	};

}


