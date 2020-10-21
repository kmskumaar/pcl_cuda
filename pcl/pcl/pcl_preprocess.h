#pragma once

#include "pcl_common.h"
#include "pcl_nn.h"

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
	};

}

