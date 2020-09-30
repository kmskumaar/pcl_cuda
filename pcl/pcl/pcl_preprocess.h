#pragma once

#include "pcl_common.h"
#include "pcl_nn.h"

namespace pcl {
	template <class T>
	class PreProcess
	{
	public:
		pcl::Clusters euclideanClustering(pcl::PointCloud<T> &inCloud, float clusterTolrence, bool useCUDATree = false);
	};

}


