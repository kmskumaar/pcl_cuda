#include "pcl_preprocess.h"

template <typename T>
pcl::Clusters pcl::PreProcess<T>::euclideanClustering(pcl::PointCloud<T>& inCloud, float tol, bool useCUDATree) {
	
	pcl::Clusters outCluster;
	outCluster.clear();

	std::vector<bool> flg;	// true if processed, false otherwise

	flg.resize(inCloud.size());
	std::fill(flg.begin(), flg.end(), false);	// Initializing the vector

	flann::Matrix<float> dists;
	flann::Matrix<int> indices;
	pcl::nn::KDTreeCPU<float> kdTreeCPU;
	kdTreeCPU.setInputCloud(inCloud);
	kdTreeCPU.buildIndex();
	int max_nn = 2000;

	for (size_t idx = 0; idx < inCloud.size(); idx++)
	{

		if (flg[idx])	// skip processed points
			continue;

		pcl::PointXYZ<T> seedPt = inCloud[idx];
		pcl::Indices queue;
		queue.indices.resize(1);
		queue.indices.at(0) = idx;

		int queueLength = 1;
		for (size_t s = 0; s < queueLength; s++)
		{
			pcl::PointXYZ<T> queryPt = inCloud.at(queue.indices.at(s));
			kdTreeCPU.knnSearch1Point(queryPt, indices, dists, tol, max_nn);

			for (size_t i = 0; i < max_nn; i++)
			{				
				if ((indices[0][i] >= 0) && (!flg[indices[0][i]]))
				{
					queue.indices.push_back(indices[0][i]);
					flg[indices[0][i]] = true;
				}
				if (indices[0][i] < 0)
					break;
								
			}
			queueLength = queue.indices.size();
		}
		outCluster.push_back(queue);	
		queue.indices.clear();			// Clear the queue for the new seed point
	}
	return outCluster;
}


template pcl::Clusters pcl::PreProcess<float>::euclideanClustering(pcl::PointCloud<float>& inCloud, float tol, bool useCUDATree);