#include "cuda_preprocess.h"

template <typename T>
pcl::Clusters cuda::PreProcess<T>::euclideanClustering(pcl::PointCloud<T>& inCloud, const float tol, const int max_nn) {

	pcl::Clusters outCluster;
	outCluster.clear();

	std::vector<bool> flg;	// true if processed, false otherwise

	flg.resize(inCloud.size());
	std::fill(flg.begin(), flg.end(), false);	// Initializing the vector

	// Setting the Kd- tree and querying all the points
	flann::Matrix<float> flannDists;
	flann::Matrix<int> flannIndices;
	pclcuda::nn::KDTreeCUDA<float> kdTreeGPU;
	kdTreeGPU.setInputCloud(inCloud);
	
	kdTreeGPU.knnSearchNPoints(inCloud, flannIndices, flannDists, tol, max_nn); //Radius search for all points

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
			//kdTreeCPU.knnSearch1Point(queryPt, indices, flannDists, tol, max_nn);

			for (size_t i = 0; i < max_nn; i++)
			{
				if ((flannIndices[queue.indices[s]][i] >= 0) && (!flg[flannIndices[queue.indices[s]][i]]))
				{
					queue.indices.push_back(flannIndices[queue.indices[s]][i]);
					flg[flannIndices[queue.indices[s]][i]] = true;
				}
				if (flannIndices[queue.indices[s]][i] < 0)
					break;

			}
			queueLength = queue.indices.size();
		}
		outCluster.push_back(queue);
		queue.indices.clear();			// Clear the queue for the new seed point
	}
	return outCluster;
}


template pcl::Clusters cuda::PreProcess<float>::euclideanClustering(pcl::PointCloud<float>& inCloud, const float tol, const int max_nn);