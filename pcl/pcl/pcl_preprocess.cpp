#include "pcl_preprocess.h"

template <typename T>
pcl::Clusters pcl::PreProcess<T>::euclideanClustering(pcl::PointCloud<T>& inCloud, const float tol, const int max_nn) {
	
	pcl::Clusters outCluster;
	outCluster.clear();

	std::vector<bool> flg;	// true if processed, false otherwise

	flg.resize(inCloud.size());
	std::fill(flg.begin(), flg.end(), false);	// Initializing the vector

	// Setting the Kd- tree and querying all the points
	flann::Matrix<float> flannDists;
	flann::Matrix<int> flannIndices;
	pcl::nn::KDTreeCPU<float> kdTreeCPU;
	kdTreeCPU.setInputCloud(inCloud);
	kdTreeCPU.buildIndex();
	kdTreeCPU.knnSearchNPoints(inCloud, flannIndices, flannDists, tol, max_nn); //Radius search for all points

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


template <typename T>
void pcl::PreProcess<T>::normalEstimation(pcl::PointCloud<T> &inCloud, pcl::NormalCloud<T> &outNormal, const int neighbors) {
	outNormal.resize(inCloud.size());

	// Setting the Kd- tree and querying all the points
	flann::Matrix<float> flannDists;
	flann::Matrix<int> flannIndices;
	pcl::nn::KDTreeCPU<float> kdTreeCPU;
	kdTreeCPU.setInputCloud(inCloud);
	kdTreeCPU.buildIndex();
	kdTreeCPU.knnSearchNPoints(inCloud, flannIndices, flannDists, neighbors); //Radius search for all points
	
	for (size_t idx = 0; idx < inCloud.size(); idx++)
	{
		pcl::Indices neighborIndices;
		neighborIndices.indices.clear();
		for (size_t n = 0; n < neighbors; n++)
			neighborIndices.indices.push_back(flannIndices[idx][n]);

		outNormal[idx] = computePointNormal(inCloud[idx], inCloud, neighborIndices);
	}
}

template <typename T>
pcl::Normal<T> pcl::PreProcess<T>::computePointNormal(pcl::PointXYZ<T> &point, pcl::PointCloud<T> &inCloud, pcl::Indices indices) {

	pcl::Normal<T> ptNormal{ 0.0,0.0,0.0 };
	pcl::PointCloud<T> demeanCld = pcl::demeanCloud(inCloud, indices, false);
	Eigen::Matrix< T, Eigen::Dynamic, 3> matEig;
	matEig.resize(demeanCld.size(), 3);
	for (size_t idx = 0; idx < demeanCld.size(); idx++)
	{
		matEig(idx,0) = demeanCld[idx].x;
		matEig(idx,1) = demeanCld[idx].y;
		matEig(idx,2) = demeanCld[idx].z;
	}
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(matEig.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::Matrix<T, 3, 1> ptNormalVec = svd.matrixU().col(2);
	ptNormal.i = ptNormalVec(0, 0);
	ptNormal.j = ptNormalVec(1, 0);
	ptNormal.k = ptNormalVec(2, 0);

	pcl::flipNormalToViewPoint(point, { 0.0,0.0,0.0 }, ptNormal);

	return ptNormal;
}

template pcl::Clusters pcl::PreProcess<float>::euclideanClustering(pcl::PointCloud<float>& inCloud, const float tol, const int max_nn);
template void pcl::PreProcess<float>::normalEstimation(pcl::PointCloud<float> &inCloud, pcl::NormalCloud<float> &outNormal, const int neighbors);

template pcl::Normal<float> pcl::PreProcess<float>::computePointNormal(pcl::PointXYZ<float> &point, pcl::PointCloud<float> &inCloud, pcl::Indices indices);