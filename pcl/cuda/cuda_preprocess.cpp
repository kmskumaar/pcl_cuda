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

template <typename T>
void cuda::PreProcess<T>::normalEstimation(pcl::PointCloud<T> &inCloud, pcl::NormalCloud<T> &outNormal, const int neighbors, const short threadToUse) {
	outNormal.resize(inCloud.size());

	// Setting the Kd- tree and querying all the points
	flann::Matrix<float> flannDists;
	flann::Matrix<int> flannIndices;
	pclcuda::nn::KDTreeCUDA<float> kdTreeGPU;
	kdTreeGPU.setInputCloud(inCloud);

	kdTreeGPU.knnSearchNPoints(inCloud, flannIndices, flannDists, neighbors);

	if (threadToUse != 1) {
		int nb_threads = threadToUse;
		if (threadToUse == 0) {
			nb_threads = std::thread::hardware_concurrency();
		}

		int batch_size = inCloud.size() / nb_threads;
		int batch_remainder = inCloud.size() % nb_threads;

		std::vector< std::thread > my_threads(nb_threads);

		for (int i = 0; i < nb_threads; ++i)
		{
			int start = i * batch_size;
			my_threads[i] = std::thread(&cuda::PreProcess<T>::normalEstimation_parallel, this, start, (start + batch_size), std::ref(inCloud), std::ref(flannIndices), neighbors, std::ref(outNormal));
		}

		// Process the remainder separately
		int start = nb_threads * batch_size;
		this->normalEstimation_parallel(start, start + batch_remainder, inCloud, flannIndices, neighbors, outNormal);

		std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
	}

	else
		this->normalEstimation_parallel(0, inCloud.size(), inCloud, flannIndices, neighbors, outNormal);
}

template <typename T>
pcl::Normal<T> cuda::PreProcess<T>::computePointNormal(pcl::PointXYZ<T> &point, pcl::PointCloud<T> &inCloud, pcl::Indices indices) {

	pcl::Normal<T> ptNormal{ 0.0,0.0,0.0 };
	pcl::PointXYZ<T> centroid;
	pcl::PointCloud<T> demeanCld = pcl::demeanCloud(inCloud, indices, centroid, false);
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
	ptNormal.i = ptNormalVec(0, 0);
	ptNormal.j = ptNormalVec(1, 0);
	ptNormal.k = ptNormalVec(2, 0);

	pcl::flipNormalToViewPoint(point, { 0.0,0.0,0.0 }, ptNormal);

	return ptNormal;
}

template <typename T>
void cuda::PreProcess<T>::normalEstimation(pcl::PointCloud<T> &inCloud, pcl::NormalCloud<T> &outNormal, const float radius, const int max_nn, const short threadToUse) {
	outNormal.resize(inCloud.size());

	// Setting the Kd- tree and querying all the points
	flann::Matrix<float> flannDists;
	flann::Matrix<int> flannIndices;	
	pclcuda::nn::KDTreeCUDA<float> kdTreeGPU;
	kdTreeGPU.setInputCloud(inCloud);
	kdTreeGPU.knnSearchNPoints(inCloud, flannIndices, flannDists, radius, max_nn); //Radius search for all points

	if (threadToUse != 1) {
		int nb_threads = threadToUse;
		if (threadToUse == 0) {
			nb_threads = std::thread::hardware_concurrency();
		}

		int batch_size = inCloud.size() / nb_threads;
		int batch_remainder = inCloud.size() % nb_threads;

		std::vector< std::thread > my_threads(nb_threads);

		for (int i = 0; i < nb_threads; ++i)
		{
			int start = i * batch_size;
			my_threads[i] = std::thread(&cuda::PreProcess<T>::normalEstimation_parallel, this, start, (start + batch_size), std::ref(inCloud), std::ref(flannIndices), max_nn, std::ref(outNormal));
		}

		// Process the remainder separately
		int start = nb_threads * batch_size;
		this->normalEstimation_parallel(start, start + batch_remainder, inCloud, flannIndices, max_nn, outNormal);

		std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
	}

	else
		this->normalEstimation_parallel(0, inCloud.size(), inCloud, flannIndices, max_nn, outNormal);
}

template <typename T>
void cuda::PreProcess<T>::normalEstimation_parallel(const int start, const int end, pcl::PointCloud<T> &inCloud, flann::Matrix<int> flannIndices, int max_nn, pcl::NormalCloud<T> &outNormal) {
	for (size_t idx = start; idx < end; idx++)
	{
		pcl::Indices neighborIndices;
		neighborIndices.indices.clear();
		for (size_t n = 0; n < max_nn; n++) {
			if (flannIndices[idx][n] >= 0)
				neighborIndices.indices.push_back(flannIndices[idx][n]);
			else
				break;
		}

		outNormal[idx] = computePointNormal(inCloud[idx], inCloud, neighborIndices);
	}
}


template pcl::Clusters cuda::PreProcess<float>::euclideanClustering(pcl::PointCloud<float>& inCloud, const float tol, const int max_nn);
template void cuda::PreProcess<float>::normalEstimation(pcl::PointCloud<float> &inCloud, pcl::NormalCloud<float> &outNormal, const int neighbors, const short threadToUse);
template void cuda::PreProcess<float>::normalEstimation(pcl::PointCloud<float> &inCloud, pcl::NormalCloud<float> &outNormal, const float radius, const int max_nn, const short threadToUse);