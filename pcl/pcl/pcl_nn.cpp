#include "pcl_nn.h"

template <typename T_CPU>
pcl::nn::KDTreeCPU<T_CPU>::KDTreeCPU() {

}

template <typename T_CPU>
pcl::nn::KDTreeCPU<T_CPU>::~KDTreeCPU() {

}

template <typename T_CPU>
void pcl::nn::KDTreeCPU<T_CPU>::setInputCloud(pcl::PointCloud<T_CPU>& inCloud) {
	size_t cloudSize = inCloud.size();
	this->cloudMatrix = flann::Matrix<T_CPU>(new T_CPU[cloudSize * 3], cloudSize, 3);
	for (size_t i = 0; i < cloudSize; i++)
	{
		this->cloudMatrix[i][0] = inCloud.at(i).x;
		this->cloudMatrix[i][1] = inCloud.at(i).y;
		this->cloudMatrix[i][2] = inCloud.at(i).z;

	}
	this->kdIndex = new flann::Index<flann::L2<T_CPU>>(this->cloudMatrix, flann::KDTreeSingleIndexParams(2));
}

template <typename T_CPU>
void pcl::nn::KDTreeCPU<T_CPU>::buildIndex() {
	this->kdIndex->buildIndex();
}

template <typename T_CPU>
int pcl::nn::KDTreeCPU<T_CPU>::knnSearch1Point(pcl::PointXYZ<T_CPU>& queryPoint, flann::Matrix<int>& indices, flann::Matrix<T_CPU>& dist, int neighbors) {
	int noOfQueryPts = 1;
	flann::Matrix<T_CPU> queryMat = flann::Matrix<T_CPU>(new T_CPU[noOfQueryPts *3], noOfQueryPts, 3);
	indices = flann::Matrix<int>(new int[noOfQueryPts * neighbors], noOfQueryPts, neighbors);
	dist = flann::Matrix<T_CPU>(new T_CPU[noOfQueryPts * neighbors], noOfQueryPts, neighbors);

	queryMat[0][0] = queryPoint.x;
	queryMat[0][1] = queryPoint.y;
	queryMat[0][2] = queryPoint.z;

	return (this->knnSearch(queryMat, indices, dist, neighbors));
}

template <typename T_CPU>
int pcl::nn::KDTreeCPU<T_CPU>::knnSearch1Point(pcl::PointXYZ<T_CPU>& queryPoint, flann::Matrix<int>& indices, flann::Matrix<T_CPU>& dist, float radius, int max_neighbors) {
	int noOfQueryPts = 1;
	flann::Matrix<T_CPU> queryMat = flann::Matrix<T_CPU>(new T_CPU[noOfQueryPts * 3], noOfQueryPts, 3);
	indices = flann::Matrix<int>(new int[noOfQueryPts * max_neighbors], noOfQueryPts, max_neighbors);
	dist = flann::Matrix<T_CPU>(new T_CPU[noOfQueryPts * max_neighbors], noOfQueryPts, max_neighbors);

	queryMat[0][0] = queryPoint.x;
	queryMat[0][1] = queryPoint.y;
	queryMat[0][2] = queryPoint.z;

	return (this->knnSearch(queryMat, indices, dist, radius, max_neighbors));
}

template <typename T_CPU>
int pcl::nn::KDTreeCPU<T_CPU>::knnSearchNPoints(pcl::PointCloud<T_CPU>& queryVec, flann::Matrix<int>& indices, flann::Matrix<T_CPU>& dist, int neighbors) {
	int noOfQueryPts = queryVec.size();
	flann::Matrix<T_CPU> queryMat = flann::Matrix<T_CPU>(new T_CPU[noOfQueryPts * 3], noOfQueryPts, 3);
	indices = flann::Matrix<int>(new int[noOfQueryPts * neighbors], noOfQueryPts, neighbors);
	dist = flann::Matrix<T_CPU>(new T_CPU[noOfQueryPts * neighbors], noOfQueryPts, neighbors);

	for (size_t i = 0; i < noOfQueryPts; i++)
	{
		queryMat[i][0] = queryVec.at(i).x;
		queryMat[i][1] = queryVec.at(i).y;
		queryMat[i][2] = queryVec.at(i).z;
	}

	return (this->knnSearch(queryMat, indices, dist, neighbors));
}

template <typename T_CPU>
int pcl::nn::KDTreeCPU<T_CPU>::knnSearchNPoints(pcl::PointCloud<T_CPU>& queryVec, flann::Matrix<int>& indices, flann::Matrix<T_CPU>& dist, float radius, int max_neighbors) {
	int noOfQueryPts = queryVec.size();
	flann::Matrix<T_CPU> queryMat = flann::Matrix<T_CPU>(new T_CPU[noOfQueryPts * 3], noOfQueryPts, 3);
	indices = flann::Matrix<int>(new int[noOfQueryPts * max_neighbors], noOfQueryPts, max_neighbors);
	dist = flann::Matrix<T_CPU>(new T_CPU[noOfQueryPts * max_neighbors], noOfQueryPts, max_neighbors);

	for (size_t i = 0; i < noOfQueryPts; i++)
	{
		queryMat[i][0] = queryVec.at(i).x;
		queryMat[i][1] = queryVec.at(i).y;
		queryMat[i][2] = queryVec.at(i).z;
	}

	return (this->knnSearch(queryMat, indices, dist, radius, max_neighbors));
}

template <typename T_CPU>
int pcl::nn::KDTreeCPU<T_CPU>::knnSearch(flann::Matrix<T_CPU>& query, flann::Matrix<int>& indices, flann::Matrix<T_CPU>& dist, int neighbors) {
	int totalProcessed = 0;
	totalProcessed = this->kdIndex->knnSearch(query, indices, dist, neighbors, flann::SearchParams(128));
	return totalProcessed;
}

template <typename T_CPU>
int pcl::nn::KDTreeCPU<T_CPU>::knnSearch(flann::Matrix<T_CPU>& query, flann::Matrix<int>& indices, flann::Matrix<T_CPU>& dist, float radius, int max_neighbors) {
	int totalProcessed = 0;
	flann::SearchParams searchParams(128);
	searchParams.max_neighbors = max_neighbors;
	searchParams.sorted = true;
	totalProcessed = this->kdIndex->radiusSearch(query, indices, dist, pow(radius, 2), searchParams);
	return totalProcessed;
}


template pcl::nn::KDTreeCPU<float>::KDTreeCPU();
template pcl::nn::KDTreeCPU<float>::~KDTreeCPU();
template void pcl::nn::KDTreeCPU<float>::setInputCloud(pcl::PointCloud<float>& inCloud);
template void pcl::nn::KDTreeCPU<float>::buildIndex();
template int pcl::nn::KDTreeCPU<float>::knnSearch(flann::Matrix<float>&, flann::Matrix<int>&, flann::Matrix<float>&, int);
template int pcl::nn::KDTreeCPU<float>::knnSearch(flann::Matrix<float>&, flann::Matrix<int>&, flann::Matrix<float>&, float, int);
template int pcl::nn::KDTreeCPU<float>::knnSearch1Point(pcl::PointXYZ<float>&, flann::Matrix<int>&, flann::Matrix<float>&, int);
template int pcl::nn::KDTreeCPU<float>::knnSearchNPoints(pcl::PointCloud<float>&, flann::Matrix<int>&, flann::Matrix<float>&, int);
template int pcl::nn::KDTreeCPU<float>::knnSearchNPoints(pcl::PointCloud<float>&, flann::Matrix<int>&, flann::Matrix<float>&, float, int);
template int pcl::nn::KDTreeCPU<float>::knnSearch1Point(pcl::PointXYZ<float>& queryPoint, flann::Matrix<int>& indices, flann::Matrix<float>& dist, float radius, int);

template pcl::nn::KDTreeCPU<double>::KDTreeCPU();
template pcl::nn::KDTreeCPU<double>::~KDTreeCPU();
template void pcl::nn::KDTreeCPU<double>::setInputCloud(pcl::PointCloud<double>& inCloud);
template void pcl::nn::KDTreeCPU<double>::buildIndex();
template int pcl::nn::KDTreeCPU<double>::knnSearch(flann::Matrix<double>&, flann::Matrix<int>&, flann::Matrix<double>&, int);
template int pcl::nn::KDTreeCPU<double>::knnSearch(flann::Matrix<double>&, flann::Matrix<int>&, flann::Matrix<double>&, float, int);
template int pcl::nn::KDTreeCPU<double>::knnSearch1Point(pcl::PointXYZ<double>&, flann::Matrix<int>&, flann::Matrix<double>&, int);
template int pcl::nn::KDTreeCPU<double>::knnSearchNPoints(pcl::PointCloud<double>&, flann::Matrix<int>&, flann::Matrix<double>&, int);
template int pcl::nn::KDTreeCPU<double>::knnSearchNPoints(pcl::PointCloud<double>&, flann::Matrix<int>&, flann::Matrix<double>&, float, int);
template int pcl::nn::KDTreeCPU<double>::knnSearch1Point(pcl::PointXYZ<double>& queryPoint, flann::Matrix<int>& indices, flann::Matrix<double>& dist, float radius,int);

