#include "cuda_nn.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template <>
void pclcuda::nn::KDTreeCUDA<float>::setInputCloud(pcl::PointCloud<float>& inCloud) {
	size_t noOfPts = inCloud.size();
	thrust::host_vector<float4> cloudHostThrust(noOfPts);

	for (int i = 0; i < noOfPts; i++)
		cloudHostThrust[i] = make_float4(inCloud[i].x, inCloud[i].y, inCloud[i].z, 0);
	
	thrust::device_vector<float4> cldDevice = cloudHostThrust;

	flann::Matrix<float> data_device_matrix((float*)thrust::raw_pointer_cast(&cldDevice[0]), noOfPts, 3, 4 * 4);

	flann::KDTreeCuda3dIndexParams index_params;
	index_params["input_is_gpu_float4"] = true;
	this->kdIndex = new flann::KDTreeCuda3dIndex<flann::L2<float>>(data_device_matrix, index_params);
	this->kdIndex->buildIndex();
}

//template <>
//void pclcuda::nn::KDTreeCUDA<double>::setInputCloud(pcl::PointCloud<double>& inCloud) {
//	size_t noOfPts = inCloud.size();
//	thrust::host_vector<double4> cloudHostThrust(noOfPts);
//
//	for (int i = 0; i < noOfPts; i++)
//		cloudHostThrust[i] = make_double4(inCloud[i].x, inCloud[i].y, inCloud[i].z, 0);
//
//	thrust::device_vector<double4> cldDevice = cloudHostThrust;
//
//	flann::Matrix<double> data_device_matrix((double*)thrust::raw_pointer_cast(&cldDevice[0]), noOfPts, 3, 4 * 4);
//
//	flann::KDTreeCuda3dIndexParams index_params;
//	index_params["input_is_gpu_float4"] = true;
//	this->kdIndex = new flann::KDTreeCuda3dIndex<flann::L2<double>>(data_device_matrix, index_params);
//	this->kdIndex->buildIndex();
//}

template <>
int pclcuda::nn::KDTreeCUDA<float>::knnSearchNPoints(pcl::PointCloud<float>& inQuery, flann::Matrix<int>& indices_host, flann::Matrix<float>& sqrDist_host, int neighbors) {
	size_t noOfPts = inQuery.size();

	thrust::host_vector<float4> query_host(noOfPts);

	for (int i = 0; i < noOfPts; i++)
		query_host[i] = make_float4(inQuery[i].x, inQuery[i].y, inQuery[i].z, 0);

	thrust::device_vector<float4> query_device = query_host;

	flann::Matrix<float> query_device_matrix((float*)thrust::raw_pointer_cast(&query_device[0]), noOfPts, 3, 4 * 4);

	thrust::host_vector<int> indices_temp(noOfPts * neighbors);
	thrust::host_vector<float> dists_temp(noOfPts * neighbors);

	this->indices_d = indices_temp;
	this->sqrDists_d = dists_temp;

	flann::Matrix<int> indices_device_matrix((int*)thrust::raw_pointer_cast(&this->indices_d[0]), noOfPts, neighbors);
	flann::Matrix<float> dists_device_matrix((float*)thrust::raw_pointer_cast(&this->sqrDists_d[0]), noOfPts, neighbors);

	flann::SearchParams sp;
	sp.matrices_in_gpu_ram = true;
	int result = this->knnSearch(query_device_matrix, indices_device_matrix, dists_device_matrix, neighbors);

	indices_host = flann::Matrix<int>(new int[noOfPts * neighbors], noOfPts, neighbors);
	sqrDist_host = flann::Matrix<float>(new float[noOfPts * neighbors], noOfPts, neighbors);

	thrust::copy(this->sqrDists_d.begin(), this->sqrDists_d.end(), sqrDist_host.ptr());
	thrust::copy(this->indices_d.begin(), this->indices_d.end(), indices_host.ptr());

	return result;
}

//template <>
//int pclcuda::nn::KDTreeCUDA<double>::knnSearchNPoints(pcl::PointCloud<double>& inQuery, flann::Matrix<int>& indices_host, flann::Matrix<double>& sqrDist_host, int neighbors) {
//	size_t noOfPts = inQuery.size();
//
//	thrust::host_vector<double4> query_host(noOfPts);
//
//	for (int i = 0; i < noOfPts; i++)
//		query_host[i] = make_double4(inQuery[i].x, inQuery[i].y, inQuery[i].z, 0);
//
//	thrust::device_vector<double4> query_device = query_host;
//
//	flann::Matrix<double> query_device_matrix((double*)thrust::raw_pointer_cast(&query_device[0]), noOfPts, 3, 4 * 4);
//
//	thrust::host_vector<int> indices_temp(noOfPts * neighbors);
//	thrust::host_vector<double> dists_temp(noOfPts * neighbors);
//
//	thrust::device_vector<int> indices_device = indices_temp;
//	thrust::device_vector<double> dists_device = dists_temp;
//
//	flann::Matrix<int> indices_device_matrix((int*)thrust::raw_pointer_cast(&indices_device[0]), noOfPts, neighbors);
//	flann::Matrix<double> dists_device_matrix((double*)thrust::raw_pointer_cast(&dists_device[0]), noOfPts, neighbors);
//
//	flann::SearchParams sp;
//	sp.matrices_in_gpu_ram = true;
//	int result = this->knnSearch(query_device_matrix, indices_device_matrix, dists_device_matrix, neighbors);
//
//	indices_host = flann::Matrix<int>(new int[noOfPts * neighbors], noOfPts, neighbors);
//	sqrDist_host = flann::Matrix<double>(new double[noOfPts * neighbors], noOfPts, neighbors);
//
//	thrust::copy(dists_device.begin(), dists_device.end(), sqrDist_host.ptr());
//	thrust::copy(indices_device.begin(), indices_device.end(), indices_host.ptr());
//
//	return result;
//}

template <>
int pclcuda::nn::KDTreeCUDA<float>::knnSearchNPoints(pcl::PointCloud<float>& inQuery, flann::Matrix<int>& indices_host, flann::Matrix<float>& sqrDist_host, float radius, int max_neighbors) {
	size_t noOfPts = inQuery.size();

	thrust::host_vector<float4> query_host(noOfPts);

	for (int i = 0; i < noOfPts; i++)
		query_host[i] = make_float4(inQuery[i].x, inQuery[i].y, inQuery[i].z, 0);

	thrust::device_vector<float4> query_device = query_host;

	flann::Matrix<float> query_device_matrix((float*)thrust::raw_pointer_cast(&query_device[0]), noOfPts, 3, 4 * 4);

	thrust::host_vector<int> indices_temp(noOfPts * max_neighbors);
	thrust::host_vector<float> dists_temp(noOfPts * max_neighbors);

	this->indices_d = indices_temp;
	this->sqrDists_d = dists_temp;

	flann::Matrix<int> indices_device_matrix((int*)thrust::raw_pointer_cast(&this->indices_d[0]), noOfPts, max_neighbors);
	flann::Matrix<float> dists_device_matrix((float*)thrust::raw_pointer_cast(&this->sqrDists_d[0]), noOfPts, max_neighbors);

	flann::SearchParams sp;
	sp.matrices_in_gpu_ram = true;
	int result = this->knnSearch(query_device_matrix, indices_device_matrix, dists_device_matrix, radius, max_neighbors);

	indices_host = flann::Matrix<int>(new int[noOfPts * max_neighbors], noOfPts, max_neighbors);
	sqrDist_host = flann::Matrix<float>(new float[noOfPts * max_neighbors], noOfPts, max_neighbors);

	thrust::copy(this->sqrDists_d.begin(), this->sqrDists_d.end(), sqrDist_host.ptr());
	thrust::copy(this->indices_d.begin(), this->indices_d.end(), indices_host.ptr());

	return result;
}

template <>
int pclcuda::nn::KDTreeCUDA<float>::knnSearch(flann::Matrix<float>& queryDeviceMatrix, flann::Matrix<int>& indicesDeviceMatrix, flann::Matrix<float>& distDeviceMatrix, int neighbors) {
	flann::SearchParams sp;
	sp.matrices_in_gpu_ram = true;	
	return (this->kdIndex->knnSearch(queryDeviceMatrix, indicesDeviceMatrix, distDeviceMatrix, neighbors, sp));
	
}

//template <>
//int pclcuda::nn::KDTreeCUDA<double>::knnSearch(flann::Matrix<double>& queryDeviceMatrix, flann::Matrix<int>& indicesDeviceMatrix, flann::Matrix<double>& distDeviceMatrix, int neighbors) {
//	flann::SearchParams sp;
//	sp.matrices_in_gpu_ram = true;
//	return (this->kdIndex->knnSearch(queryDeviceMatrix, indicesDeviceMatrix, distDeviceMatrix, neighbors, sp));
//}

template <>
int pclcuda::nn::KDTreeCUDA<float>::knnSearch(flann::Matrix<float>& queryDeviceMatrix, flann::Matrix<int>& indicesDeviceMatrix, flann::Matrix<float>& distDeviceMatrix, float radius, int max_neighbors) {
	flann::SearchParams sp;
	sp.matrices_in_gpu_ram = true;
	sp.max_neighbors = max_neighbors;
	sp.sorted = true;
	return (this->kdIndex->radiusSearch(queryDeviceMatrix, indicesDeviceMatrix, distDeviceMatrix, powf(radius, 2), sp));
}

template <>
int* pclcuda::nn::KDTreeCUDA<float>::getIndicesDevicePtr() {
	return (int*)thrust::raw_pointer_cast(&this->indices_d[0]);
}

template <>
float* pclcuda::nn::KDTreeCUDA<float>::getSqrtDistDevicePtr() {
	return (float*)thrust::raw_pointer_cast(&this->sqrDists_d[0]);
}