
#define FLANN_USE_CUDA

#include "../pcl/pcl_file_reader.h"
#include "../pcl/pcl_nn.h"
#include "../cuda/cuda_common.h"
#include "../cuda/cuda_nn.h"
#include "../pcl/pcl_common.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


int main() {
	pcl::io::FileReader fileReader;
	pcl::PointCloud<float> cld;
	cld.clear();
	//pcl::PointXYZ<float> p1{ 1.0,2.0,3.0 };
	//pcl::PointXYZ<float> p2{ 11.0,22.0,33.0 };
	//cld.push_back(p1);
	//cld.push_back(p2);
	//cld.push_back(p1*5);

	const std::string filePath = "C:/Users/R/OneDrive - Fraunhofer/private/testFiles/iotest.txt";
	fileReader.readASCIIFile(filePath, cld, ",");

	pclcuda::PointCloudHost<float4> cldHostThrust;
	pclcuda::PointCloudHost<float4> queryHostThrust;
	pclcuda::copyToHostMem<float, float4>(cld, cldHostThrust);
	pclcuda::copyToHostMem<float, float4>(cld, queryHostThrust);


	//flann::Matrix<int> indices;// (new int[noQueryPts * nn], noQueryPts, nn);
	//flann::Matrix<float> dist;// (new float[noQueryPts * nn], noQueryPts, nn);
	//int nn = 10;
	flann::Matrix<float> dists;
	flann::Matrix<int> indices;
	const int nn = 10;

	pclcuda::nn::KDTreeCUDA<float> kdTreeGPU;
	kdTreeGPU.setInputCloud(cld);
	//kdTreeGPU.buildIndex();
	kdTreeGPU.knnSearchNPoints(cld, indices, dists, nn);


	std::string delim = ",";
	for (size_t i = 0; i < indices.rows; ++i) {
		//outFile << i << delim << query_host[i].x << delim << query_host[i].y << delim << query_host[i].z << delim;
		for (size_t j = 0; j < nn; ++j) {

			//double actualDistGPU = eucDist(query_host[i], data_host[indices_host[i][j]]);
			//double actualDistCPU = eucDist(query_host[i], data_host[indices[i][j]]);
			//printf("%d \t \t%f \t%f \t%f \t%f\n", i, dists_host[i][j], dists[i][j], actualDistGPU, actualDistCPU);

			std::cout << sqrtf(dists[i][j]) << delim;

			//printf("%d \t \t%d \t%d \n", i, indices_host[i][j], indices[i][j]);



		}
		std::cout << std::endl;
	}


	/*pcl::nn::KDTreeCPU<float> kdTreeCPU;
	kdTreeCPU.setInputCloud(cld);
	kdTreeCPU.buildIndex();

	
	int noQueryPts = 10;
	

	kdTreeCPU.knnSearchNPoints(cld, indices, dist, nn);*/


	int n_points = cld.size();
	int q_points = cld.size();
	
	const int max_nn = 16;




	thrust::host_vector<float4> data_host(n_points);
	thrust::host_vector<float4> query_host(n_points);
	for (int i = 0; i < q_points; i++)
	{
		data_host[i] = make_float4(cld[i].x, cld[i].y, cld[i].z, 0);
		query_host[i] = make_float4(cld[i].x, cld[i].y, cld[i].z, 0);
	}


	thrust::device_vector<float4> data_device = cldHostThrust;
	thrust::device_vector<float4> query_device = cldHostThrust;

	dists = flann::Matrix<float>(new float[n_points*max_nn], n_points, max_nn);
	indices = flann::Matrix<int>(new int[n_points*max_nn], n_points, max_nn);

	flann::Matrix<float> data_device_matrix((float*)thrust::raw_pointer_cast(&data_device[0]), n_points, 3, 4 * 4);
	flann::Matrix<float> query_device_matrix((float*)thrust::raw_pointer_cast(&query_device[0]), n_points, 3, 4 * 4);

	flann::KDTreeCuda3dIndexParams index_params;
	index_params["input_is_gpu_float4"] = true;
	flann::KDTreeCuda3dIndex< flann::L2<float> > index(data_device_matrix, index_params);
	index.buildIndex();

	thrust::host_vector<int> indices_temp(n_points * nn);
	thrust::host_vector<float> dists_temp(n_points * nn);

	thrust::device_vector<int> indices_device = indices_temp;
	thrust::device_vector<float> dists_device = dists_temp;
	flann::Matrix<int> indices_device_matrix((int*)thrust::raw_pointer_cast(&indices_device[0]), n_points, nn);
	flann::Matrix<float> dists_device_matrix((float*)thrust::raw_pointer_cast(&dists_device[0]), n_points, nn);

	flann::SearchParams sp;
	sp.matrices_in_gpu_ram = true;
	int resul = index.knnSearch(query_device_matrix, indices_device_matrix, dists_device_matrix, nn, sp);

	flann::Matrix<int> indices_host(new int[n_points * nn], n_points, nn);
	flann::Matrix<float> dists_host(new float[n_points * nn], n_points, nn);

	thrust::copy(dists_device.begin(), dists_device.end(), dists_host.ptr());
	thrust::copy(indices_device.begin(), indices_device.end(), indices_host.ptr());

	//std::string delim = ",";
	for (size_t i = 0; i < indices_host.rows; ++i) {
		//outFile << i << delim << query_host[i].x << delim << query_host[i].y << delim << query_host[i].z << delim;
		for (size_t j = 0; j < nn; ++j) {

			//double actualDistGPU = eucDist(query_host[i], data_host[indices_host[i][j]]);
			//double actualDistCPU = eucDist(query_host[i], data_host[indices[i][j]]);
			//printf("%d \t \t%f \t%f \t%f \t%f\n", i, dists_host[i][j], dists[i][j], actualDistGPU, actualDistCPU);

			std::cout << sqrtf(dists_host[i][j]) << delim;

			//printf("%d \t \t%d \t%d \n", i, indices_host[i][j], indices[i][j]);



		}
		std::cout << std::endl;
	}

	//for (size_t i = 0; i < indices.rows; i++)
	//{
	//	std::cout << cld.at(i) << std::endl;

	//	printf("Query Pt %d:\t",i);
	//	for (size_t j = 0; j < indices.cols; j++)
	//	{
	//		printf("%d\t%f\t", indices[i][j]+1, dist[i][j]);
	//	}
	//	printf("\n");

	//	for (size_t j = 0; j < indices.cols; j++)
	//	{
	//		std::cout << cld.at(indices[i][j]) << std::endl;
	//	}
	//}

	return 0;
}