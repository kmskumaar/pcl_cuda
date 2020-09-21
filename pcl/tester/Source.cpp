
#define FLANN_USE_CUDA

#include "../pcl/pcl_file_reader.h"
#include "../pcl/pcl_nn.h"
#include "../cuda/cuda_common.h"
#include "../cuda/cuda_nn.h"
#include "../pcl/pcl_common.h"
#include <chrono>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


int main() {
	pcl::io::FileReader fileReader;
	pcl::PointCloud<float> cld;
	pcl::PointCloud<float> query;
	cld.clear();
	//pcl::PointXYZ<float> p1{ 1.0,2.0,3.0 };
	//pcl::PointXYZ<float> p2{ 11.0,22.0,33.0 };
	//cld.push_back(p1);
	//cld.push_back(p2);
	//cld.push_back(p1*5);

	const std::string filePath = "C:/Users/R/OneDrive - Fraunhofer/private/testFiles/iotest.txt";
	std::ofstream outFile;
	outFile.open("C:/Users/R/OneDrive - Fraunhofer/private/MATLAB_ws/outputCUDA_library.txt");

	fileReader.readASCIIFile(filePath, cld, ",");
	query.resize(10);
	for (size_t i = 0; i < 10; i++)
	{
		query.at(i).x = cld.at(i).x;
		query.at(i).y = cld.at(i).y;
		query.at(i).z = cld.at(i).z;
	}

	pclcuda::PointCloudHost<float4> cldHostThrust;
	pclcuda::PointCloudHost<float4> queryHostThrust;
	pclcuda::copyToHostMem<float, float4>(cld, cldHostThrust);
	pclcuda::copyToHostMem<float, float4>(cld, queryHostThrust);


	//flann::Matrix<int> indices;// (new int[noQueryPts * nn], noQueryPts, nn);
	//flann::Matrix<float> dist;// (new float[noQueryPts * nn], noQueryPts, nn);
	//int nn = 10;
	flann::Matrix<float> dists;
	flann::Matrix<int> indices;
	const int nn = 20;
	const int max_nn = 20;
	float rad = 1000;

	//pcl::nn::KDTreeCPU<float> kdTreeCPU;
	//kdTreeCPU.setInputCloud(cld);
	//kdTreeCPU.buildIndex();
	//kdTreeCPU.knnSearch1Point(cld.at(0), indices, dists, rad, max_nn);

	pclcuda::nn::KDTreeCUDA<float> kdTreeGPU;
	kdTreeGPU.setInputCloud(cld);
	kdTreeGPU.knnSearchNPoints(query, indices, dists, rad, max_nn);
	
	//pclcuda::nn::KDTreeCUDA<float> kdTreeGPU;
	//auto t1 = std::chrono::steady_clock::now();
	//kdTreeGPU.setInputCloud(cld);
	////kdTreeGPU.buildIndex();
	//kdTreeGPU.knnSearchNPoints(cld, indices, dists, nn);
	//auto t2 = std::chrono::steady_clock::now();
	//std::cout << "Operation Time : "
	//	<< std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	//	<< " ms" << std::endl;

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


	

	return 0;
}