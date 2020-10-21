
#define FLANN_USE_CUDA

#include "../pcl/pcl_file_reader.h"
#include "../pcl/pcl_nn.h"
#include "../pcl/pcl_common.h"
#include "../pcl/pcl_preprocess.h"

#include "../cuda/cuda_base.h"
#include "../cuda/cuda_common.h"
#include "../cuda/cuda_nn.h"
#include "../cuda/cuda_preprocess.h"

#include <chrono>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define NN_TEST		0
#define STL_TEST	0
#define CUDA_W_TEST 0
#define EUCL_CLUSTER	1

int main() {

	pcl::io::FileReader fileReader;

	if (NN_TEST) {		
		pcl::PointCloud<float> cld;
		pcl::PointCloud<float> query;
		cld.clear();

		const std::string filePath = "C:/Users/R/OneDrive - Fraunhofer/private/testFiles/HR_downsized.txt";
		std::ofstream outFile;
		outFile.open("C:/Users/R/OneDrive - Fraunhofer/private/MATLAB_ws/outputCUDA_library.txt");

		fileReader.readASCIIFile(filePath, cld, ",");
		query.resize(100000);
		for (size_t i = 0; i < 100000; i++)
		{
			query.at(i).x = cld.at(i).x;
			query.at(i).y = cld.at(i).y;
			query.at(i).z = cld.at(i).z;
		}

		pclcuda::PointCloudHost<float4> cldHostThrust;
		pclcuda::PointCloudHost<float4> queryHostThrust;
		pclcuda::copyToHostMem<float, float4>(cld, cldHostThrust);
		pclcuda::copyToHostMem<float, float4>(cld, queryHostThrust);

		flann::Matrix<float> dists;
		flann::Matrix<int> indices;
		const int nn = 20;
		const int max_nn = 90;
		float rad = 3;

		//pcl::nn::KDTreeCPU<float> kdTreeCPU;
		//kdTreeCPU.setInputCloud(cld);
		//kdTreeCPU.buildIndex();
		//kdTreeCPU.knnSearchNPoints(query, indices, dists, rad, max_nn);

		pclcuda::nn::KDTreeCUDA<float> kdTreeGPU;
		kdTreeGPU.setInputCloud(cld);
		kdTreeGPU.knnSearchNPoints(cld, indices, dists, rad, max_nn);

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
				std::cout << (indices[i][j]) << delim;
			}
			std::cout << std::endl;
		}
	}
	
	if (STL_TEST) {
		pcl::PolygonMesh<float> mesh;
		fileReader.readSTLFile("C:/Users/R/OneDrive - Fraunhofer/private/testFiles/stltest.stl", mesh);		
	}

	if (CUDA_W_TEST) {
		pclcuda::CudaBase* iCudaBase;
		pclcuda::CudaBase* iCudaBase2;
		iCudaBase = new pclcuda::CudaBase();
		iCudaBase2 = new pclcuda::CudaBase();

		thrust::host_vector<float> h_test;
		thrust::device_vector<float> d_test;

		int* hArray = 0;
		int* dArray = 0;

	/*	for (size_t i = 0; i < 100; i++)
		{
			hArray[i] = i;
		}
*/
		dArray = iCudaBase->copyHost2DevMem<int>(hArray, 100);

		int i = 0;

	}

	if (EUCL_CLUSTER) {
		pcl::PointCloud<float> cld;
		cld.clear();

		const std::string filePath = "C:/Users/R/OneDrive - Fraunhofer/private/testFiles/HR_downsized.txt";
		fileReader.readASCIIFile(filePath, cld, ",");

		printf("Input Point Cloud Size: %d\n", cld.size());

		/*pcl::PreProcess<float> ipreProcessCPU;
		pcl::Clusters clusters = ipreProcessCPU.euclideanClustering(cld, 3.0);*/

		cuda::PreProcess<float> ipreProcessGPU;
		pcl::Clusters clusters = ipreProcessGPU.euclideanClustering(cld, 2.0, 200);

		pcl::io::FileWriter fileWriter;
		printf("A total of %d clusters found\n", clusters.size());
		for (size_t i = 0; i < clusters.size(); i++)
		{
			std::string filePath = "C:/Users/R/OneDrive - Fraunhofer/private/testFiles/cluster/cluster_" + std::to_string(i) + ".txt";
			fileWriter.writeASCIIFile(filePath, cld, clusters[i]);
		}
	}

	return 0;
}