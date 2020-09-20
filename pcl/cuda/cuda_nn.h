#pragma once
#define FLANN_USE_CUDA

#include "cuda_common.h"
#include "flann/flann.h"
#include <thrust/device_vector.h>


namespace pclcuda {
	namespace nn {
		template <class T_CPU>
		class KDTreeCUDA
		{
		public:
			/*
			Pointer to the CUDA KD Index
			*/
			flann::KDTreeCuda3dIndex< flann::L2<T_CPU> > *kdIndex;

			/*
			Set point cloud for KD Tree
			[in] inCloud - pointer for the point cloud
			*/
			void setInputCloud(pcl::PointCloud<T_CPU>& inCloudDevice);

			/*
			Search for the nearest neighbor for "N" Points
			[in] queryVec - Pointer to pcl::PointCloud<T_CPU> point cloud to query
			[out] indices - Pointer to flann::Matrix for the indices of the nearest point
			[out] dist - Pointer to flann::Matrix for the squared distances to the nearest point
			[in] neighbors - Number of neighbors to search
			[return] - Number of nearest neighbor results. It should be equal to neighbors*querypoints
			*/
			int knnSearchNPoints(pcl::PointCloud<T_CPU>& inQueryHost, flann::Matrix<int>& indices, flann::Matrix<T_CPU>& sqrDist, int neighbors);

		private:
			/*
			Search for the nearest neighbor for "N" Points
			[in] query - Pointer to flann::Matrix<T_CPU> points to query
			[out] indices - Pointer to flann::Matrix for the indices of the nearest point
			[out] dist - Pointer to flann::Matrix for the squared distances to the nearest point
			[in] neighbors - Number of neighbors to search
			[return] - Number of nearest neighbor results. It should be equal to neighbors*querypoints
			*/
			int knnSearch(flann::Matrix<T_CPU>& queryDeviceMatrix, flann::Matrix<int>& indicesDeviceMatrix, flann::Matrix<T_CPU>& distDeviceMatrix, int neighbors);
		};
	}
	
}


