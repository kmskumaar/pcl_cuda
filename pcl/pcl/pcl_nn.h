#pragma once
#include "flann/flann.h"
#include "pcl_base.h"

namespace pcl {
	namespace nn {
		template <class T_CPU>
		class KDTreeCPU
		{
		public:

			flann::Matrix<T_CPU> cloudMatrix;
			flann::Index<flann::L2<T_CPU>> *kdIndex;

			KDTreeCPU();
			~KDTreeCPU();

			/*
			Set point cloud for KD Tree
			[in] inCloud - pointer for the point cloud
			*/
			void setInputCloud(pcl::PointCloud<T_CPU>& inCloud);

			/*
			Builds the index for the KD Tree
			*/
			void buildIndex();

			/*
			Search for the nearest neighbor for only one point
			[in] queryPoint - Pointer to pcl::PointXYZ<T_CPU> point to query
			[out] indices - Pointer to flann::Matrix for the indices of the nearest point
			[out] dist - Pointer to flann::Matrix for the squared distances to the nearest point
			[in] neighbors - Number of neighbors to search
			*/
			int knnSearch1Point(pcl::PointXYZ<T_CPU>& queryPoint, flann::Matrix<int>& indices, flann::Matrix<T_CPU>& dist, int neighbors);

			/*
			Search for the points in the given radius for only one point
			[in] queryPoint - Pointer to pcl::PointXYZ<T_CPU> point to query
			[out] indices - Pointer to flann::Matrix for the indices of the nearest point
			[out] dist - Pointer to flann::Matrix for the squared distances to the nearest point
			[in] radius - Radius in the units of the points
			*/
			int knnSearch1Point(pcl::PointXYZ<T_CPU>& queryPoint, flann::Matrix<int>& indices, flann::Matrix<T_CPU>& dist, float radius, int max_neighbors);

			/*
			Search for the nearest neighbor for "N" Points
			[in] queryVec - Pointer to pcl::PointCloud<T_CPU> point cloud to query
			[out] indices - Pointer to flann::Matrix for the indices of the nearest point
			[out] dist - Pointer to flann::Matrix for the squared distances to the nearest point
			[in] neighbors - Number of neighbors to search
			*/
			int knnSearchNPoints(pcl::PointCloud<T_CPU>& queryVec, flann::Matrix<int>& indices, flann::Matrix<T_CPU>& dist, int neighbors);

			/*
			Search for the nearest neighbor for "N" Points
			[in] queryVec - Pointer to pcl::PointCloud<T_CPU> point cloud to query
			[out] indices - Pointer to flann::Matrix for the indices of the nearest point
			[out] dist - Pointer to flann::Matrix for the squared distances to the nearest point
			[in] radius - Radius in the units of the points
			*/
			int knnSearchNPoints(pcl::PointCloud<T_CPU>& queryVec, flann::Matrix<int>& indices, flann::Matrix<T_CPU>& dist, float radius, int max_neighbors);

		private:
			/*
			Search for the nearest neighbor for "N" Points
			[in] query - Pointer to flann::Matrix<T_CPU> points to query
			[out] indices - Pointer to flann::Matrix for the indices of the nearest point
			[out] dist - Pointer to flann::Matrix for the squared distances to the nearest point
			[in] neighbors - Number of neighbors to search
			*/
			int knnSearch(flann::Matrix<T_CPU>& query, flann::Matrix<int>& indices, flann::Matrix<T_CPU>& dist, int neighbors);

			/*
			Search for the points in the given radius
			[in] query - Pointer to flann::Matrix<T_CPU> points to query
			[out] indices - Pointer to flann::Matrix for the indices of the nearest point
			[out] dist - Pointer to flann::Matrix for the squared distances to the nearest point
			[in] radius - Radius 
			*/
			int knnSearch(flann::Matrix<T_CPU>& query, flann::Matrix<int>& indices, flann::Matrix<T_CPU>& dist, float radius, int max_neighbors);
		};

	}
}


