#pragma once
#include "pcl_common.h"
#include "pcl_nn.h"

namespace pcl {
	template <class T>
	class Registration {
	public:
		/*
		Finds the best fit between two point clouds with known correspondence. Corresponding points should have same index
		Y = R * X + t
		Source: Least-squares estimation of transformation parameters between two point patterns. Author: Umeyama,Shinji
		[in] cloud_Y - input point cloud. It is the nominal point cloud that is fixed
		[in] cloud_X - input point cloud. It is the measured point cloud that is transformed
		[return] - 4x4 Transformation matrix
		*/
		pcl::TMatrix<T> pointToPointSVD(pcl::PointCloud<T>& cloud_Y, pcl::PointCloud<T>& cloud_X);

	};
}

