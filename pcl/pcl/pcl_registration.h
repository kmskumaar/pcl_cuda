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

		pcl::TMatrix<T> pointToPointSVD(pcl::PointCloud<T>& cloud_Y, pcl::PointCloud<T>& cloud_X, pcl::Indices& indices);


		/*
		Finds the root mean square deviation between the point cloud to the surface defined by the mesh file.
		[in] inCloud -  pointer to the point cloud
		[in] inMesh - pointer to the polygon mesh
		[out] outProjPtCld - cloud of projected points
		[return] - deviation as root mean squared error
		*/
		double cloudToSurfaceRMSD(pcl::PointCloud<T>& inCloud, pcl::PolygonMesh<T>& inMesh, pcl::PointCloud<T>& outProjPtCld);

		/*
		Checks if the point lies within the surface. If it lies calculate the distance of the point to the surface
		[in] inPt - input point
		[in] inMesh - pointer to the polygon mesh
		[out] distance - distance between the point and surface if the point lies on the surface. Otherwise zero
		[out] projPt - projected points
		[return] true if the point lies o the surface and a valid distance is calculated. false otherwise
		*/
		bool isPointLyingOnSurface(pcl::PointXYZ<T>& inPt, pcl::PolygonMesh<T>& inMesh, T& distance, pcl::PointXYZ<T>& projPt);
	};
}

