#pragma once
#include <vector>
#include <iostream>
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>


namespace pcl {

	/*
	Data types for storing 3D points. Allowable templates: <float> and <double>
	*/
	template<typename T>
	struct PointXYZ {
		T x;
		T y;
		T z;

		PointXYZ operator + (PointXYZ const &pt2) {
			PointXYZ newPt;
			newPt.x = x + pt2.x;
			newPt.y = y + pt2.y;
			newPt.z = z + pt2.z;
			return newPt;
		}

		PointXYZ operator - (PointXYZ const &pt2) {
			PointXYZ newPt;
			newPt.x = x - pt2.x;
			newPt.y = y - pt2.y;
			newPt.z = z - pt2.z;
			return newPt;
		}

		PointXYZ operator * (T const &scalar) {
			PointXYZ newPt;
			newPt.x = x * scalar;
			newPt.y = y * scalar;
			newPt.z = z * scalar;
			return newPt;
		}

		PointXYZ operator / (T const &scalar) {
			PointXYZ newPt;
			newPt.x = x / scalar;
			newPt.y = y / scalar;
			newPt.z = z / scalar;
			return newPt;
		}
	};

	template<typename T>
	std::ostream& operator << (std::ostream& os, const PointXYZ<T>& pt) {
		os << "X: " << pt.x << "\tY: " << pt.y << "\tZ: " << pt.z;
		return os;
	}

	/*
	Data Types for storing Point Clouds. Allowable templates: <float> and <double>
	*/
	template<typename T>
	using PointCloud = std::vector<PointXYZ<T>>;

	typedef std::vector<int> Indices;

	typedef std::vector<int> Vertices;


	/*
	Polygon Mesh Data Structure
	*/
	template<typename T>
	struct PolygonMesh {
		pcl::PointCloud<T> cloud;

		std::vector<Vertices> polygon;

	};

	//namespace cuda {
	//	template<typename T>
	//	using  PointCloudHost = thrust::host_vector<T>;

	//}
	
}