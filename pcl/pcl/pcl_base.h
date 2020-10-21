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

	/*
	Datatype for holding vectors. Allowable templates: <float> and <double>
	*/
	template<typename T>
	struct DVector {
		T i;
		T j;
		T k;

		//DVector(T i_, T j_, T k_) :i(i_), j(j_), k(k_) {}

		DVector operator + (DVector const &vec2) {
			DVector newVec;
			newVec.i = i + vec2.i;
			newVec.j = j + vec2.j;
			newVec.k = k + vec2.k;
			return newVec;
		}

		DVector operator - (DVector const &vec2) {
			DVector newVec;
			newVec.i = i - vec2.i;
			newVec.j = j - vec2.j;
			newVec.k = k - vec2.k;
			return newVec;
		}

		DVector operator * (T const &scalar) {
			DVector newVec;
			newVec.i = i * scalar;
			newVec.j = j * scalar;
			newVec.k = k * scalar;
			return newVec;
		}

		DVector operator / (T const &scalar) {
			DVector newVec;
			newVec.i = i / scalar;
			newVec.j = j / scalar;
			newVec.k = k / scalar;
			return newVec;
		}

		T norm2() {		
			return (sqrt(i * i + j * j + k * k));
		}

		DVector normalize() {
			return ((*this) / this->norm2());
		}

		DVector cross(DVector& vec2) {
			DVector result;
			result.i = (j * vec2.k) - (k * vec2.j);
			result.j = (k * vec2.i) - (i * vec2.k);
			result.k = (i * vec2.j) - (j * vec2.i);

			return result;
		}
	};

	template<typename T>
	std::ostream& operator << (std::ostream& os, const PointXYZ<T>& pt) {
		os << "X: " << pt.x << "\tY: " << pt.y << "\tZ: " << pt.z;
		return os;
	}

	template<typename T>
	std::ostream& operator << (std::ostream& os, const DVector<T>& vec) {
		os << "I: " << vec.i << "\tJ: " << vec.j << "\tK: " << vec.k;
		return os;
	}

	/*
	Data Types for storing Point Clouds. Allowable templates: <float> and <double>
	*/
	template<typename T>
	using PointCloud = std::vector<PointXYZ<T>>;


	template<typename T>
	using Normal = DVector<T>;

	template<typename T>
	using NormalCloud = std::vector<DVector<T>>;

	struct Indices {
		std::vector<int> indices;
	};

	typedef std::vector<Indices> Clusters;

	typedef std::vector<int> Vertices;


	/*
	Polygon Mesh Data Structure
	*/
	template<typename T>
	struct PolygonMesh {
		pcl::PointCloud<T> cloud;

		std::vector<Vertices> polygon;

		std::vector<pcl::Normal<T>> normals;

	};

	//namespace cuda {
	//	template<typename T>
	//	using  PointCloudHost = thrust::host_vector<T>;

	//}
	
}