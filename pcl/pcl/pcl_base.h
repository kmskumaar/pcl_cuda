#pragma once

#define _USE_MATH_DEFINES

#include <Eigen/Eigen>
#include <vector>
#include <iostream>
#include <math.h>

//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>


namespace pcl {

	template<typename T>
	using TMatrix = Eigen::Transform<T, 3, Eigen::Affine>;

	template<typename T>
	using EigenPoint = Eigen::Matrix<T, 4, 1>;

	template<typename T>
	using EigenCloud = Eigen::Matrix<T, Eigen::Dynamic, 3>;

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

		EigenPoint<T> convert2Eigen() {
			EigenPoint<T> eigenPt;
			eigenPt[0] = this->x;
			eigenPt[1] = this->y;
			eigenPt[2] = this->z;
			eigenPt[3] = 1.0;

			return eigenPt;
		}

		PointXYZ transform(TMatrix<T> t) {
			EigenPoint<T> transformedPt = t* this->convert2Eigen();
			return PointXYZ{ transformedPt[0],transformedPt[1],transformedPt[2] };
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

		T dot(DVector& vec2) {
			T result;
			result = i * vec2.i + j * vec2.j + k * vec2.k;
			return result;
		}
	};

	template<typename T>
	struct Plane {
		T A;
		T B;
		T C;
		T D;
	};

	template<typename T>
	struct TVector {
		T tx; T ty; T tz;
		T rx; T ry; T rz;

		TMatrix<T> getTMatrix() {
			T COS_X = cos(this->rx);	T SIN_X = sin(this->rx);
			T COS_Y = cos(this->ry);	T SIN_Y = sin(this->ry);
			T COS_Z = cos(this->rz);	T SIN_Z = sin(this->rz);

			TMatrix<T> t;
			t(0, 0) = COS_Y * COS_Z;							t(0, 1) = -COS_Y * SIN_Z;							t(0, 2) = SIN_Y;				t(0, 3) = this->tx;
			t(1, 0) = COS_X * SIN_Z + SIN_X * SIN_Y*COS_Z;		t(1, 1) = COS_X * COS_Z - SIN_X * SIN_Y*SIN_Z;		t(1, 2) = -SIN_X * COS_Y;		t(1, 3) = this->ty;
			t(2, 0) = SIN_X * SIN_Z - COS_X * SIN_Y*COS_Z;		t(2, 1) = COS_X * SIN_Y*SIN_Z + SIN_X * COS_Z;		t(2, 2) = COS_X * COS_Y;		t(2, 3) = this->tz;
			t(3, 0) = 0.0;										t(3, 1) = 0.0;										t(3, 2) = 0.0;					t(3, 3) = 1.0;

			return t;
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

	template<typename T>
	std::ostream& operator << (std::ostream& os, const Plane<T>& planeParam) {
		os << "A: " << planeParam.A << "\tB: " << planeParam.B << "\tC: " << planeParam.C << "\tD: " << planeParam.D;
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