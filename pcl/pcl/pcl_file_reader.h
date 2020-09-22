#pragma once
#include <fstream>
#include <string>
#include "pcl_base.h"
#include "stl_reader.h"

namespace pcl {
	namespace io {
		class FileReader
		{
		public:
			FileReader();

			/*
			Reads the file coded in ASCII format
			[in] cloudFile - File name of the cloud data encoded in ASCII
			[out] pointCloud - pcl::PointCloud<T> type to hold the point cloud. <T> - <float> or <double>
			[in] delim - delimiter. DEFAULT = ","
			*/
			template<typename T>
			int readASCIIFile(const std::string cloudFile, pcl::PointCloud<T>& pointCloud, const char* delim = ",");

			/*
			Reads the STL files
			[in] stlFile - Path to the STL file
			[out] mesh - mesh file of type pcl::PolygonMesh 
			*/
			template<typename T>
			bool readSTLFile(const std::string stlFile, pcl::PolygonMesh<T>& mesh);
			

		private:
			std::ifstream inFileStream;

		};
	}
}


