#pragma once
#include <fstream>
#include <string>
#include <iostream>
#include <iomanip>

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

		class FileWriter
		{
		public:

			/*
			Writes the point cloud data to an ASCII file
			[in] filename - File path to the ASCII file
			[in] cld - pointer to the point cloud
			[in] indices - indices to be written from the point cloud
			[in] precision - number of digits after the decimel point. DEFAULT: 6
			[in] delim - Delimiter between the x,y,z values. DEFAULT: ","
			[in] append - If the data has to be appended to the exiting file. DEFAULT false
			[return] - true when the operation succeeds, false otherwise
			*/
			template<typename T>
			bool writeASCIIFile(const std::string filePath, pcl::PointCloud<T>& cld, pcl::Indices& indices,
				const int precision = 6, const char* delim = ",", const bool append = false);


		private:
			std::ofstream outFileStream;
		};
	}
}


