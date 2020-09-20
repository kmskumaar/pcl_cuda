#pragma once
#include <fstream>
#include <string>
#include "pcl_base.h"

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

		private:
			std::ifstream inFileStream;

		};
	}
}


