#include "pcl_file_reader.h"

pcl::io::FileReader::FileReader() {

}
template<typename T>
int pcl::io::FileReader::readASCIIFile(const std::string cloudFile, pcl::PointCloud<T>& pointCloud, const char* delim /*= ","*/) {
	try {
		this->inFileStream.open(cloudFile);

		std::string line;
		int count = 0;
		pcl::PointXYZ<T> tempPt;

		pointCloud.clear();

		while (std::getline(inFileStream, line))
		{

			if (line.find(delim) != std::string::npos) {
				size_t pos = line.find(delim);
				tempPt.x = std::stof(line.substr(0, pos));
				line.erase(0, pos + 1);
				pos = line.find(delim);
				tempPt.y = std::stof(line.substr(0, pos));
				line.erase(0, pos + 1);
				tempPt.z = std::stof(line.substr(0, std::string::npos));
			}
			pointCloud.push_back(tempPt);
			count++;
		}
		this->inFileStream.close();
		return count;
	}
	
	catch (std::exception& e) {
		printf("Exception Thrown: %s\n", e.what());
		this->inFileStream.close();
		return -1;
	}
}

template int pcl::io::FileReader::readASCIIFile(const std::string cloudFile, PointCloud<float>& pointCloud, const char* delim /*= ","*/);
template int pcl::io::FileReader::readASCIIFile(const std::string cloudFile, PointCloud<double>& pointCloud, const char* delim /*= ","*/);