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

template<typename T>
bool pcl::io::FileReader::readSTLFile(const std::string stlFile, pcl::PolygonMesh<T>& mesh) {

	try {
		stl_reader::StlMesh <T, unsigned int> istlReader(stlFile);

		size_t noOfVertices = istlReader.num_vrts();
		size_t noOfPolygons = istlReader.num_tris();

		mesh.cloud.resize(noOfVertices);
		mesh.polygon.clear();
		mesh.polygon.resize(noOfPolygons);
		mesh.normals.clear();
		mesh.normals.resize(noOfPolygons);

		pcl::Vertices vertices;
		pcl::Normal<T> normal;

		const T* cloudArray = istlReader.raw_coords();
		const unsigned int* verticesArray = istlReader.raw_tris();
		const T* normalArray = istlReader.raw_normals();

		for (size_t i = 0; i < noOfVertices; i++)
		{
			// Copying the vertice points
			mesh.cloud.at(i).x = cloudArray[(i * 3) + 0];
			mesh.cloud.at(i).y = cloudArray[(i * 3) + 1];
			mesh.cloud.at(i).z = cloudArray[(i * 3) + 2];
		}	

		for (size_t i = 0; i < noOfPolygons; i++)
		{
			// Copying the index for the vertices of the polygons
			vertices.resize(3);
			vertices.at(0) = verticesArray[(i * 3) + 0];
			vertices.at(1) = verticesArray[(i * 3) + 1];
			vertices.at(2) = verticesArray[(i * 3) + 2];
			mesh.polygon.at(i) = vertices;

			// Copying the normalized normal vector of each polygons
			normal.i = normalArray[(i * 3) + 0];
			normal.j = normalArray[(i * 3) + 1];
			normal.k = normalArray[(i * 3) + 2];
			mesh.normals.at(i) = normal.normalize();
		}	
	}
	
	catch (std::exception &e) {
		std::cout << "STL Read Error: " << e.what() << "\t";
		std::cout << "Unable to read the STL file: " << stlFile << std::endl;
		return false;
	}
	
	return true;
}

template int pcl::io::FileReader::readASCIIFile(const std::string cloudFile, PointCloud<float>& pointCloud, const char* delim /*= ","*/);
template bool pcl::io::FileReader::readSTLFile(const std::string stlFile, pcl::PolygonMesh<float>& mesh);

template int pcl::io::FileReader::readASCIIFile(const std::string cloudFile, PointCloud<double>& pointCloud, const char* delim /*= ","*/);
template bool pcl::io::FileReader::readSTLFile(const std::string stlFile, pcl::PolygonMesh<double>& mesh);