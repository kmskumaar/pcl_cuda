#include "cuda_common.h"

template<>
bool pclcuda::copyPtCldToHostMem(pcl::PointCloud<float>& inCloud, pclcuda::PointCloudHost<float4>& outCloudHost) {
	size_t noOfPts = inCloud.size();
	outCloudHost.resize(noOfPts);

	for (size_t i = 0; i < noOfPts; i++)
		outCloudHost[i] = make_float4(inCloud[i].x, inCloud[i].y, inCloud[i].z, 1.0);

	return true;
}

template<>
bool pclcuda::copyPtCldToHostMem(pcl::PointCloud<double>& inCloud, pclcuda::PointCloudHost<double4>& outCloudHost) {
	size_t noOfPts = inCloud.size();
	outCloudHost.resize(noOfPts);

	for (size_t i = 0; i < noOfPts; i++)
		outCloudHost[i] = make_double4(inCloud[i].x, inCloud[i].y, inCloud[i].z, 1.0);

	return true;
}

template <>
bool pclcuda::copyMeshToHostMem(pcl::PolygonMesh<float>& inMesh, pclcuda::PointCloudHost<float4>& vertices_1,
	pclcuda::PointCloudHost<float4>& vertices_2, pclcuda::PointCloudHost<float4>& vertices_3, pclcuda::PlaneParamHost<float4>& planeParam) {

	size_t noOfPolygons = inMesh.polygon.size();
	vertices_1.resize(noOfPolygons);
	vertices_2.resize(noOfPolygons);
	vertices_3.resize(noOfPolygons);
	planeParam.resize(noOfPolygons);
	for (size_t idx = 0; idx < noOfPolygons; idx++)
	{
		planeParam[idx] = make_float4(inMesh.planes[idx].normal.i, inMesh.planes[idx].normal.j, inMesh.planes[idx].normal.k,
			inMesh.planes[idx].D);
		vertices_1[idx] = make_float4(inMesh.cloud[inMesh.polygon[idx].indices[0]].x, inMesh.cloud[inMesh.polygon[idx].indices[0]].y,
			inMesh.cloud[inMesh.polygon[idx].indices[0]].z, 1.0);
		vertices_2[idx] = make_float4(inMesh.cloud[inMesh.polygon[idx].indices[1]].x, inMesh.cloud[inMesh.polygon[idx].indices[1]].y,
			inMesh.cloud[inMesh.polygon[idx].indices[1]].z, 1.0);
		vertices_3[idx] = make_float4(inMesh.cloud[inMesh.polygon[idx].indices[2]].x, inMesh.cloud[inMesh.polygon[idx].indices[2]].y,
			inMesh.cloud[inMesh.polygon[idx].indices[2]].z, 1.0);
	}

	return true;
}

template <>
bool pclcuda::copyMeshToHostMem(pcl::PolygonMesh<double>& inMesh, pclcuda::PointCloudHost<double4>& vertices_1,
	pclcuda::PointCloudHost<double4>& vertices_2, pclcuda::PointCloudHost<double4>& vertices_3, pclcuda::PlaneParamHost<double4>& planeParam) {

	size_t noOfPolygons = inMesh.polygon.size();
	vertices_1.resize(noOfPolygons);
	vertices_2.resize(noOfPolygons);
	vertices_3.resize(noOfPolygons);
	planeParam.resize(noOfPolygons);
	for (size_t idx = 0; idx < noOfPolygons; idx++)
	{
		planeParam[idx] = make_double4(inMesh.planes[idx].normal.i, inMesh.planes[idx].normal.j, inMesh.planes[idx].normal.k,
			inMesh.planes[idx].D);
		vertices_1[idx] = make_double4(inMesh.cloud[inMesh.polygon[idx].indices[0]].x, inMesh.cloud[inMesh.polygon[idx].indices[0]].y,
			inMesh.cloud[inMesh.polygon[idx].indices[0]].z, 1.0);
		vertices_2[idx] = make_double4(inMesh.cloud[inMesh.polygon[idx].indices[1]].x, inMesh.cloud[inMesh.polygon[idx].indices[1]].y,
			inMesh.cloud[inMesh.polygon[idx].indices[1]].z, 1.0);
		vertices_3[idx] = make_double4(inMesh.cloud[inMesh.polygon[idx].indices[2]].x, inMesh.cloud[inMesh.polygon[idx].indices[2]].y,
			inMesh.cloud[inMesh.polygon[idx].indices[2]].z, 1.0);
	}

	return true;
}