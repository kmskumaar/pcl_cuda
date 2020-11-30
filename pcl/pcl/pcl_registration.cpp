#include "pcl_registration.h"

template <typename T>
pcl::TMatrix<T> pcl::Registration<T>::pointToPointSVD(pcl::PointCloud<T>& cloud_Y, pcl::PointCloud<T>& cloud_X) {

	pcl::TMatrix<T> outTMatrix;
	if (cloud_X.size() != cloud_Y.size()) {
		std::cout << "Point Cloud Size does not match. Aborting registeration operation" << std::endl;
		return outTMatrix;
	}
	
	int n_points = cloud_Y.size();

	pcl::PointXYZ<T> centroid_Y;
	pcl::PointXYZ<T> centroid_X;
	pcl::PointCloud<T> demeanCld_Y;
	pcl::PointCloud<T> demeanCld_X;
	demeanCld_Y = pcl::demeanCloud(cloud_Y, centroid_Y);
	demeanCld_X = pcl::demeanCloud(cloud_X, centroid_X);

	pcl::EigenCloud<T> eigenMatrix_Y = pcl::cloudToEigenMatrix(demeanCld_Y);
	pcl::EigenCloud<T> eigenMatrix_X = pcl::cloudToEigenMatrix(demeanCld_X);
	
	Eigen::Matrix<T, 3, 3> sigma = (1.0 / n_points)*eigenMatrix_Y.transpose()*eigenMatrix_X;

	Eigen::JacobiSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd(sigma, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::Matrix<T, 3, 3> U_mat = svd.matrixU();
	Eigen::Matrix<T, 3, 3> V_mat = svd.matrixV();

	T det_Sigma = sigma.determinant();
	T det_U = U_mat.determinant();
	T det_V = V_mat.determinant();

	Eigen::FullPivLU<Eigen::Matrix<T, 3, 3>> lu_decomp(sigma);
	int rank_sigma = lu_decomp.rank();

	Eigen::Matrix<T, 3, 3> S_Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Identity(3, 3);

	if ((det_Sigma < 0.0) || ((rank_sigma == 2) && (det_U*det_V < 0.0)))
		S_Mat(2, 2) = -1;

	Eigen::Matrix<T, 3, 3> rotMatrix = U_mat * S_Mat*V_mat.transpose();
	Eigen::Matrix<T, 3, 1> translation = centroid_Y.convert2Eigen().block<3, 1>(0, 0) - rotMatrix * centroid_X.convert2Eigen().block<3, 1>(0, 0);

	outTMatrix.matrix().block<3, 3>(0, 0) = rotMatrix;
	outTMatrix.matrix().block<3, 1>(0, 3) = translation;

	return outTMatrix;
}

template <typename T>
double pcl::Registration<T>::cloudToSurfaceRMSD(pcl::PointCloud<T>& inCloud, pcl::PolygonMesh<T>& inMesh, pcl::PointCloud<T>& outProjPtCld) {
	double sumSqrError = 0.0;
	int count = 0;
	outProjPtCld.resize(inCloud.size());
	for (int idx = 0; idx < inCloud.size(); idx++)
	{
		T distance = 0.0;
		if (this->isPointLyingOnSurface(inCloud[idx], inMesh, distance, outProjPtCld[idx])) {
			count++;
			sumSqrError = sumSqrError + pow(distance, 2);
		}
	}
	return sqrt(sumSqrError / static_cast<double>(count));
}

template<typename T>
pcl::TMatrix<T> pcl::Registration<T>::pointToPointSVD(pcl::PointCloud<T>& cloud_Y, pcl::PointCloud<T>& cloud_X, pcl::Indices& indices) {
	pcl::PointCloud<T> Y_reduced(indices.indices.size());
	pcl::PointCloud<T> X_reduced(indices.indices.size());
	
	for (size_t idx = 0; idx < indices.indices.size(); idx++)
	{
		Y_reduced[idx] = cloud_Y[indices.indices[idx]];
		X_reduced[idx] = cloud_X[indices.indices[idx]];
	}

	return this->pointToPointSVD(Y_reduced, X_reduced);
}

template <typename T>
bool pcl::Registration<T>::isPointLyingOnSurface(pcl::PointXYZ<T>& inPt, pcl::PolygonMesh<T>& inMesh, T& distance, pcl::PointXYZ<T>& projPt) {
	distance = 0.0;
	projPt = { 0.0,0.0,0.0 };
	bool flgFound = false;
	for (int idx = 0; idx < inMesh.polygon.size(); idx++) {
		pcl::Plane<T> plane = inMesh.planes[idx];
		pcl::PointXYZ<T> tempPt = inPt.projectToPlane(inMesh.planes[idx]);

		pcl::DVector<T> vec1 = reinterpret_cast<pcl::DVector<T>&>(inMesh.cloud[inMesh.polygon[idx].indices[0]] - tempPt);
		pcl::DVector<T> vec2 = reinterpret_cast<pcl::DVector<T>&>(inMesh.cloud[inMesh.polygon[idx].indices[1]] - tempPt);
		pcl::DVector<T> vec3 = reinterpret_cast<pcl::DVector<T>&>(inMesh.cloud[inMesh.polygon[idx].indices[2]] - tempPt);

		T totalAngle = (vec1.angle(vec2) + vec2.angle(vec3) + vec3.angle(vec1))*180.0 / M_PI;

		if (round(totalAngle) == 360.0) {
			if (distance == 0.0) {
				distance = inPt.distance2Plane(inMesh.planes[idx]);
				projPt = tempPt;
			}
			if (abs(inPt.distance2Plane(inMesh.planes[idx])) < abs(distance)) {
				distance = inPt.distance2Plane(inMesh.planes[idx]);
				projPt = tempPt;
			}
			flgFound = true;
		}
	}
	return flgFound;
}

template pcl::TMatrix<float> pcl::Registration<float>::pointToPointSVD(pcl::PointCloud<float>& cloud_Y, pcl::PointCloud<float>& cloud_X);
template pcl::TMatrix<double> pcl::Registration<double>::pointToPointSVD(pcl::PointCloud<double>& cloud_Y, pcl::PointCloud<double>& cloud_X);

template pcl::TMatrix<float> pcl::Registration<float>::pointToPointSVD(pcl::PointCloud<float>& cloud_Y, pcl::PointCloud<float>& cloud_X, pcl::Indices& indices);
template pcl::TMatrix<double> pcl::Registration<double>::pointToPointSVD(pcl::PointCloud<double>& cloud_Y, pcl::PointCloud<double>& cloud_X, pcl::Indices& indices);

template double pcl::Registration<float>::cloudToSurfaceRMSD(pcl::PointCloud<float>& inCloud, pcl::PolygonMesh<float>& inMesh, pcl::PointCloud<float>& outProjPtCld);
template double pcl::Registration<double>::cloudToSurfaceRMSD(pcl::PointCloud<double>& inCloud, pcl::PolygonMesh<double>& inMesh, pcl::PointCloud<double>& outProjPtCld);

template bool pcl::Registration<float>::isPointLyingOnSurface(pcl::PointXYZ<float>& inPt, pcl::PolygonMesh<float>& inMesh, float& distance, pcl::PointXYZ<float>& projPt);
template bool pcl::Registration<double>::isPointLyingOnSurface(pcl::PointXYZ<double>& inPt, pcl::PolygonMesh<double>& inMesh, double& distance, pcl::PointXYZ<double>& projPt);