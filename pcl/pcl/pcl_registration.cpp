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

template pcl::TMatrix<float> pcl::Registration<float>::pointToPointSVD(pcl::PointCloud<float>& cloud_Y, pcl::PointCloud<float>& cloud_X);
template pcl::TMatrix<double> pcl::Registration<double>::pointToPointSVD(pcl::PointCloud<double>& cloud_Y, pcl::PointCloud<double>& cloud_X);