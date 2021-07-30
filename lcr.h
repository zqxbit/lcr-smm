#ifndef LCR_H_
#define LCR_H_

#include <vector>

#include <sophus/se3.hpp>
#include <sophus/types.hpp>
#include <sophus/common.hpp>
#include <pcl/registration/icp.h>
#include <Eigen/Geometry>

namespace lcrsmm {

template <size_t N>
class EmIterativeClosestPoint {
 public:
  typedef pcl::PointXYZL PointT;
  typedef typename pcl::PointCloud<PointT> PointCloud;
  typedef typename PointCloud::Ptr PointCloudPtr;

  typedef std::vector<Eigen::Matrix3d,
                      Eigen::aligned_allocator<Eigen::Matrix3d>>
                      MatricesVector;
  typedef std::vector<Eigen::Matrix<double, 6, 6>,
                      Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>>
                      CovarianceVector;
  typedef std::vector<Eigen::Matrix<double, N, 1>,
                               Eigen::aligned_allocator<Eigen::Matrix<double, N, 1>>>
                               DistVector;
  typedef std::shared_ptr< MatricesVector > MatricesVectorPtr;
  typedef std::shared_ptr< const MatricesVector > MatricesVectorConstPtr;
  typedef std::shared_ptr< DistVector > DistVectorPtr;

  typedef typename pcl::KdTreeFLANN<PointT> KdTree;
  typedef typename KdTree::Ptr KdTreePtr;

  typedef Eigen::Matrix<double, 6, 1> Vector6d;

  EmIterativeClosestPoint(int k = 20,
                          double epsilon = 0.001) :
  kCorrespondences_(k),
  kEpsilon_(epsilon) {
    Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
    base_transformation_ = Sophus::SE3d(mat);
    memset(relabel_i,0,sizeof(relabel_i));
  }

  inline void
  setSourceCloud(const PointCloudPtr &cloud ) {
    source_cloud_ = cloud;
    source_kd_tree_ = KdTreePtr(new KdTree());
    source_kd_tree_->setInputCloud(source_cloud_);
    source_covariances_ = MatricesVectorPtr(new MatricesVector());
    source_distributions_ = DistVectorPtr(new DistVector());
  }

  inline void
  setTargetCloud(const PointCloudPtr &cloud ) {
    target_cloud_ = cloud;
    target_kd_tree_ = KdTreePtr(new KdTree());
    target_kd_tree_->setInputCloud(target_cloud_);
    target_covariances_ = MatricesVectorPtr(new MatricesVector());
    target_distributions_ = DistVectorPtr(new DistVector());
  }

  inline void
  setConfusionMatrix(const Eigen::Matrix<double, N, N> &in) {
    confusion_matrix_ = in;
  }

  void
  align(PointCloudPtr finalCloud);

  void
  align(PointCloudPtr finalCloud, const Sophus::SE3d &initTransform);

  Sophus::SE3d
  getFinalTransFormation() {
    Sophus::SE3d temp = final_transformation_;
    return temp;
  }
  void
  setiniterrormode(double mode = 2)
  {
      init_error_mode = mode;
  }
  int
  getOuterIter() 
  {
      return outer_iter;
  }
  void set_max_distance(double max_distance = 5.0)
  {
    max_match_distance = max_distance;
  }
  void set_correspond_num(int num = 4)
  {
    correspond_num = num;
  }
  void set_probability_mode(int mode = 1)
  {
    probability_mode = mode;
  }
  void set_range(double r1=-0.5, double r2=0.5, double x1=-8, double x2=8, double y1=-8, double y2=8)
  {
    rotation_inf = r1;
    rotation_sup = r2;
    x_inf = x1;
    x_sup = x2;
    y_inf = y1;
    y_sup = y2;
  }
  void errorcompute(double* errorvecptr,  int branch_r = 5, int branch_x = 5, int branch_y = 5);
  protected:
  int kNumClasses_;
  int kCorrespondences_;
  double kEpsilon_;
  double kTranslationEpsilon_;
  double kRotationEpsilon_;
  int kMaxInnerIterations_;
  double rotation_sup;
  double rotation_inf;
  double x_sup;
  double x_inf;
  double y_sup;
  double y_inf;  

  int outer_iter;

  int relabel_i[N];

  double init_error_mode;

  double max_match_distance;
  int correspond_num;
  int probability_mode;
  Sophus::SE3d base_transformation_;
  Sophus::SE3d final_transformation_;

  PointCloudPtr source_cloud_;
  KdTreePtr source_kd_tree_;
  MatricesVectorPtr source_covariances_;
  DistVectorPtr source_distributions_;

  PointCloudPtr target_cloud_;
  KdTreePtr target_kd_tree_;
  MatricesVectorPtr  target_covariances_;
  DistVectorPtr target_distributions_;
  

  Eigen::Matrix<double, N, N> confusion_matrix_;

  void ComputeCovariances(const PointCloudPtr cloudptr,
                          KdTreePtr treeptr,
                          MatricesVectorPtr matvec,
                          DistVectorPtr distvec);
};

}  // namespace lcrsmm

#include "lcr.hpp"

#endif  // LCR_H_
