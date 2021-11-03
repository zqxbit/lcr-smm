/****************************************************************
LCR-SMM Large Convergence Region Semantic Map Matching Algorithm
Last modified: Nov 3, 2021

The code for calculating cost function are derived from semantic-icp by Steven Parkison.
https://bitbucket.org/saparkison/semantic-icp
****************************************************************/


#ifndef LCR_HPP_
#define LCR_HPP_

#include <iostream>
#include <ceres/ceres.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "cost_function.h"
#include <math.h>
#include <Eigen/StdVector>
#include <ceres/gradient_checker.h>


template <size_t N>
void LCR_SMM<N>::align(PointCloudPtr final_cloud,
                                    const Sophus::SE3d & init_transform) {

  ComputeCovariances(source_cloud_, source_kd_tree_, source_covariances_, source_distributions_);
  ComputeCovariances(target_cloud_, target_kd_tree_, target_covariances_, target_distributions_);
  Sophus::SE3d current_transform = init_transform;
  bool converged = false;
  size_t outter_itter = 0;
  double cov_larger = 1;
  std::vector <Eigen::Vector3d>  transforms_R;
  std::vector <Eigen::Vector3d>  transforms_t;
  std::vector <Eigen::Vector3d>  delta_transform_R;
  std::vector <Eigen::Vector3d>  delta_transform_t;
  enum state
  {
    just_start,
    converging,
    far_from_converge,
    nearly_converged
  };
  state current_state = just_start;
  while(converged!=true) {
    ceres::Problem problem;
    Sophus::SO3d current_transform_R(current_transform.rotationMatrix());
    transforms_R.push_back(current_transform_R.log());
    transforms_t.push_back(current_transform.translation());
    if(outter_itter>0)
    {
        delta_transform_R.push_back(transforms_R[outter_itter]-transforms_R[outter_itter-1]);
        delta_transform_t.push_back(transforms_t[outter_itter]-transforms_t[outter_itter-1]);
    }
      if(outter_itter > 5&&current_state != nearly_converged)
      {
        if(outter_itter%10==0)
        {
          if((delta_transform_R[outter_itter-1].squaredNorm()/delta_transform_R[outter_itter-6].squaredNorm()>0.1)&&(delta_transform_t[outter_itter-1].squaredNorm()/delta_transform_t[outter_itter-6].squaredNorm()>0.1))
          {
            current_state = far_from_converge;
          }
          else
          {
            current_state = converging;
          }
          
        }
        if(delta_transform_R[outter_itter-1].squaredNorm()<0.0001&&delta_transform_t[outter_itter-1].squaredNorm()<0.001)
        {
          current_state = nearly_converged;
        }
      }
      switch (current_state)
      {
      case  just_start :
        break;
      case  converging:
        break;
      case far_from_converge:
        break;
      case nearly_converged:
        correspond_num =1;
        break;
      default:
        break;
      }
      if(outter_itter <= 5)
      {
        cov_larger = init_error_mode - (double)outter_itter*(init_error_mode-1)/6;
      }
      else
      {
        cov_larger = 1;
      }
      if (cov_larger < 1)
      {
        cov_larger = 1;
      }
      
      std::cout << "current_state:"<< current_state << std::endl;
      std::cout << "current_cov_larger" << cov_larger << std::endl;
      if(current_state == far_from_converge)
      {
        current_state = converging;
      }
      if (correspond_num > 8)
      {
        correspond_num = 8; 
      }
      std::cout << "current_correspond_num:"<<correspond_num<<std::endl;
    Sophus::SE3d est_transform = current_transform;
    problem.AddParameterBlock(est_transform.data(), Sophus::SE3d::num_parameters,
                              new LocalParameterizationSE3);

    double mse_high = 0;

    typename pcl::PointCloud<PointT>::Ptr transformed_source (new pcl::PointCloud<PointT>());
    Eigen::Matrix4d trans_mat = current_transform.matrix();
    pcl::transformPointCloud(*source_cloud_,
                                *transformed_source,
                                trans_mat);

    std::vector<int> target_index;
    std::vector<float> dist_sq;

    for(int source_index = 0; source_index != transformed_source->size(); source_index++) {
      const PointT &transformed_source_pt = transformed_source->points[source_index];

      target_kd_tree_->nearestKSearch(transformed_source_pt, correspond_num,
                                      target_index, dist_sq);
      for(int correspondence_index = 0;
          correspondence_index < correspond_num;
          correspondence_index++) {
        if( dist_sq[correspondence_index] < max_match_distance){
          const PointT &source_pt =
            source_cloud_->points[source_index];
          const pcl::PointXYZ s_pt(source_pt.x, source_pt.y, source_pt.z);
          const Eigen::Matrix3d &source_cov =
           cov_larger*(source_covariances_->at(source_index));
          const PointT &target_pt =
            target_cloud_->points[target_index[correspondence_index]];
          const pcl::PointXYZ t_pt(target_pt.x, target_pt.y, target_pt.z);
          const Eigen::Matrix3d &target_cov =
           cov_larger*( target_covariances_->at(target_index[correspondence_index]));
          const Eigen::Matrix<double,N, 1> target_dist =
            target_distributions_->at(target_index[correspondence_index]);
          const Eigen::Matrix<double,N, 1> source_dist =
            source_distributions_->at(source_index);
 
          double prob =0;
          if(probability_mode)
          {
            for(size_t s = 0; s<N; s++){
              double temp = target_dist.transpose()*confusion_matrix_.col(s);
              temp *= source_dist.transpose()*confusion_matrix_.col(s);
              prob += temp;
            }
          }
          else
          {
            prob = 1;
          }
             GICPCostFunction* cost_function = new GICPCostFunction(s_pt,
                                                                    t_pt,
                                                                    source_cov,
                                                                    target_cov,
                                                                    base_transformation_);     
          prob *=cost_function->Probability( est_transform);
          problem.AddResidualBlock(cost_function,
                                   new ceres::ComposedLoss(
                                   new ceres::ScaledLoss(new ceres::CauchyLoss(3.0),
                                                         prob,
                                                         ceres::TAKE_OWNERSHIP),
                                   ceres::TAKE_OWNERSHIP,
                                   new SQLoss(),
                                   ceres::TAKE_OWNERSHIP),
                                   est_transform.data());

        } 
      } 
    } 
    ceres::Solver::Options options;
    options.gradient_tolerance = 0.1 * Sophus::Constants<double>::epsilon();
    options.function_tolerance = 0.1 * Sophus::Constants<double>::epsilon();
    options.linear_solver_type = ceres::DENSE_QR;
    options.num_threads = 8;
    options.max_num_iterations = 400;
    options.gradient_check_numeric_derivative_relative_step_size = 1e-8;
    options.gradient_check_relative_precision = 1e-6;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);


    double mse = (current_transform.inverse()*est_transform).log().squaredNorm();
    if(mse < 1e-5 || outter_itter>49)
        converged = true;
    current_transform = est_transform;
    outter_itter++;
  }
  final_transformation_ = current_transform;
   
  Sophus::SE3d trans = final_transformation_*base_transformation_;
  Eigen::Matrix4f mat = (trans.matrix()).cast<float>();
  if( final_cloud != nullptr ) {
      pcl::transformPointCloud(*source_cloud_,
                               *final_cloud,
                               mat);
  }

  outer_iter=outter_itter;
}


template <size_t N>
void LCR_SMM<N>::ComputeCovariances(
    const PointCloudPtr cloudptr,
    KdTreePtr treeptr,
    MatricesVectorPtr matvecptr,
    DistVectorPtr distvecptr) {
  // Variables for computing Covariances
  Eigen::Vector3d mean;
  Eigen::Matrix<double, N, 1> dist = Eigen::Matrix<double, N, 1>::Zero();
  double increment = 1.0/static_cast<double>(kCorrespondences_);

  std::vector<int> nn_idecies; nn_idecies.reserve (kCorrespondences_);
  std::vector<float> nn_dist_sq; nn_dist_sq.reserve (kCorrespondences_);

  // Set up Itteration
  matvecptr->resize(cloudptr->size());
  distvecptr->resize(cloudptr->size());

  for(size_t itter = 0; itter < cloudptr->size(); itter++) {
    const PointT &query_pt = (*cloudptr)[itter];

    Eigen::Matrix3d cov;
    cov.setZero();
    mean.setZero();
    dist.setZero();

    treeptr->nearestKSearch(query_pt, kCorrespondences_, nn_idecies, nn_dist_sq);

    for( int index: nn_idecies) {
      const PointT &pt = (*cloudptr)[index];
      int label_i;

      if(probability_mode)
      {     
        for(label_i=0;(relabel_i[label_i]!=0)&&(relabel_i[label_i]!=pt.label);label_i++);
        if(label_i<N)
                relabel_i[label_i] = pt.label;
        dist(label_i,0) += increment;
      }
      mean[0] += pt.x;
      mean[1] += pt.y;
      mean[2] += pt.z;

      cov(0,0) += pt.x*pt.x;

      cov(1,0) += pt.y*pt.x;
      cov(1,1) += pt.y*pt.y;

      cov(2,0) += pt.z*pt.x;
      cov(2,1) += pt.z*pt.y;
      cov(2,2) += pt.z*pt.z;
    }

    mean /= static_cast<double> (kCorrespondences_);
    for (int k = 0; k < 3; k++) {
      for (int l =0; l <= k; l++) {
        cov(k,l) /= static_cast<double> (kCorrespondences_);
        cov(k,l) -= mean[k]*mean[l];
        cov(l,k) = cov(k,l);
      }
    }
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU);
    cov.setZero();
    Eigen::Matrix3d U = svd.matrixU();

    for (int k = 0; k<3; k++) {
      Eigen::Vector3d col = U.col(k);
      double v = 1.;
      if (k == 2) {
        v = kEpsilon_;
      }
      cov+= v*col*col.transpose();
    }
    (*matvecptr)[itter] = cov;
    (*distvecptr)[itter] = dist;
  }


}
template <size_t N>
void LCR_SMM<N>:: errorcompute(double* errorvecptr, int branch_r, int branch_x, int branch_y)
{
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>  test_trans_mat ;
  double delta_r, delta_x, delta_y;
  delta_r = (rotation_sup - rotation_inf)/branch_r;
  delta_x = (x_sup - x_inf)/branch_x;
  delta_y = (y_sup - y_inf)/branch_y;
  double* errorptr = errorvecptr;
  double error = 0 ;
  ComputeCovariances(source_cloud_, source_kd_tree_, source_covariances_, source_distributions_);
  ComputeCovariances(target_cloud_, target_kd_tree_, target_covariances_, target_distributions_);
  typename pcl::PointCloud<PointT>::Ptr test_transformed_source (new pcl::PointCloud<PointT>());
  for(int rad = 0 ; rad < branch_r; rad ++)
  {
    for (int  x = 0; x <  branch_x; x++)
    {
      for (int y = 0; y < branch_y; y++)
      {
        Eigen::Vector3d test_init_rotation_n(0,0,1);
        Eigen::Vector3d test_init_t(x_inf + 0.5*(2*x + 1)*delta_x  ,      y_inf + 0.5*(2*y + 1)*delta_y,             0);
      //  std::cout << test_init_t <<std::endl;
        Eigen::AngleAxisd test_init_rotation_vector(rotation_inf + 0.5*(2*rad + 1)*delta_r , test_init_rotation_n);
        Eigen::Matrix3d test_init_R = test_init_rotation_vector.toRotationMatrix();
        Sophus::SE3d test_init_transform(test_init_R,test_init_t);
        Eigen::Matrix4d test_init_mat =test_init_transform.matrix();
        test_trans_mat.push_back(test_init_mat);
      }
    }
  }
  std::cout << test_trans_mat.size()<< std::endl;
  for (int test_trans_index = 0; test_trans_index <  125; test_trans_index++)
  {
    error =0;
    
    pcl::transformPointCloud(*source_cloud_,*test_transformed_source, test_trans_mat[test_trans_index]);
    std::vector<int> target_index;
    std::vector<float> dist_sq;
    for(int source_index = 0; source_index < test_transformed_source->size(); source_index+=10) {
      const PointT &transformed_source_pt = test_transformed_source->points[source_index];

      target_kd_tree_->nearestKSearch(transformed_source_pt, 4,
                                      target_index, dist_sq);
      for(int correspondence_index = 0;
          correspondence_index < 4;
          correspondence_index++) {
        if( dist_sq[correspondence_index] < max_match_distance ) {
          const PointT &source_pt =
            source_cloud_->points[source_index];
          const pcl::PointXYZ s_pt(source_pt.x, source_pt.y, source_pt.z);
          const Eigen::Matrix3d &source_cov =
            source_covariances_->at(source_index);
          const PointT &target_pt =
            target_cloud_->points[target_index[correspondence_index]];
          const pcl::PointXYZ t_pt(target_pt.x, target_pt.y, target_pt.z);
          const Eigen::Matrix3d &target_cov =
            target_covariances_->at(target_index[correspondence_index]);

          const Eigen::Matrix<double,N, 1> target_dist =
            target_distributions_->at(target_index[correspondence_index]);
          const Eigen::Matrix<double,N, 1> source_dist =
            source_distributions_->at(source_index);

          //double prob = confusion_matrix_(source_pt.label-1, target_pt.label-1)*
          //              dist(target_pt.label-1, 0);
          double prob =0;
          if(probability_mode)
          {
            for(size_t s = 0; s<N; s++){
              double temp = target_dist.transpose()*confusion_matrix_.col(s);
              temp *= source_dist.transpose()*confusion_matrix_.col(s);
              prob += temp;
              
            }
          }
          else
          {
            prob = 1;
          }
          
          Sophus::SE3<double> test_Transform(test_trans_mat[test_trans_index]);
          Eigen::Matrix3d R = test_Transform.rotationMatrix();
          Eigen::Matrix3d M = (target_cov+R*source_cov*R.transpose()).inverse();
          Eigen::Vector3d point_source_ (s_pt.x ,s_pt.y ,s_pt.z);
          Eigen::Vector3d point_target_  (t_pt.x ,t_pt.y ,t_pt.z);
          Eigen::Vector3d transformed_point_source_ = test_Transform * point_source_;
          Eigen::Vector3d res = point_target_-transformed_point_source_;
          Eigen::Vector3d dT = M*res;
          double malhalanobis= -1.0/2.0*double(res.transpose() * dT);
          double probability = pow((2*M_PI*M.inverse()).determinant(), -1.0/2.0)*exp(malhalanobis);
          prob *= probability;
          error += prob ;
          
        }
  
      }
   }
     errorvecptr[test_trans_index] =error;
 }
}


#endif
