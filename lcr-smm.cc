#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>
#include <string>
#include <dirent.h>
#include <algorithm>
#include <fstream>
#include <vector>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include "lcr.h"

int
main (int argc, char** argv)
{
    typedef pcl::PointXYZL PointT;
    std::string strSource;
    std::string strTarget;
    double delta_r, delta_x, delta_y;
    double errorvector[1000]={0};
    int branch_x = 5, branch_y = 5, branch_r = 5;
    double x_inf = -8 , x_sup = 8, y_inf = -8, y_sup = 8, r_inf = -0.5, r_sup = 0.5; 
    delta_r = (r_sup - r_inf)/branch_r;
    delta_x = (x_sup - x_inf)/branch_x;
    delta_y = (y_sup - y_inf)/branch_y;
    if ( !pcl::console::parse_argument(argc, argv, "-s", strSource) ) {
        std::cout << "Need source file (-s)\n";
        return (-1);
    }
    if ( !pcl::console::parse_argument(argc, argv, "-t", strTarget) ) {
        std::cout << "Need target file (-t)\n";
        return (-1);
    }
    pcl::PointCloud<PointT>::Ptr cloudA (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr mycloudA (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr mycloudA_filtered (new pcl::PointCloud<PointT>); 
    if (pcl::io::loadPCDFile<pcl::PointXYZL> (strSource, *cloudA) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file cloudA.pcd \n");
        return (-1);
    }
    pcl::PointCloud<PointT>::Ptr cloudB (new pcl::PointCloud<PointT>);

    if (pcl::io::loadPCDFile<PointT> (strTarget, *cloudB) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file cloudB.pcd \n");
        return (-1);
    }
    Eigen::Vector3d my_init_rotation_n(0,0,1);
    Eigen::Vector3d my_init_t(0,4,0);
    Eigen::AngleAxisd my_init_rotation_vector(15*M_PI/180 , my_init_rotation_n);
    Eigen::Matrix3d my_init_R = my_init_rotation_vector.toRotationMatrix();
    Sophus::SE3d my_init_transform(my_init_R,my_init_t);
    Eigen::Matrix4d my_init_mat =my_init_transform.matrix();
    Eigen::Matrix4d my_true_mat =my_init_transform.matrix().inverse();
    std::cout <<my_init_mat<<std::endl;
    std::cout <<std::endl;
    std::cout <<my_true_mat<<std::endl;
    pcl::transformPointCloud(*cloudA,*mycloudA, my_init_mat);
    auto begin = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    Eigen::Matrix4d temp = Eigen::Matrix4d::Identity();
    
    Sophus::SE3d initTransform(temp);
    Eigen::Matrix<double,20,20> cm = Eigen::Matrix<double,20,20>::Identity();
    lcrsmm::EmIterativeClosestPoint<20> lcr;
    pcl::PointCloud<PointT>::Ptr
        finalCloudem( new pcl::PointCloud<PointT> );
    begin = std::chrono::steady_clock::now();
    lcr.set_max_distance(5);
    lcr.setSourceCloud(mycloudA);//_filtered);
    lcr.setTargetCloud(cloudB);
    lcr.setConfusionMatrix(cm);
    lcr.set_correspond_num(4);
    lcr.set_probability_mode(1);//0:geometric matching 1:semantic matching
    lcr.set_range(r_inf, r_sup, x_inf, x_sup, y_inf, y_sup);
    lcr.errorcompute(errorvector, branch_r, branch_x, branch_y);    
    //lcr.setSourceCloud(mycloudA);
    double error_max = 0;
    int error_index = 0;
    int error_num;
    error_num = branch_r * branch_x * branch_y;
    for (int i = 0; i < error_num; i++)
    {
        std::cout <<errorvector[i] <<" ";
        if (i%5==4)
        {
            std::cout <<std::endl;
        }
        if (i%25==24)
        {
        std::cout << std::endl;
        }
        if(error_max < errorvector[i])
        {
            error_max = errorvector[i];
            error_index = i;
        }
        
    }
    std::sort(errorvector,errorvector+error_num);
    double init_error_mean = (errorvector[error_num-5]+errorvector[error_num-4]+errorvector[error_num-3]+errorvector[error_num-2]+errorvector[error_num-1])/5;
    double init_error_mode = 0;
    if (errorvector[error_num-1]-init_error_mean<=0)
    {
        init_error_mode = 50;
    }
    else
    {
        init_error_mode = 2*init_error_mean/(errorvector[error_num-1]-init_error_mean);    
    }
    if (init_error_mode  >50)
    {
        init_error_mode = 50;
    }
    
    std::cout << "init_error_mode:"<<init_error_mode<< std::endl;
    lcr.setiniterrormode(init_error_mode);
    
    Eigen::Vector3d init_rotation_n(0,0,1);
    Eigen::Vector3d init_t(x_inf + 0.5*(1 + 2*error_index%(branch_x*branch_y)/branch_y)*delta_x,       y_inf + 0.5*(1 + 2*error_index % branch_y)*delta_y ,       0);
    Eigen::AngleAxisd init_rotation_vector(r_inf + 0.5*(1 + 2 * error_index/(branch_x*branch_y))*delta_r,init_rotation_n);
    Eigen::Matrix3d init_R = init_rotation_vector.toRotationMatrix();
    Sophus::SE3d init_transform(init_R,init_t);
    Eigen::Matrix4d init_mat =init_transform.matrix();
    lcr.align(finalCloudem, init_transform);
    end = std::chrono::steady_clock::now();
    auto time_lcr = std::chrono::duration_cast<std::chrono::seconds>(end-begin).count();
    Sophus::SE3d em_finaltransform=lcr.getFinalTransFormation();
    Sophus::SE3d true_transform(my_init_mat.inverse());
    Sophus::SE3d transformdifferent_em = true_transform*em_finaltransform.inverse();
    Sophus::SO3d transformdifferent_R_em(transformdifferent_em.rotationMatrix());
    double error_R_em = sqrt(transformdifferent_R_em.log().squaredNorm())*180/M_PI;
    double error_t_em = sqrt(transformdifferent_em.translation().squaredNorm());
    //double transformerror_em =transformdifferent_em.log().squaredNorm();
    std::cout <<error_R_em<<std::endl;
    std::cout <<error_t_em<<std::endl;
    std::cout <<"LCR TIME\n" <<time_lcr<<std::endl;

    pcl::io::savePCDFileASCII ("init.pcd", *mycloudA+*cloudB);
    pcl::io::savePCDFileASCII ("LCR.pcd", *finalCloudem+*cloudB);
    return (0);
    
}
