/****************************************************************
LCR-SMM Large Convergence Region Semantic Map Matching Algorithm
Last modified: Nov 3, 2021

The code for calculating cost function are derived from semantic-icp by Steven Parkison.
https://bitbucket.org/saparkison/semantic-icp
****************************************************************/


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

    Eigen::Matrix4d I3 = Eigen::Matrix4d::Identity();
    
    Sophus::SE3d initTransform(I3);

    Eigen::Matrix<double,20,20> cm = Eigen::Matrix<double,20,20>::Identity();

    LCR_SMM<20> lcr;
    pcl::PointCloud<PointT>::Ptr
        finalCloudem( new pcl::PointCloud<PointT> );
    lcr.set_max_distance(5);
    lcr.setSourceCloud(cloudA);
    lcr.setTargetCloud(cloudB);
    lcr.setConfusionMatrix(cm);
    lcr.set_correspond_num(4);
    lcr.set_probability_mode(1);//0:geometric matching 1:semantic matching
    lcr.set_range(r_inf, r_sup, x_inf, x_sup, y_inf, y_sup);
    lcr.errorcompute(errorvector, branch_r, branch_x, branch_y);    
    double error_max = 0;
    int error_index = 0;
    int error_num;
    error_num = branch_r * branch_x * branch_y;
    for (int i = 0; i < error_num; i++)
    {
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

    lcr.setiniterrormode(init_error_mode);
    
    Eigen::Vector3d init_rotation_n(0,0,1);
    Eigen::Vector3d init_t(x_inf + 0.5*(1 + 2*error_index%(branch_x*branch_y)/branch_y)*delta_x,       y_inf + 0.5*(1 + 2*error_index % branch_y)*delta_y ,       0);
    Eigen::AngleAxisd init_rotation_vector(r_inf + 0.5*(1 + 2 * error_index/(branch_x*branch_y))*delta_r,init_rotation_n);
    Eigen::Matrix3d init_R = init_rotation_vector.toRotationMatrix();
    Sophus::SE3d init_transform(init_R,init_t);
    Eigen::Matrix4d init_mat =init_transform.matrix();
    lcr.align(finalCloudem, init_transform);



    std::cout <<"Estimated Transformation:\n" << lcr.getFinalTransFormation().matrix() << std::endl;

    pcl::io::savePCDFileASCII ("init.pcd", *cloudA+*cloudB);
    pcl::io::savePCDFileASCII ("LCR.pcd", *finalCloudem+*cloudB);
    return (0);
    
}
