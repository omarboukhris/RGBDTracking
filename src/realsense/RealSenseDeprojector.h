/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

//#include <CImgPlugin/CImgData.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/BoundingBox.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/OptionsGroup.h>

#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <fstream>
#include <algorithm>
#include <iostream>
#include <string>
#include <map>

#include <sofa/opencvplugin/OpenCVWidget.h>
#include <RGBDTracking/src/realsense/RealSenseCam.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>

namespace sofa {

namespace rgbdtracking {

class RealSenseDeprojector : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS( RealSenseDeprojector , core::objectmodel::BaseObject);
    typedef core::objectmodel::BaseObject Inherited;

    Data<opencvplugin::ImageData> d_depth ;
    Data<opencvplugin::ImageData> d_color ;
    Data<int> d_downsampler ;
    Data<bool> d_drawpcl ;
    Data<helper::vector<defaulttype::Vector3> > d_output ;

    core::objectmodel::SingleLink<
        RealSenseDeprojector,
        RealSenseCam,
        BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK
    > l_rs_cam ; //for intrinsics
    DataCallback c_image ;

    rs2_intrinsics cam_intrinsics ;

    pcl::PointCloud<pcl::PointXYZ>::Ptr m_pointcloud ;
    pcl::PointCloud<pcl::Normal>::Ptr m_cloud_normals ;

    helper::vector<defaulttype::Vector3> m_colors ;

    RealSenseDeprojector()
        : Inherited()
        , d_depth(initData(&d_depth, "depth", "segmented depth data image"))
        , d_color(initData(&d_color, "color", "segmented color data image"))
        , d_downsampler(initData(&d_downsampler, 5, "downsample", "point cloud downsampling"))
        , d_output(initData(&d_output, "output", "output 3D position"))
        , d_drawpcl(initData(&d_drawpcl, false, "drawpcl", "true if you want to draw the point cloud"))
        , l_rs_cam(initLink("rscam", "link to realsense camera component - used for getting camera intrinsics"))
        , m_pointcloud(new pcl::PointCloud<pcl::PointXYZ>)
        , m_cloud_normals(new pcl::PointCloud<pcl::Normal>)
    {
        c_image.addInputs({&d_color, &d_depth});
        c_image.addCallback(std::bind(&RealSenseDeprojector::deproject_image, this));
    }

    virtual ~RealSenseDeprojector () {
    }

    void erode_mask (const cv::Mat & src, cv::Mat & erosion_dst, int erosion_size) {
        cv::Mat element = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),
            cv::Point( erosion_size, erosion_size )
        );

        /// Apply the erosion operation
        cv::dilate( src, erosion_dst , element );
    }

    /*!
     * \brief deproject_image : deprojects depth image in a 3D point cloud
     *
     */
    void get_point_cloud(
        helper::vector<defaulttype::Vector3>& colors,
        helper::vector<defaulttype::Vector3>& output,
        pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud,
        const cv::Mat color_im,
        const cv::Mat depth_im,
        rs2::depth_frame depth
    ) {
        //cv::Mat color_gray ;
        //cv::cvtColor(color_im, color_gray, cv::COLOR_BGR2GRAY);

        //output.clear () ;
        //colors.clear() ;
        m_pointcloud->clear();

        int downSample = d_downsampler.getValue() ;
        for (size_t i = 0 ; i < depth_im.rows/downSample ; ++i) {
            for (size_t j = 0 ; j < depth_im.cols/downSample ; ++j) {
                if (depth_im.at<const uchar>(downSample*i,downSample*j) > 0) {
                    // deprojection
                    float dist = depth.get_distance(downSample*j, downSample*i) ;
                    float
                        point3d[3] = {0.f, 0.f, 0.f},
                        point2d[2] = {downSample*i, downSample*j};
                    rs2_deproject_pixel_to_point(
                        point3d,
                        &cam_intrinsics,
                        point2d,
                        dist
                    );

                    // set units
                    pcl::PointXYZ pclpoint = pcl::PointXYZ(point3d[1], point3d[0], point3d[2]) ;
                    //cv::Vec3b color = color_im.at<cv::Vec3b>(downSample*i,downSample*j) ;
                    //defaulttype::Vector3 deprojected_point = defaulttype::Vector3(point3d[0], point3d[1], point3d[2]) ;
                    //defaulttype::Vector3 deprojected_color = defaulttype::Vector3(color[0], color[1], color[2]) ;

                    // add units to result
                    pointcloud->push_back(pclpoint);
                    //output.push_back(deprojected_point) ;
                    //colors.push_back(deprojected_color) ;
                }
            }
        }
    }

    void deproject_image () {
        if (!l_rs_cam) {
        // we need a valid link to realsense cam sofa component
            std::cerr <<
                "(RealSenseDeprojector) link to realsense cam component is broken" <<
            std::endl ;
            return ;
        }
        // get intrinsics from link to rs-cam component
        rs2::depth_frame depth = *l_rs_cam->depth ;
        cam_intrinsics = depth.get_profile().as<rs2::video_stream_profile>().get_intrinsics() ;

        std::fstream fstr ;
        fstr.open("intrins.log", std::fstream::out); // | std::fstream::app) ;

        fstr
            << cam_intrinsics.fx  << std::endl
            << cam_intrinsics.fy  << std::endl
            << cam_intrinsics.ppx  << std::endl
            << cam_intrinsics.ppy  << std::endl
            << cam_intrinsics.height  << std::endl
            << cam_intrinsics.width  << std::endl <<
        std::endl ;

        fstr.close();

        // get depth & color image
        cv::Mat color_im, color_tmp = d_color.getValue().getImage(),
                depth_im = d_depth.getValue().getImage() ;

        // setup output
        helper::vector<defaulttype::Vector3> & output = *d_output.beginEdit() ;
        //this->erode_mask(color_tmp, color_im, 15) ;
        this->get_point_cloud(m_colors, output, m_pointcloud, color_im, depth_im, depth);
        d_output.endEdit();

        // once pcl extracted compute normals
        // compute_pcl_normals() ;
    }

    void compute_pcl_normals () {
        // compute normals for remeshing ?
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne ;
        ne.setInputCloud(m_pointcloud);
        // Create an empty kdtree representation, and pass it to the normal estimation object.
        // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
        ne.setSearchMethod (tree);

        // Use all neighbors in a sphere of radius 3cm
        ne.setRadiusSearch (0.03);

        // Compute the features
        m_cloud_normals->clear();
        ne.compute (*m_cloud_normals);
    }

    void draw(const core::visual::VisualParams* vparams) {
        if (!d_drawpcl.getValue()) {
        // don't draw point cloud
            return ;
        }

        if (m_colors.size() == m_pointcloud->size()) {
            size_t i = 0 ;
            for (const auto & pt : *m_pointcloud) {
                auto color = m_colors[i++] ;
                vparams->drawTool()->drawPoint(
                    defaulttype::Vector3(pt.x, pt.y, pt.z),
                    sofa::defaulttype::Vector4 (color[0],color[1],color[2],0)
                );
            }
        } else {
            for (const auto & pt : *m_pointcloud) {
                vparams->drawTool()->drawPoint(
                    defaulttype::Vector3(pt.x, pt.y, pt.z),
                    sofa::defaulttype::Vector4 (0, 0, 255, 0)
                );
            }
        }
    }
};

}

}

