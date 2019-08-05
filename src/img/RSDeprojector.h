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
#include <sofa/opencvplugin/utils/OpenCVMouseEvents.h>
#include <RGBDTracking/src/img/RealSense.h>

namespace sofa {

namespace rgbdtracking {

using pcl_ptr = pcl::PointCloud<pcl::PointXYZ>::Ptr ;

class RSDeprojector : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS( RSDeprojector , core::objectmodel::BaseObject);
    typedef core::objectmodel::BaseObject Inherited;

    Data<opencvplugin::ImageData> d_depth ;
    Data<opencvplugin::ImageData> d_color ;
    Data<int> d_downsampler ;
    Data<bool> d_drawpcl ;
    Data<helper::vector<defaulttype::Vector3> > d_output ;

    core::objectmodel::SingleLink<
        RSDeprojector,
        RealSense,
        BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK
    > l_rs_cam ; //for intrinsics

    rs2_intrinsics cam_intrinsics ;

    DataCallback c_image ;

    RSDeprojector()
        : Inherited()
        , d_depth(initData(&d_depth, "depth", "segmented depth data image"))
        , d_color(initData(&d_color, "color", "segmented color data image"))
        , d_output(initData(&d_output, "output", "output 3D position"))
        , d_drawpcl(initData(&d_drawpcl, false, "drawpcl", "true if you want to draw the point cloud"))
        , d_downsampler(initData(&d_downsampler, 5, "downsample", "point cloud downsampling"))
        , l_rs_cam(initLink("rscam", "link to realsense camera component - used for getting camera intrinsics"))
    {
        c_image.addInputs({&d_depth});
        c_image.addCallback(std::bind(&RSDeprojector::deproject_image, this));
    }

    virtual ~RSDeprojector () {
    }

    void deproject_image () {
        // get intrinsics from link to rs-cam component
        if (!l_rs_cam) {
            std::cerr <<
                "(RSDeprojector) link to realsense cam component is broken" <<
            std::endl ;
            return ;
        }

        rs2::depth_frame depth = *l_rs_cam->depth ;
        cam_intrinsics = depth.get_profile().as<rs2::video_stream_profile>().get_intrinsics() ;

        // get depth image
        cv::Mat depth_im = d_depth.getValue().getImage() ;
        cv::Mat color_im = d_color.getValue().getImage() ;
        cv::cvtColor(color_im, color_im, cv::COLOR_BGR2GRAY);

        // setup output
        helper::vector<defaulttype::Vector3> & output = *d_output.beginEdit() ;
        output.clear () ;
        int downSample = d_downsampler.getValue() ;
        for (size_t i = 0 ; i < depth_im.rows/downSample ; ++i) {
            for (size_t j = 0 ; j < depth_im.cols/downSample ; ++j) {
                //if (depth_im.at<const float>(downSample*i,downSample*j) > 0) {
                // if depth value @[i,j] greater than 0
                // deprojection point-wise happens here
                // idea #2 : filter pixel-wise depending on grabcut results
                    float dist = depth.get_distance(downSample*i, downSample*j) ;
                    float
                        point3d[3] = {0.f, 0.f, 0.f},
                        point2d[2] = {downSample*i, downSample*j};
                    rs2_deproject_pixel_to_point(
                        point3d,
                        &cam_intrinsics,
                        point2d,
                        dist
                    );
                    defaulttype::Vector3 deprojected_point = defaulttype::Vector3(point3d[0], point3d[1], point3d[2]) ;
                    output.push_back(deprojected_point) ;
                //}
            }
        }
        // the end
        d_output.endEdit();
    }

    void deproject_image2 () {
        if (!l_rs_cam) {
            std::cerr <<
                "(RSDeprojector) link to realsense cam component is broken" <<
            std::endl ;
            return ;
        }
        // std::vector<pcl_ptr> points (l_rs_cam->points) ;
        helper::vector<defaulttype::Vector3> & output = *d_output.beginEdit() ;
        output.clear () ;
        pointcloud2vectors (output, l_rs_cam->points) ;
        d_output.endEdit() ;
    }

    void draw(const core::visual::VisualParams* vparams) {
        if (!d_drawpcl.getValue()) {
        // don't draw point cloud
            return ;
        }

        helper::vector<defaulttype::Vector3> output = d_output.getValue() ;
        if (output.size()){
            std::cout << output.size() << std::endl ;
            std::vector< sofa::defaulttype::Vector3 > points;

            points.resize(0);
            for (unsigned int i=0; i< output.size(); i++) {
                sofa::defaulttype::Vector3 point = output[i] ;
                vparams->drawTool()->drawSphere(point, 0.0008);
                points.push_back(point);
            }
        }
    }

protected :
    void pointcloud2vectors (helper::vector<defaulttype::Vector3> & out, const rs2::points& points) {
        auto ptr = points.get_vertices() ;
        int downSample = d_downsampler.getValue() ;
        for (size_t i = 0; i < points.size() ; i++, ptr++ ) {
            if (i % downSample == 0 && ptr->z) {
                out.push_back (
                    defaulttype::Vector3 (ptr->x, ptr->y, ptr->z)
                ) ;
            }
        }
    }

};

}

}

