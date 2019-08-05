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
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <algorithm>
#include <iostream>
#include <string>
#include <map>

#include <sofa/opencvplugin/OpenCVWidget.h>
#include <sofa/opencvplugin/BaseOpenCVStreamer.h>

namespace sofa
{

namespace rgbdtracking
{

using namespace cimg_library;
using defaulttype::Vec;
using defaulttype::Vector3;


using namespace std;
using namespace cv;
using namespace boost;
using namespace rs2;


class RealSense : public opencvplugin::streamer::BaseOpenCVStreamer //core::objectmodel::BaseObject
{
public:
    SOFA_CLASS( RealSense , opencvplugin::streamer::BaseOpenCVStreamer );
    typedef opencvplugin::streamer::BaseOpenCVStreamer Inherited;

    Data<int> depthMode;
    Data<int> depthScale;

    Data<opencvplugin::ImageData> d_color ;
    Data<opencvplugin::ImageData> d_depth ;

    rs2_intrinsics cam_intrinsics ;
    rs2::pipeline_profile selection ;

    // Declare depth colorizer for pretty visualization of depth data
	rs2::colorizer color_map;

    // for pointcloud extraction
    rs2::pointcloud pc ;
    rs2::points points ;

	// Declare RealSense pipeline, encapsulating the actual device and sensors
	rs2::pipeline pipe;
	// Start streaming with default recommended configuration


	// Using the context to create a rs2::align object.
	// rs2::align allows you to perform aliment of depth frames to others

    RealSense()
        : Inherited()
        , depthMode ( initData ( &depthMode,1,"depthMode","depth mode" ))
        , depthScale(initData(&depthScale,1,"depthScale","scale for the depth values, 1 for SR300, 10 for 435"))
        , d_color(initData(&d_color, "color", "RGB data image"))
        , d_depth(initData(&d_depth, "depth", "depth data image"))
    {
        this->f_listening.setValue(true) ;
    }

    ~RealSense () {
    }

    void init() {
        initAlign();
    }


    void decodeImage(cv::Mat & /*img*/) {
        acquireAligned();
    }

protected:

    void calc_intrinsics (rs2::video_stream_profile video_stream) {
        try
        {
            //If the stream is indeed a video stream, we can now simply call get_intrinsics()
            cam_intrinsics = video_stream.get_intrinsics();

//            auto principal_point = std::make_pair(intrinsics.ppx, intrinsics.ppy);
//            auto focal_length = std::make_pair(intrinsics.fx, intrinsics.fy);
//            rs2_distortion model = intrinsics.model;

//            std::cout << "Principal Point         : " << principal_point.first << ", " << principal_point.second << std::endl;
//            std::cout << "Focal Length            : " << focal_length.first << ", " << focal_length.second << std::endl;
//            std::cout << "Distortion Model        : " << model << std::endl;
//            std::cout << "Distortion Coefficients : [" << intrinsics.coeffs[0] << "," << intrinsics.coeffs[1] << "," <<
//                intrinsics.coeffs[2] << "," << intrinsics.coeffs[3] << "," << intrinsics.coeffs[4] << "]" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to get intrinsics for the given stream. " << e.what() << std::endl;
        }
    }

    void initAlign() {
        selection = pipe.start();
        auto depth_stream = selection.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>() ;
        calc_intrinsics(depth_stream);

        rs2::align align(RS2_STREAM_COLOR);
        rs2::frameset frameset;

        for (int it= 0; it < 100 ; it++) {
            frameset = pipe.wait_for_frames();
        }

        while (
            (!frameset.first_or_default(RS2_STREAM_DEPTH) ||
             !frameset.first_or_default(RS2_STREAM_COLOR))
        ) {
            frameset = pipe.wait_for_frames();
        }

        auto processed = align.process(frameset);

        // Trying to get both color and aligned depth frames
        rs2::video_frame color = processed.get_color_frame();
        rs2::depth_frame depth = processed.get_depth_frame();

        int widthd = depth.get_width();
        int heightd = depth.get_height();

        int widthc = color.get_width();
        int heightc = color.get_height();

        // extract pointcloud
        getpointcloud(color, depth) ;

        // Create depth image

        cv::Mat
            rgb0(heightc,widthc, CV_8UC3, (void*) color.get_data()),
            & bgr_image = *d_color.beginEdit() ;
        cvtColor (rgb0, bgr_image, cv::COLOR_RGB2BGR); // bgr_image is output
        d_color.endEdit();

        cv::Mat
            depth16 = cv::Mat(heightd, widthd, CV_16U, (void*)depth.get_data()),
            & depth8 = *d_depth.beginEdit() ;
        depth16.convertTo(depth8, CV_8U, 1.f/64*depthScale.getValue()); //depth32 is output
        d_depth.endEdit();

    }

    void acquireAligned() {
        auto depth_stream = selection.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>() ;
        calc_intrinsics(depth_stream);

		rs2::align align(RS2_STREAM_COLOR);

		rs2::frameset frameset;

		while (
			!frameset.first_or_default(RS2_STREAM_DEPTH) || 
			!frameset.first_or_default(RS2_STREAM_COLOR)
		) {
			frameset = pipe.wait_for_frames();
		}

		auto processed = align.process(frameset);

		rs2::video_frame color = processed.get_color_frame();
		rs2::depth_frame depth = processed.get_depth_frame();

        int widthd = depth.get_width();
        int heightd = depth.get_height();

        int widthc = color.get_width();
        int heightc = color.get_height();

        // extract pointcloud
        getpointcloud(color, depth) ;

        // Create depth & color images
        cv::Mat
            rgb0(heightc,widthc, CV_8UC3, (void*) color.get_data()),
            & bgr_image = *d_color.beginEdit() ;
        cvtColor (rgb0, bgr_image, cv::COLOR_RGB2BGR); // bgr_image is output
        d_color.endEdit();

        cv::Mat depth160( heightd, widthd, CV_16U, (void*)depth.get_data() );
        cv::Mat
            depth16 = depth160.clone(),
            & depth8 = *d_depth.beginEdit() ;
        depth16.convertTo(depth8, CV_8U, 1.f/64*depthScale.getValue()); //depth32 is output
        d_depth.endEdit();

    }

protected :
    void getpointcloud (rs2::frame color, rs2::frame depth) {
        if (color) {
            pc.map_to (color) ;
        }

        points = pc.calculate(depth) ;
    }

};

}

}

