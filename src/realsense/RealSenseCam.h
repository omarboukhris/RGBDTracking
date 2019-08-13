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


#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <librealsense2/rs.hpp>
//#include <librealsense/wrappers/opencv/cv-helpers.hpp>

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


class RealSenseCam : public opencvplugin::streamer::BaseOpenCVStreamer //core::objectmodel::BaseObject
{
public:
    SOFA_CLASS( RealSenseCam , opencvplugin::streamer::BaseOpenCVStreamer );
    typedef opencvplugin::streamer::BaseOpenCVStreamer Inherited;

    Data<int> depthMode;
    Data<int> depthScale;

    Data<opencvplugin::ImageData> d_color ;
    Data<opencvplugin::ImageData> d_depth ;

    // Declare depth colorizer for pretty visualization of depth data
	rs2::colorizer color_map;

    // for pointcloud extraction
    rs2::pointcloud pc ;
    rs2::points points ;

    rs2::video_frame *color ;
    rs2::depth_frame *depth ;

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Start streaming with default recommended configuration

    // Using the context to create a rs2::align object.
	// rs2::align allows you to perform aliment of depth frames to others

    RealSenseCam()
        : Inherited()
        , depthMode ( initData ( &depthMode,1,"depthMode","depth mode" ))
        , depthScale(initData(&depthScale,1,"depthScale","scale for the depth values, 1 for SR300, 10 for 435"))
        , d_color(initData(&d_color, "color", "RGB data image"))
        , d_depth(initData(&d_depth, "depth", "depth data image"))
    {
        this->f_listening.setValue(true) ;
        color = nullptr ;
        depth = nullptr ;
    }

    ~RealSenseCam () {
    }

    void init() {
        initAlign();
    }


    void decodeImage(cv::Mat & /*img*/) {
        acquireAligned();
    }

//    void handleEvent(sofa::core::objectmodel::Event */*event*/) {
//        acquireAligned();
//    }

protected:

    void initAlign() {
        //rs2::pipeline_profile selection =
        pipe.start();

        rs2::align align(RS2_STREAM_COLOR);
        rs2::frameset frameset;

        while (
            (!frameset.first_or_default(RS2_STREAM_DEPTH) ||
             !frameset.first_or_default(RS2_STREAM_COLOR))
        ) {
            frameset = pipe.wait_for_frames();
        }

        rs2::frameset processed = align.process(frameset);

        // Trying to get both color and aligned depth frames
        color = new rs2::video_frame(processed.get_color_frame());
        depth = new rs2::depth_frame(processed.get_depth_frame());

        // extract pointcloud
        getpointcloud(*color, *depth) ;

        // Create depth and color image
        frame_to_cvmat(*color, *depth);

    }

    void acquireAligned() {
		rs2::align align(RS2_STREAM_COLOR);

		rs2::frameset frameset;

		while (
			!frameset.first_or_default(RS2_STREAM_DEPTH) || 
			!frameset.first_or_default(RS2_STREAM_COLOR)
		) {
			frameset = pipe.wait_for_frames();
		}

        rs2::frameset processed = align.process(frameset);

        // Trying to get both color and aligned depth frames
        if (color) delete color ;
        if (depth) delete depth ;
        color = new rs2::video_frame(processed.get_color_frame());
        depth = new rs2::depth_frame(processed.get_depth_frame());

        // extract pointcloud
        getpointcloud(*color, *depth) ;

        // Create depth and color image
        frame_to_cvmat(*color, *depth);
    }

protected :
    void getpointcloud (rs2::frame color, rs2::frame depth) {
        if (color) {
            pc.map_to (color) ;
        }
        points = pc.calculate(depth) ;
    }

    void frame_to_cvmat(rs2::video_frame color, rs2::depth_frame depth) {
        int widthc = color.get_width();
        int heightc = color.get_height();

        cv::Mat
            rgb0(heightc,widthc, CV_8UC3, (void*) color.get_data()),
            & bgr_image = *d_color.beginEdit() ;
        cv::cvtColor (rgb0, bgr_image, cv::COLOR_RGB2BGR); // bgr_image is output
//        cv::flip(bgr_image, rgb0, AXIS) ;
//        bgr_image = rgb0.clone() ;
        d_color.endEdit();

        int widthd = depth.get_width();
        int heightd = depth.get_height();
        cv::Mat
            depth16 = cv::Mat(heightd, widthd, CV_16U, (void*)depth.get_data()),
            & depth8 = *d_depth.beginEdit() ;
        depth16.convertTo(depth8, CV_8U, 1.f/64*depthScale.getValue()); //depth32 is output
//        cv::flip(depth8, depth16, AXIS) ;
//        depth8 = depth16.clone() ;
        d_depth.endEdit();


    }


};

}

}

