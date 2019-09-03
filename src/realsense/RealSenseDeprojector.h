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

class RealSenseDistFrame {
public :
    typedef struct {
        size_t _width ;
        size_t _height ;
        float* frame ;
    } RealSenseDistStruct ;
    RealSenseDistStruct _distdata ;

    RealSenseDistFrame () {}

    RealSenseDistFrame (RealSenseDistStruct diststr) {
        _distdata = diststr ;
    }

    operator RealSenseDistStruct & () {
        return getFrame() ;
    }
    operator const RealSenseDistStruct & () {
        return getFrame() ;
    }

    inline size_t width() const {
        return _distdata._width ;
    }

    inline size_t height() const {
        return _distdata._height;
    }

    inline float* data() {
        return _distdata.frame ;
    }

    inline RealSenseDistStruct & getFrame () {
        return _distdata ;
    }

    friend std::istream& operator >> ( std::istream& in, RealSenseDistFrame&  )
    {
        return in;
    }

    friend std::ostream& operator << ( std::ostream& out, const RealSenseDistFrame&  )
    {
        return out;
    }
} ;

class RealSenseDistFrameExporter : public core::objectmodel::BaseObject
{

public:
    SOFA_CLASS( RealSenseDistFrameExporter , core::objectmodel::BaseObject);
    typedef core::objectmodel::BaseObject Inherited;

    Data<std::string>  d_filename ;
    Data<RealSenseDistFrame>  d_distframe ;

    Data<int> d_fpf; // frame per file

    DataCallback c_distframe ;
    DataCallback c_filename ;

    std::FILE* filestream ;
    size_t frame_count, file_id ;

    RealSenseDistFrameExporter()
        : d_filename (initData(&d_filename, "filename", "output filename"))
        , d_distframe (initData(&d_distframe, "distframe", "link to distFrame data"))
        , d_fpf(initData(&d_fpf, 18000, "fpf", "frame per file"))
    {
        c_distframe.addInput(&d_distframe);
        c_distframe.addCallback(std::bind(&RealSenseDistFrameExporter::saveFrame, this));
        filestream = nullptr ;
        frame_count = 0 ;
        file_id = 0 ;
    }

    ~RealSenseDistFrameExporter() {
        std::fclose(filestream) ;
    }

    void updateFileStream () {
        if (filestream == nullptr) {
            std::string filename = processFileName() ;
            filestream = std::fopen(filename.c_str(), "wb") ;
            frame_count = 0 ;
        }
    }

    void saveFrameToStream() {
        RealSenseDistFrame distframe = d_distframe.getValue() ;
        RealSenseDistFrame::RealSenseDistStruct diststruct = distframe.getFrame();

        // write width and height
        std::fwrite(&diststruct._width, sizeof(size_t), 1, filestream) ;
        std::fwrite(&diststruct._height, sizeof(size_t), 1, filestream) ;

        // write frame data
        std::fwrite (
            diststruct.frame,
            sizeof(float),
            diststruct._width * diststruct._height,
            filestream
        ) ;
//        dump frame
//        std::cout << "w/h" << diststruct._width << " " << diststruct._height << std::endl
//                  << "##" << diststruct.frame[0] << std::endl ; ;
//        for (int i = 0 ; i < diststruct._width * diststruct._height ; ++i) {
//            std::cout << diststruct.frame[i] << ", " ;
//        }
//        std::cout << std::endl ;
    }

    void saveFrame () {
        updateFileStream();
        if (filestream == nullptr) {
            std::cerr << "stream is unopened. check rights on file" << std::endl ;
            return ;
        }

        saveFrameToStream();

        if (++frame_count >= d_fpf.getValue()) {
            std::fclose(filestream) ;
            filestream = nullptr ;
        }
    }

    std::string processFileName () {
        std::string
            extension = d_filename.getValue() , // starts as whole filename but ends up as extension
            delimiter = "." ;
        std::string filename = extension.substr(0, extension.find_last_of(delimiter)) ;
        extension.erase(0, extension.find_last_of(delimiter)) ;
        return filename + std::to_string(++file_id) + extension ;

    }
} ;

class RealSenseDistFrameStreamer : public opencvplugin::streamer::BaseOpenCVStreamer
{

public:
    SOFA_CLASS( RealSenseDistFrameStreamer, opencvplugin::streamer::BaseOpenCVStreamer);
    typedef opencvplugin::streamer::BaseOpenCVStreamer Inherited;

    Data<std::string>  d_filename ;
    Data<RealSenseDistFrame>  d_distframe ;

    DataCallback c_filename ;
    std::FILE* filestream ;

    RealSenseDistFrameStreamer()
        : Inherited()
        , d_filename (initData(&d_filename, "filename", "output filename"))
        , d_distframe (initData(&d_distframe, "distframe", "link to distFrame data"))
    {
        c_filename.addInput(&d_filename);
        c_filename.addCallback(std::bind(&RealSenseDistFrameStreamer::updateFileStream, this));
        filestream = nullptr ;
    }

    virtual void decodeImage(cv::Mat & /*img*/) {
        readFrame();
    }

    void updateFileStream () {
        if (filestream == nullptr) {
            filestream = std::fopen(d_filename.getValue().c_str(), "rb") ;
        }
    }

    void readFrame () {
        updateFileStream();
        if (filestream == nullptr) {
            std::cerr << "stream is unopened. check stream state before passing to function" << std::endl ;
            return ;
        }

        RealSenseDistFrame::RealSenseDistStruct diststruct ;
        // write width and height
        std::fread(&diststruct._width, sizeof(size_t), 1, filestream) ;
        std::fread(&diststruct._height, sizeof(size_t), 1, filestream) ;

        // write frame data
        diststruct.frame = new float[diststruct._width * diststruct._height] ;
        std::fread (
            diststruct.frame,
            sizeof(float),
            diststruct._width * diststruct._height,
            filestream
        ) ;
        RealSenseDistFrame distFrm (diststruct) ;
        d_distframe.setValue(distFrm);

//        dump frame
//        std::cout << "w/h" << diststruct._width << " " << diststruct._height << std::endl
//                  << "##" << diststruct.frame[0] << std::endl ;
//        std::cout <<"frame=" << std::endl;
//        for (int i = 0 ; i < diststruct._width * diststruct._height ; ++i) {
//            std::cout << diststruct.frame[i] << "," ;
//        }
//        std::cout << std::endl ;

    }
} ;

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

    Data<RealSenseDistFrame> d_distframe ;
    Data<std::string> d_intrinsics ;
    DataCallback c_intrinsics ;

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

        //offline reco
        , d_distframe(initData(&d_distframe, "distframe", "frame encoding pixel's distance from camera. used for offline deprojection"))
        , d_intrinsics(initData(&d_intrinsics, std::string("intrinsics.log"), "intrinsics", "path to realsense intrinsics file to read from"))

        // needed for online reco
        , l_rs_cam(initLink("rscam", "link to realsense camera component - used for getting camera intrinsics"))
        , m_pointcloud(new pcl::PointCloud<pcl::PointXYZ>)
        , m_cloud_normals(new pcl::PointCloud<pcl::Normal>)
    {
        c_image.addInputs({&d_color, &d_depth});
        c_image.addCallback(std::bind(&RealSenseDeprojector::deproject_image, this));

        c_intrinsics.addInput({&d_intrinsics});
        c_intrinsics.addCallback(std::bind(&RealSenseDeprojector::readIntrinsics, this));
        readIntrinsics();
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


    void readIntrinsics () {
        std::FILE* filestream = std::fopen("/home/omar/projects/sofa/build/bin/intrinsics.log", "rb") ;
        if (filestream == NULL) {
            std::cout << "Check rights on intrins.log file" << std::endl ;
            return ;
        }
        std::fread(&cam_intrinsics.width, sizeof(int), 1, filestream) ;
        std::fread(&cam_intrinsics.height, sizeof(int), 1, filestream) ;
        std::fread(&cam_intrinsics.ppx, sizeof(float), 1, filestream) ;
        std::fread(&cam_intrinsics.ppy, sizeof(float), 1, filestream) ;
        std::fread(&cam_intrinsics.fx, sizeof(float), 1, filestream) ;
        std::fread(&cam_intrinsics.fy, sizeof(float), 1, filestream) ;
        std::fread(&cam_intrinsics.model, sizeof(rs2_distortion), 1, filestream) ;
        std::fread(cam_intrinsics.coeffs, sizeof(float), 5, filestream) ;
        std::fclose(filestream) ;
    }

    void deproject_image_offline () {
        cv::Mat depth_im = d_depth.getValue().getImage() ;
        RealSenseDistFrame distframe = d_distframe.getValue() ;
        RealSenseDistFrame::RealSenseDistStruct diststruct = distframe.getFrame() ;
        int downSample = d_downsampler.getValue() ;

        if (diststruct._width != (size_t)(depth_im.cols/downSample) ||
            diststruct._height != (size_t)(depth_im.rows/downSample)) {
            return ;
        }

        m_pointcloud->clear();
        for (size_t i = 0 ; i < depth_im.rows/downSample ; ++i) {
            for (size_t j = 0 ; j < depth_im.rows/downSample ; ++j) {
                if (depth_im.at<const uchar>(downSample*i,downSample*j) > 0) {
                    // deprojection
                    float dist = diststruct.frame[i*diststruct._width+j] ;
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
                    // add units to result
                    m_pointcloud->push_back(pclpoint);
                }
            }
        }

    }

    void deproject_image () {
        if (!l_rs_cam) {
        // we need a valid link to realsense cam sofa component
            deproject_image_offline () ;
            return ;
        }
        // get intrinsics from link to rs-cam component
        rs2::depth_frame depth = *l_rs_cam->depth ;
        cam_intrinsics = l_rs_cam->cam_intrinsics ;

        // get depth
        cv::Mat depth_im = d_depth.getValue().getImage() ;

        // setup output
        m_pointcloud->clear();

        int downSample = d_downsampler.getValue() ;
        RealSenseDistFrame::RealSenseDistStruct & diststruct = *d_distframe.beginEdit();
        diststruct._width = depth_im.cols/downSample ;
        diststruct._height = depth_im.rows/downSample ;
        diststruct.frame = new float[
            depth_im.cols/downSample *
            depth_im.rows/downSample
        ] ;
        for (size_t i = 0 ; i < diststruct._height; ++i) {
            for (size_t j = 0 ; j < diststruct._width ; ++j) {
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
                    // set
                    diststruct.frame[i*diststruct._width+j] = dist ;
                    // set units
                    pcl::PointXYZ pclpoint = pcl::PointXYZ(point3d[1], point3d[0], point3d[2]) ;
                    // add units to result
                    m_pointcloud->push_back(pclpoint);
                }
            }
        }
        d_distframe.endEdit();

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

