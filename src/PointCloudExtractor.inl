/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <limits>
#include <iterator>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/gpu/features/features.hpp>
#include <pcl/search/impl/search.hpp>


//#include "DataSource.hpp"

#include <sofa/helper/gl/Color.h>
#include <sofa/core/ObjectFactory.h>
#include <SofaBaseVisual/BaseCamera.h>
#include <SofaBaseVisual/InteractiveCamera.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/simulation/Simulation.h>
#include <sofa/helper/AdvancedTimer.h>

#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/pfh.h>
#include <pcl/features/pfhrgb.h>
#include <pcl/features/3dsc.h>
#include <pcl/features/shot_omp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>

#include <algorithm>

#include "PointCloudExtractor.h"

using std::cerr;
using std::endl;

namespace sofa {

namespace rgbdtracking {

template <class DataTypes>
PointCloudExtractor<DataTypes>::PointCloudExtractor( )
 : Inherit()
    , d_foreground(initData(&d_foreground, "foreground", "data link to segmented foreground"))
    , d_depth(initData(&d_depth, "depth", "data link to depth image"))
    , d_color(initData(&d_color, "color", "data link to rgb image"))

    , useCurvature(initData(&useCurvature,false,"useCurvature"," "))
    , useContour(initData(&useContour,false,"useContour","Emphasize forces close to the target contours"))

    , niterations(initData(&niterations,1,"niterations","Number of iterations in the tracking process"))
    , samplePCD(initData(&samplePCD,4,"samplePCD","Sample step for the point cloud"))
    , borderThdPCD(initData(&borderThdPCD,4,"borderThdPCD","border threshold on the target silhouette"))
    , sigmaWeight(initData(&sigmaWeight,(Real)4,"sigmaWeight","sigma weights"))

    //display params
    , drawPointCloud(initData(&drawPointCloud,false,"drawPointCloud"," "))
    , displayBackgroundImage(initData(&displayBackgroundImage,false,"displayBackgroundImage"," "))

    //output
    , targetPositions(initData(&targetPositions,"targetPositions","Points of the target point cloud."))
    , targetNormals(initData(&targetNormals,"targetNormals","normals of the target point cloud."))
    , targetContourPositions(initData(&targetContourPositions,"targetContourPositions","Contour points of the target point cloud."))
    , targetBorder(initData(&targetBorder,"targetBorder","Border of the target point cloud."))
    , targetWeights(initData(&targetWeights,"targetWeights","weigths of the target point cloud."))
    , curvatures(initData(&curvatures, "curvatures", "output curvatures"))

    , cameraIntrinsicParameters(initData(&cameraIntrinsicParameters,Vector4(),"cameraIntrinsicParameters","camera parameters"))
    , cameraPosition(initData(&cameraPosition,"cameraPosition","Position of the camera w.r.t the point cloud"))
    , cameraOrientation(initData(&cameraOrientation,"cameraOrientation","Orientation of the camera w.r.t the point cloud"))
    , cameraChanged(initData(&cameraChanged,false,"cameraChanged","If the camera has changed or not"))

    , safeModeSeg(initData(&safeModeSeg,false,"safeModeSeg","safe mode when segmentation fails"))
    , segTolerance(initData(&segTolerance,0.5,"segTolerance","tolerance or segmentation"))
{
    this->f_listening.setValue(true);

    c_image.addInputs({&d_foreground, &d_depth, &d_color});
    c_image.addCallback(std::bind(&PointCloudExtractor<DataTypes>::setDataInput, this));

}

template <class DataTypes>
void PointCloudExtractor<DataTypes>::setDataInput () {
    foreground = d_foreground.getValue().getImage() ;
    depth = d_depth.getValue().getImage() ;
    color = d_color.getValue().getImage() ;
}

template <class DataTypes>
PointCloudExtractor<DataTypes>::~PointCloudExtractor()
{
}

template <class DataTypes>
void PointCloudExtractor<DataTypes>::init()
{
    // these lines fuck up the display on sofa
    //(tested with only opencv viewer as a graphical component)
//    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//    glEnable(GL_CULL_FACE);
//    glEnable(GL_DEPTH_TEST);
//    glDepthMask(GL_TRUE);

    Vector4 camParam = cameraIntrinsicParameters.getValue();

    rgbIntrinsicMatrix(0,0) = camParam[0];
    rgbIntrinsicMatrix(1,1) = camParam[1];
    rgbIntrinsicMatrix(0,2) = camParam[2];
    rgbIntrinsicMatrix(1,2) = camParam[3];
}

template <class DataTypes>
pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloudExtractor<DataTypes>::PCDFromRGBD(cv::Mat& depthImage, cv::Mat& rgbImage)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr outputPointcloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    //pcl::PointCloud<pcl::PointXYZRGB> pointcloud;
    outputPointcloud->points.resize(0);

    int sample;

    if (samplePCD.getValue() > 1) {
        sample = samplePCD.getValue();
    } else {
        sample = 2;
    }

    float rgbFocalInvertedX = 1/rgbIntrinsicMatrix(0,0);	// 1/fx
    float rgbFocalInvertedY = 1/rgbIntrinsicMatrix(1,1);	// 1/fy
    pcl::PointXYZRGB newPoint;
    //std::cout << "inrgbpcl" <<std::endl ;
    for (int i=0;i<(int)depthImage.rows/sample;i++) {
        for (int j=0;j<(int)depthImage.cols/sample;j++) {

            float depthValue = (float)depthImage.at<float>(sample*i,sample*j);//*0.819;
            int avalue = (int)rgbImage.at<Vec4b>(sample*i,sample*j)[3];

            if (avalue > 0 && depthValue>0) {
                //std::cout << "IN" << std::endl ;
                // if depthValue is not NaN
                // Find 3D position respect to rgb frame:
                newPoint.z = depthValue;
                newPoint.x = (sample*j - rgbIntrinsicMatrix(0,2)) * newPoint.z * rgbFocalInvertedX;
                newPoint.y = (sample*i - rgbIntrinsicMatrix(1,2)) * newPoint.z * rgbFocalInvertedY;
                newPoint.r = rgbImage.at<cv::Vec4b>(sample*i,sample*j)[2];
                newPoint.g = rgbImage.at<cv::Vec4b>(sample*i,sample*j)[1];
                newPoint.b = rgbImage.at<cv::Vec4b>(sample*i,sample*j)[0];
                outputPointcloud->points.push_back(newPoint);
                //std::cout << "OUT" << std::endl ;
            }
        }
    }
    std::cout << "outrgbpcl.size = " << outputPointcloud->size() <<std::endl ;

    if (useCurvature.getValue()){
        int sample1 = samplePCD.getValue();
        pcl::PointCloud<pcl::PointXYZ>::Ptr outputPointcloud1(new pcl::PointCloud<pcl::PointXYZ>);
        outputPointcloud1->points.resize(0);

        pcl::PointXYZ newPoint1;

        for (int i=0;i<(int)depthImage.rows/sample1;i++) {
            for (int j=0;j<(int)depthImage.cols/sample1;j++) {
                float depthValue = (float)depthImage.at<float>(sample1*i,sample1*j);//*0.819;
                int avalue = (int)rgbImage.at<Vec4b>(sample1*i,sample1*j)[3];
                if (avalue > 0 && depthValue>0) {
                    // if depthValue is not NaN
                        // Find 3D position respect to rgb frame:
                        newPoint1.z = depthValue;
                        newPoint1.x = (sample1*j - rgbIntrinsicMatrix(0,2)) * newPoint.z * rgbFocalInvertedX;
                        newPoint1.y = (sample1*i - rgbIntrinsicMatrix(1,2)) * newPoint.z * rgbFocalInvertedY;
                        outputPointcloud1->points.push_back(newPoint1);
                }
            }
        }


        // Compute the normals
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
        normalEstimation.setInputCloud (outputPointcloud1);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
        normalEstimation.setSearchMethod (tree);
        pcl::PointCloud<pcl::Normal>::Ptr cloudWithNormals (new pcl::PointCloud<pcl::Normal>);
        normalEstimation.setRadiusSearch (0.02);
        normalEstimation.compute (*cloudWithNormals);

        // Setup the principal curvatures computation
        pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> principalCurvaturesEstimation;

        // Provide the original point cloud (without normals)
        principalCurvaturesEstimation.setInputCloud (outputPointcloud1);

        // Provide the point cloud with normals
        principalCurvaturesEstimation.setInputNormals(cloudWithNormals);

        // Use the same KdTree from the normal estimation
        principalCurvaturesEstimation.setSearchMethod (tree);
        principalCurvaturesEstimation.setRadiusSearch(0.03);

        // Actually compute the principal curvatures
        pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principalCurvatures (new pcl::PointCloud<pcl::PrincipalCurvatures> ());
        principalCurvaturesEstimation.compute (*principalCurvatures);

        std::cout << "output points.size (): " << principalCurvatures->points.size () << std::endl;

        // Display and retrieve the shape context descriptor vector for the 0th point.
        std::vector<double> curvs;
        for (int k = 0; k < principalCurvatures->points.size (); k++) {
            pcl::PrincipalCurvatures descriptor0 = principalCurvatures->points[k];
            double curv = abs(descriptor0.pc1*descriptor0.pc1);
            curvs.push_back(curv);
        }

        curvatures.setValue(curvs);
    }
    pcl::io::savePCDFile( "cloud.pcd", *outputPointcloud, true );
    return outputPointcloud;
}


template <class DataTypes>
void PointCloudExtractor<DataTypes>::extractTargetPCD() {

    int t = (int)this->getContext()->getTime();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr targetP ;

    targetP.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    targetP = PCDFromRGBD(depth,foreground);
    VecCoord targetpos;

    if (targetP->size() <= 10) {
        return ;
    }

    target.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    target = targetP;
    targetpos.resize(target->size());

    Vector3 pos;
    for (unsigned int i=0; i<target->size(); i++) {
        pos[0] = (double)target->points[i].x;
        pos[1] = (double)target->points[i].y;
        pos[2] = (double)target->points[i].z;
        targetpos[i]=pos;
        //std::cout << " target " << pos[0] << " " << pos[1] << " " << pos[2] << std::endl;
    }
    const VecCoord&  p = targetpos;

    static int sizeinit = p.size() ;

    if (safeModeSeg.getValue()) {
        bool guard =
            abs((double)p.size() - (double)sizeinit)/(double)sizeinit
            < segTolerance.getValue() ;
        if (t<20*niterations.getValue()) {
            sizeinit = p.size();
            targetPositions.setValue(p);
        } else if (guard) {
            targetPositions.setValue(p);
        }
    } else {
        targetPositions.setValue(p);
    }

}

template <class DataTypes>
pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloudExtractor<DataTypes>::PCDContourFromRGBD(cv::Mat& depthImage, cv::Mat& rgbImage, cv::Mat& distImage, cv::Mat& dotImage) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr outputPointcloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    //pcl::PointCloud<pcl::PointXYZRGB> pointcloud;

    cv::Mat
        frgd = rgbImage,
        distimg = distImage,
        dotimg = dotImage;

    helper::vector<bool> targetborder;
    targetborder.resize(0);

    helper::vector<double> targetweights;
    targetweights.resize(0);

    int sample = samplePCD.getValue();

    float rgbFocalInvertedX = 1/rgbIntrinsicMatrix(0,0);// 1/fx
    float rgbFocalInvertedY = 1/rgbIntrinsicMatrix(1,1);// 1/fy
    ntargetcontours = 0;
    int jj = 0;
    double totalweights = 0;
    pcl::PointXYZRGB newPoint;

    for (int i=0;i<(int)depthImage.rows/sample;i++) {
        for (int j=0;j<(int)depthImage.cols/sample;j++) {
            float depthValue = (float)depthImage.at<float>(sample*i,sample*j);//*0.819;
            int avalue = (int)frgd.at<Vec4b>(sample*i,sample*j)[3];
            int bvalue = (int)distimg.at<uchar>(sample*i,sample*(j));
            int dvalue = (int)dotimg.at<uchar>(sample*i,sample*(j));

            if (dvalue == 0 && depthValue>0) {
                // if depthValue is not NaN
                // Find 3D position respect to rgb frame:
                newPoint.z = depthValue;
                newPoint.x = (sample*j - rgbIntrinsicMatrix(0,2)) * newPoint.z * rgbFocalInvertedX;
                newPoint.y = (sample*i - rgbIntrinsicMatrix(1,2)) * newPoint.z * rgbFocalInvertedY;
                newPoint.r = frgd.at<cv::Vec4b>(sample*i,sample*j)[2];
                newPoint.g = frgd.at<cv::Vec4b>(sample*i,sample*j)[1];
                newPoint.b = frgd.at<cv::Vec4b>(sample*i,sample*j)[0];
                outputPointcloud->points.push_back(newPoint);

                targetweights.push_back((double)exp(-bvalue/sigmaWeight.getValue()));
                totalweights += targetweights[jj++];

                if (avalue > 0 && bvalue < borderThdPCD.getValue()) {
                    targetborder.push_back(true);
                    ntargetcontours++;
                } else {
                    targetborder.push_back(false);
                }
            }
        }
    }

    for (int i=0; i < targetweights.size();i++) {
        targetweights[i]*=((double)targetweights.size()/totalweights);
        //std::cout << " weights " << totalweights << " " << (double)targetweights[i] << std::endl;
    }

    targetWeights.setValue(targetweights);
    targetBorder.setValue(targetborder);

    if (useCurvature.getValue()) {
        int sample1 = samplePCD.getValue();
        pcl::PointCloud<pcl::PointXYZ>::Ptr outputPointcloud1(new pcl::PointCloud<pcl::PointXYZ>);
        outputPointcloud1->points.resize(0);

        pcl::PointXYZ newPoint1;

        for (int i=0;i<(int)depthImage.rows/sample1;i++) {
            for (int j=0;j<(int)depthImage.cols/sample1;j++) {
                float depthValue = (float)depthImage.at<float>(sample1*i,sample1*j);//*0.819;
                int avalue = (int)rgbImage.at<Vec4b>(sample1*i,sample1*j)[3];
                if (avalue > 0 && depthValue>0) {
                    // if depthValue is not NaN
                    // Find 3D position respect to rgb frame:
                    newPoint1.z = depthValue;
                    newPoint1.x = (sample1*j - rgbIntrinsicMatrix(0,2)) * newPoint.z * rgbFocalInvertedX;
                    newPoint1.y = (sample1*i - rgbIntrinsicMatrix(1,2)) * newPoint.z * rgbFocalInvertedY;
                    outputPointcloud1->points.push_back(newPoint1);
                }
            }
        }


        // Compute the normals
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
        normalEstimation.setInputCloud (outputPointcloud1);

        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
        normalEstimation.setSearchMethod (tree);

        pcl::PointCloud<pcl::Normal>::Ptr cloudWithNormals (new pcl::PointCloud<pcl::Normal>);

        normalEstimation.setRadiusSearch (0.02);

        normalEstimation.compute (*cloudWithNormals);

        // Setup the principal curvatures computation
        pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> principalCurvaturesEstimation;

        // Provide the original point cloud (without normals)
        principalCurvaturesEstimation.setInputCloud (outputPointcloud1);

        // Provide the point cloud with normals
        principalCurvaturesEstimation.setInputNormals(cloudWithNormals);

        // Use the same KdTree from the normal estimation
        principalCurvaturesEstimation.setSearchMethod (tree);
        principalCurvaturesEstimation.setRadiusSearch(0.03);

        // Actually compute the principal curvatures
        pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principalCurvatures (new pcl::PointCloud<pcl::PrincipalCurvatures> ());
        principalCurvaturesEstimation.compute (*principalCurvatures);

        std::cout << "output points.size (): " << principalCurvatures->points.size () << std::endl;

        // Display and retrieve the shape context descriptor vector for the 0th point.
        pcl::PrincipalCurvatures descriptor = principalCurvatures->points[0];

        std::vector<double> curvs;

        for (int k = 0; k < principalCurvatures->points.size (); k++) {
            pcl::PrincipalCurvatures descriptor0 = principalCurvatures->points[k];
            double curv = abs(descriptor0.pc1*descriptor0.pc1);
            curvs.push_back(curv);
        }

        curvatures.setValue(curvs);
        //std::cout << " curvature " << descriptor << std::endl;
    }
    //pcl::io::savePCDFile( "cloud.pcd", *outputPointcloud, true ) ;
    return outputPointcloud;
}

template <class DataTypes>
void PointCloudExtractor<DataTypes>::extractTargetPCDContour() {
#define CANNY_TH1 150
#define CANNY_TH2 80
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr targetP ;

    targetP.reset(new pcl::PointCloud<pcl::PointXYZRGB>);


    cv::Mat contour, dist, dist0;

    //cv::imwrite("depthmap.png", seg.distImage);
    cv::Canny(dotimage, contour, CANNY_TH1, CANNY_TH2, 3);
    contour = cv::Scalar::all(255) - contour;
    cv::distanceTransform(contour, dist, CV_DIST_L2, 3);
    dist.convertTo(dist0, CV_8U, 1, 0);
    targetP = PCDContourFromRGBD(depth,foreground, distimage, dotimage);
    VecCoord targetpos;

    if (targetP->size() <= 10) {
        return ;
    }
    target.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    target = targetP;
    targetpos.resize(target->size());

    for (unsigned int i=0; i<target->size(); i++) {
        Vector3 pos (
            (double)target->points[i].x,
            (double)target->points[i].y,
            (double)target->points[i].z
        ) ;
        targetpos[i]=pos;
    }

    VecCoord targetContourpos;
    targetContourpos.resize(ntargetcontours);
    int kk = 0;
    for (unsigned int i=0; i<target->size(); i++) {
        if (targetBorder.getValue()[i]) {
            Vector3 pos (
                (double)target->points[i].x,
                (double)target->points[i].y,
                (double)target->points[i].z
            );
            targetContourpos[kk++]=pos;
        }
    }

    const VecCoord&  p0 = targetpos;
    targetPositions.setValue(p0);
    const VecCoord&  p1 = targetContourpos;
    targetContourPositions.setValue(p1);
//    std::cout << " target contour " << p1.size() << std::endl;

}

template<class DataTypes>
void PointCloudExtractor<DataTypes>::setCameraPose()
{
    pcl::PointCloud<pcl::PointXYZRGB>& point_cloud = *target;
    if (point_cloud.size() > 0) {
        Vec3 cameraposition;
        Quat cameraorientation;
        cameraposition[0] = point_cloud.sensor_origin_[0];
        cameraposition[1] = point_cloud.sensor_origin_[1];
        cameraposition[2] = point_cloud.sensor_origin_[2];
        cameraorientation[0] = point_cloud.sensor_orientation_.w ();
        cameraorientation[1] = point_cloud.sensor_orientation_.x ();
        cameraorientation[2] = point_cloud.sensor_orientation_.y ();
        cameraorientation[3] = point_cloud.sensor_orientation_.z ();

        cameraPosition.setValue(cameraposition);
        cameraOrientation.setValue(cameraorientation);
        cameraChanged.setValue(true);
    }
}

template <class DataTypes>
void PointCloudExtractor<DataTypes>::handleEvent(sofa::core::objectmodel::Event *event) {
    if (dynamic_cast<simulation::AnimateBeginEvent*>(event)) {
        helper::AdvancedTimer::stepBegin("PointCloudExtractor") ;
        static bool initsegmentation = true ;

        if (!useContour.getValue()) {
            extractTargetPCD();
        } else {
            extractTargetPCDContour();
        }
        if (initsegmentation) {
            setCameraPose();
            initsegmentation = false;
        }
        cameraChanged.setValue(false);
        helper::AdvancedTimer::stepEnd("PointCloudExtractor") ;
    }
}

template <class DataTypes>
void PointCloudExtractor<DataTypes>::draw(const core::visual::VisualParams* vparams)
{

//    helper::ReadAccessor< Data< VecCoord > > xtarget(targetPositions);
//    vparams->drawTool()->saveLastState();

//    if (displayBackgroundImage.getValue()) {
//        GLfloat projectionMatrixData[16];
//        glGetFloatv(GL_PROJECTION_MATRIX, projectionMatrixData);
//        GLfloat modelviewMatrixData[16];
//        glGetFloatv(GL_MODELVIEW_MATRIX, modelviewMatrixData);

//        cv::Mat colorrgb = color.clone();
//        if (!color.empty())
//        cv::cvtColor(color, colorrgb, CV_RGB2BGR);

//        std::stringstream imageString;
//        imageString.write((const char*)colorrgb.data, colorrgb.total()*colorrgb.elemSize());
//        // PERSPECTIVE

//        glMatrixMode(GL_PROJECTION);	//init the projection matrix
//        glPushMatrix();
//        glLoadIdentity();
//        glOrtho(0, 1, 0, 1, -1, 1);  // orthogonal view
//        glMatrixMode(GL_MODELVIEW);
//        glPushMatrix();
//        glLoadIdentity();

//        // BACKGROUND TEXTURING
//        //glDepthMask (GL_FALSE);		// disable the writing of zBuffer
//        glDisable(GL_DEPTH_TEST);
//        glEnable(GL_TEXTURE_2D);	// enable the texture
//        glDisable(GL_LIGHTING);		// disable the light

//        glBindTexture(GL_TEXTURE_2D, 0);  // texture bind
//        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, colorrgb.cols, colorrgb.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, imageString.str().c_str());

//        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);	// Linear Filtering

//                                                                            // BACKGROUND DRAWING
//                                                                            //glEnable(GL_DEPTH_TEST);

//        glBegin(GL_QUADS); //we draw a quad on the entire screen (0,0 1,0 1,1 0,1)
//        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
//        glTexCoord2f(0, 1);		glVertex2f(0, 0);
//        glTexCoord2f(1, 1);		glVertex2f(1, 0);
//        glTexCoord2f(1, 0);		glVertex2f(1, 1);
//        glTexCoord2f(0, 0);		glVertex2f(0, 1);
//        glEnd();

//        //glEnable(GL_DEPTH_TEST);
//        glEnable(GL_LIGHTING);		// enable light
//        glDisable(GL_TEXTURE_2D);	// disable texture 2D
//        glEnable(GL_DEPTH_TEST);
//        //glDepthMask (GL_TRUE);		// enable zBuffer

//        glPopMatrix();
//        glMatrixMode(GL_PROJECTION);
//        glPopMatrix();
//        glMatrixMode(GL_MODELVIEW);

//        vparams->drawTool()->restoreLastState();
//    }

//    if (drawPointCloud.getValue() && xtarget.size() > 0){
//        std::vector< sofa::defaulttype::Vector3 > points;
//        sofa::defaulttype::Vector3 point;

//        for (unsigned int i=0; i< xtarget.size(); i++) {
//            points.resize(0);
//            point = DataTypes::getCPos(xtarget[i]);
//            points.push_back(point);
//            // std::cout << curvatures.getValue()[i] << std::endl;
//            //if (targetWeights.getValue().size()>0) vparams->drawTool()->drawPoints(points, 10, sofa::defaulttype::Vec<4,float>(0.5*targetWeights.getValue()[i],0,0,1));
//            vparams->drawTool()->drawPoints(points, 10, sofa::defaulttype::Vec<4,float>(1,0.5,0.5,1));
//        }

//    }

}

} // rgbdtracking

} // namespace sofa

