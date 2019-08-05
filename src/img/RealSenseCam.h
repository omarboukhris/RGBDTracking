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
#ifndef SOFA_IMAGE_REALSENSECAM_H
#define SOFA_IMAGE_REALSENSECAM_H


//#include <image/config.h>
//#include "ImageTypes.h"
#include <CImgPlugin/CImgData.h>
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
#include <iostream>
#include <string>
#include <map>
#include <boost/thread.hpp>
#include <sys/times.h>

#include <fstream>
#include <algorithm>
#include <cstring>

#include <chrono>
#include <thread>

#ifdef Success
  #undef Success
#endif

namespace sofa
{

namespace component
{

namespace container
{

using namespace cimg_library;
using defaulttype::Vec;
using defaulttype::Vector3;


using namespace std;
using namespace cv;
using namespace boost;
using namespace rs2;


class RealSenseCam : public virtual core::objectmodel::BaseObject
{
public:
    typedef core::objectmodel::BaseObject Inherited;
    SOFA_CLASS( RealSenseCam , Inherited);

    // image data
    typedef defaulttype::ImageUC ImageTypes;
    typedef ImageTypes::T T;
    typedef ImageTypes::imCoord imCoord;
    typedef helper::WriteAccessor<Data< ImageTypes > > waImage;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    Data< ImageTypes > imageO;

    // transform data
    typedef SReal Real;
    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef helper::WriteAccessor<Data< TransformType > > waTransform;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    Data< TransformType > transform;

    // depth data
    typedef defaulttype::ImageF DepthTypes;
    typedef DepthTypes::T dT;
    typedef DepthTypes::imCoord dCoord;
    typedef helper::WriteAccessor<Data< DepthTypes > > waDepth;
    typedef helper::ReadAccessor<Data< DepthTypes > > raDepth;
    Data< DepthTypes > depthImage;
    Data< TransformType > depthTransform;

    Data<helper::OptionsGroup> resolution;
    Data<int> depthMode;
    Data<bool> drawBB;
    Data<float> showArrowSize;
    Data<int> depthScale;

cv::Mat_<cv::Vec3f> points3d;

int niterations;

// Declare depth colorizer for pretty visualization of depth data
rs2::colorizer color_map;

// Declare RealSense pipeline, encapsulating the actual device and sensors
rs2::pipeline pipe;
// Start streaming with default recommended configuration


// Using the context to create a rs2::align object.
// rs2::align allows you to perform aliment of depth frames to others

public:

RealSenseCam();// : Inherited();
virtual void clear();
virtual ~RealSenseCam();

virtual std::string getTemplateName() const	{ return templateName(this); }
static std::string templateName(const RealSenseCam* = NULL) {	return std::string(); }


virtual void init();
void initRaw();
void initAlign();


protected:

void acquireRaw()
{

    rs2::frameset data;

    data = pipe.wait_for_frames(); // Wait for next set of frames from the camera

    // Trying to get both color and depth frames
     rs2::video_frame color = data.get_color_frame();
     rs2::depth_frame depth = data.get_depth_frame();

    // Create depth image
    if (depth && color){

     int widthd, heightd, widthc, heightc;

     widthc = color.get_width();
     heightc = color.get_height();

     cv::Mat rgb0(heightc,widthc, CV_8UC3, (void*) color.get_data());

     widthd = depth.get_width();
     heightd = depth.get_height();

     cv::Mat depth16,depth32;

     cv::Mat depth160( heightd, widthd, CV_16U, (void*)depth.get_data() );
     depth16=depth160.clone();

     depth16.convertTo(depth32, CV_32F,(float)1/8190);

     widthc = color.get_width();
     heightc = color.get_height();

    //std::cout << " widthc " << widthc << " heigthc " << heightc << std::endl;
    //std::cout << " widthd " << widthd << " heigthd " << heightd << std::endl;

     //depth8u = depth16;
     //depth8u.convertTo( depth8u, CV_8UC1, 255.0/1000);


    // Read the color buffer and display
    int32_t w, h, w_depth, h_depth;

    w = widthc;
    h = heightc;
    w_depth = widthd;
    h_depth = heightd;
    waImage wimage(this->imageO);
    CImg<T>& img =wimage->getCImg(0);
    //img.resize(widthc,heightc,1,3);

    waDepth wdepth(this->depthImage);
    waTransform wdt(this->depthTransform);
    CImg<dT>& depthimg =wdepth->getCImg(0);
    //depthimg.resize(widthd,heightd,1,1);

    cv::Mat bgr_image;
    cvtColor (rgb0, bgr_image, cv::COLOR_RGB2BGR);

            if(img.spectrum()==3)
            {
            unsigned char* rgb0 = (unsigned char*)bgr_image.data;
            //unsigned char* rgb0 = (unsigned char*)color.get_data();
            unsigned char *ptr_r = img.data(0,0,0,2), *ptr_g = img.data(0,0,0,1), *ptr_b = img.data(0,0,0,0);
            for ( int siz = 0 ; siz<img.width()*img.height(); siz++)    { *(ptr_r++) = *(rgb0++); *(ptr_g++) = *(rgb0++); *(ptr_b++) = *(rgb0++);
                        }
            }
            else memcpy(img.data(),  bgr_image.data, img.width()*img.height()*sizeof(T));

    waImage wimage1(this->imageO);
    CImg<T>& img1 =wimage1->getCImg(0);

    memcpy(depthimg.data(), (float*)depth32.data , w_depth*h_depth*sizeof(float));

    std::cout << " ok acquire raw " << std::endl;

}

}

void acquireAligned()
{
    rs2::align align(RS2_STREAM_COLOR);

    rs2::frameset frameset;

    //double timeAcq0 = (double)getTickCount();

    while (!frameset.first_or_default(RS2_STREAM_DEPTH) || !frameset.first_or_default(RS2_STREAM_COLOR))
    {
        frameset = pipe.wait_for_frames();
    }

    auto processed = align.process(frameset);

   //double timeAcq1 = (double)getTickCount();
   //cout <<"time process frames " << (timeAcq1 - timeAcq0)/getTickFrequency() << endl;
   // Trying to get both color and aligned depth frames
    rs2::video_frame color = processed.get_color_frame();
    rs2::depth_frame depth = processed.get_depth_frame();

    if (depth && color)
    {
    double timePCD = (double)getTickCount();


    int widthd, heightd, widthc, heightc;

    widthc = color.get_width();
    heightc = color.get_height();

    cv::Mat rgb0(heightc,widthc, CV_8UC3, (void*) color.get_data());
    cv::Mat depth16, depth32,depth8u;
    //cv::imwrite("rgb12.png", rgb0);

    // Create depth image
    widthd = depth.get_width();
    heightd = depth.get_height();

    cv::Mat depth160( heightd, widthd, CV_16U, (void*)depth.get_data() );

    depth160.convertTo(depth32, CV_32F,(float)1/8190*depthScale.getValue());
    // Read the color buffer and display
    int32_t w, h, w_depth, h_depth;

    w = widthc;
    h = heightc;
    w_depth = widthd;
    h_depth = heightd;

    waImage wimage(this->imageO);
    CImg<T>& img =wimage->getCImg(0);
    img.resize(widthc,heightc,1,3);

    waDepth wdepth(this->depthImage);
    waTransform wdt(this->depthTransform);
    CImg<dT>& depthimg =wdepth->getCImg(0);
    depthimg.resize(widthd,heightd,1,1);

    //cv::Mat bgr_image;
    //cvtColor (rgb0, bgr_image, CV_RGB2BGR);

    /*depth160.convertTo( depth8u, CV_8UC1, 255.0/1000 );
    cv::imwrite("bgr0.png", depth8u);*/

        if(img.spectrum()==3)
        {
        //unsigned char* rgb0 = (unsigned char*)bgr_image.data;
        //unsigned char* rgb = (unsigned char*)color.get_data();
        unsigned char* rgb1 = (unsigned char*)rgb0.data;
        unsigned char *ptr_b = img.data(0,0,0,2), *ptr_g = img.data(0,0,0,1), *ptr_r = img.data(0,0,0,0);
        for ( int siz = 0 ; siz<img.width()*img.height(); siz++)    { *(ptr_r++) = *(rgb1++); *(ptr_g++) = *(rgb1++); *(ptr_b++) = *(rgb1++);
                    }
        }
        else memcpy(img.data(),  rgb0.data, img.width()*img.height()*sizeof(T));

    //memcpy(depthimg.data(), (ushort*)depth160.data , w_depth*h_depth*sizeof(ushort));
    memcpy(depthimg.data(), (float*)depth32.data , w_depth*h_depth*sizeof(float));
    timePCD = ((double)getTickCount() - timePCD)/getTickFrequency();
    //boost::this_thread::sleep( boost::posix_time::milliseconds(10) );
    std::cout << " TIME ACQUIRE " << timePCD << std::endl;
    }

}

void handleEvent(sofa::core::objectmodel::Event *event)
{
if (dynamic_cast<simulation::AnimateEndEvent*>(event))
{
       if(this->depthMode.getValue()==0) acquireRaw();
        else acquireAligned();
}

}

void getCorners(Vec<8,Vector3> &c) // get image corners
    {
        raDepth rimage(this->depthImage);
        const imCoord dim= rimage->getDimensions();

        Vec<8,Vector3> p;
        p[0]=Vector3(-0.5,-0.5,-0.5);
        p[1]=Vector3(dim[0]-0.5,-0.5,-0.5);
        p[2]=Vector3(-0.5,dim[1]-0.5,-0.5);
        p[3]=Vector3(dim[0]-0.5,dim[1]-0.5,-0.5);
        p[4]=Vector3(-0.5,-0.5,dim[2]-0.5);
        p[5]=Vector3(dim[0]-0.5,-0.5,dim[2]-0.5);
        p[6]=Vector3(-0.5,dim[1]-0.5,dim[2]-0.5);
        p[7]=Vector3(dim[0]-0.5,dim[1]-0.5,dim[2]-0.5);

        raTransform rtransform(this->depthTransform);
        for(unsigned int i=0; i<p.size(); i++) c[i]=rtransform->fromImage(p[i]);

        //std::cout << " c0 " << c[0] << std::endl;
    }

    virtual void computeBBox(const core::ExecParams*  params )
    {
        if (!drawBB.getValue()) return;
        Vec<8,Vector3> c;
        getCorners(c);

        Real bbmin[3]  = {c[0][0],c[0][1],c[0][2]} , bbmax[3]  = {c[0][0],c[0][1],c[0][2]};
        for(unsigned int i=1; i<c.size(); i++)
            for(unsigned int j=0; j<3; j++)
            {
                if(bbmin[j]>c[i][j]) bbmin[j]=c[i][j];
                if(bbmax[j]<c[i][j]) bbmax[j]=c[i][j];
            }
        this->f_bbox.setValue(params,sofa::defaulttype::TBoundingBox<Real>(bbmin,bbmax));
    }

void draw(const core::visual::VisualParams* vparams)
    {
        /*glPushAttrib( GL_LIGHTING_BIT | GL_ENABLE_BIT | GL_LINE_BIT );
        glPushMatrix();

        if (drawBB.getValue())
        {
            const float color[]= {1.,0.5,0.5,0.}, specular[]= {0.,0.,0.,0.};
            glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,color);
            glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,specular);
            glMaterialf(GL_FRONT_AND_BACK,GL_SHININESS,0.0);
            glColor4fv(color);
            glLineWidth(2.0);

            Vec<8,Vector3> c;
            getCorners(c);
            glBegin(GL_LINE_LOOP);	glVertex3d(c[0][0],c[0][1],c[0][2]); glVertex3d(c[1][0],c[1][1],c[1][2]); glVertex3d(c[3][0],c[3][1],c[3][2]); glVertex3d(c[2][0],c[2][1],c[2][2]);	glEnd ();
            glBegin(GL_LINE_LOOP);  glVertex3d(c[0][0],c[0][1],c[0][2]); glVertex3d(c[4][0],c[4][1],c[4][2]); glVertex3d(c[6][0],c[6][1],c[6][2]); glVertex3d(c[2][0],c[2][1],c[2][2]);	glEnd ();
            glBegin(GL_LINE_LOOP);	glVertex3d(c[0][0],c[0][1],c[0][2]); glVertex3d(c[1][0],c[1][1],c[1][2]); glVertex3d(c[5][0],c[5][1],c[5][2]); glVertex3d(c[4][0],c[4][1],c[4][2]);	glEnd ();
            glBegin(GL_LINE_LOOP);	glVertex3d(c[1][0],c[1][1],c[1][2]); glVertex3d(c[3][0],c[3][1],c[3][2]); glVertex3d(c[7][0],c[7][1],c[7][2]); glVertex3d(c[5][0],c[5][1],c[5][2]);	glEnd ();
            glBegin(GL_LINE_LOOP);	glVertex3d(c[7][0],c[7][1],c[7][2]); glVertex3d(c[5][0],c[5][1],c[5][2]); glVertex3d(c[4][0],c[4][1],c[4][2]); glVertex3d(c[6][0],c[6][1],c[6][2]);	glEnd ();
            glBegin(GL_LINE_LOOP);	glVertex3d(c[2][0],c[2][1],c[2][2]); glVertex3d(c[3][0],c[3][1],c[3][2]); glVertex3d(c[7][0],c[7][1],c[7][2]); glVertex3d(c[6][0],c[6][1],c[6][2]);	glEnd ();
        }

        glPopMatrix ();
        glPopAttrib();*/
    }
};


RealSenseCam::RealSenseCam() : Inherited()
        , depthImage(initData(&depthImage,DepthTypes(),"depthImage","depth map"))
        , depthTransform(initData(&depthTransform, TransformType(), "depthTransform" , ""))
        , imageO(initData(&imageO,ImageTypes(),"image","image"))
        , transform(initData(&transform, TransformType(), "transform" , ""))
        , resolution ( initData ( &resolution,"resolution","resolution" ))
        , depthMode ( initData ( &depthMode,1,"depthMode","depth mode" ))
        , drawBB(initData(&drawBB,false,"drawBB","draw bounding box"))
        , depthScale(initData(&depthScale,1,"depthScale","scale for the depth values, 1 for SR300, 10 for 435"))
    {
        this->addAlias(&imageO, "inputImage");
        this->addAlias(&transform, "inputTransform");
        transform.setGroup("Transform");
        depthTransform.setGroup("Transform");
        f_listening.setValue(true);  // to update camera during animate
        drawBB = false;
    }


void RealSenseCam::clear()
    {
        waImage wimage(this->imageO);
        wimage->clear();
        waDepth wdepth(this->depthImage);
        wdepth->clear();
    }

RealSenseCam::~RealSenseCam()
    {
    clear();
    }

/*----------------------------------------------------------------------------*/

void RealSenseCam::initRaw()
{

    pipe.start();
    rs2::frameset data;
    for (int i = 0; i < 100 ; i++)
    data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
    rs2::video_frame depth = data.get_depth_frame();
    rs2::video_frame color = data.get_color_frame();
    while(!depth || !color)
    {
    data = pipe.wait_for_frames(); // Wait for next set of frames from the camera

    depth = data.get_depth_frame(); // Find and colorize the depth data
    color = data.get_color_frame(); // Find the color data
    }

    int widthd = depth.get_width();
    int heightd = depth.get_height();

    int widthc = color.get_width();
    int heightc = color.get_height();

    //std::cout << " widthc " << widthc << " heigthc " << heightc << std::endl;
    //std::cout << " widthd " << widthd << " heigthd " << heightd << std::endl;

    waDepth wdepth(this->depthImage);
    waTransform wdt(this->depthTransform);
    if(wdepth->isEmpty()) wdepth->getCImgList().push_back(CImg<dT>());
    CImg<dT>& depthimg=wdepth->getCImg(0);
    depthimg.resize(widthd,heightd,1,1);

    wdt->setCamPos((Real)(wdepth->getDimensions()[0]-1)/2.0,(Real)(wdepth->getDimensions()[1]-1)/2.0); // for perspective transforms
    wdt->update(); // update of internal data

     waImage wimage(this->imageO);
     waTransform wit(this->transform);
    if(wimage->isEmpty()) wimage->getCImgList().push_back(CImg<T>());
    CImg<T>& img = wimage->getCImg(0);
    img.resize(widthc,heightc,1,3);

    wit->setCamPos((Real)(wimage->getDimensions()[0]-1)/2.0,(Real)(wimage->getDimensions()[1]-1)/2.0); // for perspective transforms
    wit->update(); // update of internal data

    cv::Mat rgb,depth8u,depth16, depth32;


    // Create depth image

    cv::Mat depth160( heightd, widthd, CV_16U, (void*)depth.get_data() );
    depth16=depth160.clone();
    depth16.convertTo(depth32, CV_32F, (float)1/8190);
    cv::Mat rgb0(heightc,widthc, CV_8UC3, (void*) color.get_data());

    // Read the color buffer and display
    int32_t w, h, w_depth, h_depth;

    w = widthc;
    h = heightc;
    w_depth = widthd;
    h_depth = heightd;

    cv::Mat bgr_image;
    cvtColor (rgb0, bgr_image, cv::COLOR_RGB2BGR);

    //depth16.convertTo( depth8u, CV_8UC1, 255.0/1000 );
    //cv::imwrite("bgr.png", depth8u);

            if(img.spectrum()==3)
            {
            unsigned char* rgb0 = (unsigned char*)bgr_image.data;
            //unsigned char* rgb = (unsigned char*)color.get_data();
            unsigned char *ptr_r = img.data(0,0,0,2), *ptr_g = img.data(0,0,0,1), *ptr_b = img.data(0,0,0,0);
            for ( int siz = 0 ; siz<img.width()*img.height(); siz++)    { *(ptr_r++) = *(rgb0++); *(ptr_g++) = *(rgb0++); *(ptr_b++) = *(rgb0++);
                        }
            }
            else memcpy(img.data(),  bgr_image.data, img.width()*img.height()*sizeof(T));

    memcpy(depthimg.data(), (float*)depth32.data , w_depth*h_depth*sizeof(float));

    std::cout << " ok init raw " << std::endl;

}

void RealSenseCam::initAlign()
{
    pipe.start();
    rs2::align align(RS2_STREAM_COLOR);
    rs2::frameset frameset;

        for (int it= 0; it < 100 ; it++)
        frameset = pipe.wait_for_frames();

        while ((!frameset.first_or_default(RS2_STREAM_DEPTH) || !frameset.first_or_default(RS2_STREAM_COLOR)))
        {
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

    waDepth wdepth(this->depthImage);
    waTransform wdt(this->depthTransform);
    if(wdepth->isEmpty()) wdepth->getCImgList().push_back(CImg<dT>());
    CImg<dT>& depthimg=wdepth->getCImg(0);
    depthimg.resize(widthd,heightd,1,1);

    wdt->setCamPos((Real)(wdepth->getDimensions()[0]-1)/2.0,(Real)(wdepth->getDimensions()[1]-1)/2.0); // for perspective transforms
    wdt->update(); // update of internal data

     waImage wimage(this->imageO);
     waTransform wit(this->transform);
    if(wimage->isEmpty()) wimage->getCImgList().push_back(CImg<T>());
    CImg<T>& img = wimage->getCImg(0);
    img.resize(widthc,heightc,1,3);

    wit->setCamPos((Real)(wimage->getDimensions()[0]-1)/2.0,(Real)(wimage->getDimensions()[1]-1)/2.0); // for perspective transforms
    wit->update(); // update of internal data
    cv::Mat rgb,depth8u,depth16, depth32;

    // Create depth image

    cv::Mat depth160( heightd, widthd, CV_16U, (void*)depth.get_data() );
    depth16=depth160.clone();
    depth16.convertTo(depth32, CV_32F, (float)1/8190);
    cv::Mat rgb0(heightc,widthc, CV_8UC3, (void*) color.get_data());

    // Read the color buffer and display
    int32_t w, h, w_depth, h_depth;

    w = widthc;
    h = heightc;
    w_depth = widthd;
    h_depth = heightd;

    cv::Mat bgr_image;
    cvtColor (rgb0, bgr_image, cv::COLOR_RGB2BGR);

    //depth16.convertTo( depth8u, CV_8UC1, 255.0/1000 );
    //cv::imwrite("bgr.png", depth8u);

            if(img.spectrum()==3)
            {
            unsigned char* rgb0 = (unsigned char*)bgr_image.data;
            //unsigned char* rgb = (unsigned char*)color.get_data();
            unsigned char *ptr_r = img.data(0,0,0,2), *ptr_g = img.data(0,0,0,1), *ptr_b = img.data(0,0,0,0);
            for ( int siz = 0 ; siz<img.width()*img.height(); siz++)    { *(ptr_r++) = *(rgb0++); *(ptr_g++) = *(rgb0++); *(ptr_b++) = *(rgb0++);
                        }
            }
            else memcpy(img.data(),  bgr_image.data, img.width()*img.height()*sizeof(T));

    memcpy(depthimg.data(), (float*)depth32.data , w_depth*h_depth*sizeof(float));

    std::cout << " ok init align " << std::endl;


}

void RealSenseCam::init()
{

   if(this->depthMode.getValue()==0) initRaw();
   else initAlign();

}


}

}

}


#endif ///*IMAGE_SOFTKINETIC_H*/
