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

#ifndef SOFA_RGBDTRACKING_PointCloudExtractor_H
#define SOFA_RGBDTRACKING_PointCloudExtractor_H

#include <RGBDTracking/config.h>
//#include <boost/thread.hpp>

#include <sofa/core/core.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/core/objectmodel/Link.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/helper/accessor.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Vec.h>
#include <SofaBaseTopology/TopologyData.h>
#include <sofa/gui/BaseGUI.h>
#include <sofa/gui/BaseViewer.h>
#include <sofa/gui/GUIManager.h>

#define GL_GLEXT_PROTOTYPES 1
#define GL4_PROTOTYPES 1
#include <GL/glew.h>
#include <GL/freeglut.h>
//#include <GL/glext.h>
#include <GL/glu.h>

#include <set>

#ifdef WIN32
#include <process.h>
#else
#include <pthread.h>
#endif

#include <stdlib.h>
#include <vector>
#include <iostream>
#include <string>

#include <sys/times.h>

#include <visp/vpIoTools.h>
#include <visp/vpImageIo.h>
#include <visp/vpParseArgv.h>
#include <visp/vpMatrix.h>
#include <visp/vpKltOpencv.h>

#include <pcl/search/impl/search.hpp>

#include <sofa/opencvplugin/BaseOpenCVComponent.h>

using namespace std;
using namespace cv;


namespace sofa {

namespace rgbdtracking {

using helper::vector;
using namespace sofa::defaulttype;
using namespace sofa::component::topology;

template<class DataTypes>
class PointCloudExtractor : public opencvplugin::BaseOpenCVComponent
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PointCloudExtractor,DataTypes),opencvplugin::BaseOpenCVComponent);

    typedef opencvplugin::BaseOpenCVComponent Inherit;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef Data<typename DataTypes::VecCoord> DataVecCoord;
    typedef Data<typename DataTypes::VecDeriv> DataVecDeriv;
    typedef sofa::defaulttype::Vector4 Vector4;
    typedef sofa::defaulttype::Vector3 Vec3;
    typedef helper::fixed_array <unsigned int,3> tri;
//    typedef defaulttype::ImageF DepthTypes;

    typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud ;

    typedef defaulttype::Mat<
        Vec3dTypes::spatial_dimensions,
        Vec3dTypes::spatial_dimensions,
        Real
    > Mat;

    Data<opencvplugin::ImageData>
        d_foreground,
        d_depth,
        d_color ;
    DataCallback c_image ;

    Data<bool> useCurvature;
    Data<bool> useContour;

    Data<int> niterations;
    Data<int> samplePCD;
    Data<int> borderThdPCD;
    Data<Real> sigmaWeight;


    //display param
    Data<bool> drawPointCloud;
    Data<bool> displayBackgroundImage;

    //outputs
    Data< VecCoord > targetPositions;
    Data< VecCoord > targetNormals;
    Data< VecCoord > targetContourPositions;
    Data< VecCoord > targetGtPositions;
    Data< helper::vector< bool > > targetBorder;
    Data< helper::vector< double > > targetWeights;
    Data< helper::vector< double > > curvatures;

    Data<Vector4> cameraIntrinsicParameters;
    Data<Vec3> cameraPosition;
    Data<Quat> cameraOrientation;
    Data<bool> cameraChanged;

    Data<bool> safeModeSeg;
    Data<double> segTolerance;

    Eigen::Matrix3f rgbIntrinsicMatrix;

    int ntargetcontours;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr target ;


    PointCloudExtractor();
    virtual ~PointCloudExtractor();

    void init();
    void handleEvent(sofa::core::objectmodel::Event *event);

    void extractTargetPCD();
    void extractTargetPCDContour();
    void setCameraPose();

    void draw(const core::visual::VisualParams* vparams) ;
private :
    cv::Mat foreground;
    cv::Mat depth, color ;
    cv::Mat distimage, dotimage;


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr PCDFromRGBD(cv::Mat& depthImage, cv::Mat& rgbImage);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr PCDContourFromRGBD(cv::Mat& depthImage, cv::Mat& rgbImage, cv::Mat& distImage, cv::Mat& dotImage);

    void setDataInput () ;
};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(PointCloudExtractor_CPP)
#ifndef SOFA_FLOAT
    extern template class SOFA_RGBDTRACKING_API PointCloudExtractor<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
    extern template class SOFA_RGBDTRACKING_API PointCloudExtractor<defaulttype::Vec3fTypes>;
#endif
#endif


} // rgbdtracking

} // namespace sofa

#endif
