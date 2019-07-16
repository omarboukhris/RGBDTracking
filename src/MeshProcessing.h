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

#ifndef SOFA_RGBDTRACKING_MESHPROCESSING_H
#define SOFA_RGBDTRACKING_MESHPROCESSING_H

#include <opencv/cv.h>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <sofa/core/core.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/core/objectmodel/Link.h>
#include <sofa/helper/gl/FrameBufferObject.h>

#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/helper/accessor.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/vector.h>
#include <SofaBaseTopology/TopologyData.h>
#include <RGBDTracking/config.h>
#include <algorithm>    // std::max

#include <stdlib.h>
#include <vector>
#include <math.h>
#include <sys/times.h>

#define GL_GLEXT_PROTOTYPES 1
#define GL4_PROTOTYPES 1
#include <GL/glew.h>
#include <GL/freeglut.h>
//#include <GL/glext.h>
#include <GL/glu.h>

#include <boost/thread.hpp>

#include <image/ImageTypes.h>
#include "RenderingManager.h"

using namespace std;
using namespace cv;

using namespace sofa::defaulttype;
using namespace sofa::component::topology;

namespace sofa {

namespace rgbdtracking {

template<class DataTypes>
class MeshProcessing : public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(MeshProcessing,DataTypes),sofa::core::objectmodel::BaseObject);
	
    typedef sofa::core::objectmodel::BaseObject Inherit;

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef Data<typename DataTypes::VecCoord> DataVecCoord;
    typedef Data<typename DataTypes::VecDeriv> DataVecDeriv;
    typedef sofa::defaulttype::Vector4 Vector4;
    typedef sofa::defaulttype::Vector2 Vec2;
	
    typename core::behavior::MechanicalState<DataTypes> *mstate;

    core::objectmodel::SingleLink<
        MeshProcessing<DataTypes>,
        RenderingManager,
        BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> l_renderingmanager ;

    Eigen::Matrix3f rgbIntrinsicMatrix;
    cv::Mat depthrend, color, depthMap;
    cv::Rect rectRtt;
    int hght;
    int wdth;

    // source mesh data
    Data<Vector4> cameraIntrinsicParameters;

//    Data< VecCoord > sourceSurfacePositions;
//    Data< VecCoord > sourcePositions;
    //output
    Data< VecCoord > sourceVisiblePositions;
    Data<helper::vector< bool > > sourceVisible;  // flag ignored vertices
    Data< helper::vector< bool > > sourceBorder;


    Data< helper::vector<int> > indicesVisible;
    Data< VecCoord > sourceContourPositions;
    Data< helper::vector< Vec2 > > sourceContourNormals;
    Data< helper::vector< double > > sourceWeights;

    Data<Real> sigmaWeight;
    Data<bool> useContour;
    Data<bool> useVisible;
    Data<Real> visibilityThreshold;
    Data<int> niterations;

    Data<int> borderThdSource;
    Data<Vector4> BBox;
    Data<bool> drawVisibleMesh;
    Data<bool> useSIFT3D;

    // unused
//    Data< helper::vector< tri > > sourceTriangles;
//    Data< VecCoord > sourceNormals;
//    Data< VecCoord > sourceSurfaceNormals;
//    Data< VecCoord > sourceSurfaceNormalsM;
//    vector< bool > sourceIgnored;  // flag ignored vertices
//    vector< bool > sourceSurface;

    MeshProcessing();
    virtual ~MeshProcessing();

    void init();
    void handleEvent(sofa::core::objectmodel::Event *event);
    void draw(const core::visual::VisualParams* vparams);

//    VecCoord getSourcePositions(){return sourcePositions.getValue();}
//    VecCoord getSourceVisiblePositions(){return sourceVisiblePositions.getValue();}
//    VecCoord getSourceContourPositions(){return sourceContourPositions.getValue();}

private :
    // step 1 : case 1
    void extractSourceContour();

    // step 1 : case 2 #1
    void getSourceVisible(double znear, double zfar);
    // step 1 : case 2 #2
    void extractSourceVisibleContour();
    void extractSourceSIFT3D();

    //step 2 : case 1
    void updateSourceVisibleContour();
    // step 2 : case 2
    void updateSourceVisible();
    // /!\ unused
    void updateSourceSurface(); // built k-d tree and identify border vertices


};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(MeshProcessing_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_RGBDTRACKING_API MeshProcessing<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_RGBDTRACKING_API MeshProcessing<defaulttype::Vec3fTypes>;
#endif
#endif


} //

} // namespace sofa

#endif
