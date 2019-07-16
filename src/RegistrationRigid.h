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

#ifndef SOFA_RGBDTRACKING_REGISTRATIONRIGID_H
#define SOFA_RGBDTRACKING_REGISTRATIONRIGID_H

#include <RGBDTracking/config.h>
#include <image/ImageTypes.h>
#include <sofa/core/core.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Event.h>

#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>

#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/helper/accessor.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/vector.h>
#include <visp/vpKltOpencv.h>
#include <SofaGeneralEngine/NormalsFromPoints.h>

//#include <sofa/helper/kdTree.inl>
//#include "KalmanFilter.h"

#include <visp/vpDisplayX.h>
#include <algorithm>    

#ifdef WIN32
    #include <process.h>
#else
    #include <pthread.h>
#endif

#include <stdlib.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <sys/times.h>

#include <opencv/cv.h>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <visp/vpIoTools.h>
#include <visp/vpImageIo.h>
#include <visp/vpParseArgv.h>
#include <visp/vpMatrix.h>

#include <string>
#include <boost/thread.hpp>
//#include "ccd.h"
#include "RGBDDataProcessing.h"
#include "MeshProcessing.h"
#include "ImageConverter.h"


using namespace std;
using namespace cv;

namespace sofa {

namespace rgbdtracking {

using helper::vector;
using namespace sofa::defaulttype;

template<class DataTypes>
class RegistrationRigid : public virtual core::objectmodel::BaseObject {
public:
    SOFA_CLASS(SOFA_TEMPLATE(RegistrationRigid,DataTypes),sofa::core::objectmodel::BaseObject);

    typedef sofa::core::objectmodel::BaseObject Inherit;
    typedef defaulttype::ImageF DepthTypes;

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecReal VecReal;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef Data<typename DataTypes::VecCoord> DataVecCoord;
    typedef Data<typename DataTypes::VecDeriv> DataVecDeriv;
    typedef sofa::defaulttype::Vector4 Vector4;
    typedef core::behavior::MechanicalState<DataTypes> MechanicalState;
    enum { N=DataTypes::spatial_dimensions };
    typedef defaulttype::Mat<N,N,Real> Mat;
    typedef helper::fixed_array <unsigned int,3> tri;

    typename core::behavior::MechanicalState<DataTypes> *mstate;

public:
    RegistrationRigid();
    virtual ~RegistrationRigid();
	
    static std::string templateName(const RegistrationRigid<DataTypes>* = NULL) { return DataTypes::Name();    }
    virtual std::string getTemplateName() const    { return templateName(this);    }

    // -- Base object interface
    void reinit();
    void init();
    void handleEvent(sofa::core::objectmodel::Event *event);
    void RegisterRigid();
	
protected :
	
//    core::objectmodel::SingleLink<
//        RegistrationRigid<DataTypes>,
//        RGBDDataProcessing<DataTypes>,
//        BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> l_rgbddataprocessing;
//    core::objectmodel::SingleLink<
//        RegistrationRigid<DataTypes>,
//        MeshProcessing<DataTypes>,
//        BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> l_meshprocessing;

    // Input
    Data<VecCoord> d_targetPositions ;
    Data<VecCoord> d_sourceVisiblePositions ;

    // Method config
    Data<bool> useVisible;
    Data<bool> forceRegistration;
    Data<int> niterations;
    Data<int> startimage;
    Data<int> stopAfter;
    Data<bool> MeshToPointCloud;

    // Output
    Data< VecReal > translation;
    Data< VecReal > rotation;
    Data< VecCoord > rigidForces;


    sofa::core::behavior::MechanicalState< DataTypes > *mstateRigid;

    void determineRigidTransformation ();
    void determineRigidTransformationVisible ();
    double determineErrorICP();
};


/*#if defined(SOFA_EXTERN_TEMPLATE) && !defined(RegistrationRigid_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_RGBDTRACKING_API RegistrationRigid<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_RGBDTRACKING_API RegistrationRigid<defaulttype::Vec3fTypes>;
#endif
#endif*/


//

} //

} // namespace sofa

#endif
