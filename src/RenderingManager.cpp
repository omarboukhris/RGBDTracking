/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#define SOFA_RGBDTRACKING_RENDERINGMANAGER_CPP

#include "RenderingManager.h"
#include <sofa/simulation/VisualVisitor.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/core/core.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/helper/accessor.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/vector.h>
#include <SofaBaseTopology/TopologyData.h>

#include <sofa/helper/gl/Color.h>
#include <sofa/core/ObjectFactory.h>
#include <SofaBaseVisual/InteractiveCamera.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/simulation/Simulation.h>

#include <opencv2/opencv.hpp>


namespace sofa {

namespace rgbdtracking {

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(RenderingManager)

// Register in the Factory
int RenderingManagerClass = core::RegisterObject("Compute forces based on closest points from/to a target surface/point set")
    .add< RenderingManager >()
;

using namespace core::visual;

const std::string RenderingManager::DEPTH_OF_FIELD_VERTEX_SHADER = "shaders/depthOfField.vert";
const std::string RenderingManager::DEPTH_OF_FIELD_FRAGMENT_SHADER = "shaders/depthOfField.frag";

RenderingManager::RenderingManager()
    :zNear(initData(&zNear, (double) 1.0, "zNear", "Set zNear distance (for Depth Buffer)"))
    ,zFar(initData(&zFar, (double) 100.0, "zFar", "Set zFar distance (for Depth Buffer)"))
    ,useBBox(initData(&useBBox, true, "useBBox", "Option to use a bounding box around the rendered scene for glreadpixels"))
    ,BBox(initData(&BBox, "BBox", "Bounding box around the rendered scene for glreadpixels"))
    ,niterations(initData(&niterations,1,"niterations","Number of iterations in the tracking process"))
    ,useRenderAR(initData(&useRenderAR, false, "useRenderAR", "Option to enable augmented reality overlay"))
{
    // TODO Auto-generated constructor stub
}

RenderingManager::~RenderingManager()
{

}


void RenderingManager::init()
{
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);

}

void RenderingManager::initVisual()
{

}

void RenderingManager::preDrawScene(VisualParams* vp)
{

}

bool RenderingManager::drawScene(VisualParams* vp)
{
    return false;
}

void RenderingManager::postDrawScene(VisualParams* /*vp*/) {
//    std::cout << "IN" << std::endl ;

    helper::AdvancedTimer::stepBegin("Rendering") ;

    double znear, zfar ;
    if (l_currentcamera) {
        znear = l_currentcamera->getZNear();
        zfar = l_currentcamera->getZFar();
    } else {
        sofa::simulation::Node::SPtr root = dynamic_cast<simulation::Node*>(this->getContext());
        sofa::component::visualmodel::BaseCamera::SPtr currentCamera;
        root->get(currentCamera);

        znear = currentCamera->getZNear();
        zfar = currentCamera->getZFar();
    }
    zNear.setValue(znear);
    zFar.setValue(zfar);
    std::cout << zNear << " " << zFar << std::endl ;

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT,viewport);

    int x_1 = viewport[0];
    int y_1 = viewport[1];

    int wdth = viewport[2];
    int hght = viewport[3];

    int wdth_1,hght_1/*, x_1, y_1;*/ ;
    if (useBBox.getValue() && BBox.getValue()[2]>0 &&
        (int)this->getContext()->getTime()>5) {
        x_1 = BBox.getValue()[0];
        y_1 = BBox.getValue()[1];
        wdth_1 = BBox.getValue()[2];
        hght_1 = BBox.getValue()[3];
    } else {
//        x_1 = x,
//        y_1 = y;
        wdth_1 = wdth;
        hght_1 = hght;
    }
    float *depths = new float[wdth_1 * hght_1 ];
    cv::Mat depthm;
    depthm.create(hght, wdth, CV_32F);

    //std::cout << " znear1 " << znear << " zfar1 " << zfar << std::endl;
    //std::cout << " viewport1 " << x_1 << " "<< x_1 + wdth_1 << " " << y_1 << " " <<y_1 + hght_1 << std::endl;

    if ( (int)this->getContext()->getTime()%niterations.getValue() == 0) {
        glReadPixels(x_1, y_1, wdth_1, hght_1, GL_DEPTH_COMPONENT, GL_FLOAT, depths);

        for (int j = x_1; j < x_1 + wdth_1; ++j) {
            for (int i = y_1; i< y_1 + hght_1; ++i) {
                if ((float)depths[(j-x_1)+(i-y_1)*wdth_1] < 1  &&
                    (float)depths[(j-x_1)+(i-y_1)*wdth_1] > 0 // between [0, 1]
                ) {
                    depthm.at<float>(hght-i-1,j) = depths[(j-x_1)+(i-y_1)*wdth_1];
                }
            }
        }

        depthmat = depthm.clone();
        cv::imshow("img2", depthm) ;
//        std::cout << depthm << std::endl ;

        if (useRenderAR.getValue()) {
            texturemat.create(hght,wdth, CV_8UC3);
            glReadBuffer(GL_FRONT);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glReadPixels(viewport[0], viewport[1], viewport[2], viewport[3], GL_RGB, GL_UNSIGNED_BYTE, texturemat.data);
            glReadBuffer(GL_BACK);
        }
    }
    helper::AdvancedTimer::stepEnd("Rendering") ;
    std::cout << "OUT" << std::endl ;
    delete depths;

}

void RenderingManager::handleEvent(sofa::core::objectmodel::Event* /*event*/)
{
}

} //rgbdtracking

} //sofa
