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

#ifndef SOFA_RGBDTRACKING_CLOSESTPOINT_H
#define SOFA_RGBDTRACKING_CLOSESTPOINT_H

#include <RGBDTracking/config.h>
#include <image/ImageTypes.h>
#include <sofa/core/core.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Event.h>

#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/helper/accessor.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/helper/kdTree.inl>

#include <vector>
#include <opencv/cv.h>
#include <boost/thread.hpp>

#include <pcl/search/impl/search.hpp>

using namespace std;
using namespace cv;


namespace sofa {

namespace rgbdtracking {


using helper::vector;
using namespace sofa::defaulttype;
using namespace sofa::component::topology;


template<class DataTypes>
class ClosestPointInternalData
{
public:
};

/*
* DetectBorder
* initSourceSurface
* updateSourceSurface
* updateClosestPointsContoursNormals
* 
* 
* aren't used anymore
*/

template<class DataTypes>
class ClosestPoint : public sofa::core::objectmodel::BaseObject
{
public:
	SOFA_CLASS(SOFA_TEMPLATE(ClosestPoint,DataTypes),sofa::core::objectmodel::BaseObject);

	typedef sofa::core::objectmodel::BaseObject Inherit;
	
	typedef typename DataTypes::Real Real;
	typedef typename DataTypes::Coord Coord;
	typedef typename DataTypes::Deriv Deriv;
	typedef typename DataTypes::VecCoord VecCoord;
	typedef typename DataTypes::VecReal VecReal;
	typedef typename DataTypes::VecDeriv VecDeriv;
	typedef Data<typename DataTypes::VecCoord> DataVecCoord;
	typedef Data<typename DataTypes::VecDeriv> DataVecDeriv;
	typedef sofa::defaulttype::Vector4 Vector4;
	typedef sofa::defaulttype::Vector2 Vec2;

	typedef core::behavior::MechanicalState<DataTypes> MechanicalState;
	typename core::behavior::MechanicalState<DataTypes> *mstate;
	
	enum { N=DataTypes::spatial_dimensions };
	typedef defaulttype::Mat<N,N,Real> Mat;

	typedef helper::fixed_array <unsigned int,3> tri;
	typedef helper::kdTree<Coord> KDT;
	typedef typename KDT::distanceSet distanceSet;

public:
	ClosestPoint();
	virtual ~ClosestPoint();

	/*static std::string templateName(const ClosestPoint<DataTypes >* = NULL) { return DataTypes::Name()+ std::string(",");    }
	virtual std::string getTemplateName() const    { return templateName(this);    }*/
	
	// -- baseobject interface
	void init();

	void updateClosestPoints(); //
	void updateClosestPointsContours(); //

	void initSource(); // built k-d tree and identify border vertices
	void initSourceVisible(); // built k-d tree and identify border vertices

	// accessors
	vector< distanceSet > getClosestSource(){return closestSource;}
	vector< distanceSet > getClosestTarget(){return closestTarget;}

	vector< bool > getSourceIgnored(){return sourceIgnored;}
	vector< bool > getTargetIgnored(){return targetIgnored;}
	
	vector< int > getIndices(){return indices;}
	vector< int > getIndicesTarget(){return indicesTarget;}
	
	// algorithm param, should be changed to simple data or something
	Data<Real> blendingFactor;
	Data<Real> outlierThreshold;
	Data<bool> rejectBorders;
	Data<bool> useVisible;
	Data<bool> useDistContourNormal;

	defaulttype::BoundingBox targetBbox;


	// source mesh data
	Data< VecCoord > sourcePositions;
	Data< VecCoord > sourceVisiblePositions;
	Data< helper::vector< tri > > sourceTriangles;
	Data< VecCoord > sourceNormals;
	Data< VecCoord > sourceSurfacePositions;
	Data< VecCoord > sourceSurfaceNormals;
	Data< VecCoord > sourceSurfaceNormalsM;
	Data< VecCoord > sourceContourPositions;
	Data< helper::vector< Vec2 > > sourceContourNormals;

	vector< distanceSet >  closestSource; // CacheSize-closest target points from source
	vector< Real > cacheDist;
	vector< Real > cacheDist2;
	VecCoord previousX; // storage for cache acceleration

	KDT sourceKdTree;
	vector< bool > sourceBorder;
	vector< bool > sourceIgnored;  // flag ignored vertices
	vector< bool > sourceSurface;
	std::vector<int> indices;

	// target point cloud data
	Data< VecCoord > targetPositions;
	Data< VecCoord > targetGtPositions;
	Data< VecCoord > targetNormals;
	vector< bool > targetIgnored;  // flag ignored vertices
	Data< VecCoord > targetContourPositions;

	KDT targetKdTree;
	KDT targetContourKdTree;

	std::vector<int> indicesTarget;
	helper::vector< bool > targetBorder;
	helper::vector< distanceSet >  closestTarget; // CacheSize-closest source points from target

	void initTarget();  // built k-d tree and identify border vertices
	void initTargetContour();  // built k-d tree and identify border vertices

	Eigen::Matrix3f rgbIntrinsicMatrix;

private :

    static void getNClosest (KDT & kdtree, vector<distanceSet> & closestgeom, const VecCoord & veccoord, const VecCoord & positions) ;

    static void rejectBorder (const vector<distanceSet> & closestGeom, const vector<bool> & borderGeom, vector<bool> & ignoredGeom) ;
    static void ignoreElement (const vector<distanceSet> & closestGeom, vector<bool> & ignoredGeom, Real mean) ;
    static void sum_meanStdev (const vector<distanceSet> & closestGeom, Real & mean, Real & stdev, Real & count) ;

};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(ClosestPoint_CPP)
//#ifndef SOFA_FLOAT
extern template class SOFA_RGBDTRACKING_API ClosestPoint<defaulttype::Vec3dTypes>;
//#endif
//#ifndef SOFA_DOUBLE
//extern template class SOFA_RGBDTRACKING_API ClosestPoint<defaulttype::Vec3fTypes>;
//#endif
#endif


} //

} // namespace sofa

#endif
