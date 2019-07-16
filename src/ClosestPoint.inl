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

#define SOFA_RGBDTRACKING_CLOSESTPOINT_INL

#include <limits>
#include <iterator>
#include <sofa/helper/gl/Color.h>
#include <sofa/simulation/Simulation.h>
#include <iostream>
#include <map>

#ifdef USING_OMP_PRAGMAS
	#include <omp.h>
#endif

#include <algorithm> 
#include "ClosestPoint.h"

using std::cerr;
using std::endl;


namespace sofa {

namespace rgbdtracking {


using namespace sofa::defaulttype;
using namespace helper;

template <class DataTypes>
ClosestPoint<DataTypes>::ClosestPoint()
	: Inherit()
	, blendingFactor(initData(&blendingFactor,(Real)1,"blendingFactor","blending between projection (=0) and attraction (=1) forces."))
	, outlierThreshold(initData(&outlierThreshold,(Real)7,"outlierThreshold","suppress outliers when distance > (meandistance + threshold*stddev)."))
	, rejectBorders(initData(&rejectBorders,false,"rejectBorders","ignore border vertices."))
	, useVisible(initData(&useVisible,true,"useVisible","Use the vertices of the visible surface of the source mesh"))
	, useDistContourNormal(initData(&useDistContourNormal,false,"useVisible","Use the vertices of the visible surface of the source mesh"))
{
}

template <class DataTypes>
ClosestPoint<DataTypes>::~ClosestPoint()
{
}

template <class DataTypes>
void ClosestPoint<DataTypes>::init()
{
	this->Inherit::init();
	// Get source normals
	//if(!sourceNormals.getValue().size()) serr<<"normals of the source model not found"<<sendl;		
}

template<class DataTypes>
void ClosestPoint<DataTypes>::initSource()
{
	// build k-d tree
	const VecCoord&  p = sourcePositions.getValue();
	sourceKdTree.build(p);
	
	// detect border
/* if(sourceBorder.size()!=p.size())
	{ sourceBorder.resize(p.size()); 
	//detectBorder(sourceBorder,sourceTriangles.getValue()); 
	}*/
}

template<class DataTypes>
void ClosestPoint<DataTypes>::initSourceVisible()
{
	// build k-d tree
	
	const VecCoord&  p = sourceVisiblePositions.getValue();
	
	sourceKdTree.build(p);
	// detect border
	/*if(sourceBorder.size()!=p.size())
	{
		sourceBorder.resize(p.size());

	//detectBorder(sourceBorder,sourceTriangles.getValue()); 
	}*/
}

template<class DataTypes>
void ClosestPoint<DataTypes>::initTarget()
{
	const VecCoord&  p = targetPositions.getValue();
	
	targetKdTree.build(p);

	// updatebbox
	for(unsigned int i=0;i<p.size();++i) {
		targetBbox.include(p[i]);
	}
	// detect border
	//if(targetBorder.size()!=p.size()) { targetBorder.resize(p.size()); detectBorder(targetBorder,targetTriangles.getValue()); }

}

template<class DataTypes>
void ClosestPoint<DataTypes>::initTargetContour()
{

	const VecCoord&  p = targetContourPositions.getValue();
	targetContourKdTree.build(p);

	// updatebbox
	//for(unsigned int i=0;i<p.size();++i)    targetBbox.include(p[i]);

	// detect border
	//if(targetBorder.size()!=p.size()) { targetBorder.resize(p.size()); detectBorder(targetBorder,targetTriangles.getValue()); }

}

template<class DataTypes>
void ClosestPoint<DataTypes>::updateClosestPoints()
{
	// load input (geometries to match)
	VecCoord x;
	if (!useVisible.getValue()) {
		x = sourcePositions.getValue();
	} else {
		x = sourceVisiblePositions.getValue();
	}
	const VecCoord&  tp = targetPositions.getValue();
	unsigned int nbs=x.size(), nbt=tp.size();

	// init source
	distanceSet emptyset;	
	if(nbs!=closestSource.size()) {
		// this is disposable
		if (!useVisible.getValue()) {
			initSource();
		} else {
			initSourceVisible();
		} // ends here
		closestSource.resize(nbs);
		closestSource.fill(emptyset);
		cacheDist.resize(nbs);
		cacheDist.fill((Real)0.);
		cacheDist2.resize(nbs);
		cacheDist2.fill((Real)0.);
		previousX.assign(x.begin(),x.end());
	}

	/*if(nbtc!=closestSourceContour.size()) {initSource();  closestSourceContour.resize(nbtc);
	closestSourceContour.fill(emptyset); 
	cacheDist.resize(nbtc); 
	cacheDist.fill((Real)0.); 
	cacheDist2.resize(nbtc); 
	cacheDist2.fill((Real)0.); 
	previousX.assign(x.begin(),x.end());}*/

	// init target
	if(nbt!=closestTarget.size()) {
		initTarget();
		closestTarget.resize(nbt);
		closestTarget.fill(emptyset);
	}
	
	if(blendingFactor.getValue()<1 && nbt>0) {
	//unsigned int count=0;
#ifdef USING_OMP_PRAGMAS
#pragma omp parallel for
#endif
        ClosestPoint<DataTypes>::getNClosest(targetKdTree, closestSource, x, targetPositions.getValue()) ;
	}
		
	// closest source points from target points
	if(blendingFactor.getValue()>0) {
		if (!useVisible.getValue()) {
			initSource();
            ClosestPoint<DataTypes>::getNClosest(sourceKdTree, closestTarget, tp, sourcePositions.getValue()) ;
		} else {
			initSourceVisible();
            ClosestPoint<DataTypes>::getNClosest(sourceKdTree, closestTarget, tp, sourceVisiblePositions.getValue()) ;
		}
	}

    // setup all to false
    this->sourceIgnored.resize(nbs);
    this->targetIgnored.resize(nbt);
	sourceIgnored.fill(false);
	targetIgnored.fill(false);

	// prune outliers
	if(outlierThreshold.getValue()!=0) {
		Real mean=0,stdev=0,count=0;
        ClosestPoint<DataTypes>::sum_meanStdev(closestSource, mean, stdev, count) ;
        ClosestPoint<DataTypes>::sum_meanStdev(closestTarget, mean, stdev, count) ;

        //post computing
		mean=mean/count; 
		stdev=(Real)sqrt(stdev/count-mean*mean);
		mean+=stdev*outlierThreshold.getValue();

        ClosestPoint<DataTypes>::ignoreElement(closestSource, sourceIgnored, mean) ;
        ClosestPoint<DataTypes>::ignoreElement(closestTarget, targetIgnored, mean) ;
		
		if(rejectBorders.getValue()) {
            ClosestPoint<DataTypes>::rejectBorder(closestSource, targetBorder, sourceIgnored) ;
            ClosestPoint<DataTypes>::rejectBorder(closestTarget, sourceBorder, targetIgnored) ;
		}
	}
}

template<class DataTypes>
void ClosestPoint<DataTypes>::getNClosest (KDT & kdtree, vector<distanceSet> & closestgeom, const VecCoord & veccoord, const VecCoord & positions) {
    for (size_t i = 0 ; i < closestgeom.size() ; ++i) {
        kdtree.getNClosest(closestgeom[i], veccoord[i], positions, 1);
    }
}

template<class DataTypes>
void ClosestPoint<DataTypes>::sum_meanStdev (const vector<distanceSet> & closestGeom, Real & mean, Real & stdev, Real & count) {
    for(size_t i = 0 ; i < closestGeom.size() ; ++i) {
        if(closestGeom[i].size() ) {
            count++;
            stdev+=(closestGeom[i].begin()->first)*(closestGeom[i].begin()->first);
            mean+=(Real)(closestGeom[i].begin()->first);
        }
    }
}

template<class DataTypes>
void ClosestPoint<DataTypes>::ignoreElement (const vector<distanceSet> & closestGeom, vector<bool> & ignoredGeom, Real mean) {
    for(size_t i = 0 ; i < closestGeom.size() ; ++i) {
        if(closestGeom[i].size() && closestGeom[i].begin()->first > mean ) {
            ignoredGeom[i]=true;
        }
    }
}

template<class DataTypes>
void ClosestPoint<DataTypes>::rejectBorder (const vector<distanceSet> & closestGeom, const vector<bool> & borderGeom, vector<bool> & ignoredGeom) {
    for(size_t i = 0 ; i < closestGeom.size() ; ++i) {
        if (closestGeom[i].size() &&
			borderGeom[closestGeom[i].begin()->second]) {
				ignoredGeom[i]=true ;
		}
	}
}

template<class DataTypes>
void ClosestPoint<DataTypes>::updateClosestPointsContours()
{
	
	VecCoord x,x0;
	if (!useVisible.getValue()) {
		x = sourcePositions.getValue();
	} else {
		x = sourceVisiblePositions.getValue();
	}
	x0 = sourcePositions.getValue();
	
	const VecCoord& tp = targetPositions.getValue();
	const VecCoord& tcp = targetContourPositions.getValue();

	unsigned int nbs=x.size(), nbt=tp.size(), nbs0=x0.size();

	distanceSet emptyset;
	
	if(nbs!=closestSource.size()) {
		if (!useVisible.getValue()) {
			initSource();
		} else {
			initSourceVisible();
		}
		closestSource.resize(nbs);
		closestSource.fill(emptyset);
		cacheDist.resize(nbs);
		cacheDist.fill((Real)0.);
		cacheDist2.resize(nbs);
		cacheDist2.fill((Real)0.);
		previousX.assign(x.begin(),x.end());
	}

	if(nbt!=closestTarget.size()) {
		initTarget();
		initTargetContour();
		closestTarget.resize(nbt);
		closestTarget.fill(emptyset);
	}

	indicesTarget.resize(0);
		
	// closest target points from source points
	if(blendingFactor.getValue()<1) {
		for(int i=0;i<(int)nbt;i++) {
			//int id = indicesVisible[i];
			if(targetBorder[i]) {
			// && t%niterations.getValue() == 0)
				double distmin = 10;
				double dist;
				int kmin;
				for (int k = 0; k < x0.size(); k++) {
					if (sourceBorder[k]) {
						dist = (tp[i][0] - x0[k][0])*(tp[i][0] - x0[k][0]) +
							(tp[i][1] - x0[k][1])*(tp[i][1] - x0[k][1]) +
							(tp[i][2] - x0[k][2])*(tp[i][2] - x0[k][2]);

						if (dist < distmin) {
							distmin = dist;
							kmin = k;
						}
					}
				}
				indicesTarget.push_back(kmin);
			}
		}
		
#ifdef USING_OMP_PRAGMAS
		#pragma omp parallel for
#endif
        ClosestPoint<DataTypes>::getNClosest(targetKdTree, closestSource, x, targetPositions.getValue()) ;
	}
	indices.resize(0);
		
#ifdef USING_OMP_PRAGMAS
		#pragma omp parallel for
#endif
	int kc = 0;
	for(int i=0;i<(int)nbs0;i++) {
		if(sourceBorder[i]) {
			// && t%niterations.getValue() == 0)

			double distmin = 1000;
			double distmin1;
			double dist, dist1,dist2;
			int kmin2,kmin1;

			for (int k = 0; k < tcp.size(); k++) {
				dist = (x0[i][0] - tcp[k][0])*(x0[i][0] - tcp[k][0]) +
					(x0[i][1] - tcp[k][1])*(x0[i][1] - tcp[k][1]) +
					(x0[i][2] - tcp[k][2])*(x0[i][2] - tcp[k][2]);
				if (dist < distmin) {
					distmin = dist;
					kmin1 = k;
				}
			}
			double x_u_1 = ((x0[i][0])*rgbIntrinsicMatrix(0,0)/x0[i][2] + rgbIntrinsicMatrix(0,2)) - ((tcp[kmin1][0])*rgbIntrinsicMatrix(0,0)/tcp[kmin1][2] + rgbIntrinsicMatrix(0,2));
			double x_v_1 = ((x0[i][1])*rgbIntrinsicMatrix(1,1)/x0[i][2] + rgbIntrinsicMatrix(1,2)) - ((tcp[kmin1][1])*rgbIntrinsicMatrix(1,1)/tcp[kmin1][2] + rgbIntrinsicMatrix(1,2));
			double distmin0 = distmin;
			distmin = 1000;
			for (int k = 0; k < tcp.size(); k++) {
				dist = (x0[i][0] - tcp[k][0])*(x0[i][0] - tcp[k][0]) + (x0[i][1] - tcp[k][1])*(x0[i][1] - tcp[k][1]) + (x0[i][2] - tcp[k][2])*(x0[i][2] - tcp[k][2]);
				double x_u_2 = ((x0[i][0])*rgbIntrinsicMatrix(0,0)/x0[i][2] + rgbIntrinsicMatrix(0,2)) - ((tcp[k][0])*rgbIntrinsicMatrix(0,0)/tcp[k][2] + rgbIntrinsicMatrix(0,2));
				double x_v_2 = ((x0[i][1])*rgbIntrinsicMatrix(1,1)/x0[i][2] + rgbIntrinsicMatrix(1,2)) - ((tcp[k][1])*rgbIntrinsicMatrix(1,1)/tcp[k][2] + rgbIntrinsicMatrix(1,2));

				dist2 = abs(sourceContourNormals.getValue()[kc][1]*x_u_2 - sourceContourNormals.getValue()[kc][0]*x_v_2);
				dist1 = x_u_2*x_u_1 + x_v_2*x_v_1;

				if (dist2 < distmin && sqrt(dist) < 0.10 &&
					dist1 > 0 && sqrt(dist)/sqrt(distmin0)< 5
				) {
					distmin = dist2;
					kmin2 = k;
					distmin1 = dist1;
				}
			}

			if (useDistContourNormal.getValue()) {
				indices.push_back(kmin2);
			} else {
				indices.push_back(kmin1);
			}
			kc++;

		}
	}
	//std::cout << " indices size " << indices.size() << " tcp size " << tcp.size() << " xcp size " << xcp.size() << std::endl;

	// closest source points from target points
	if(blendingFactor.getValue()>0) {
		if (!useVisible.getValue()) {
			initSource();
		} else {
			initSourceVisible();
		}
	#ifdef USING_OMP_PRAGMAS
			#pragma omp parallel for
	#endif
        ClosestPoint<DataTypes>::getNClosest(sourceKdTree, closestTarget, tp, sourcePositions.getValue()) ;
	}
    // setup all to false
    this->sourceIgnored.resize(nbs);
    this->targetIgnored.resize(nbt);
    sourceIgnored.fill(false);
    targetIgnored.fill(false);

    // prune outliers
    if(outlierThreshold.getValue()!=0) {
        Real mean=0,stdev=0,count=0;
        ClosestPoint<DataTypes>::sum_meanStdev(closestSource, mean, stdev, count) ;
        ClosestPoint<DataTypes>::sum_meanStdev(closestTarget, mean, stdev, count) ;

        //post computing
        mean = mean/count;
        stdev = (Real)sqrt(stdev/count-mean*mean);
        mean += stdev*outlierThreshold.getValue();
        mean *= mean;

        ClosestPoint<DataTypes>::ignoreElement(closestSource, sourceIgnored, mean) ;
        ClosestPoint<DataTypes>::ignoreElement(closestTarget, targetIgnored, mean) ;

        if(rejectBorders.getValue()) {
            ClosestPoint<DataTypes>::rejectBorder(closestSource, targetBorder, sourceIgnored) ;
            ClosestPoint<DataTypes>::rejectBorder(closestTarget, sourceBorder, targetIgnored) ;
        }
    }

}

}

} // namespace sofa



